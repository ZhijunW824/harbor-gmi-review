"""Offline unit tests for the GMI environment; the SDK client is faked."""

import asyncio
import logging
import shlex
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("sandbox_sdk", reason="requires the optional 'gmi' extra")

from sandbox_sdk.errors import (  # noqa: E402
    ConflictError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TransportError,
)

from harbor.environments import gmi  # noqa: E402
from harbor.environments.gmi import GMIEnvironment, GMIExecTimeoutError  # noqa: E402
from harbor.models.task.config import EnvironmentConfig  # noqa: E402
from harbor.models.trial.paths import TrialPaths  # noqa: E402

IMAGE = "python:3.12-slim"
RUNNING = {"status": "running"}
STALE = ConflictError(
    409,
    "The template created by this idempotency key no longer exists; "
    "use a new key to create another",
)
SANDBOX = {
    "id": "sb-1",
    "domain": "sandbox.test",
    "sandbox_key": "key-1",
    "sandbox_access_token": "token",
}


def done(exit_code=0, **fields):
    return {
        "status": "succeeded",
        "exit_code": exit_code,
        "stdout": "",
        "stderr": "",
        **fields,
    }


class Record:
    def __init__(self, **data):
        self.data = data


class FakeExecution:
    def __init__(self, states, refresh_errors, delay):
        self._states = list(states) or [done()]
        self.data = self._states.pop(0)
        self.execution_id = "exec-1"
        self.cancels = 0
        self._refresh_errors = refresh_errors
        self._delay = delay

    status = property(lambda self: self.data.get("status"))
    exit_code = property(lambda self: self.data.get("exit_code"))
    stdout = property(lambda self: self.data.get("stdout"))
    stderr = property(lambda self: self.data.get("stderr"))

    def refresh(self):
        threading.Event().wait(self._delay)
        if self._refresh_errors:
            raise self._refresh_errors.pop(0)
        if self._states:
            self.data = self._states.pop(0)
        return self

    def cancel(self):
        self.cancels += 1


class FakeCommands:
    def __init__(self):
        self.calls, self.executions = [], []
        self.scripts, self.errors, self.refresh_errors = [], [], []
        self.refresh_delay = 0.0

    def run(self, command, *, envs=None, cwd=None, wait=False):
        self.calls.append(
            SimpleNamespace(command=command, envs=envs, cwd=cwd, wait=wait)
        )
        if self.errors:
            raise self.errors.pop(0)
        states = self.scripts.pop(0) if self.scripts else []
        execution = FakeExecution(states, self.refresh_errors, self.refresh_delay)
        self.executions.append(execution)
        return execution


class FakeSandbox(Record):
    def __init__(self, **data):
        super().__init__(**{**SANDBOX, **data})
        self.commands = FakeCommands()
        self.files = SimpleNamespace(contents={})
        self.files.write = lambda path, content: self.files.contents.__setitem__(
            path, content
        )
        self.files.read = lambda path: self.files.contents[path]


class FakeClient:
    """Enough of SandboxClient; `_request` serves the SDK's own Sandbox/Template.delete()."""

    def __init__(self):
        self.sandbox = FakeSandbox()
        self.listed, self.build_states, self.sandbox_states = [], ["ready"], ["running"]
        self.template_results, self.sandbox_results = [], []
        self.template_creates, self.sandbox_creates, self.requests = [], [], []
        self.templates = SimpleNamespace(
            list=self._list_templates,
            create=self._create_template,
            get=self._get_template,
        )
        self.sandboxes = SimpleNamespace(
            create=self._create_sandbox, get=self._get_sandbox
        )

    @staticmethod
    def _next(queue, default):
        result = queue.pop(0) if queue else default
        if isinstance(result, BaseException):
            raise result
        return result

    def _list_templates(self, *, page, page_size, idc_name=None):
        chunk = self.listed[(page - 1) * page_size : page * page_size]
        return SimpleNamespace(items=[Record(**item) for item in chunk])

    def _create_template(self, **kwargs):
        self.template_creates.append(kwargs)
        return Record(**self._next(self.template_results, {"id": "tpl-new"}))

    def _get_template(self, template_id, *, idc_name=None):
        states = self.build_states
        return Record(
            id=template_id,
            latest_build_status=states.pop(0) if len(states) > 1 else states[0],
        )

    def _create_sandbox(self, **kwargs):
        self.sandbox_creates.append(kwargs)
        return self._next(self.sandbox_results, self.sandbox)

    def _get_sandbox(self, sandbox_id):
        states = self.sandbox_states
        return Record(
            id=sandbox_id, state=states.pop(0) if len(states) > 1 else states[0]
        )

    def _request(self, method, path, **kwargs):
        self.requests.append((method, path))
        return {}

    @property
    def deletes(self):
        return [path for method, path in self.requests if method == "DELETE"]


@pytest.fixture
def client(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(gmi, "SandboxClient", lambda: fake)
    monkeypatch.setenv("GMI_SANDBOX_IDC_NAME", "test-idc")
    return fake


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    real_sleep = asyncio.sleep

    async def instant(delay, result=None):
        await real_sleep(0)
        return result

    monkeypatch.setattr(asyncio, "sleep", instant)
    monkeypatch.setattr(gmi.time, "sleep", lambda _delay: None)


def make_env(
    path, *, dockerfile="FROM python:3.12-slim\nWORKDIR /app\n", files=(), **config
):
    env_dir = path / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    if dockerfile is not None:
        (env_dir / "Dockerfile").write_text(dockerfile)
    for name in files:
        (env_dir / name).write_text("services: {}\n")
    kwargs = {
        key: config.pop(key)
        for key in ("template_id", "product", "mounts")
        if key in config
    }
    config.setdefault("docker_image", IMAGE)
    trial_paths = TrialPaths(trial_dir=path / "trial")
    trial_paths.mkdir()
    return GMIEnvironment(
        environment_dir=env_dir,
        environment_name="task",
        session_id="task__abc__env",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(**config),
        **kwargs,
    )


def ready_env(path, client):
    env = make_env(path)
    env._sandbox, env._sandbox_id = client.sandbox, "sb-1"
    return env


def alias_of(env):
    resources = {"type": "preset", "product": env._product}
    build = {"source": {"type": "image", "image": IMAGE}}
    return gmi._template_alias(env.environment_id, resources, build)


def keys(client):
    return [create["idempotency_key"] for create in client.template_creates]


async def eventually(predicate):
    for _ in range(300):
        if predicate():
            return True
        await asyncio.to_thread(threading.Event().wait, 0.01)
    return predicate()


# --- definition ---


def test_dockerfile_only_task_is_rejected(tmp_path, client):
    with pytest.raises(ValueError, match="docker_image"):
        make_env(tmp_path, docker_image=None)


def test_task_without_any_definition_is_rejected(tmp_path, client):
    with pytest.raises(FileNotFoundError):
        make_env(tmp_path, dockerfile=None, docker_image=None)


def test_compose_task_is_rejected(tmp_path, client):
    with pytest.raises(ValueError, match="Compose"):
        make_env(tmp_path, files=["docker-compose.yaml"])


def test_template_id_needs_no_other_definition(tmp_path, client):
    make_env(tmp_path, dockerfile=None, docker_image=None, template_id="tpl-1")


def test_workdir_variable_needs_an_explicit_workdir(tmp_path, client):
    dockerfile = "FROM python:3.12-slim\nWORKDIR $APP_HOME\n"
    with pytest.raises(ValueError, match="WORKDIR"):
        make_env(tmp_path, dockerfile=dockerfile)
    make_env(tmp_path / "ok", dockerfile=dockerfile, workdir="/srv")


@pytest.mark.parametrize(
    ("given", "expected"),
    [(None, None), ("", None), (0, "0"), (12345, "12345"), ("t", "t")],
)
def test_template_id_is_normalized(tmp_path, client, given, expected):
    assert make_env(tmp_path, template_id=given)._template_id == expected


# --- templates ---


def test_template_name_matches_the_previous_adapter():
    # Same name, so Templates that the previous adapter built are reused.
    resources = {"type": "preset", "product": "gmi.sandbox.medium"}
    build = {"source": {"type": "image", "image": IMAGE}}
    alias = gmi._template_alias("0123456789abcdef_0123456789abcdef", resources, build)
    assert alias == "harbor-0123456789abcdef-012-1b21eff7"


async def test_ready_template_is_reused(tmp_path, client):
    env = make_env(tmp_path)
    client.listed = [
        {"id": "tpl-1", "name": alias_of(env), "latest_build_status": "ready"}
    ]
    assert await env._ensure_template() == "tpl-1"
    assert client.template_creates == [] and client.deletes == []


@pytest.mark.parametrize("status", ["building", "waiting", "", "something-new"])
async def test_unfinished_template_is_awaited_not_deleted(tmp_path, client, status):
    env = make_env(tmp_path)
    client.listed = [
        {"id": "tpl-1", "name": alias_of(env), "latest_build_status": status}
    ]
    client.build_states = ["building", "ready"]
    assert await env._ensure_template() == "tpl-1"
    assert client.template_creates == [] and client.deletes == []


async def test_missing_template_is_built_from_the_image(tmp_path, client):
    env = make_env(tmp_path)
    assert await env._ensure_template() == "tpl-new"
    [create] = client.template_creates
    body = {k: v for k, v in create.items() if k != "idempotency_key"}
    assert body == {
        "name": alias_of(env),
        "idc_name": "test-idc",
        "resources": {"type": "preset", "product": "gmi.sandbox.small"},
        "build": {"source": {"type": "image", "image": IMAGE}},
        "description": "Harbor task task",
    }
    # A replayed key must carry the same body, so the key digests all of it.
    assert create["idempotency_key"] == gmi._payload_key(alias_of(env), body)


async def test_failed_template_is_rebuilt_under_the_next_key(tmp_path, client):
    env = make_env(tmp_path)
    client.listed = [
        {"id": "tpl-bad", "name": alias_of(env), "latest_build_status": "failed"}
    ]
    # Deleting tpl-bad retired the first key.
    client.template_results = [STALE, {"id": "tpl-new"}]
    assert await env._ensure_template() == "tpl-new"
    assert client.deletes == ["/templates/tpl-bad"]
    first, second = keys(client)
    assert second == f"{first}-1"


async def test_retired_keys_are_skipped_in_a_fixed_order(tmp_path, client):
    client.template_results = [STALE, STALE, {"id": "tpl-3"}]
    assert await make_env(tmp_path)._ensure_template() == "tpl-3"
    base = keys(client)[0]
    assert keys(client) == [base, f"{base}-1", f"{base}-2"]


async def test_running_out_of_keys_is_reported(tmp_path, client):
    client.template_results = [STALE] * gmi._KEY_GENERATIONS
    with pytest.raises(RuntimeError, match="retired"):
        await make_env(tmp_path)._ensure_template()
    assert len(keys(client)) == gmi._KEY_GENERATIONS
    assert max(map(len, keys(client))) <= 64


async def test_template_already_deleted_by_another_trial_is_quiet(
    tmp_path, client, caplog
):
    env = make_env(tmp_path)
    client.listed = [
        {"id": "tpl-bad", "name": alias_of(env), "latest_build_status": "failed"}
    ]

    def already_gone(method, path, **kwargs):
        raise NotFoundError(404, "template not found")

    client._request = already_gone
    with caplog.at_level(logging.WARNING):
        assert await env._ensure_template() == "tpl-new"
    assert "Failed to delete" not in caplog.text


async def test_ready_copy_wins_among_duplicates(tmp_path, client):
    env = make_env(tmp_path)
    alias = alias_of(env)
    client.listed = [
        {"id": "a", "name": alias, "latest_build_status": "failed"},
        {"id": "b", "name": alias, "latest_build_status": "ready"},
        {"id": "c", "name": alias, "latest_build_status": "building"},
    ]
    assert await env._ensure_template() == "b"


async def test_build_capacity_error_is_waited_out(tmp_path, client, caplog):
    busy = RateLimitError(429, "maximum concurrent template builds reached")
    client.template_results = [busy, {"id": "tpl-2"}]
    with caplog.at_level(logging.WARNING):
        assert await make_env(tmp_path)._ensure_template() == "tpl-2"
    assert len(client.template_creates) == 2
    assert "capacity is full" in caplog.text


async def test_failed_build_is_reported(tmp_path, client):
    client.build_states = ["failed"]
    with pytest.raises(RuntimeError, match="ended as 'failed'"):
        await make_env(tmp_path)._ensure_template()


# --- sandbox lifecycle ---


async def test_start_creates_and_prepares_the_sandbox(tmp_path, client):
    mount = {"type": "bind", "source": str(tmp_path), "target": "/logs/agent"}
    env = make_env(tmp_path, mounts=[mount])
    await env.start(force_build=False)
    [create] = client.sandbox_creates
    assert create["template_id"] == "tpl-new" and create["timeout"] == 86_400
    assert create["metadata"] == {
        "harbor_environment": "task",
        "harbor_session": "task__abc__env",
    }
    sshd, dirs = client.sandbox.commands.calls
    assert "/run/sshd.pid" in sshd.command and sshd.cwd == "/"
    assert (
        "test -d /app" in dirs.command and "/logs/" in dirs.command and dirs.cwd == "/"
    )


async def test_force_build_is_ignored_with_a_warning(tmp_path, client, caplog):
    with caplog.at_level(logging.WARNING):
        await make_env(tmp_path).start(force_build=True)
    assert "force_build is ignored" in caplog.text


async def test_directory_setup_failure_fails_start(tmp_path, client):
    client.sandbox.commands.scripts = [[done()], [done(exit_code=1, stderr="denied")]]
    with pytest.raises(RuntimeError, match="prepare directories: denied"):
        await make_env(tmp_path).start(force_build=False)


async def test_create_retries_a_read_timeout_under_the_same_key(tmp_path, client):
    client.sandbox_results = [
        TimeoutError("The read operation timed out"),
        client.sandbox,
    ]
    await make_env(tmp_path).start(force_build=False)
    first, second = client.sandbox_creates
    assert first["idempotency_key"] == second["idempotency_key"]


async def test_quota_error_names_the_fix(tmp_path, client):
    client.sandbox_results = [RateLimitError(429, "sandbox quota exceeded (5/5)")]
    with pytest.raises(RuntimeError, match="Keep -n within"):
        await make_env(tmp_path).start(force_build=False)


async def test_sandbox_without_a_data_plane_endpoint_is_deleted(tmp_path, client):
    client.sandbox_results = [FakeSandbox(sandbox_access_token=None)]
    with pytest.raises(RuntimeError, match="data-plane"):
        await make_env(tmp_path).start(force_build=False)
    assert client.deletes == ["/sandboxes/sb-1"]


async def test_cancelled_create_deletes_the_sandbox_that_lands_later(tmp_path, client):
    env = make_env(tmp_path)
    entered, release = threading.Event(), threading.Event()
    create = client.sandboxes.create

    def slow_create(**kwargs):
        entered.set()
        release.wait(5)
        return create(**kwargs)

    client.sandboxes.create = slow_create
    task = asyncio.create_task(env._create_sandbox("tpl-1"))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert await eventually(lambda: client.deletes == ["/sandboxes/sb-1"])
    assert env._sandbox is None


async def test_readiness_waits_through_starting_states(tmp_path, client):
    client.sandbox_states = ["", "starting", "RUNNING"]
    await make_env(tmp_path).start(force_build=False)


async def test_sandbox_that_comes_up_failed_is_reported(tmp_path, client):
    client.sandbox_states = ["failed"]
    with pytest.raises(RuntimeError, match="came up as 'failed'"):
        await make_env(tmp_path).start(force_build=False)


async def test_stop_deletes_the_sandbox(tmp_path, client):
    env = ready_env(tmp_path, client)
    await env.stop(delete=True)
    assert client.deletes == ["/sandboxes/sb-1"] and env._sandbox is None


async def test_stop_keeps_the_sandbox_when_asked(tmp_path, client):
    env = ready_env(tmp_path, client)
    await env.stop(delete=False)
    assert client.deletes == [] and env._sandbox is None


@pytest.mark.parametrize(
    ("error", "logged"),
    [(ServerError(500, "boom"), True), (NotFoundError(404, "gone"), False)],
)
async def test_stop_never_raises(tmp_path, client, caplog, error, logged):
    env = ready_env(tmp_path, client)

    def failing_request(method, path, **kwargs):
        raise error

    client._request = failing_request
    with caplog.at_level(logging.ERROR):
        await env.stop(delete=True)
    assert ("Error stopping GMI sandbox sb-1" in caplog.text) is logged
    assert env._sandbox is None


# --- exec ---


@pytest.mark.parametrize(
    ("user", "expected"),
    [
        (None, "sudo -n bash -c 'echo hi'"),
        ("root", "sudo -n bash -c 'echo hi'"),
        (0, "sudo -n bash -c 'echo hi'"),
        ("user", "echo hi"),
        ("agent", "sudo -n -u agent bash -c 'echo hi'"),
        (1000, "sudo -n -u '#1000' bash -c 'echo hi'"),
    ],
)
def test_users_other_than_the_sandbox_default_go_through_sudo(
    tmp_path, client, user, expected
):
    assert make_env(tmp_path)._wrap("echo hi", None, user) == expected


def test_env_is_exported_inside_sudo(tmp_path, client):
    wrapped = make_env(tmp_path)._wrap("printenv A", {"A": "x y"}, None)
    assert shlex.split(wrapped)[-1] == "export A='x y'; printenv A"


@pytest.mark.parametrize("name", ["BAD-NAME", "FOO\n", "1X"])
def test_invalid_env_names_are_rejected(tmp_path, client, name):
    with pytest.raises(RuntimeError, match="invalid environment variable name"):
        make_env(tmp_path)._wrap("true", {name: "v"}, None)


async def test_exec_polls_until_the_command_finishes(tmp_path, client):
    env = ready_env(tmp_path, client)
    client.sandbox.commands.scripts = [
        [RUNNING, {"status": ""}, done(3, stdout="o", stderr="e")]
    ]
    result = await env.exec("false", env={"K": "v"})
    assert (result.return_code, result.stdout, result.stderr) == (3, "o", "e")
    [call] = client.sandbox.commands.calls
    assert call.wait is False and call.cwd == "/app" and call.envs == {"K": "v"}


async def test_missing_exit_code_is_reported(tmp_path, client):
    env = ready_env(tmp_path, client)
    client.sandbox.commands.scripts = [[{"status": "timed_out", "exit_code": None}]]
    result = await env.exec("sleep 1")
    assert result.return_code == 1
    assert "execution ended as 'timed_out' with no exit code" in result.stderr


async def test_own_timeout_cancels_the_command_and_is_not_a_timeout_error(
    tmp_path, client
):
    env = ready_env(tmp_path, client)
    client.sandbox.commands.scripts = [[RUNNING]]
    client.sandbox.commands.refresh_delay = 0.05
    with pytest.raises(GMIExecTimeoutError) as caught:
        await env.exec("sleep 100", timeout_sec=1)
    assert not isinstance(caught.value, TimeoutError)
    assert client.sandbox.commands.executions[0].cancels == 1


async def test_cancelled_exec_cancels_the_command(tmp_path, client):
    env = ready_env(tmp_path, client)
    client.sandbox.commands.scripts = [[RUNNING]]
    client.sandbox.commands.refresh_delay = 0.01
    task = asyncio.create_task(env.exec("sleep 100"))
    assert await eventually(lambda: client.sandbox.commands.executions)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.sandbox.commands.executions[0].cancels == 1


async def test_polling_rides_out_transient_failures(tmp_path, client):
    env = ready_env(tmp_path, client)
    client.sandbox.commands.scripts = [[RUNNING, done()]]
    client.sandbox.commands.refresh_errors = [TransportError("reset")] * 3
    assert (await env.exec("true")).return_code == 0


@pytest.mark.parametrize(
    ("error", "attempts"),
    [
        (TransportError("[Errno 61] Connection refused"), 3),
        (TransportError("timed out"), 3),
        (RateLimitError(429, "Too many concurrently active executions."), 3),
        (TimeoutError("The read operation timed out"), 1),
    ],
)
async def test_dispatch_retries_only_what_never_arrived(
    tmp_path, client, error, attempts
):
    env = ready_env(tmp_path, client)
    client.sandbox.commands.errors = [error] * 3
    with pytest.raises((TransportError, RateLimitError)):
        await env.exec("true")
    assert len(client.sandbox.commands.calls) == attempts


async def test_cancelled_dispatch_cancels_the_command_once_it_lands(tmp_path, client):
    env = ready_env(tmp_path, client)
    commands = client.sandbox.commands
    entered, release = threading.Event(), threading.Event()
    run = commands.run

    def slow_run(command, **kwargs):
        entered.set()
        release.wait(5)
        return run(command, **kwargs)

    commands.run = slow_run
    task = asyncio.create_task(env.exec("sleep 100"))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert await eventually(
        lambda: commands.executions and commands.executions[0].cancels == 1
    )


async def test_sdk_calls_wrap_a_bare_timeout(tmp_path, client):
    attempts = []

    def read_times_out():
        attempts.append(1)
        raise TimeoutError("The read operation timed out")

    with pytest.raises(TransportError, match="Probing failed"):
        await make_env(tmp_path)._call("Probing", read_times_out)
    assert len(attempts) == 2


# --- files ---


async def test_file_round_trip(tmp_path, client):
    env = ready_env(tmp_path, client)
    (tmp_path / "a.bin").write_bytes(b"\x00\x01")
    await env.upload_file(tmp_path / "a.bin", "/tmp/a.bin")
    await env.download_file("/tmp/a.bin", tmp_path / "deep" / "b.bin")
    assert (tmp_path / "deep" / "b.bin").read_bytes() == b"\x00\x01"


async def test_upload_dir_unpacks_and_cleans_up_in_one_command(tmp_path, client):
    env = ready_env(tmp_path, client)
    (tmp_path / "src" / "sub").mkdir(parents=True)
    (tmp_path / "src" / "sub" / "f.txt").write_text("x")
    await env.upload_dir(tmp_path / "src", "/data")
    [archive] = client.sandbox.files.contents
    script = shlex.split(client.sandbox.commands.calls[0].command)[-1]
    assert "-C /data" in script and script.endswith(
        f"; rc=$?; rm -f {archive}; exit $rc"
    )


async def test_upload_dir_failure_is_raised(tmp_path, client):
    env = ready_env(tmp_path, client)
    client.sandbox.commands.scripts = [[done(exit_code=2, stderr="no space")]]
    with pytest.raises(RuntimeError, match="no space"):
        await env.upload_dir(tmp_path, "/data")


async def test_upload_dir_needs_an_absolute_target(tmp_path, client):
    with pytest.raises(ValueError, match="absolute"):
        await ready_env(tmp_path, client).upload_dir(tmp_path, "")
