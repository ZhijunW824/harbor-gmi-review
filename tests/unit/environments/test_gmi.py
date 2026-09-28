"""Unit tests for the GMI environment against a mocked SDK client."""

import asyncio
import shlex
import threading
from unittest.mock import MagicMock

import pytest

from harbor.environments import gmi
from harbor.environments.gmi import GMIEnvironment, GMIExecTimeoutError
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

errors = pytest.importorskip("sandbox_sdk.errors", reason="needs the 'gmi' extra")

IMAGE = "python:3.12-slim"
BUILD = {"source": {"type": "image", "image": IMAGE}}
SANDBOX = dict(id="sb-1", domain="d", sandbox_key="k", sandbox_access_token="t")
STALE = errors.ConflictError(409, "The template for this key no longer exists")
DOCKERFILE = {"Dockerfile": "FROM x\nWORKDIR /app\n"}
RUNNING = {"status": "running"}


def rec(**data):
    return MagicMock(data=data)


def done(code=0, **fields):
    return {"status": "succeeded", "exit_code": code, **fields}


class Execution:
    """A command handle whose refresh() steps through `states`, repeating the last."""

    def __init__(self, *states, delay=0.0, poll_errors=()):
        self.states, self.poll_errors = list(states) or [done()], list(poll_errors)
        self.delay, self.cancel, self.execution_id = delay, MagicMock(), "e1"
        self.refresh(first=True)

    def refresh(self, first=False):
        threading.Event().wait(0 if first else self.delay)
        if self.poll_errors and not first:
            raise self.poll_errors.pop(0)
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        self.__dict__.update(dict.fromkeys(["exit_code", "stdout", "stderr"]), **state)
        return self


@pytest.fixture
def client(monkeypatch):
    sdk = MagicMock()
    sdk.templates.list.return_value.items = []
    sdk.templates.create.return_value = rec(id="tpl-new")
    sdk.templates.get.return_value = rec(latest_build_status="ready")
    sdk.sandboxes.create.return_value = sdk.sandbox = MagicMock(data=SANDBOX)
    sdk.sandboxes.get.return_value = rec(state="running")
    sdk.sandbox.commands.run.side_effect = lambda *args, **kwargs: Execution()
    monkeypatch.setattr(gmi, "SandboxClient", lambda: sdk)
    monkeypatch.setenv("GMI_SANDBOX_IDC_NAME", "idc")
    sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda *_: sleep(0))
    monkeypatch.setattr(gmi.time, "sleep", lambda _: None)
    return sdk


def make_env(path, files=None, sandbox=None, **config):
    (env_dir := path / "environment").mkdir(parents=True, exist_ok=True)
    for name, text in (DOCKERFILE if files is None else files).items():
        (env_dir / name).write_text(text)
    kwargs = {k: config.pop(k) for k in ("template_id", "mounts") if k in config}
    (trial_paths := TrialPaths(trial_dir=path / "trial")).mkdir()
    env_config = EnvironmentConfig(**{"docker_image": IMAGE, **config})
    env = GMIEnvironment(env_dir, "task", "s", trial_paths, env_config, **kwargs)
    env._sandbox, env._sandbox_id = sandbox, sandbox and "sb-1"
    return env


def list_templates(client, env, *templates):
    resources = {"type": "preset", "product": env._product}
    alias = gmi._template_alias(env.environment_id, resources, BUILD)
    items = [rec(id=i, name=alias, latest_build_status=s) for i, s in templates]
    other_task = rec(id="other-task", name="harbor-other", latest_build_status="ready")
    client.templates.list.return_value.items = [other_task, *items]


def deleted(client):
    return [c.args[1] for c in client._request.call_args_list if c.args[0] == "DELETE"]


async def eventually(predicate, tries=300):
    while not predicate() and (tries := tries - 1):
        await asyncio.to_thread(threading.Event().wait, 0.01)
    return predicate()


async def cancel_while_blocked(target, result, start):
    """Cancels start() while the mock `target` runs, then lets it return `result`."""
    entered, release = threading.Event(), threading.Event()

    def blocked(*args, **kwargs):
        entered.set()
        release.wait(5)
        return result

    target.side_effect = blocked
    task = asyncio.create_task(start())
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()


@pytest.mark.parametrize(
    ("config", "error", "match"),
    [
        ({"docker_image": None}, ValueError, "docker_image"),
        ({"docker_image": None, "files": {}}, FileNotFoundError, None),
        ({"files": {"Dockerfile": "FROM x\nWORKDIR $APP\n"}}, ValueError, "WORKDIR"),
    ],
)
def test_invalid_task_definition(tmp_path, client, config, error, match):
    with pytest.raises(error, match=match):
        make_env(tmp_path, **config)


async def test_numeric_template_id_skips_template_management(tmp_path, client):
    env = make_env(tmp_path, docker_image=None, files={}, template_id=12345)
    await env.start(force_build=False)
    assert not client.templates.list.called and not client.templates.create.called
    assert client.sandboxes.create.call_args.kwargs["template_id"] == "12345"


@pytest.mark.parametrize(
    "templates",
    [
        [("t1", "waiting")],
        [("t0", "failed"), ("t1", "ready"), ("t2", "building")],
    ],
)
async def test_existing_template_is_used_not_deleted(tmp_path, client, templates):
    env = make_env(tmp_path)
    list_templates(client, env, *templates)
    pending = [rec(latest_build_status=s) for s in ("", "ready")]
    client.templates.get.side_effect = pending
    assert await env._ensure_template() == "t1"
    assert not client.templates.create.called and deleted(client) == []


async def test_failed_template_is_rebuilt_under_the_next_key(tmp_path, client, caplog):
    env = make_env(tmp_path)
    list_templates(client, env, ("bad", "failed"))
    busy = errors.RateLimitError(429, "maximum concurrent template builds reached")
    # Capacity is full once; then "bad" and an external delete retired two keys.
    client.templates.create.side_effect = [busy, STALE, STALE, rec(id="t3")]
    with caplog.at_level("WARNING"):
        assert await env._ensure_template() == "t3"
    assert "capacity is full" in caplog.text and deleted(client) == ["/templates/bad"]
    calls = client.templates.create.call_args_list
    base, *rest = [call.kwargs["idempotency_key"] for call in calls]
    assert rest == [base, f"{base}-1", f"{base}-2"] and len(rest[-1]) <= 64
    # The server rejects a replayed key whose body differs, so the key covers it.
    request = dict(client.templates.create.call_args.kwargs)
    assert request.pop("idempotency_key") == rest[-1]
    assert base == gmi._payload_key(request["name"], request)
    assert request["build"] == BUILD and request["idc_name"] == "idc"
    # Existing Templates are found by this name; it changes with the image.
    assert request["name"] == "harbor-9c3c956a7072466ae9db-d459ee07"


async def test_start_prepares_the_sandbox(tmp_path, client, caplog):
    mount = {"type": "bind", "source": str(tmp_path), "target": "/logs/agent"}
    states = [rec(state=s, latest_build_status=s) for s in ("", "starting", "READY")]
    client.sandboxes.get.side_effect = client.templates.get.side_effect = states
    with caplog.at_level("WARNING"):
        await make_env(tmp_path, mounts=[mount]).start(force_build=True)
    assert "force_build is ignored" in caplog.text
    assert client.templates.get.call_count == client.sandboxes.get.call_count == 3
    sshd, dirs = client.sandbox.commands.run.call_args_list
    assert "/run/sshd.pid" in sshd.args[0] and dirs.kwargs["cwd"] == "/"
    assert "test -d /app" in dirs.args[0] and "/logs/agent" in dirs.args[0]


async def test_sandbox_create(tmp_path, client):
    timeout = TimeoutError("The read operation timed out")
    client.sandboxes.create.side_effect = [timeout, client.sandbox]
    await make_env(tmp_path).start(force_build=False)
    first, second = client.sandboxes.create.call_args_list
    assert first.kwargs["idempotency_key"] == second.kwargs["idempotency_key"]
    assert first.kwargs["timeout"] == 86_400 and first.kwargs["idc_name"] == "idc"
    # A bare TimeoutError would be reported as an environment start timeout.
    client.sandboxes.create.side_effect = [timeout, timeout]
    with pytest.raises(errors.TransportError, match="Creating sandbox failed"):
        await make_env(tmp_path / "b")._create_sandbox("t")
    # Without its data-plane endpoint a sandbox is unusable, so it is deleted.
    no_token = MagicMock(data={**SANDBOX, "sandbox_access_token": None})
    client.sandboxes.create.side_effect = [no_token]
    with pytest.raises(RuntimeError, match="data-plane"):
        await make_env(tmp_path / "c")._create_sandbox("t")
    assert deleted(client) == ["/sandboxes/sb-1"]


async def test_work_landing_after_a_cancel_is_undone(tmp_path, client):
    env, handle = make_env(tmp_path, sandbox=client.sandbox), Execution()
    create, run = client.sandboxes.create, client.sandbox.commands.run
    await cancel_while_blocked(create, client.sandbox, lambda: env._create_sandbox("t"))
    await cancel_while_blocked(run, handle, lambda: env.exec("sleep 9"))
    # The sandbox would bill for a day; the command would outlive the agent phase.
    assert await eventually(lambda: deleted(client) == ["/sandboxes/sb-1"])
    assert await eventually(lambda: handle.cancel.called)


@pytest.mark.parametrize(
    ("delete", "error", "attempts"),
    [
        (True, None, 1),
        (False, None, 0),
        (True, errors.ServerError(500, "boom"), 2),
    ],
)
async def test_stop_never_raises(tmp_path, client, caplog, delete, error, attempts):
    client.sandboxes.get.side_effect = RuntimeError("boot")  # start fails after create
    with pytest.raises(RuntimeError, match="boot"):
        await (env := make_env(tmp_path)).start(force_build=False)
    client._request.side_effect = error
    with caplog.at_level("ERROR"):
        await env.stop(delete=delete)
    assert env._sandbox is None and len(deleted(client)) == attempts
    assert ("Error stopping" in caplog.text) == isinstance(error, errors.ServerError)


@pytest.mark.parametrize(
    ("user", "wrapped"),
    [
        (None, "sudo -n bash -c 'echo hi'"),
        ("user", "echo hi"),
        (1000, "sudo -n -u '#1000' bash -c 'echo hi'"),
    ],
)
def test_sudo_wrapping(tmp_path, client, user, wrapped):
    assert make_env(tmp_path)._wrap("echo hi", None, user) == wrapped


def test_env_is_exported_inside_sudo_and_names_are_checked(tmp_path, client):
    env = make_env(tmp_path)
    wrapped = env._wrap("printenv A", {"A": "x y"}, None)
    assert shlex.split(wrapped)[-1] == "export A='x y'; printenv A"
    with pytest.raises(RuntimeError, match="invalid environment variable name"):
        env._wrap("true", {"FOO\n": "v"}, None)


@pytest.mark.parametrize(
    ("states", "code", "stderr"),
    [
        ([RUNNING, {"status": ""}, done(3, stdout="o", stderr="e")], 3, "e"),
        ([{"status": "timed_out", "stdout": "o"}], 1, "'timed_out' with no exit code"),
    ],
)
async def test_exec_result(tmp_path, client, states, code, stderr):
    # Failed polls say nothing about the command, so the wait goes on.
    failures = [errors.TransportError("reset"), errors.RateLimitError(429, "slow")] * 2
    client.sandbox.commands.run.side_effect = [Execution(*states, poll_errors=failures)]
    env = make_env(tmp_path, sandbox=client.sandbox, env={"P": "1"})
    result = await env.exec("cmd", env={"K": "v"})
    assert result.return_code == code and stderr in result.stderr
    assert result.stdout == "o"
    call = client.sandbox.commands.run.call_args
    assert call.kwargs == {"envs": {"P": "1", "K": "v"}, "cwd": "/app", "wait": False}


async def test_command_is_cancelled_when_the_wait_ends_early(tmp_path, client):
    env = make_env(tmp_path, sandbox=client.sandbox)
    slow, stuck = Execution(RUNNING, delay=0.05), Execution(RUNNING)
    client.sandbox.commands.run.side_effect, stuck.refresh = [slow, stuck], MagicMock()
    with pytest.raises(GMIExecTimeoutError) as caught:
        await env.exec("sleep 9", timeout_sec=1)
    # The trial would report a plain TimeoutError as an agent timeout.
    assert not isinstance(caught.value, TimeoutError) and slow.cancel.called
    await cancel_while_blocked(stuck.refresh, stuck, lambda: env.exec("sleep 9"))
    assert stuck.cancel.called


@pytest.mark.parametrize(
    ("error", "attempts"),
    [
        (errors.TransportError("[Errno 61] Connection refused"), 3),
        (errors.TransportError("timed out"), 3),
        (errors.RateLimitError(429, "Too many concurrently active executions."), 3),
        (TimeoutError("The read operation timed out"), 1),
        (errors.ServerError(502, "Bad Gateway"), 1),
    ],
)
async def test_dispatch_retries_only_undelivered(tmp_path, client, error, attempts):
    client.sandbox.commands.run.side_effect = error
    with pytest.raises(errors.SandboxSDKError):
        await make_env(tmp_path, sandbox=client.sandbox).exec("true")
    assert client.sandbox.commands.run.call_count == attempts


async def test_file_transfer(tmp_path, client):
    env = make_env(tmp_path, sandbox=client.sandbox)
    (tmp_path / "src" / "logs").mkdir(parents=True)
    await env.upload_dir(tmp_path / "src", "/data")
    archive, payload = client.sandbox.files.write.call_args.args
    script = shlex.split(client.sandbox.commands.run.call_args.args[0])[-1]
    assert "-C /data" in script and script.endswith(f"rm -f {archive}; exit $rc")
    # Rewards come back through download_dir: serve the uploaded archive back.
    client.sandbox.files.read.return_value = payload
    await env.download_dir("/logs/verifier", tmp_path / "out")
    assert (tmp_path / "out" / "logs").is_dir()
    client.sandbox.commands.run.side_effect = [Execution(done(2, stderr="no space"))]
    with pytest.raises(RuntimeError, match="no space"):
        await env.upload_dir(tmp_path / "src", "/data")
