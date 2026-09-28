"""GMI Cloud sandbox environment."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, override

from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random_exponential,
)

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.environments.definition import (
    COMPOSE_FILE_NAME,
    effective_exec_cwd,
    parse_dockerfile_workdir,
    require_agent_environment_definition,
)
from harbor.environments.tar_transfer import pack_dir_to_bytes, remote_unpack_command
from harbor.models.environment_type import EnvironmentType
from harbor.utils.optional_import import MissingExtraError

try:
    from sandbox_sdk.client import Sandbox, SandboxClient, Template
    from sandbox_sdk.errors import (
        NotFoundError,
        RateLimitError,
        ServerError,
        TransportError,
    )

    _HAS_GMI = True
except ImportError:
    _HAS_GMI = False

_IDC_ENV = "GMI_SANDBOX_IDC_NAME"
_DEFAULT_PRODUCT = "gmi.sandbox.small"
_SANDBOX_TIMEOUT_SEC = 86_400
# The sandbox runs commands as this unprivileged user; any other user, root
# included, goes through sudo.
_SANDBOX_USER = "user"
# Task scripts use `source`, which dash does not provide.
_SUDO_SHELL = "bash"
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Template names must fit the server's 64-character limit.
_TEMPLATE_HASH_LEN = 20
_BUILD_HASH_LEN = 8
_TEMPLATE_PAGE_SIZE = 100
_TEMPLATE_PAGE_LIMIT = 50
_BUILD_TIMEOUT_SEC = 1_800
_BUILD_FAILED = frozenset({"failed", "error", "cancelled", "canceled"})
_READY_TIMEOUT_SEC = 180.0
_READY_STATES = frozenset({"running", "ready", "active"})
_NOT_READY_STATES = frozenset({"pending", "creating", "provisioning", "starting"})
_EXEC_RUNNING = frozenset({"pending", "queued", "running"})
# How long exec keeps polling through errors: a failed status poll does not mean
# the command failed.
_POLL_FAILURE_BUDGET_SEC = 300.0

_BUILD_CAPACITY_MARKERS = ("maximum concurrent template builds",)
_STALE_KEY_MARKERS = ("no longer exists", "use a new key")
# Each deleted Template retires one idempotency key; see _create_template.
_KEY_GENERATIONS = 8
_SANDBOX_QUOTA_MARKERS = ("sandbox quota exceeded",)
_QUOTA_HINT = (
    "Lower --n-concurrent to fit the organization's GMI sandbox quota, "
    "or ask GMI Cloud to raise the quota."
)
# Connection-setup failures: the request never reached the server, so resending
# a command cannot run it twice. The SDK reports all of them as TransportError.
_DISPATCH_RETRYABLE_MARKERS = (
    "handshake operation timed out",
    "name or service not known",
    "nodename nor servname",
    "temporary failure in name resolution",
    "connection refused",
    "network is unreachable",
    "no route to host",
)


class GMIExecTimeoutError(RuntimeError):
    """A command outlived its timeout_sec.

    Not a TimeoutError: the trial would report one as a phase timeout.
    """


class _CreateHandoff:
    """Passes a new sandbox from the SDK thread to its caller.

    If the caller is cancelled, whichever side acts second deletes the sandbox.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._abandoned = False
        self._delivered: Any = None

    def deliver(self, sandbox: Any) -> bool:
        with self._lock:
            self._delivered = sandbox
            return not self._abandoned

    def abandon(self) -> Any:
        with self._lock:
            self._abandoned = True
            return self._delivered


def _matches(exc: BaseException, markers: tuple[str, ...]) -> bool:
    return any(marker in str(exc).lower() for marker in markers)


def _is_safe_to_redispatch(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    # The SDK raises a connect timeout as TransportError("timed out"), before the
    # request is sent. A read timeout is a TimeoutError, which _dispatch_command
    # rewraps with a longer message, so it is never retried.
    return isinstance(exc, TransportError) and (
        str(exc).lower() == "timed out" or _matches(exc, _DISPATCH_RETRYABLE_MARKERS)
    )


def _digest(*parts: Any) -> str:
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _template_alias(environment_id: str, *build_parts: Any) -> str:
    prefix = environment_id[:_TEMPLATE_HASH_LEN]
    return f"harbor-{prefix}-{_digest(*build_parts)[:_BUILD_HASH_LEN]}"


def _payload_key(alias: str, *parts: Any) -> str:
    return f"{alias}-{_digest(*parts)[:16]}"


class GMIEnvironment(BaseEnvironment):
    """GMI Cloud sandboxes, started from a prebuilt image or an existing Template.

    Requires GMI_SANDBOX_API_KEY and GMI_SANDBOX_IDC_NAME, the GMI data center
    to run in. With ``template_id``, every sandbox starts from that Template.
    Otherwise a Template is built from the task's ``docker_image`` and reused
    across trials; ``product`` sets its size (default ``gmi.sandbox.small``).

    A Template build accepts no build context, so a Dockerfile is read only for
    its WORKDIR. ``force_build`` is ignored: to pick up a re-pushed image tag,
    delete the Template described as ``Harbor task <name>``. Docker Compose
    tasks are not supported.
    """

    @classmethod
    @override
    def preflight(cls) -> None:
        if not _HAS_GMI:
            raise MissingExtraError(package="gmi-sandbox-sdk", extra="gmi")
        if not os.environ.get("GMI_SANDBOX_API_KEY"):
            raise SystemExit(
                "GMI sandboxes require GMI_SANDBOX_API_KEY to be set. "
                "Please set this environment variable and try again."
            )
        if not os.environ.get(_IDC_ENV):
            raise SystemExit(
                f"GMI sandboxes require {_IDC_ENV} to be set to the GMI data "
                "center (IDC) to run in. Please set this environment variable "
                "and try again."
            )

    def __init__(
        self,
        *args: Any,
        template_id: str | int | None = None,
        product: str = _DEFAULT_PRODUCT,
        **kwargs: Any,
    ) -> None:
        if not _HAS_GMI:
            raise MissingExtraError(package="gmi-sandbox-sdk", extra="gmi")
        # Set before super().__init__(), which calls _validate_definition. --ek
        # turns template_id=12345 into an int and template_id= into "".
        self._template_id = None if template_id in (None, "") else str(template_id)
        self._product = product
        super().__init__(*args, **kwargs)
        self._client = SandboxClient()
        self._sandbox: Any = None
        self._sandbox_id: str | None = None
        self._idc_name = os.environ.get(_IDC_ENV) or None
        self._workdir = parse_dockerfile_workdir(self._environment_definition_path)
        if self._workdir and "$" in self._workdir and not self.task_env_config.workdir:
            # start() creates the working directory, and would create "$APP_HOME"
            # verbatim; nothing here expands Dockerfile variables.
            raise ValueError(
                f"WORKDIR {self._workdir!r} uses a variable; "
                "set [environment].workdir in task.toml"
            )

    @staticmethod
    @override
    def type() -> EnvironmentType:
        return EnvironmentType.GMI

    @property
    def _environment_definition_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities()

    @classmethod
    @override
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities()

    @override
    def _validate_definition(self) -> None:
        if self._template_id is not None:
            return
        require_agent_environment_definition(
            self.environment_dir, docker_image=self.task_env_config.docker_image
        )
        if (self.environment_dir / COMPOSE_FILE_NAME).exists():
            raise ValueError(
                "Docker Compose tasks are not supported by the gmi environment."
            )
        if not self.task_env_config.docker_image:
            raise ValueError(
                "GMI sandboxes run prebuilt images, because a GMI Template build "
                "cannot take a build context. Set [environment].docker_image in "
                "task.toml, or pass --ek template_id=<id>."
            )

    def _require_sandbox(self) -> Any:
        if self._sandbox is None:
            raise RuntimeError("GMI sandbox is not running; call start() first.")
        return self._sandbox

    async def _ensure_template(self) -> str:
        image = self.task_env_config.docker_image
        resources = {"type": "preset", "product": self._product}
        build = {"source": {"type": "image", "image": image}}
        alias = _template_alias(self.environment_id, resources, build)
        template_id, status = await self._call(
            f"Looking up template {alias}", self._find_template, alias
        )
        if template_id and status == "ready":
            self.logger.debug(f"Reusing template {alias}")
            return template_id
        if template_id and status not in _BUILD_FAILED:
            # Another trial may be building it; deleting it would strand that trial.
            await self._await_build(template_id)
            return template_id
        if template_id:
            await self._delete_template(template_id)
        self.logger.debug(f"Building template {alias} from {image}")
        template_id = await self._create_template(
            dict(
                name=alias,
                idc_name=self._idc_name,
                resources=resources,
                build=build,
                description=f"Harbor task {self.environment_name}",
            )
        )
        await self._await_build(template_id)
        return template_id

    def _find_template(self, alias: str) -> tuple[str | None, str]:
        """Id and build status of the Template named ``alias``, preferring a ready one."""
        found: list[tuple[str, str]] = []
        for page in range(1, _TEMPLATE_PAGE_LIMIT + 1):
            items = self._client.templates.list(
                page=page, page_size=_TEMPLATE_PAGE_SIZE, idc_name=self._idc_name
            ).items
            found += [
                (t.data["id"], (t.data.get("latest_build_status") or "").lower())
                for t in items
                if t.data.get("name") == alias
            ]
            if len(items) < _TEMPLATE_PAGE_SIZE:
                break
        # A Template past the last scanned page is not lost: _create_template sends
        # the same request and key, and the server returns the existing Template.
        rank = {"ready": 0, **dict.fromkeys(_BUILD_FAILED, 2)}
        return min(found, key=lambda t: rank.get(t[1], 1), default=(None, ""))

    @retry(
        retry=retry_if_exception(lambda exc: _matches(exc, _BUILD_CAPACITY_MARKERS)),
        wait=wait_exponential(multiplier=5, max=60),
        stop=stop_after_delay(_BUILD_TIMEOUT_SEC),
        before_sleep=lambda state: state.args[0].logger.warning(
            "GMI template build capacity is full; "
            f"retrying in {state.upcoming_sleep:.0f}s"
        ),
        reraise=True,
    )
    async def _create_template(self, request: dict[str, Any]) -> str:
        alias = request["name"]
        # The server rejects a replayed key whose body differs, so the key covers
        # the whole body.
        key = _payload_key(alias, request)
        # Deleting a Template retires its key. Every trial walks the same sequence
        # of keys, so concurrent rebuilds still share one Template.
        for generation in range(_KEY_GENERATIONS):
            try:
                created = await self._call(
                    f"Creating template {alias}",
                    self._client.templates.create,
                    **request,
                    idempotency_key=f"{key}-{generation}" if generation else key,
                )
            except Exception as exc:
                if not _matches(exc, _STALE_KEY_MARKERS):
                    raise
                continue
            if not created.data.get("id"):
                raise RuntimeError(f"Template create for {alias} returned no id")
            return str(created.data["id"])
        raise RuntimeError(
            f"Cannot create template {alias}: all {_KEY_GENERATIONS} of its "
            "idempotency keys belong to deleted Templates"
        )

    async def _await_build(self, template_id: str) -> None:
        deadline = asyncio.get_running_loop().time() + _BUILD_TIMEOUT_SEC
        delay = 1.0
        while True:
            template = await self._call(
                f"Polling template {template_id}",
                self._client.templates.get,
                template_id,
                idc_name=self._idc_name,
            )
            status = (template.data.get("latest_build_status") or "").lower()
            if status == "ready":
                return
            if status in _BUILD_FAILED:
                raise RuntimeError(
                    f"Template build for {template_id} ended as {status!r}"
                )
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(
                    f"Template {template_id} was still {status!r} after {_BUILD_TIMEOUT_SEC}s"
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10.0)

    async def _delete_template(self, template_id: str) -> None:
        try:
            await self._call(
                f"Deleting template {template_id}",
                Template(self._client, {"id": template_id}).delete,
                idc_name=self._idc_name,
            )
        except NotFoundError:
            pass  # a concurrent trial deleted it first
        except Exception as exc:  # noqa: BLE001 - best effort
            # If the failed Template survives under our key, the create replays it
            # and _await_build reports the failure.
            self.logger.warning(f"Failed to delete template {template_id}: {exc}")

    async def _create_sandbox(self, template_id: str) -> None:
        request: dict[str, Any] = dict(
            template_id=template_id,
            idc_name=self._idc_name,
            timeout=_SANDBOX_TIMEOUT_SEC,
            env_vars=dict(self._startup_env()) or None,
            metadata={
                "harbor_environment": self.environment_name,
                "harbor_session": self.session_id,
            },
            # Replaying the key returns the same sandbox, so retries are safe.
            idempotency_key=str(uuid.uuid4()),
        )
        handoff = _CreateHandoff()

        def create() -> Any:
            sandbox = self._client.sandboxes.create(**request)
            if not handoff.deliver(sandbox):
                # The caller was cancelled, and the event loop may already be gone.
                self._reap(sandbox.data.get("id"))
            return sandbox

        try:
            sandbox = await self._call("Creating sandbox", create)
        except asyncio.CancelledError:
            if (orphan := handoff.abandon()) is not None:
                threading.Thread(
                    target=self._reap, args=(orphan.data.get("id"),)
                ).start()
            raise
        except RateLimitError as exc:
            if _matches(exc, _SANDBOX_QUOTA_MARKERS):
                raise RuntimeError(f"{exc}. {_QUOTA_HINT}") from exc
            raise
        data = sandbox.data
        # Commands and files go to https://{sandbox_key}.{domain} with the access
        # token; without these fields the sandbox is unreachable.
        if not all(
            data.get(k) for k in ("id", "domain", "sandbox_key", "sandbox_access_token")
        ):
            await asyncio.get_running_loop().run_in_executor(
                None, self._reap, data.get("id")
            )
            raise RuntimeError("Sandbox create returned no usable data-plane endpoint")
        self._sandbox, self._sandbox_id = sandbox, data["id"]

    def _delete_now(self, sandbox_id: str) -> None:
        """Delete the sandbox, retrying in-thread so a cancel cannot cut it short."""
        for attempt in (1, 2):
            try:
                Sandbox(self._client, {"id": sandbox_id}).delete()
                return
            except NotFoundError:
                return
            except (TransportError, TimeoutError, ConnectionError, ServerError):
                if attempt == 2:
                    raise
                time.sleep(1)

    def _reap(self, sandbox_id: str | None) -> None:
        if not sandbox_id:
            return
        try:
            self._delete_now(sandbox_id)
        except Exception as exc:  # noqa: BLE001 - cleanup must not raise
            self.logger.warning(
                f"Could not delete GMI sandbox {sandbox_id}; it keeps running until "
                f"its {_SANDBOX_TIMEOUT_SEC}s timeout: {exc}"
            )

    async def _await_sandbox_ready(self) -> None:
        # Read state from a separate object: Sandbox.refresh() replaces all of its
        # data with the GET response, which lacks the data-plane fields.
        deadline = asyncio.get_running_loop().time() + _READY_TIMEOUT_SEC
        delay = 0.5
        while True:
            current = await self._call(
                "Checking sandbox state", self._client.sandboxes.get, self._sandbox_id
            )
            state = str(current.data.get("state") or current.data.get("status") or "")
            state = state.lower()
            if state in _READY_STATES:
                return
            if state and state not in _NOT_READY_STATES:
                raise RuntimeError(f"Sandbox {self._sandbox_id} came up as {state!r}")
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(
                    f"Sandbox {self._sandbox_id} was still {state or 'unknown'!r} "
                    f"after {_READY_TIMEOUT_SEC:.0f}s"
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 4.0)

    async def _release_sshd_pidfile(self) -> None:
        # The platform's own sshd writes /run/sshd.pid, possibly after the sandbox
        # reports ready. While that file exists, a task's `service ssh start`
        # assumes sshd is running, so wait up to 10 s for it and remove it.
        script = (
            "for _ in $(seq 1 20); do [ -e /run/sshd.pid ] && break; sleep 0.5; done; "
            "rm -f /run/sshd.pid"
        )
        try:
            await self.exec(script, cwd="/", timeout_sec=60, user="root")
        except Exception as exc:  # noqa: BLE001 - best effort
            self.logger.debug(f"Could not remove /run/sshd.pid: {exc}")

    @override
    async def start(self, force_build: bool) -> None:
        if force_build:
            self.logger.warning(
                "force_build is ignored: the gmi environment runs prebuilt images"
            )
        template_id = self._template_id or await self._ensure_template()
        await self._create_sandbox(template_id)
        await self._await_sandbox_ready()
        await self._release_sshd_pidfile()
        # Create the working directory and the writable mount targets in one exec.
        # A working directory from the image keeps its ownership; a new one is
        # owned by the sandbox user.
        steps = []
        if cwd := effective_exec_cwd(None, self.task_env_config.workdir, self._workdir):
            q = shlex.quote(cwd)
            steps.append(
                f"{{ test -d {q} || {{ mkdir -p {q} && chown {_SANDBOX_USER} {q}; }}; }}"
            )
        if dirs := self._mount_targets(writable_only=True):
            steps.append(self._ensure_dirs_command(dirs))
        if steps:
            result = await self.exec(
                " && ".join(steps), cwd="/", timeout_sec=60, user="root"
            )
            if result.return_code != 0:
                raise RuntimeError(
                    f"Failed to prepare directories: {result.stderr or result.stdout}"
                )
        await self._upload_environment_dir_after_start()

    @override
    async def stop(self, delete: bool) -> None:
        if self._sandbox is None:
            return
        if not delete:
            self.logger.debug(
                f"Keeping GMI sandbox {self._sandbox_id} alive (delete=False)"
            )
            self._sandbox = None
            return
        try:
            # shield() keeps a cancelled caller from cancelling the DELETE, and an
            # executor job, unlike a task, is awaited rather than cancelled when
            # asyncio.run shuts down.
            loop = asyncio.get_running_loop()
            await asyncio.shield(
                loop.run_in_executor(None, self._delete_now, self._sandbox_id)
            )
        except asyncio.CancelledError:
            self.logger.warning(
                f"Stop cancelled; GMI sandbox {self._sandbox_id} is still being deleted"
            )
            raise
        except Exception as exc:  # noqa: BLE001 - a raise here would be recorded as a trial error
            self.logger.error(f"Error stopping GMI sandbox {self._sandbox_id}: {exc}")
        finally:
            self._sandbox = None

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        user = self._resolve_user(user)
        env = self._merge_env(env)
        cwd = effective_exec_cwd(cwd, self.task_env_config.workdir, self._workdir)
        execution = await self._dispatch_command(
            self._wrap(command, env, user), env=env, cwd=cwd
        )
        execution = await self._wait_for_execution(execution, timeout_sec)
        if execution.exit_code is None:
            status = (execution.status or "unknown").lower()
            return ExecResult(
                stdout=execution.stdout or "",
                stderr=(execution.stderr or "")
                + f"\n[gmi] execution ended as {status!r} with no exit code",
                return_code=1,
            )
        return ExecResult(
            stdout=execution.stdout or "",
            stderr=execution.stderr or "",
            return_code=execution.exit_code,
        )

    def _wrap(
        self, command: str, env: dict[str, str] | None, user: str | int | None
    ) -> str:
        if env:
            for name in env:
                if not _ENV_NAME.fullmatch(name):
                    raise RuntimeError(f"invalid environment variable name {name!r}")
            # The request carries these too, but sudo resets the environment.
            exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in env.items())
            command = exports + command
        requested = "root" if user is None else str(user)
        if requested == _SANDBOX_USER:
            return command
        if requested in ("root", "0"):
            as_user = ""
        else:
            # sudo reads a bare number as a user name; a uid needs the # prefix.
            target = f"#{requested}" if requested.isdigit() else requested
            as_user = f"-u {shlex.quote(target)} "
        return f"sudo -n {as_user}{_SUDO_SHELL} -c {shlex.quote(command)}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=10),
        retry=retry_if_exception(_is_safe_to_redispatch),
        reraise=True,
    )
    async def _dispatch_command(
        self, command: str, *, env: dict[str, str] | None, cwd: str | None
    ) -> Any:
        """Start ``command`` detached; retry only errors proving it never started."""
        # The server holds a synchronous request for at most 25 seconds, so start
        # the command detached and poll for its result.
        dispatch = asyncio.create_task(
            asyncio.to_thread(
                self._require_sandbox().commands.run,
                command,
                envs=env,
                cwd=cwd,
                wait=False,
            )
        )
        try:
            return await asyncio.shield(dispatch)
        except asyncio.CancelledError:
            # The command may already have reached the sandbox. Cancel it once the
            # dispatch returns: after an agent timeout the verifier runs in this
            # sandbox, so the agent's command must not keep running.
            dispatch.add_done_callback(self._cancel_late_dispatch)
            raise
        except TimeoutError as exc:
            # Not retried: the command may already be running. Wrapped so the trial
            # does not report it as an agent timeout.
            raise TransportError(f"Command dispatch got no response: {exc}") from exc

    def _cancel_late_dispatch(self, task: asyncio.Task[Any]) -> None:
        if not task.cancelled() and task.exception() is None:
            threading.Thread(target=self._cancel_now, args=(task.result(),)).start()

    async def _wait_for_execution(self, execution: Any, timeout_sec: int | None) -> Any:
        loop = asyncio.get_running_loop()
        # None and 0 both mean no limit.
        deadline = loop.time() + timeout_sec if timeout_sec else None
        delay = 0.2
        failing_since: float | None = None
        try:
            while True:
                status = (execution.status or "").lower()
                if status and status not in _EXEC_RUNNING:
                    return execution
                if deadline is not None and loop.time() >= deadline:
                    raise GMIExecTimeoutError(
                        f"Command timed out after {timeout_sec}s "
                        f"(execution {execution.execution_id})"
                    )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 2.0)
                try:
                    execution = await self._call(
                        f"Polling execution {execution.execution_id}", execution.refresh
                    )
                    failing_since = None
                except (TransportError, ServerError, RateLimitError, OSError) as exc:
                    failing_since = failing_since or loop.time()
                    if loop.time() - failing_since > _POLL_FAILURE_BUDGET_SEC:
                        raise
                    self.logger.debug(
                        f"Polling execution failed ({exc}); still waiting"
                    )
        except BaseException:
            # Whatever ends the wait, stop the command: the next phase (the
            # verifier, after an agent timeout) reuses this sandbox.
            await self._cancel_execution(execution)
            raise

    async def _cancel_execution(self, execution: Any) -> None:
        await asyncio.shield(asyncio.to_thread(self._cancel_now, execution))

    def _cancel_now(self, execution: Any) -> None:
        try:
            execution.cancel()
        except Exception as exc:  # noqa: BLE001 - best effort
            self.logger.warning(
                f"Failed to cancel execution {execution.execution_id}: {exc}"
            )

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._call(
            f"Uploading {target_path}",
            self._require_sandbox().files.write,
            target_path,
            Path(source_path).read_bytes(),
        )

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        data = await self._call(
            f"Downloading {source_path}",
            self._require_sandbox().files.read,
            source_path,
        )
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        target = PurePosixPath(target_dir)
        if not target.is_absolute():
            raise ValueError(f"upload_dir needs an absolute target, got {target_dir!r}")
        remote_tar = f"/tmp/harbor-upload-{uuid.uuid4().hex}.tar.gz"
        buffer = await asyncio.to_thread(
            pack_dir_to_bytes, Path(source_dir), compress=True
        )
        await self._call(
            f"Uploading {remote_tar}",
            self._require_sandbox().files.write,
            remote_tar,
            buffer.getvalue(),
        )
        unpack = remote_unpack_command(remote_tar, str(target))
        result = await self.exec(
            f"{unpack}; rc=$?; rm -f {shlex.quote(remote_tar)}; exit $rc",
            timeout_sec=300,
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to unpack into {target_dir!r}: {result.stderr or result.stdout}"
            )

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._download_dir_with_exclusions_impl(
            source_dir=source_dir, target_dir=target_dir, exclude=[], service=None
        )

    async def _call(
        self, what: str, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """Run a blocking SDK call in a thread, retrying transient failures."""
        retrying = AsyncRetrying(
            # Every call routed here is a read, is idempotent, or carries an
            # idempotency key, so resending it is safe.
            retry=retry_if_exception_type(
                (TransportError, TimeoutError, ConnectionError, ServerError)
            ),
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            before_sleep=lambda state: self.logger.debug(
                f"{what} failed ({state.outcome.exception() if state.outcome else None}); "
                "retrying"
            ),
            reraise=True,
        )
        try:
            return await retrying(asyncio.to_thread, fn, *args, **kwargs)
        except (TransportError, TimeoutError) as exc:
            # A read timeout surfaces as a bare TimeoutError, which the trial would
            # otherwise report as a phase timeout.
            raise TransportError(f"{what} failed: {exc}") from exc
