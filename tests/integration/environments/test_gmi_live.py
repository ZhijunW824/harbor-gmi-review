"""Live GMI smoke tests for the exec and file-transfer contract.

Requires GMI_SANDBOX_API_KEY and GMI_SANDBOX_IDC_NAME. Skipped automatically when
either is unset.
"""

import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sandbox_sdk")

from harbor.environments.gmi import GMIEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

pytestmark = pytest.mark.integration

requires_gmi = pytest.mark.skipif(
    not (
        os.environ.get("GMI_SANDBOX_API_KEY") and os.environ.get("GMI_SANDBOX_IDC_NAME")
    ),
    reason="GMI_SANDBOX_API_KEY or GMI_SANDBOX_IDC_NAME is not set",
)


def _make_live_env(tmp_path: Path) -> GMIEnvironment:
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    # Read only for WORKDIR; the image itself is never rebuilt.
    (env_dir / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /srv/app\n")
    trial_paths = TrialPaths(trial_dir=tmp_path / "trial")
    trial_paths.mkdir()
    return GMIEnvironment(
        environment_dir=env_dir,
        environment_name="harbor-gmi-smoke",
        session_id="gmi-smoke",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(docker_image="python:3.12-slim"),
    )


async def _out(gmi: GMIEnvironment, command: str, **kwargs: Any) -> str:
    return ((await gmi.exec(command, **kwargs)).stdout or "").strip()


@requires_gmi
@pytest.mark.asyncio
async def test_gmi_exec_runs_as_the_requested_user_in_the_workdir(tmp_path):
    env = _make_live_env(tmp_path)
    try:
        await env.start(force_build=True)  # ignored with a warning

        assert await _out(env, "pwd") == "/srv/app"
        assert await _out(env, "id -u", user="root") == "0"
        assert await _out(env, "id -un", user="user") == "user"
        # Per-exec env reaches the command verbatim, through sudo.
        assert await _out(env, "printenv X", env={"X": "$HOME/x"}) == "$HOME/x"
        assert (await env.exec("exit 7")).return_code == 7
    finally:
        await env.stop(delete=True)


@requires_gmi
@pytest.mark.asyncio
async def test_gmi_file_and_directory_round_trip(tmp_path):
    env = _make_live_env(tmp_path)
    (tmp_path / "tree" / "sub").mkdir(parents=True)
    (tmp_path / "tree" / "sub" / "a.txt").write_text("a\n")
    payload = os.urandom(4096)
    (tmp_path / "blob").write_bytes(payload)
    try:
        await env.start(force_build=False)

        await env.upload_file(tmp_path / "blob", "/tmp/blob")
        await env.download_file("/tmp/blob", tmp_path / "back" / "blob")
        assert (tmp_path / "back" / "blob").read_bytes() == payload

        await env.upload_dir(tmp_path / "tree", "/tmp/tree")
        await env.download_dir("/tmp/tree", tmp_path / "tree-back")
        assert (tmp_path / "tree-back" / "sub" / "a.txt").read_text() == "a\n"
    finally:
        await env.stop(delete=True)
