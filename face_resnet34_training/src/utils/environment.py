"""Environment capture and the git-clean guard.

capture_environment() records everything needed to reproduce a run: library
versions, GPU, node, Slurm job ID, git commit, and full pip freeze. Call it
once per run and save the result into experiment_config.yaml /
environment.json.

check_git_clean() is the safety gate described in the spec: training refuses
to start on a dirty working tree unless --allow-dirty is passed, and that
override is recorded in the environment capture so it's visible later.
"""
from __future__ import annotations

import platform
import socket
import subprocess
import sys

import torch


class DirtyWorkingTreeError(RuntimeError):
    pass


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def git_commit_hash() -> str | None:
    return _run(["git", "rev-parse", "HEAD"])


def git_is_dirty() -> bool:
    status = _run(["git", "status", "--porcelain"])
    return bool(status)


def check_git_clean(allow_dirty: bool) -> bool:
    """Returns whether the tree was dirty. Raises unless allow_dirty is set."""
    dirty = git_is_dirty()
    if dirty and not allow_dirty:
        raise DirtyWorkingTreeError(
            "Git working tree has uncommitted changes. Commit first, or pass "
            "--allow-dirty to proceed anyway (this will be recorded in "
            "environment.json)."
        )
    return dirty


def capture_environment(allow_dirty_used: bool = False) -> dict:
    gpu_name = None
    cuda_version = None
    cudnn_version = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()

    pip_freeze = _run([sys.executable, "-m", "pip", "freeze"])

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "gpu_name": gpu_name,
        "node_name": socket.gethostname(),
        "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
        "git_commit": git_commit_hash(),
        "git_was_dirty": git_is_dirty(),
        "allow_dirty_used": allow_dirty_used,
        "pip_freeze": pip_freeze.splitlines() if pip_freeze else None,
    }
