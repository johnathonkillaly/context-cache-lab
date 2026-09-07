"""Provenance stamped onto every result row.

Every number this repo reports must be traceable to a commit, a seed and a corpus draw.
Cheap to record, impossible to reconstruct afterwards.
"""

from __future__ import annotations

import datetime as _dt
import platform
import subprocess
from typing import Any

_REPO_ROOT = __file__.rsplit("/src/ccl/", 1)[0]


def git_commit() -> str:
    """Short commit hash, with a '-dirty' suffix if the tree has uncommitted changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if head.returncode != 0:
            return "unknown"
        commit = head.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if status.returncode == 0 and status.stdout.strip():
            commit += "-dirty"
        return commit
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def env_info() -> dict[str, Any]:
    """Environment fingerprint. Torch is imported lazily so corpus tools stay light."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["mps_available"] = bool(torch.backends.mps.is_available())
    except ImportError:
        info["torch"] = None
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = None
    return info


def stamp(**extra: Any) -> dict[str, Any]:
    """Standard provenance block merged into every emitted record."""
    return {
        "git_commit": git_commit(),
        "timestamp_utc": utc_now(),
        "env": env_info(),
        **extra,
    }
