"""Ensure script entry points run in the project conda environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_NAME = "patent_sim"
DEFAULT_CONDA_ROOT = Path("/home/ubuntu/miniconda3")
REEXEC_FLAG = "_PATENT_SIM_CONDA_REEXECED"
SKIP_FLAG = "PATENT_SIM_SKIP_CONDA_REEXEC"


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _env_prefix(env_name: str) -> Path:
    override = os.environ.get("PATENT_SIM_CONDA_PREFIX")
    if override:
        return Path(override)
    conda_root = Path(os.environ.get("PATENT_SIM_CONDA_ROOT", str(DEFAULT_CONDA_ROOT)))
    return conda_root / "envs" / env_name


def ensure_patent_sim_env() -> None:
    """Re-exec the current script with the patent_sim conda Python when needed."""
    if os.environ.get(SKIP_FLAG):
        return

    env_name = os.environ.get("PATENT_SIM_CONDA_ENV", ENV_NAME)
    prefix = _env_prefix(env_name)
    target_python = Path(
        os.environ.get("PATENT_SIM_CONDA_PYTHON", str(prefix / "bin" / "python"))
    )

    current_paths = [Path(sys.executable), Path(sys.prefix)]
    base_prefix = getattr(sys, "base_prefix", None)
    if base_prefix:
        current_paths.append(Path(base_prefix))

    if any(_path_is_relative_to(path, prefix) for path in current_paths):
        return

    if not target_python.exists():
        raise RuntimeError(
            f"Required conda environment {env_name!r} was not found at {target_python}. "
            "Set PATENT_SIM_CONDA_PYTHON to override the interpreter path."
        )

    if os.environ.get(REEXEC_FLAG):
        raise RuntimeError(
            f"Tried to re-exec with {target_python}, but the active interpreter is still "
            f"{sys.executable}."
        )

    env = os.environ.copy()
    env[REEXEC_FLAG] = "1"
    env["CONDA_DEFAULT_ENV"] = env_name
    env["CONDA_PREFIX"] = str(prefix)
    env["PATH"] = str(target_python.parent) + os.pathsep + env.get("PATH", "")
    os.execve(str(target_python), [str(target_python), *sys.argv], env)
