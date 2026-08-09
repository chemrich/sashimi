"""Locating the APBS binary.

Order: `$SASHIMI_APBS_PATH`, then the active pixi/conda environment, then bare
`shutil.which`. The pinned environment deliberately outranks `which` — a
system-wide APBS (Homebrew's, say) otherwise shadows the pinned one silently
and "reproducible environment" stops meaning anything. The resolved path
travels in `SolveResult.backend` so a shadowed binary is visible after the fact.

Every APBS invocation writes an `io.mc` log into the working directory,
`--version` included, so the version probe runs inside a temp dir.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sashimi.errors import SolverNotFound

__all__ = ["ApbsBinary", "ApbsNotFound", "discover_apbs"]

_VERSION_RE = re.compile(r"APBS\s+(\d+\.\d+\.\d+)")
_INSTALL_HINT = (
    "Install it with `pixi add apbs` (or `conda install -c conda-forge apbs`), "
    "or point $SASHIMI_APBS_PATH at an existing binary."
)


class ApbsNotFound(SolverNotFound):
    """The APBS binary could not be located, or is not runnable."""


@dataclass(frozen=True)
class ApbsBinary:
    path: Path
    version: str

    @property
    def label(self) -> str:
        """Provenance string for `SolveResult.backend`."""
        return f"apbs-{self.version}"


def _candidates() -> list[Path]:
    found: list[Path] = []

    if explicit := os.environ.get("SASHIMI_APBS_PATH"):
        found.append(Path(explicit).expanduser())

    # pixi run / conda activate both export CONDA_PREFIX.
    if prefix := os.environ.get("CONDA_PREFIX"):
        found.append(Path(prefix) / "bin" / "apbs")

    # A pixi project whose env exists but is not activated.
    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidate = parent / ".pixi" / "envs" / "default" / "bin" / "apbs"
        if candidate.exists():
            found.append(candidate)
            break

    if on_path := shutil.which("apbs"):
        found.append(Path(on_path))

    return found


def _probe_version(path: Path) -> str | None:
    """Run `apbs --version` in a temp dir; return the version or None."""
    with tempfile.TemporaryDirectory(prefix="sashimi-probe-") as tmp:
        try:
            proc = subprocess.run(
                [str(path), "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=tmp,
            )
        except (OSError, subprocess.SubprocessError):
            return None
    match = _VERSION_RE.search(proc.stdout + proc.stderr)
    return match.group(1) if match else None


@lru_cache(maxsize=8)
def _discover_cached(explicit: str | None, conda_prefix: str | None, cwd: str) -> ApbsBinary:
    tried: list[str] = []
    for candidate in _candidates():
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            tried.append(f"{candidate} (missing or not executable)")
            continue
        version = _probe_version(candidate)
        if version is None:
            tried.append(f"{candidate} (did not report a version)")
            continue
        return ApbsBinary(path=candidate.resolve(), version=version)

    detail = "\n  ".join(tried) if tried else "no candidate paths"
    raise ApbsNotFound(f"No usable APBS binary found. Tried:\n  {detail}\n\n{_INSTALL_HINT}")


def discover_apbs() -> ApbsBinary:
    """Locate APBS. Cached per (SASHIMI_APBS_PATH, CONDA_PREFIX, cwd)."""
    return _discover_cached(
        os.environ.get("SASHIMI_APBS_PATH"),
        os.environ.get("CONDA_PREFIX"),
        str(Path.cwd()),
    )
