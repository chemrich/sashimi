"""Locating TABI-PB, and the mesher it shells out to.

Two executables, not one. `tabipb` computes the boundary-element solve, but the
triangulation comes from NanoShaper, which `tabipb` invokes **by bare name**
through a shell — so NanoShaper being installed is not enough, it has to be on
`PATH` at the moment of the call. `run.py` arranges that; this module's job is
to find both and to fail with a message naming which one is missing, since the
two have entirely different remedies.

TABI-PB's own build fetches a NanoShaper for the platform (`-DGET_NanoShaper=ON`)
and leaves it beside the binary, so looking next to `tabipb` finds it in the
common case.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sashimi.errors import BackendUnavailable

__all__ = ["TabipbBinary", "TabipbNotFound", "discover_tabipb"]

_TABIPB_NAMES = ("tabipb", "tabi")
_MESHER_NAME = "NanoShaper"
_MESHER_VERSION_RE = re.compile(r"Starting NanoShaper\s+([0-9][0-9.]*)")

_INSTALL_HINT = (
    "Build it from https://github.com/Treecodes/TABI-PB (BSD-3-Clause) with\n"
    "  cmake -DGET_NanoShaper=ON -DCMAKE_BUILD_TYPE=Release .. && make\n"
    "which also places a NanoShaper executable beside the binary, then point\n"
    "$SASHIMI_TABIPB_PATH at build/bin/tabipb."
)


class TabipbNotFound(BackendUnavailable):
    """TABI-PB, or the mesher it depends on, could not be located."""


@dataclass(frozen=True)
class TabipbBinary:
    path: Path
    mesher_path: Path
    mesher_version: str | None = None

    @property
    def label(self) -> str:
        """Provenance string. The mesher is part of the identity of a result.

        Two runs of the same `tabipb` against different triangulation codes are
        not the same calculation — the published comparison of MSMS against
        NanoShaper in TABI exists precisely because the surface moves the
        answer.
        """
        mesher = f"+nanoshaper-{self.mesher_version}" if self.mesher_version else "+nanoshaper"
        return f"tabipb{mesher}"

    @property
    def sha256(self) -> str:
        return _checksum(str(self.path))


@lru_cache(maxsize=8)
def _checksum(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidates() -> list[Path]:
    found: list[Path] = []
    if explicit := os.environ.get("SASHIMI_TABIPB_PATH"):
        found.append(Path(explicit).expanduser())
    found.extend(Path(on_path) for name in _TABIPB_NAMES if (on_path := shutil.which(name)))
    return found


def _find_mesher(near: Path) -> Path | None:
    """NanoShaper beside the solver, else on PATH."""
    sibling = near.parent / _MESHER_NAME
    if sibling.is_file():
        return sibling
    located = shutil.which(_MESHER_NAME)
    return Path(located) if located else None


def _probe_mesher(path: Path) -> str | None:
    """NanoShaper's version, from the banner it prints before complaining.

    Run with no configuration file, which it treats as an error — after
    announcing itself, which is all this needs. In a temp dir because it writes
    its own working files beside wherever it is run.
    """
    with tempfile.TemporaryDirectory(prefix="sashimi-mesher-probe-") as tmp:
        try:
            proc = subprocess.run(
                [str(path)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=tmp,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
    match = _MESHER_VERSION_RE.search(proc.stdout + proc.stderr)
    return match.group(1) if match else None


@lru_cache(maxsize=8)
def _discover_cached(_explicit: str | None, _path: str | None) -> TabipbBinary:
    tried: list[str] = []
    for candidate in _candidates():
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            tried.append(f"{candidate} (missing or not executable)")
            continue
        mesher = _find_mesher(candidate.resolve())
        if mesher is None:
            tried.append(
                f"{candidate} (found, but no {_MESHER_NAME} beside it or on PATH; "
                "TABI-PB cannot triangulate a surface without it)"
            )
            continue
        if not os.access(mesher, os.X_OK):
            tried.append(f"{mesher} (not executable — the fetched copy often needs chmod +x)")
            continue
        return TabipbBinary(
            path=candidate.resolve(),
            mesher_path=mesher.resolve(),
            mesher_version=_probe_mesher(mesher),
        )

    detail = "\n  ".join(tried) if tried else "no candidate paths"
    raise TabipbNotFound(f"No usable TABI-PB found. Tried:\n  {detail}\n\n{_INSTALL_HINT}")


def discover_tabipb() -> TabipbBinary:
    """Locate TABI-PB and its mesher. Cached per (SASHIMI_TABIPB_PATH, PATH)."""
    return _discover_cached(os.environ.get("SASHIMI_TABIPB_PATH"), os.environ.get("PATH"))
