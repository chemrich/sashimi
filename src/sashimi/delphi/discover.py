"""Locating a DelPhi executable, and deciding which of the two it is.

Order: `$SASHIMI_DELPHI_PATH`, then the C++ builds on PATH, then
`pydelphi-static`. The explicit variable wins because the C++ program has no
canonical installed name — its makefile emits `delphicpp_mac`,
`delphicpp_release`, `delphicpp_omp_release` or `delphicpp_mpi_release`
depending on which folder it was built in — so pointing at it directly is the
normal case rather than the escape hatch it is for APBS.

Neither flavour has a package: the C++ source ships as a tarball from Clemson
that has to be compiled (its makefile hardcodes `g++`, assumes x86 flags and
needs boost headers), and pyDelPhi installs from a git checkout. Version
probing therefore reads what the executable says about itself rather than
trusting any filename: the v8.5.0 tarball builds a binary that reports 8.6.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from sashimi.errors import BackendUnavailable

__all__ = ["DelphiBinary", "DelphiFlavour", "DelphiNotFound", "discover_delphi"]

# The C++ banner is printed before any input is parsed, so an unusable argument
# is a perfectly good version probe.
_CPP_VERSION_RE = re.compile(r"DelPhi C\+\+ V\.\s*([0-9][0-9.]*)")
_PY_VERSION_RE = re.compile(r"PyDelphi:\s*v?([0-9][0-9.]*)")

# Build-folder names from the shipped makefiles, in the order the README lists.
_CPP_NAMES = (
    "delphicpp",
    "delphicpp_release",
    "delphicpp_mac",
    "delphicpp_omp_release",
    "delphicpp_mpi_release",
)
_PY_NAME = "pydelphi-static"

_INSTALL_HINT = (
    "DelPhi has no package: build the C++ release from the Clemson tarball "
    "(https://compbio.clemson.edu/lab/delphicpp_release/) and point "
    "$SASHIMI_DELPHI_PATH at the executable, or `pip install "
    "git+https://github.com/delphi001/pyDelPhi` for the pure-Python flavour, "
    "which needs no compiler and runs on every platform."
)


class DelphiNotFound(BackendUnavailable):
    """No usable DelPhi executable could be located, or it did not run."""


class DelphiFlavour(StrEnum):
    """Which DelPhi implementation an executable is.

    They differ in three places that matter and nowhere else: how a surface
    model is named, where the energies come out, and how long they take. See
    `sashimi.delphi.options` and `sashimi.delphi.run`.
    """

    CPP = "delphicpp"  # the C++ reference implementation, built from source
    PYDELPHI = "pydelphi"  # the Python/Numba reimplementation, pip-installable


@dataclass(frozen=True)
class DelphiBinary:
    path: Path
    version: str
    flavour: DelphiFlavour

    @property
    def label(self) -> str:
        """Provenance string for `Provenance.backend`.

        The flavour is part of the identity, not a footnote: two executables
        both calling themselves DelPhi 8.x do not support the same surface
        models, so a result that records only "delphi-8.6" cannot be checked
        for comparability later.
        """
        return f"{self.flavour.value}-{self.version}"

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


def classify(path: Path) -> DelphiFlavour:
    """Guess a flavour from the executable's name, before running it."""
    return DelphiFlavour.PYDELPHI if _PY_NAME in path.name else DelphiFlavour.CPP


def _candidates() -> list[Path]:
    found: list[Path] = []

    if explicit := os.environ.get("SASHIMI_DELPHI_PATH"):
        found.append(Path(explicit).expanduser())

    found.extend(
        Path(on_path) for name in (*_CPP_NAMES, _PY_NAME) if (on_path := shutil.which(name))
    )
    return found


def _run_probe(path: Path, args: list[str]) -> str:
    """Run a short command in a temp dir and return everything it printed.

    In a temp dir because the C++ program writes working files next to itself
    on some paths, and a version probe must not litter the caller's cwd — the
    same reason `sashimi.apbs.discover` probes inside one.
    """
    with tempfile.TemporaryDirectory(prefix="sashimi-delphi-probe-") as tmp:
        try:
            proc = subprocess.run(
                [str(path), *args],
                capture_output=True,
                text=True,
                timeout=120,  # pyDelPhi imports numba, which is not instant
                cwd=tmp,
                check=False,  # a non-zero exit still prints the banner
            )
        except (OSError, subprocess.SubprocessError):
            return ""
    return proc.stdout + proc.stderr


def _probe_version(path: Path, flavour: DelphiFlavour) -> str | None:
    if flavour is DelphiFlavour.PYDELPHI:
        match = _PY_VERSION_RE.search(_run_probe(path, ["--version"]))
        return match.group(1) if match else None

    # The C++ program has no --version flag; it prints its banner and then
    # complains about the input file, which is all this needs.
    output = _run_probe(path, [str(Path(tempfile.gettempdir()) / "sashimi-no-such-file.prm")])
    match = _CPP_VERSION_RE.search(output)
    return match.group(1) if match else None


@lru_cache(maxsize=8)
def _discover_cached(_explicit: str | None, _path: str | None) -> DelphiBinary:
    """Arguments are cache keys only; `_candidates` reads the environment itself."""
    tried: list[str] = []
    for candidate in _candidates():
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            tried.append(f"{candidate} (missing or not executable)")
            continue
        flavour = classify(candidate)
        version = _probe_version(candidate, flavour)
        if version is None:
            tried.append(f"{candidate} (did not report a DelPhi version)")
            continue
        return DelphiBinary(path=candidate.resolve(), version=version, flavour=flavour)

    detail = "\n  ".join(tried) if tried else "no candidate paths"
    raise DelphiNotFound(
        f"No usable DelPhi executable found. Tried:\n  {detail}\n\n{_INSTALL_HINT}"
    )


def discover_delphi() -> DelphiBinary:
    """Locate DelPhi. Cached per (SASHIMI_DELPHI_PATH, PATH)."""
    return _discover_cached(os.environ.get("SASHIMI_DELPHI_PATH"), os.environ.get("PATH"))
