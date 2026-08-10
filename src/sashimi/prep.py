"""Structure preparation: PDB -> PQR, via pdb2pqr.

A subprocess wrapper, not a use of pdb2pqr's Python API. Its internals are not
a stability contract, and process isolation means a pdb2pqr hang cannot take
down the MCP server.

pdb2pqr rebuilds structures — it adds missing heavy atoms, debumps clashes and
places hydrogens — and it is opinionated about it. Those decisions change the
charges that go into the solver, so they are surfaced as a structured summary
rather than left in a log nobody reads. An agent should know three sidechains
were rebuilt before trusting the energies downstream.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sashimi.errors import SashimiError
from sashimi.pqr import parse_pqr
from sashimi.protocol import PQRData

__all__ = ["ForceField", "PrepResult", "PreparationFailed", "prepare_structure"]

ForceField = Literal["AMBER", "CHARMM", "PARSE", "TYL06", "PEOEPB", "SWANSON"]

DEFAULT_TIMEOUT = 300.0

# pdb2pqr logs `LEVEL:message` to stderr.
_LOG_RE = re.compile(r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL):(.*)$")

# Structural edits worth reporting, keyed by the phrase pdb2pqr uses.
_INTERESTING = {
    "missing_atoms": re.compile(r"missing atom (\S+) in residue (.+)", re.IGNORECASE),
    "added_atoms": re.compile(r"added atom (\S+) to residue (\S+ \S+ \S+)", re.IGNORECASE),
    "debumped": re.compile(r"debump(?:ing|ed)\s+(\S+ \S+ \S+)", re.IGNORECASE),
    "multiple_occupancy": re.compile(r"multiple occupanc\w+ found", re.IGNORECASE),
    "unknown_residue": re.compile(r"unable to (?:find|assign) .*residue (.+)", re.IGNORECASE),
}

# Noise about output formatting, not about the structure.
_IGNORED = re.compile(r"ignoring \d+ header lines", re.IGNORECASE)


class PreparationFailed(SashimiError):
    """pdb2pqr could not produce a usable PQR."""


@dataclass
class PrepResult:
    """A prepared structure plus what pdb2pqr changed to get there."""

    pqr: PQRData
    pqr_text: str
    warnings: tuple[str, ...] = ()
    edits: dict[str, list[str]] = field(default_factory=dict)

    @property
    def structure_was_modified(self) -> bool:
        """True when pdb2pqr rebuilt something, not merely reformatted it."""
        return any(self.edits.values())

    def summary(self) -> dict[str, object]:
        """Compact, JSON-shaped account for the MCP layer to surface."""
        return {
            "n_atoms": self.pqr.n_atoms,
            "total_charge": round(self.pqr.total_charge, 4),
            "structure_was_modified": self.structure_was_modified,
            "edits": {k: v for k, v in self.edits.items() if v},
            "n_warnings": len(self.warnings),
            "warnings": list(self.warnings),
        }


def _classify(lines: list[str]) -> tuple[tuple[str, ...], dict[str, list[str]]]:
    warnings: list[str] = []
    edits: dict[str, list[str]] = {key: [] for key in _INTERESTING}

    for raw in lines:
        match = _LOG_RE.match(raw.strip())
        if match is None:
            continue
        level, message = match.group(1), match.group(2).strip()
        if _IGNORED.search(message):
            continue

        # pdb2pqr emits the same record on stdout and stderr, so the combined
        # stream repeats every line; dedupe while preserving first-seen order.
        if level in ("WARNING", "ERROR", "CRITICAL") and message not in warnings:
            warnings.append(message)

        for key, pattern in _INTERESTING.items():
            found = pattern.search(message)
            if found:
                detail = " ".join(g for g in found.groups() if g) or message
                if detail not in edits[key]:
                    edits[key].append(detail)

    return tuple(warnings), edits


def prepare_structure(
    pdb_path: str | os.PathLike[str],
    *,
    forcefield: ForceField = "AMBER",
    ph: float | None = None,
    drop_water: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
) -> PrepResult:
    """Assign charges and radii to a PDB structure.

    `ph` opts into propka titration-state prediction; leaving it None keeps the
    forcefield's default protonation, which is faster and deterministic.
    """
    source = Path(pdb_path)
    if not source.is_file():
        raise PreparationFailed(f"no such structure file: {source}")

    with tempfile.TemporaryDirectory(prefix="sashimi-prep-") as tmp:
        work = Path(tmp)
        out = work / "prepared.pqr"

        # `-m pdb2pqr` rather than a PATH lookup: it pins execution to the same
        # interpreter this package is installed into, so a stray pdb2pqr on
        # PATH cannot answer instead.
        argv = [sys.executable, "-m", "pdb2pqr", f"--ff={forcefield}"]
        if drop_water:
            argv.append("--drop-water")
        if ph is not None:
            argv += ["--titration-state-method=propka", "--with-ph", str(ph)]
        argv += [str(source.resolve()), str(out)]

        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PreparationFailed(
                f"pdb2pqr timed out after {timeout:g}s on {source.name}"
            ) from exc
        except OSError as exc:
            raise PreparationFailed(f"could not run pdb2pqr: {exc}") from exc

        combined = proc.stdout + proc.stderr
        warnings, edits = _classify(combined.splitlines())

        if not out.exists():
            tail = "\n".join(combined.strip().splitlines()[-15:])
            raise PreparationFailed(
                f"pdb2pqr exited {proc.returncode} without writing a PQR for "
                f"{source.name}.\nLast output:\n{tail}"
            )

        pqr_text = out.read_text()

    try:
        pqr = parse_pqr(pqr_text)
    except ValueError as exc:
        raise PreparationFailed(f"pdb2pqr wrote an unparseable PQR: {exc}") from exc

    return PrepResult(pqr=pqr, pqr_text=pqr_text, warnings=warnings, edits=edits)
