"""Running APBS.

Each solve gets a fresh temporary directory: APBS writes `io.mc` and its DX
output into the working directory, so anything less leaks files into the
caller's cwd and lets concurrent solves overwrite each other.

Success is verified structurally rather than from the exit code, because APBS
exits 0 on several failures. The run counts as successful only if the expected
DX file exists and parses; stdout is additionally scanned for error signatures
so a plausible-looking-but-wrong grid is caught rather than returned.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from sashimi.apbs.discover import ApbsBinary
from sashimi.apbs.input import POTENTIAL_STEM
from sashimi.dx import read_dx
from sashimi.errors import ConvergenceFailure, SolverCrash
from sashimi.protocol import PotentialGrid

__all__ = ["ApbsCrash", "ApbsRun", "run_apbs"]

DEFAULT_TIMEOUT = 300.0

_ENERGY_RE = re.compile(r"Global net ELEC energy\s*=\s*([-+0-9.eE]+)\s*kJ/mol")
_TOTAL_ENERGY_RE = re.compile(r"Total electrostatic energy\s*=\s*([-+0-9.eE]+)\s*kJ/mol")
_ITERATIONS_RE = re.compile(r"Vpmg_solve:\s*solution\s*took\s*([0-9.]+)\s*sec", re.IGNORECASE)

# APBS exits 0 on these; they must be caught by reading stdout.
_CONVERGENCE_SIGNATURES = (
    "did not converge",
    "iteration failed",
    "Bad iteration",
)
_ERROR_SIGNATURES = (
    "Vpmg_ctor2:",
    "Vnm_print: Error",
    "APBS ERROR",
    "Error while parsing input file",
    "Fatal error",
)


class ApbsCrash(SolverCrash):
    """APBS exited abnormally, timed out, or produced no usable output."""


@dataclass
class ApbsRun:
    potential: PotentialGrid | None
    energy_kj_mol: float | None
    stdout: str
    wall_seconds: float
    returncode: int


def _tail(text: str, lines: int = 25) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def find_potential(work: Path) -> Path | None:
    """Locate the DX grid APBS wrote, whichever name this build chose.

    Serial builds write `<stem>.dx`. Builds compiled with MPI support append
    the processing-element rank — `<stem>-PE0.dx` — even for a single-process
    `mg-auto` run. Debian's apbs 3.4.1 does this and Homebrew's does not, so
    the same version and the same input file produce different filenames on
    different platforms. The contents are identical.
    """
    exact = work / f"{POTENTIAL_STEM}.dx"
    if exact.exists():
        return exact
    ranked = sorted(work.glob(f"{POTENTIAL_STEM}-PE*.dx"))
    return ranked[0] if ranked else None


def _as_text(raw: object) -> str:
    """Captured output is str under `text=True` and bytes otherwise; typeshed
    annotates `TimeoutExpired.stdout` as bytes regardless, so accept both."""
    if isinstance(raw, bytes):
        return raw.decode(errors="replace")
    return raw if isinstance(raw, str) else ""


def run_apbs(
    binary: ApbsBinary,
    *,
    pqr_text: str,
    input_text: str,
    timeout: float = DEFAULT_TIMEOUT,
    expect_energy: bool = False,
    expect_potential: bool = True,
) -> ApbsRun:
    with tempfile.TemporaryDirectory(prefix="sashimi-solve-") as tmp:
        work = Path(tmp)
        (work / "mol.pqr").write_text(pqr_text)
        (work / "input.in").write_text(input_text)

        started = time.monotonic()
        try:
            proc = subprocess.run(
                [str(binary.path), "input.in"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work,
                # Deliberately not check=True: APBS exits 0 on several failures,
                # so the exit code is not the signal. Success is verified below
                # from the output that was actually produced.
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            partial = _tail(_as_text(exc.stdout))
            raise ApbsCrash(
                f"APBS timed out after {timeout:g}s. Reduce resolution or raise the timeout.\n"
                f"Last output:\n{partial}"
            ) from exc
        except OSError as exc:
            raise ApbsCrash(f"could not execute {binary.path}: {exc}") from exc
        wall = time.monotonic() - started

        combined = proc.stdout + proc.stderr

        for signature in _CONVERGENCE_SIGNATURES:
            if signature.lower() in combined.lower():
                raise ConvergenceFailure(
                    f"APBS reported a convergence problem ({signature!r}).\n"
                    f"Last output:\n{_tail(combined)}"
                )
        for signature in _ERROR_SIGNATURES:
            if signature.lower() in combined.lower():
                raise ApbsCrash(
                    f"APBS reported an error ({signature!r}), exit code {proc.returncode}.\n"
                    f"Last output:\n{_tail(combined)}"
                )

        potential = None
        dx_path = find_potential(work)
        if dx_path is None and expect_potential:
            # Name what the run *did* leave behind. A build that writes the grid
            # under another name, or writes nothing at all, are different faults
            # and the message should not make the caller guess which happened.
            produced = sorted(p.name for p in work.iterdir())
            raise ApbsCrash(
                f"APBS exited {proc.returncode} without writing {POTENTIAL_STEM}.dx. "
                f"Exit code alone is not reliable here, so this is treated as failure.\n"
                f"Files in the working directory: {produced}\n"
                f"Last output:\n{_tail(combined)}"
            )
        if dx_path is not None:
            try:
                potential = read_dx(dx_path)
            except ValueError as exc:
                raise ApbsCrash(f"APBS wrote an unparseable {dx_path.name}: {exc}") from exc

        energy: float | None = None
        if expect_energy:
            match = _ENERGY_RE.search(combined)
            if match is None:
                raise ApbsCrash(
                    "energy was requested but APBS printed no 'Global net ELEC energy' line.\n"
                    f"Last output:\n{_tail(combined)}"
                )
            energy = float(match.group(1))

        return ApbsRun(
            potential=potential,
            energy_kj_mol=energy,
            stdout=combined,
            wall_seconds=wall,
            returncode=proc.returncode,
        )


def parse_block_energies(stdout: str) -> list[float]:
    """Per-elec-block total electrostatic energies, in order. Diagnostics only."""
    return [float(v) for v in _TOTAL_ENERGY_RE.findall(stdout)]
