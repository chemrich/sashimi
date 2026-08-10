"""Running DelPhi.

Each solve gets a fresh temporary directory, for the same reason APBS does: both
flavours write output next to the parameter file, so anything less leaks files
into the caller's cwd and lets concurrent solves collide.

As with APBS, success is verified **structurally rather than from the exit
code** — but DelPhi's silent failures are its own. The C++ program exits 0 after
printing `[FATAL ERROR]` and after a parse failure that leaves it having
computed nothing, so the checks here look for what was produced and scan the
output for its error markers. It also has no timeout of its own: with `linit`
and `maxc` at their defaults of 0 it iterates forever, which the options layer
prevents by always writing both, and the timeout here catches anything else.

Energies come from different places. The C++ program prints them, at two
decimal places, which is the precision floor for anything compared against a
C++ DelPhi number. pyDelPhi writes a tab-separated CSV with four, and that is
preferred whenever it exists.

**What the energy is, and why it is not one thing.** Both flavours label their
answer a *corrected reaction field energy*, and they do not mean the same by it.

- The C++ program's printed line is the polarization term alone. It does not
  move with salt: measured on a Born ion, -92.22 kT at both 0 M and 0.5 M, even
  though the solver plainly received the salt (it reports a Debye length of
  4.307 A). Asking it for the ionic term as well shifts its aggregate to
  -92.56 kT.
- pyDelPhi's `E_rxn_corr_tot` does move with salt — -219.27 vs -220.63 kJ/mol
  between 0 M and 0.5 M on the same request — so it behaves like the C++
  *aggregate* rather than the C++ line of the same name.

APBS is a third case again: `Global net ELEC energy` is a difference between a
solvated state and a uniform-dielectric, ion-free reference, so it carries the
mobile-ion contribution by construction. Three backends, three definitions of
"the solvation energy", agreeing to ~2% at zero salt and diverging in a way no
tolerance would distinguish from a bug. `energy_term_description` puts the
distinction in `SolveResult.diagnostics` rather than leaving a reader to assume
the numbers are interchangeable.
"""

from __future__ import annotations

import csv
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from sashimi.delphi.cube import read_cube
from sashimi.delphi.discover import DelphiBinary, DelphiFlavour
from sashimi.delphi.input import POTENTIAL_FILENAME, PQR_FILENAME
from sashimi.delphi.units import kt_to_kj_per_mol
from sashimi.errors import ConvergenceFailure, MalformedStructure, SolverCrash
from sashimi.protocol import PotentialGrid

__all__ = ["DEFAULT_TIMEOUT", "DelphiCrash", "DelphiRun", "run_delphi"]

DEFAULT_TIMEOUT = 300.0

ENERGY_CSV = "outputs.csv"
_CSV_ENERGY_COLUMN = "E_rxn_corr_tot"  # total corrected reaction-field energy, kT

_CPP_ENERGY_RE = re.compile(
    r"Corrected reaction field energy\s*:\s*([-+0-9.eE]+)\s*kT", re.IGNORECASE
)

# Both flavours exit 0 on these.
_ERROR_SIGNATURES = (
    "[FATAL ERROR]",
    "PROGRAM ABORTS",
    "parseBEM error",
    "❌",  # pyDelPhi's parameter-validation marker
    "Invalid choice",
)
_CONVERGENCE_SIGNATURES = (
    "did not converge",
    "failed to converge",
)

_MIN_CSV_ROWS = 2  # a header plus at least one record


class DelphiCrash(SolverCrash):
    """DelPhi exited abnormally, timed out, or produced no usable output."""


@dataclass
class DelphiRun:
    potential: PotentialGrid | None
    energy_kj_mol: float | None
    stdout: str
    wall_seconds: float
    returncode: int


def _tail(text: str, lines: int = 25) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def parse_cpp_energy(stdout: str) -> float | None:
    """The C++ program's reaction-field energy, in kT. None if it printed none."""
    match = _CPP_ENERGY_RE.search(stdout)
    return float(match.group(1)) if match else None


def parse_csv_energy(path: Path) -> float | None:
    """pyDelPhi's reaction-field energy, in kT, from its results CSV.

    Preferred over the printed value because it carries four decimals rather
    than two. The file leads with a `#` comment describing the columns.
    """
    if not path.is_file():
        return None
    rows = [
        line for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")
    ]
    if len(rows) < _MIN_CSV_ROWS:
        return None
    reader = csv.DictReader(rows, delimiter="\t")
    for record in reader:
        value = record.get(_CSV_ENERGY_COLUMN)
        if value:
            return float(value)
    return None


def energy_term_description(flavour: DelphiFlavour) -> str:
    """What this flavour's reported energy actually includes.

    Travels in diagnostics because the two flavours use the same words for
    different quantities, and because neither matches APBS's definition. See the
    module docstring for the measurements.
    """
    if flavour is DelphiFlavour.CPP:
        return (
            "corrected reaction field (polarization only; does not move with ionic "
            "strength, and excludes the mobile-ion osmotic term APBS includes)"
        )
    return (
        "corrected reaction field total (moves with ionic strength, so it includes a "
        "mobile-ion contribution the C++ flavour's line of the same name omits)"
    )


def _raise_on_reported_failure(output: str, flavour: DelphiFlavour, returncode: int) -> None:
    """Turn a self-reported failure into an exception. Both flavours exit 0 on these."""
    lowered = output.lower()
    for signature in _CONVERGENCE_SIGNATURES:
        if signature.lower() in lowered:
            raise ConvergenceFailure(
                f"{flavour.value} reported a convergence problem ({signature!r}).\n"
                f"Last output:\n{_tail(output)}"
            )
    for signature in _ERROR_SIGNATURES:
        if signature.lower() in lowered:
            raise DelphiCrash(
                f"{flavour.value} reported an error ({signature!r}), exit code "
                f"{returncode}.\nLast output:\n{_tail(output)}"
            )


def run_delphi(
    binary: DelphiBinary,
    *,
    pqr_text: str,
    input_text: str,
    temperature: float,
    timeout: float = DEFAULT_TIMEOUT,
    expect_energy: bool = True,
    expect_potential: bool = True,
) -> DelphiRun:
    with tempfile.TemporaryDirectory(prefix="sashimi-delphi-") as tmp:
        work = Path(tmp)
        (work / PQR_FILENAME).write_text(pqr_text)
        (work / "param.prm").write_text(input_text)

        command = [str(binary.path)]
        if binary.flavour is DelphiFlavour.PYDELPHI:
            command += ["--param-file", "param.prm", "--platform", "cpu"]
        else:
            command += ["param.prm"]

        started = time.monotonic()
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work,
                stdin=subprocess.DEVNULL,  # the C++ program reads stdin if its input is unusable
                check=False,  # exit code is not the signal; see the module docstring
            )
        except subprocess.TimeoutExpired as exc:
            partial = _tail(_as_text(exc.stdout))
            raise DelphiCrash(
                f"{binary.flavour.value} timed out after {timeout:g}s. DelPhi has no iteration "
                "limit of its own when `linit`/`maxc` are unset; sashimi always writes both, so "
                "this is more likely a grid that is simply too large.\n"
                f"Last output:\n{partial}"
            ) from exc
        except OSError as exc:
            raise DelphiCrash(f"could not execute {binary.path}: {exc}") from exc
        wall = time.monotonic() - started

        combined = proc.stdout + proc.stderr
        _raise_on_reported_failure(combined, binary.flavour, proc.returncode)

        potential = None
        cube = work / POTENTIAL_FILENAME
        if expect_potential and not cube.is_file():
            produced = sorted(p.name for p in work.iterdir())
            raise DelphiCrash(
                f"{binary.flavour.value} exited {proc.returncode} without writing "
                f"{POTENTIAL_FILENAME}. Exit code alone is not reliable here, so this is "
                f"treated as failure.\nFiles in the working directory: {produced}\n"
                f"Last output:\n{_tail(combined)}"
            )
        if cube.is_file():
            try:
                potential = read_cube(cube)
            except MalformedStructure as exc:
                raise DelphiCrash(
                    f"{binary.flavour.value} wrote an unparseable cube: {exc}"
                ) from exc

        energy_kt = parse_csv_energy(work / ENERGY_CSV)
        if energy_kt is None:
            energy_kt = parse_cpp_energy(combined)

        energy = None
        if expect_energy:
            if energy_kt is None:
                raise DelphiCrash(
                    "energy was requested but DelPhi reported no corrected reaction field "
                    f"energy.\nLast output:\n{_tail(combined)}"
                )
            # DelPhi reports multiples of kT at the run's temperature; the
            # protocol is kJ/mol. See `sashimi.delphi.units`.
            energy = kt_to_kj_per_mol(energy_kt, temperature)

        return DelphiRun(
            potential=potential,
            energy_kj_mol=energy,
            stdout=combined,
            wall_seconds=wall,
            returncode=proc.returncode,
        )


def _as_text(raw: object) -> str:
    if isinstance(raw, bytes):
        return raw.decode(errors="replace")
    return raw if isinstance(raw, str) else ""
