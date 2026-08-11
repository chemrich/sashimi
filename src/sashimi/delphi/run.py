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

**What the energy is.** DelPhi's headline "corrected reaction field energy" is
*not* APBS's solvation energy: it is the polarization term alone, and it does
not move with ionic strength at all — -92.22 kT on a Born ion at both 0 M and
0.5 M, despite the solver reporting a Debye length of 4.307 A at 0.5 M. APBS's
`Global net ELEC energy` is a difference against a uniform-dielectric, ion-free
reference and so carries the mobile-ion atmosphere by construction.

The C++ build can be asked for the matching quantity, which is what sashimi
does. `parse_cpp_polar_solvation` reconstructs it from the aggregate line; see
that function for the arithmetic and `input.py` for why the request has to say
`energy(s,c,ion)` rather than the `s` or `s,ion` one might expect.

That it is the right term is not assumed. The reconstructed contribution is zero
at zero salt and grows monotonically with it (-0.20 kT at 0.15 M, -0.34 at 0.5 M
on a Born ion), and adding it makes the gap to APBS **salt-independent** —
2.30 / 2.59 / 2.70% across 0 / 0.15 / 0.5 M becomes 2.30 / 2.38 / 2.34%, leaving
only the constant discretization difference between two codes on different
grids. A missing term produces exactly that signature; a coincidence does not.

pyDelPhi has no such line: its results CSV carries no ion-atmosphere column, so
it stays on the reaction-field term and `ENERGY_TERMS` records the difference.
`sashimi.validate` reads that and compares the two only where they coincide.
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
from sashimi.protocol import EnergyTerm, PotentialGrid

__all__ = [
    "DEFAULT_TIMEOUT",
    "ENERGY_TERMS",
    "ENERGY_TERM_DETAIL",
    "DelphiCrash",
    "DelphiRun",
    "run_delphi",
]

DEFAULT_TIMEOUT = 300.0

ENERGY_CSV = "outputs.csv"
_CSV_ENERGY_COLUMN = "E_rxn_corr_tot"  # total corrected reaction-field energy, kT

_CPP_ENERGY_RE = re.compile(
    r"Corrected reaction field energy\s*:\s*([-+0-9.eE]+)\s*kT", re.IGNORECASE
)
_CPP_COULOMBIC_RE = re.compile(r"Coulombic energy\s*:\s*([-+0-9.eE]+)\s*kT", re.IGNORECASE)
_CPP_TOTAL_RE = re.compile(
    r"All required energy terms but grid energy\s*:\s*([-+0-9.eE]+)\s*kT", re.IGNORECASE
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

# What each flavour's reported energy is. These differ, and for once that is a
# capability difference rather than a mistake: the C++ build can be asked for
# the ion-inclusive quantity and pyDelPhi cannot, since its results CSV carries
# no ion-atmosphere column.
ENERGY_TERMS: dict[DelphiFlavour, EnergyTerm] = {
    DelphiFlavour.CPP: EnergyTerm.POLAR_SOLVATION,
    DelphiFlavour.PYDELPHI: EnergyTerm.REACTION_FIELD,
}

ENERGY_TERM_DETAIL: dict[DelphiFlavour, str] = {
    DelphiFlavour.CPP: (
        "reaction field plus the mobile-ion atmosphere (DelPhi's aggregate less its "
        "Coulombic term), which is the quantity APBS reports"
    ),
    DelphiFlavour.PYDELPHI: (
        "corrected reaction field (polarization only; pyDelPhi reports no ion-atmosphere "
        "term, so this does not move with ionic strength and is not APBS's quantity)"
    ),
}


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


def parse_cpp_reaction_field(stdout: str) -> float | None:
    """The C++ program's reaction-field line, in kT. Polarization only."""
    match = _CPP_ENERGY_RE.search(stdout)
    return float(match.group(1)) if match else None


def parse_cpp_polar_solvation(stdout: str) -> float | None:
    """Reaction field *plus* the mobile-ion atmosphere, in kT — APBS's quantity.

    DelPhi's aggregate line is
    `Nonlinear + Coulombic + Solvation + SolvToChgIn + SolvToChgOut`
    (`energy_run.cpp`), where the last two are the ion-atmosphere terms and
    `Nonlinear` is zero for the linear equation. Subtracting the separately
    printed Coulombic term therefore leaves exactly the polar solvation energy.

    The alternative would be DelPhi's dedicated "solvent and boundary pol." line,
    which computes precisely this — but it is commented out in 8.5.0/8.6.

    Costs a little precision: both inputs are printed to two decimals in kT, so
    the result carries up to 0.01 kT (0.025 kJ/mol) of rounding. That is an
    absolute error, so it matters least where it would matter most — on a large
    solute whose Coulombic term is huge.
    """
    total = _CPP_TOTAL_RE.search(stdout)
    coulombic = _CPP_COULOMBIC_RE.search(stdout)
    if total is None or coulombic is None:
        return None
    return float(total.group(1)) - float(coulombic.group(1))


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

        # pyDelPhi's CSV carries only a reaction-field column, so it stays on
        # that term; the C++ build is asked for the ion-inclusive quantity.
        energy_kt = parse_csv_energy(work / ENERGY_CSV)
        if energy_kt is None:
            energy_kt = parse_cpp_polar_solvation(combined) or parse_cpp_reaction_field(combined)

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
