"""DelphiSolver — the second backend, and the protocol's first real test.

Implements `Solver[FiniteDifferenceRequest]` with no changes to the protocol at
all: the same request type APBS takes, the same `SolveResult` out. Everything
DelPhi does differently — a cubic grid instead of a multigrid lattice, energies
in kT, geometry in Bohr, a temperature parameter in Celsius, a surface-model
vocabulary that overlaps APBS's in exactly one member — is absorbed below this
line. That is the claim ROADMAP.md section 2 makes for the protocol layer, now
tested by something other than the backend it was designed around.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sashimi.delphi.discover import DelphiBinary, discover_delphi
from sashimi.delphi.grid import size_grid
from sashimi.delphi.input import build_input, resolved_parameters
from sashimi.delphi.options import DelphiOptions, check_equation
from sashimi.delphi.run import (
    DEFAULT_TIMEOUT,
    ENERGY_TERM_DETAIL,
    ENERGY_TERMS,
    parse_cpp_assignment,
    run_delphi,
)
from sashimi.errors import SolverError
from sashimi.pqr import format_pqr
from sashimi.protocol import FiniteDifferenceRequest, PQRData, Provenance, SolveResult

__all__ = ["DelphiSolver"]


@dataclass
class DelphiSolver:
    """Solves the linearized Poisson-Boltzmann equation by shelling out to DelPhi.

    Binary discovery is lazy, so constructing a solver never fails and the
    useful error surfaces on the first solve where the caller can act on it —
    the same contract `ApbsSolver` offers.
    """

    options: DelphiOptions = field(default_factory=DelphiOptions)
    timeout: float = DEFAULT_TIMEOUT
    _binary: DelphiBinary | None = field(default=None, repr=False)

    @property
    def binary(self) -> DelphiBinary:
        if self._binary is None:
            self._binary = discover_delphi()
        return self._binary

    def solve(self, request: FiniteDifferenceRequest) -> SolveResult:
        binary = self.binary
        flavour = binary.flavour
        check_equation(request.equation, flavour)

        solvent = request.solvent
        # Raises UnsupportedRequest before anything runs when this flavour has
        # no equivalent of the requested surface — the default
        # SMOOTHED_MOLECULAR is APBS-only, and substituting silently would move
        # the answer by more than 2,000x the corpus tolerance.
        grid = size_grid(request.structure, request.grid)

        input_text = build_input(
            grid,
            solvent,
            self.options,
            flavour=flavour,
            compute_energy=request.want_energy,
            write_potential=request.want_potential,
        )

        run = run_delphi(
            binary,
            pqr_text=format_pqr(request.structure),
            input_text=input_text,
            temperature=solvent.temperature,
            timeout=self.timeout,
            expect_energy=request.want_energy,
            expect_potential=request.want_potential,
        )

        _verify_delphi_read_what_was_sent(run.stdout, request.structure)

        provenance = Provenance(
            backend=binary.label,
            binary_path=str(binary.path),
            binary_sha256=binary.sha256,
            resolved_parameters=resolved_parameters(
                grid, solvent, self.options, flavour=flavour, equation=request.equation
            ),
            wall_seconds=round(run.wall_seconds, 3),
            # Differs by flavour: only the C++ build can be asked for the
            # ion-inclusive quantity. See `sashimi.delphi.run`.
            energy_term=ENERGY_TERMS[flavour],
        )

        result = SolveResult(
            provenance=provenance,
            energy_kj_mol=run.energy_kj_mol,
            potential=run.potential,
            diagnostics={
                **grid.as_diagnostics(),
                "flavour": flavour.value,
                # Not the same quantity APBS reports; see `sashimi.delphi.run`.
                "energy_term": ENERGY_TERM_DETAIL[flavour],
                "resolution_requested": request.grid.resolution,
                "resolution_relaxed": any(s > request.grid.resolution + 1e-9 for s in grid.spacing),
            },
        )
        result.check_satisfies(request)
        return result


# DelPhi's echo agrees with the file to four figures on every structure tested,
# so the band only absorbs its printed precision. A real misparse is not close:
# the acetate that motivated this arrived as +80.84 e against -1.
_CHARGE_TOLERANCE = 0.05


def _verify_delphi_read_what_was_sent(stdout: str, structure: PQRData) -> None:
    """Check DelPhi's own echo of the PQR against the PQR, or say nothing.

    Structural verification of the output rather than trust in the exit code —
    the same discipline ROADMAP.md §13 applies to APBS, which also exits 0 on
    failure. DelPhi reads by fixed column, so a shifted field is not an error to
    it, and every downstream number is then computed from charges that are not
    the caller's.

    Silent for a flavour that prints no such line: pyDelPhi reports through a
    CSV and has no equivalent echo, and inventing a check it cannot fail would
    be worse than admitting the gap.
    """
    assignment = parse_cpp_assignment(stdout)
    if assignment is None:
        return

    reported_charge, reported_count = assignment
    expected_charge = float(structure.charges.sum())
    expected_count = int(np.count_nonzero(structure.charges))

    if (
        abs(reported_charge - expected_charge) > _CHARGE_TOLERANCE
        or reported_count != expected_count
    ):
        raise SolverError(
            f"DelPhi read a different structure from the one it was given: it reports "
            f"{reported_count} charged atoms totalling {reported_charge:+.4f} e, where the "
            f"PQR holds {expected_count} totalling {expected_charge:+.4f} e. DelPhi parses "
            "PQR by fixed column, so this is a formatting mismatch rather than a solver "
            "failure, and every energy and potential from this run would be computed from "
            "the wrong charges. Please report it: sashimi wrote the file."
        )
