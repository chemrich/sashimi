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

from sashimi.delphi.discover import DelphiBinary, discover_delphi
from sashimi.delphi.grid import size_grid
from sashimi.delphi.input import build_input, resolved_parameters
from sashimi.delphi.options import DelphiOptions, check_equation
from sashimi.delphi.run import DEFAULT_TIMEOUT, ENERGY_TERM, run_delphi
from sashimi.pqr import format_pqr
from sashimi.protocol import EnergyTerm, FiniteDifferenceRequest, Provenance, SolveResult

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

        provenance = Provenance(
            backend=binary.label,
            binary_path=str(binary.path),
            binary_sha256=binary.sha256,
            resolved_parameters=resolved_parameters(
                grid, solvent, self.options, flavour=flavour, equation=request.equation
            ),
            wall_seconds=round(run.wall_seconds, 3),
            # Polarization only, and it does not move with ionic strength;
            # see `sashimi.delphi.run`. Not APBS's quantity.
            energy_term=EnergyTerm.REACTION_FIELD,
        )

        result = SolveResult(
            provenance=provenance,
            energy_kj_mol=run.energy_kj_mol,
            potential=run.potential,
            diagnostics={
                **grid.as_diagnostics(),
                "flavour": flavour.value,
                # Not the same quantity APBS reports; see `sashimi.delphi.run`.
                "energy_term": ENERGY_TERM,
                "resolution_requested": request.grid.resolution,
                "resolution_relaxed": any(s > request.grid.resolution + 1e-9 for s in grid.spacing),
            },
        )
        result.check_satisfies(request)
        return result
