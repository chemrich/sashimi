"""ApbsSolver — the subprocess backend, and the only thing debye replaces."""

from __future__ import annotations

from dataclasses import dataclass, field

from sashimi.apbs.discover import ApbsBinary, discover_apbs
from sashimi.apbs.grid import size_grid
from sashimi.apbs.input import build_input, resolved_parameters
from sashimi.apbs.options import ApbsOptions
from sashimi.apbs.run import DEFAULT_TIMEOUT, parse_block_energies, run_apbs
from sashimi.errors import UnsupportedRequest
from sashimi.pqr import format_pqr
from sashimi.protocol import (
    Equation,
    FiniteDifferenceRequest,
    Provenance,
    SolveResult,
)

__all__ = ["ApbsSolver"]


@dataclass
class ApbsSolver:
    """Solves the Poisson-Boltzmann equation by shelling out to APBS.

    Implements `Solver[FiniteDifferenceRequest]`. Binary discovery is lazy so
    that constructing a solver never fails; the useful error surfaces on the
    first solve, where the caller can act on it.
    """

    options: ApbsOptions = field(default_factory=ApbsOptions)
    timeout: float = DEFAULT_TIMEOUT
    _binary: ApbsBinary | None = field(default=None, repr=False)

    @property
    def binary(self) -> ApbsBinary:
        if self._binary is None:
            self._binary = discover_apbs()
        return self._binary

    def solve(self, request: FiniteDifferenceRequest) -> SolveResult:
        if request.equation is not Equation.LINEAR:
            # Representable, not implemented (ROADMAP.md phase 4). The keyword
            # mapping exists, so enabling this means adding corpus cases and
            # settling the NPBE-vs-LPBE energy comparability rule -- not
            # changing the protocol. Refusing beats returning untested numbers.
            raise UnsupportedRequest(
                f"sashimi can express the {request.equation.value} equation but does not yet "
                "solve it; APBS supports it, and wiring it up is tracked as ROADMAP.md "
                "section 14 'still open'. Use Equation.LINEAR."
            )

        binary = self.binary
        solvent = request.solvent
        grid = size_grid(request.structure, request.grid)

        input_text = build_input(
            grid,
            solvent,
            self.options,
            equation=request.equation,
            compute_energy=request.want_energy,
            write_potential=request.want_potential,
        )

        run = run_apbs(
            binary,
            pqr_text=format_pqr(request.structure),
            input_text=input_text,
            timeout=self.timeout,
            expect_energy=request.want_energy,
            expect_potential=request.want_potential,
        )

        provenance = Provenance(
            backend=binary.label,
            binary_path=str(binary.path),
            binary_sha256=binary.sha256,
            resolved_parameters=resolved_parameters(
                grid, solvent, self.options, equation=request.equation
            ),
            wall_seconds=round(run.wall_seconds, 3),
        )

        result = SolveResult(
            provenance=provenance,
            energy_kj_mol=run.energy_kj_mol,
            potential=run.potential,
            diagnostics={
                **grid.as_diagnostics(),
                "block_energies_kj_mol": parse_block_energies(run.stdout),
                "resolution_requested": request.grid.resolution,
                "resolution_relaxed": any(s > request.grid.resolution + 1e-9 for s in grid.spacing),
            },
        )
        result.check_satisfies(request)
        return result
