"""TabipbSolver — the first backend that is not a grid.

Implements `Solver[BoundaryElementRequest]`, and that type parameter is the
whole point. ROADMAP.md section 2 calls the FD/BEM split "the acid test": a
protocol built around volumetric grids would have had to be bent to admit a
solver whose native output is a triangulated surface. Nothing was bent here.
`BoundaryElementRequest` and `SurfacePotential` were designed in phase 4 and
exercised only by `bem_stub.StubBemSolver` until now; this is the same shapes
carrying a real solver's real answer.

What that costs is honest and worth stating: this backend cannot answer
`sashimi_potential_at`, because there is no volume to interpolate. It returns
potentials on the dielectric interface, which is what a boundary-element method
computes and all it computes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sashimi.pqr import format_pqr
from sashimi.protocol import (
    BoundaryElementRequest,
    EnergyTerm,
    Provenance,
    SolveResult,
)
from sashimi.tabipb.discover import TabipbBinary, discover_tabipb
from sashimi.tabipb.input import build_input, resolved_parameters
from sashimi.tabipb.options import TabipbOptions
from sashimi.tabipb.run import DEFAULT_TIMEOUT, run_tabipb

__all__ = ["TabipbSolver"]


@dataclass
class TabipbSolver:
    """Solves the linearized Poisson-Boltzmann equation by boundary integrals.

    Discovery is lazy, as with the other backends, so constructing a solver
    never fails and the actionable error arrives on the first solve.
    """

    options: TabipbOptions = field(default_factory=TabipbOptions)
    timeout: float = DEFAULT_TIMEOUT
    _binary: TabipbBinary | None = field(default=None, repr=False)

    @property
    def binary(self) -> TabipbBinary:
        if self._binary is None:
            self._binary = discover_tabipb()
        return self._binary

    def solve(self, request: BoundaryElementRequest) -> SolveResult:
        binary = self.binary
        solvent = request.solvent

        # Raises UnsupportedRequest for a surface model with no mesh equivalent,
        # before anything runs.
        input_text = build_input(
            solvent,
            request.mesh_density,
            self.options,
            write_potential=request.want_potential,
        )

        run = run_tabipb(
            binary,
            pqr_text=format_pqr(request.structure),
            input_text=input_text,
            n_atoms=request.structure.n_atoms,
            mesh_density=request.mesh_density,
            timeout=self.timeout,
            expect_potential=request.want_potential,
        )

        provenance = Provenance(
            backend=binary.label,
            binary_path=str(binary.path),
            binary_sha256=binary.sha256,
            resolved_parameters=resolved_parameters(solvent, request.mesh_density, self.options),
            wall_seconds=round(run.wall_seconds, 3),
            # Verified against salt: TABI-PB's solvation energy moves with ionic
            # strength by -0.52 kJ/mol from 0 to 0.15 M on ALA-GLY, against
            # APBS's -0.49 on the same structure, so it carries the mobile-ion
            # contribution as APBS's difference-of-blocks does.
            energy_term=EnergyTerm.POLAR_SOLVATION,
        )

        result = SolveResult(
            provenance=provenance,
            energy_kj_mol=run.energy_kj_mol,
            potential=run.potential,
            diagnostics={
                "family": "boundary-element",
                "mesh_density": request.mesh_density,
                "n_vertices": run.potential.n_vertices if run.potential else None,
                "free_energy_kj_mol": run.free_energy_kj_mol,
                "mesher": binary.mesher_path.name,
                "mesher_version": binary.mesher_version,
                "energy_term": (
                    "solvation energy including the mobile-ion contribution, in kJ/mol "
                    "as TABI-PB reports it — no unit conversion applied"
                ),
            },
        )
        result.check_satisfies(request)
        return result
