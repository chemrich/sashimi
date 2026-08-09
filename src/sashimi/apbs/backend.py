"""ApbsSolver — the subprocess backend, and the only thing debye replaces."""

from __future__ import annotations

from dataclasses import dataclass, field

from sashimi.apbs.discover import ApbsBinary, discover_apbs
from sashimi.apbs.grid import size_grid
from sashimi.apbs.input import build_input
from sashimi.apbs.run import DEFAULT_TIMEOUT, parse_block_energies, run_apbs
from sashimi.pqr import format_pqr
from sashimi.protocol import GridSpec, PQRData, SolventModel, SolveResult

__all__ = ["ApbsSolver"]


@dataclass
class ApbsSolver:
    """Solves the LPBE by shelling out to APBS.

    Implements `sashimi.protocol.Solver`. Binary discovery is lazy so that
    constructing a solver never fails; the useful error surfaces on the first
    solve, where the caller can act on it.
    """

    timeout: float = DEFAULT_TIMEOUT
    _binary: ApbsBinary | None = field(default=None, repr=False)

    @property
    def binary(self) -> ApbsBinary:
        if self._binary is None:
            self._binary = discover_apbs()
        return self._binary

    def solve_lpbe(
        self,
        pqr: PQRData,
        grid: GridSpec,
        solvent: SolventModel = SolventModel(),  # noqa: B008 — frozen dataclass
        *,
        compute_energy: bool = False,
    ) -> SolveResult:
        binary = self.binary
        apbs_grid = size_grid(pqr, grid)
        input_text = build_input(apbs_grid, solvent, compute_energy=compute_energy)

        run = run_apbs(
            binary,
            pqr_text=format_pqr(pqr),
            input_text=input_text,
            timeout=self.timeout,
            expect_energy=compute_energy,
        )

        diagnostics = {
            **apbs_grid.as_diagnostics(),
            "wall_seconds": round(run.wall_seconds, 3),
            "binary_path": str(binary.path),
            "block_energies_kj_mol": parse_block_energies(run.stdout),
            "resolution_requested": grid.resolution,
            "resolution_relaxed": any(s > grid.resolution + 1e-9 for s in apbs_grid.spacing),
        }

        return SolveResult(
            potential=run.potential,
            energy_kj_mol=run.energy_kj_mol,
            backend=binary.label,
            diagnostics=diagnostics,
        )
