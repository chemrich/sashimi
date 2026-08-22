"""DebyeSolver — the clean-room finite-difference solver, at M1.

The second backend with no binary and the first that is both in-process *and*
in the reference tier: `sashimi.gb` needs nothing installed but approximates the
equation, and every backend that discretizes it needs a compiled program. That
combination is the whole point — ROADMAP.md section 12 exists because protean's
default field path has no binary available, so an approximation was the only
thing sashimi could offer it, and known-wrong physics is not something to hand a
consumer that will come to depend on it.

**Two solves, not one, and that is the design rather than an inefficiency.** A
point charge on a grid carries an enormous self-energy that has nothing to do
with solvation and diverges as the grid refines. The solvation energy is the
*difference* between the solvated system and the same charges in a uniform
solute dielectric with no mobile ions, on the same grid, with the same charge
assignment — and the self-energy cancels between them exactly. That is what APBS
does with its two `elec` blocks, and it is why the reported term is
`POLAR_SOLVATION` rather than DelPhi's corrected reaction field.

**Where M1 stops.** The van der Waals boundary, the linearized equation, and
nothing else; `sashimi.debye.options` refuses the rest by name. The solvent
excluded surface is M4 and is the hardest single construction on the ladder,
which is why the ladder climbs the sharp boundary first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from sashimi.constants import AVOGADRO, BOLTZMANN, JOULES_PER_KJ
from sashimi.debye.dielectric import bjerrum_length_a
from sashimi.debye.grid import DebyeGrid, size_grid
from sashimi.debye.linear import SolveReport, build_levels, solve_system
from sashimi.debye.options import DebyeOptions, check_equation, check_surface
from sashimi.debye.sources import (
    debye_huckel_boundaries,
    interpolate_at_atoms,
    source_term,
)
from sashimi.protocol import (
    AccuracyTier,
    Diagnostics,
    EnergyTerm,
    FiniteDifferenceRequest,
    FloatArray,
    PotentialGrid,
    PQRData,
    Provenance,
    SolventModel,
    SolveResult,
)

__all__ = ["BACKEND_VERSION", "DebyeSolver"]

# Bumped when this solver's answer changes. The corpus records numbers against
# it, so "debye changed" and "debye's version changed" have to be the same
# event or a recording cannot be read.
BACKEND_VERSION = "0.1"


def _solve_state(
    grid: DebyeGrid,
    structure: PQRData,
    solvent: SolventModel,
    options: DebyeOptions,
    *,
    boundary: FloatArray,
) -> tuple[FloatArray, SolveReport]:
    """One of the two states, returning the total potential in kT/e.

    The Dirichlet data is held out of the linear system and added back at the
    end: every vector the solver touches then has a zero boundary, so the
    coarse-grid corrections need no boundary handling of their own and cannot
    accidentally inherit the fine grid's Debye-Huckel tail.

    **The boundary is passed in rather than computed here**, because the two
    states share the distance matrix it is built from and that matrix is 1.5
    billion pairs on a protein. See `debye_huckel_boundaries`.
    """
    levels = build_levels(grid, structure, solvent, options.dielectric_smoothing)
    rhs = source_term(grid, structure, solvent) - levels[0].apply(boundary)
    rhs[0, :, :] = rhs[-1, :, :] = 0.0
    rhs[:, 0, :] = rhs[:, -1, :] = 0.0
    rhs[:, :, 0] = rhs[:, :, -1] = 0.0

    interior, report = solve_system(
        levels,
        rhs,
        tolerance=options.tolerance,
        max_cycles=options.max_cycles,
        smoothing_sweeps=options.smoothing_sweeps,
    )
    return interior + boundary, report


@dataclass
class DebyeSolver:
    """Linearized Poisson-Boltzmann on a Cartesian grid, in this process.

    Takes a `FiniteDifferenceRequest`, so a type checker refuses to hand it a
    boundary-element request; it is `Solver[FiniteDifferenceRequest]` exactly
    as APBS and DelPhi are, which is the protocol's claim that a clean-room
    solver needs no new vocabulary being cashed rather than asserted.
    """

    options: DebyeOptions = field(default_factory=DebyeOptions)

    @property
    def label(self) -> str:
        return f"debye-{BACKEND_VERSION}"

    def solve(self, request: FiniteDifferenceRequest) -> SolveResult:
        solvent = request.solvent
        check_surface(solvent.surface_model)
        check_equation(request.equation)

        started = time.perf_counter()
        grid = size_grid(request.structure, request.grid)

        # The reference state: solute dielectric everywhere, no mobile ions.
        # Built by replacing the solvent's properties rather than by a second
        # code path, so the two states cannot drift apart in anything but the
        # physics being varied. Named up here because both states' boundary
        # values come out of one pass over the distances — and only the states
        # actually being solved are asked for, so a `want_energy=False` request
        # still pays for one.
        reference_solvent = replace(
            solvent,
            solvent_dielectric=solvent.solute_dielectric,
            ionic_strength=0.0,
        )
        wanted = [(solvent, False)]
        if request.want_energy:
            wanted.append((reference_solvent, True))
        boundaries = debye_huckel_boundaries(grid, request.structure, wanted)

        solvated, solvated_report = _solve_state(
            grid, request.structure, solvent, self.options, boundary=boundaries[0]
        )

        energy: float | None = None
        reference_report: SolveReport | None = None
        if request.want_energy:
            uniform, reference_report = _solve_state(
                grid, request.structure, reference_solvent, self.options, boundary=boundaries[1]
            )
            energy = _polar_solvation_energy(
                grid, request.structure, solvated - uniform, solvent.temperature
            )

        wall_seconds = time.perf_counter() - started

        potential = None
        if request.want_potential:
            potential = PotentialGrid(
                values=solvated,
                origin=np.asarray(grid.origin, dtype=float),
                spacing=np.asarray(grid.spacing, dtype=float),
            )

        result = SolveResult(
            provenance=Provenance(
                backend=self.label,
                binary_path=None,
                binary_sha256=None,
                resolved_parameters=_resolved(grid, solvent, self.options),
                wall_seconds=round(wall_seconds, 4),
                energy_term=EnergyTerm.POLAR_SOLVATION,
                accuracy_tier=AccuracyTier.REFERENCE,
            ),
            energy_kj_mol=energy,
            potential=potential,
            diagnostics=_diagnostics(
                request.structure, grid, solvated_report, reference_report, solvent
            ),
        )
        result.check_satisfies(request)
        return result


def _polar_solvation_energy(
    grid: DebyeGrid, structure: PQRData, difference: FloatArray, temperature: float
) -> float:
    """1/2 sum q_i (phi_solvated - phi_reference)(r_i), in kJ/mol.

    The potential is read back with the same trilinear weights the charge was
    spread with, which is what makes the grid self-energy cancel between the two
    states rather than merely mostly cancel.
    """
    at_atoms = interpolate_at_atoms(grid, difference, structure)
    energy_kt = 0.5 * float(np.dot(structure.charges, at_atoms))
    return energy_kt * BOLTZMANN * temperature * AVOGADRO / JOULES_PER_KJ


def _resolved(grid: DebyeGrid, solvent: SolventModel, options: DebyeOptions) -> Diagnostics:
    return {
        "surface_model": solvent.surface_model.value,
        "solute_dielectric": solvent.solute_dielectric,
        "solvent_dielectric": solvent.solvent_dielectric,
        "ionic_strength": solvent.ionic_strength,
        "ion_radius": solvent.ion_radius,
        "temperature": solvent.temperature,
        "equation": "linear",
        "grid": grid.as_diagnostics(),
        "debye": {
            "tolerance": options.tolerance,
            # `max_cycles` is here because it decides whether the answer exists
            # at all: a case that only converges at a raised budget and one that
            # converges at the default produced byte-identical provenance until
            # a review asked. ROADMAP.md section 4 wants provenance to be enough
            # to reproduce the number.
            "max_cycles": options.max_cycles,
            "smoothing_sweeps": options.smoothing_sweeps,
            "boundary_condition": "multiple Debye-Huckel on the box face",
            "bjerrum_length_vacuum_a": round(bjerrum_length_a(solvent.temperature), 4),
        },
    }


def _diagnostics(
    structure: PQRData,
    grid: DebyeGrid,
    solvated: SolveReport,
    reference: SolveReport | None,
    solvent: SolventModel,
) -> Diagnostics:
    diagnostics: Diagnostics = {
        "family": "finite-difference",
        "n_atoms": structure.n_atoms,
        "grid_points": grid.n_points,
        "spacing_achieved_a": [round(s, 6) for s in grid.spacing],
        "solvated_solve": solvated.as_diagnostics(),
        "surface_model": solvent.surface_model.value,
    }
    if reference is not None:
        # Named for what it is rather than "reference": a reader comparing this
        # against APBS's two `elec` blocks should see the same two states.
        diagnostics["uniform_dielectric_solve"] = reference.as_diagnostics()
    return diagnostics
