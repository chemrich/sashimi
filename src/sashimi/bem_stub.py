"""A boundary-element backend that computes nothing.

This exists to answer one question: does the protocol admit a BEM solver
without APBS-shaped concessions? ROADMAP.md phase 4's exit criterion. It
returns a `SurfacePotential` on a sphere-tessellated surface with analytic
Debye-Huckel values — physically a toy, structurally the real thing.

It is not a solver and must never be registered as one. When TABI-PB or PyGBe
arrive (ROADMAP.md §8) they replace this file; if they need the protocol to
change to fit, the protocol was wrong and this stub failed to catch it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sashimi.constants import ANGSTROM, AVOGADRO, ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY
from sashimi.errors import UnsupportedRequest
from sashimi.protocol import (
    DIMENSIONS,
    BoundaryElementRequest,
    FloatArray,
    Provenance,
    SolventModel,
    SolveResult,
    SurfaceModel,
    SurfacePotential,
)

__all__ = ["StubBemSolver"]


def fibonacci_sphere(n: int) -> FloatArray:
    """Near-uniform points on a unit sphere. Stands in for a surface mesh."""
    indices = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    return np.asarray(
        np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]),
        dtype=np.float64,
    )


@dataclass
class StubBemSolver:
    """Implements `Solver[BoundaryElementRequest]`. Computes a toy field."""

    name: str = "stub-bem-0"

    def solve(self, request: BoundaryElementRequest) -> SolveResult:
        # A BEM solver has no volumetric field to give. Declining is the honest
        # answer, and the protocol has to allow it -- that is the whole test.
        if request.want_potential and request.solvent.surface_model is SurfaceModel.GAUSSIAN:
            raise UnsupportedRequest(
                "boundary-element methods need a sharp dielectric interface; "
                "a Gaussian dielectric has none."
            )

        structure = request.structure
        centre = structure.center()
        radius = float(structure.extent().max()) / 2 + request.solvent.surface_radius

        n_vertices = max(DIMENSIONS + 1, int(4 * math.pi * radius**2 * request.mesh_density))
        directions = fibonacci_sphere(n_vertices)
        vertices = centre + directions * radius

        charge = structure.total_charge
        values = np.full(n_vertices, _screened_potential(charge, radius, request.solvent))

        energy = _born_energy(charge, radius, request.solvent) if request.want_energy else None
        potential = (
            SurfacePotential(vertices=vertices, values=values, normals=directions)
            if request.want_potential
            else None
        )

        result = SolveResult(
            provenance=Provenance(
                backend=self.name,
                resolved_parameters={
                    "surface_model": request.solvent.surface_model.value,
                    "mesh_density": request.mesh_density,
                    "n_vertices": n_vertices,
                },
            ),
            energy_kj_mol=energy,
            potential=potential,
            diagnostics={"effective_radius_a": radius},
        )
        result.check_satisfies(request)
        return result


def _screened_potential(charge_e: float, radius_a: float, solvent: SolventModel) -> float:
    eps, temperature = solvent.solvent_dielectric, solvent.temperature
    q, r = charge_e * ELEMENTARY_CHARGE, radius_a * ANGSTROM
    volts = q / (4 * math.pi * VACUUM_PERMITTIVITY * eps * r) if r else 0.0
    kt_over_e = 1.380649e-23 * temperature / ELEMENTARY_CHARGE
    return volts / kt_over_e


def _born_energy(charge_e: float, radius_a: float, solvent: SolventModel) -> float:
    eps_s, eps_p = solvent.solvent_dielectric, solvent.solute_dielectric
    q, a = charge_e * ELEMENTARY_CHARGE, radius_a * ANGSTROM
    if a == 0:
        return 0.0
    joules = -(q**2 / (8 * math.pi * VACUUM_PERMITTIVITY * a)) * (1 / eps_p - 1 / eps_s)
    return joules * AVOGADRO / 1000.0
