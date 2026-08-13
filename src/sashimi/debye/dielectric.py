"""Geometry to coefficients: where the solvent is, and how strongly it screens.

Two maps, and they are not the same map. The **dielectric** boundary is the van
der Waals surface — the union of the atomic spheres — and it lives on the faces
between nodes, because a flux through a face is what the finite-volume operator
integrates. The **ion-accessible** region is further out: mobile ions have a
radius of their own and cannot approach the solute closer than that, so the
Boltzmann term switches on at the union of spheres inflated by `ion_radius`.
Both incumbents make this distinction and it is invisible at zero salt, which
is exactly the configuration M1 is graded on — so it is written now rather than
discovered at M3.

The dielectric is sampled at face centres rather than averaged over the face.
That is the same first-order choice APBS makes with `srfm mol`, and it is worth
naming as a choice: a volume-fraction average would put the boundary error at
second order and is the obvious place to look if M1's 1% turns out to be out of
reach. It is not free — the fraction of a face lying inside a union of spheres
has no closed form — so it is not paid for before the measurement says it is
needed.
"""

from __future__ import annotations

import math

import numpy as np

from sashimi.analytic import debye_length_a
from sashimi.constants import (
    ANGSTROM,
    BOLTZMANN,
    ELEMENTARY_CHARGE,
    VACUUM_PERMITTIVITY,
)
from sashimi.debye.grid import DebyeGrid, axis_coordinates
from sashimi.protocol import DIMENSIONS, FloatArray, PQRData, SolventModel

__all__ = [
    "bjerrum_length_a",
    "dielectric_faces",
    "inside_union_of_spheres",
    "screening_nodes",
]


def bjerrum_length_a(temperature: float) -> float:
    """e^2 / (4 pi eps0 kT), in angstroms: the vacuum Bjerrum length.

    The one constant that converts this module's charges into this module's
    potentials. In water at 298.15 K the familiar number is 7.14 A, which is
    this divided by 78.54 — the dielectric is not folded in here because the
    solver carries it in the operator, where it varies with position.

    Built from `sashimi.constants` rather than quoted, for the reason that
    module exists: the Born closed form is computed from the same CODATA 2018
    values, so a solver that agrees with it to six digits is agreeing about the
    physics rather than about a rounding.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    metres = ELEMENTARY_CHARGE**2 / (4.0 * math.pi * VACUUM_PERMITTIVITY * BOLTZMANN * temperature)
    return metres / ANGSTROM


def inside_union_of_spheres(
    axes: list[FloatArray],
    coords: FloatArray,
    radii: FloatArray,
) -> np.ndarray:
    """Boolean mask over the lattice spanned by `axes`: inside any sphere.

    Marked atom by atom over each sphere's own index window rather than by
    evaluating every atom against every point. The difference is not a
    micro-optimisation: the whole-grid version took 64 s on a 1,960-atom
    protein in `sashimi.analysis` before it was fixed (ROADMAP.md section 7),
    because its cost is atoms x points where this one is the volume the
    spheres actually occupy.
    """
    shape = tuple(len(axis) for axis in axes)
    mask = np.zeros(shape, dtype=bool)
    for center, radius in zip(coords, radii, strict=True):
        if radius <= 0.0:
            continue  # a zero-radius atom bounds no volume; Kirkwood's has one
        window = []
        for axis in range(DIMENSIONS):
            lo = int(np.searchsorted(axes[axis], center[axis] - radius, side="left"))
            hi = int(np.searchsorted(axes[axis], center[axis] + radius, side="right"))
            window.append(slice(lo, hi))
        if any(w.start >= w.stop for w in window):
            continue  # the sphere falls between nodes, or outside the box
        offsets = [(axes[axis][window[axis]] - center[axis]) ** 2 for axis in range(DIMENSIONS)]
        squared = offsets[0][:, None, None] + offsets[1][None, :, None] + offsets[2][None, None, :]
        mask[tuple(window)] |= squared <= radius * radius
    return mask


def dielectric_faces(
    grid: DebyeGrid, structure: PQRData, solvent: SolventModel
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Dielectric at the face centres, one array per axis.

    `faces[axis]` has the grid's shape with `axis` one shorter: entry (i, j, k)
    of `faces[0]` is the dielectric halfway between nodes (i, j, k) and
    (i+1, j, k), which is the coefficient of the flux the operator sums there.
    """
    faces = []
    for axis in range(DIMENSIONS):
        axes = axis_coordinates(grid, staggered=axis)
        inside = inside_union_of_spheres(axes, structure.coords, structure.radii)
        eps = np.where(inside, solvent.solute_dielectric, solvent.solvent_dielectric)
        faces.append(np.ascontiguousarray(eps, dtype=np.float64))
    return faces[0], faces[1], faces[2]


def screening_nodes(
    grid: DebyeGrid, structure: PQRData, solvent: SolventModel
) -> tuple[FloatArray, float]:
    """The Boltzmann term's coefficient at each node, and the bulk value it takes.

    Returns `eps_s * kappa^2` in 1/A^2 — zero inside the ion-exclusion region,
    bulk outside it. Zero everywhere at zero ionic strength, which is every case
    M1 is graded on; the array is still built, because a solver that only works
    at zero salt is not a Poisson-Boltzmann solver and would not say so.

    The exclusion radius is the atomic radius plus `ion_radius`, not plus the
    solvent probe: the ion is the thing being excluded. `sashimi.analytic`'s
    screened Born expression evaluates its screening term at `a + ion_radius`
    for the same reason, so the two agree about what the Stern layer is.
    """
    if solvent.ionic_strength <= 0.0:
        return np.zeros(grid.shape, dtype=np.float64), 0.0

    kappa = 1.0 / debye_length_a(
        solvent.ionic_strength, solvent.solvent_dielectric, solvent.temperature
    )
    bulk = solvent.solvent_dielectric * kappa * kappa  # 1/A^2

    axes = axis_coordinates(grid)
    excluded = inside_union_of_spheres(axes, structure.coords, structure.radii + solvent.ion_radius)
    return np.where(excluded, 0.0, bulk), bulk
