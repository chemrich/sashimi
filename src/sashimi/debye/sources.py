"""Where the charge goes onto the grid, and what the box edge is held at.

The two ends of the discretization that are not the operator. Both are more
consequential than they look:

**Charge assignment and interpolation are the same weights, deliberately.** A
point charge on a grid has an infinite self-energy, and what a finite-difference
solver actually computes is a large grid-dependent self-term plus the small
solvation term wanted. The self-term cancels *exactly* between the solvated and
the homogeneous-reference solve only if the charge is spread and the potential
is read back with the same operator — assignment and interpolation adjoint to
each other. Read the potential back with a different scheme, say a higher-order
interpolation, and the cancellation leaves a residue that scales like 1/h: the
Born ion would then get *worse* under refinement, which is the failure mode M1's
"converging monotonically" is there to catch.

**The boundary condition is the whole reason `padding` exists.** debye solves
the box directly, with no coarse grid and no focusing, so the Dirichlet values
on the box face are the only place the outside world enters. They come from the
Debye-Huckel expression summed over every atom — APBS's `bcfl mdh` — which is
exact for a lone ion at any distance and asymptotically right for a molecule.
"""

from __future__ import annotations

import numpy as np

from sashimi.analytic import debye_length_a
from sashimi.debye.dielectric import bjerrum_length_a
from sashimi.debye.grid import DebyeGrid
from sashimi.protocol import FloatArray, PQRData, SolventModel

__all__ = [
    "assign_charges",
    "boundary_mask",
    "debye_huckel_boundary",
    "interpolate_at_atoms",
    "source_term",
    "trilinear_weights",
]

# Boundary points are evaluated against every atom, so the intermediate is
# points x atoms. Chunked at a few million entries, which is tens of megabytes
# and keeps a 129^3 box over a 2,000-atom protein from allocating gigabytes.
_PAIR_CHUNK = 4_000_000


def trilinear_weights(grid: DebyeGrid, coords: FloatArray) -> tuple[np.ndarray, FloatArray]:
    """The eight surrounding nodes of each point, and their weights.

    Returns base indices (N, 3) and weights (N, 2, 2, 2). A point outside the
    grid is an error rather than a clamp: every caller here is placing an atom
    inside a box built to contain it with `padding` to spare, so a point outside
    means the grid and the structure have come apart, and a clamped charge would
    show up as a plausible wrong energy.
    """
    origin = np.asarray(grid.origin)
    spacing = np.asarray(grid.spacing)
    fractional = (coords - origin) / spacing
    base = np.floor(fractional).astype(int)
    shape = np.asarray(grid.shape)
    if np.any(base < 0) or np.any(base > shape - 2):
        outside = int(np.sum(np.any((base < 0) | (base > shape - 2), axis=1)))
        raise ValueError(
            f"{outside} atom(s) fall outside the grid built for them; the box is "
            f"{grid.shape} nodes from {grid.origin} at spacing {grid.spacing}"
        )

    t = fractional - base
    weights = np.empty((len(coords), 2, 2, 2), dtype=np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                weights[:, dx, dy, dz] = (
                    (t[:, 0] if dx else 1.0 - t[:, 0])
                    * (t[:, 1] if dy else 1.0 - t[:, 1])
                    * (t[:, 2] if dz else 1.0 - t[:, 2])
                )
    return base, weights


def assign_charges(grid: DebyeGrid, structure: PQRData) -> FloatArray:
    """Charge per node, in e. Cloud-in-cell: each charge onto its eight corners.

    Conserves total charge to floating-point exactly, which
    `tests/test_debye_discretization.py` asserts — a charge that leaks is a
    solver that quietly solves for a different molecule.
    """
    base, weights = trilinear_weights(grid, structure.coords)
    rho = np.zeros(grid.shape, dtype=np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                np.add.at(
                    rho,
                    (base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz),
                    structure.charges * weights[:, dx, dy, dz],
                )
    return rho


def interpolate_at_atoms(grid: DebyeGrid, values: FloatArray, structure: PQRData) -> FloatArray:
    """Read a nodal field back at the atom centres, (N,) — adjoint of `assign_charges`."""
    base, weights = trilinear_weights(grid, structure.coords)
    out = np.zeros(len(structure.charges), dtype=np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                out += (
                    weights[:, dx, dy, dz]
                    * values[base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz]
                )
    return out


def boundary_mask(shape: tuple[int, int, int]) -> np.ndarray:
    """True on the six faces of the box: the nodes Dirichlet data fixes."""
    mask = np.zeros(shape, dtype=bool)
    mask[0, :, :] = mask[-1, :, :] = True
    mask[:, 0, :] = mask[:, -1, :] = True
    mask[:, :, 0] = mask[:, :, -1] = True
    return mask


def debye_huckel_boundary(
    grid: DebyeGrid,
    structure: PQRData,
    solvent: SolventModel,
    *,
    homogeneous: bool = False,
) -> FloatArray:
    """Dirichlet values on the box face, kT/e, zero in the interior.

    Every atom contributes a screened Coulomb tail, evaluated with its own
    ion-exclusion radius:

        phi(r) = sum_i (l_B q_i / eps) * exp(-kappa (d_i - b_i)) / (1 + kappa b_i) / d_i

    With `homogeneous=True` this describes the reference state the solvation
    energy is measured against — solute dielectric everywhere, no mobile ions —
    so it reduces to a plain Coulomb sum. Using the *same* box and the same
    expression for both states is what makes their difference a solvation
    energy rather than two unrelated numbers subtracted.
    """
    eps = solvent.solute_dielectric if homogeneous else solvent.solvent_dielectric
    kappa = (
        0.0
        if homogeneous or solvent.ionic_strength <= 0.0
        else 1.0
        / debye_length_a(solvent.ionic_strength, solvent.solvent_dielectric, solvent.temperature)
    )
    prefactor = bjerrum_length_a(solvent.temperature) / eps

    mask = boundary_mask(grid.shape)
    indices = np.argwhere(mask)
    points = np.asarray(grid.origin) + indices * np.asarray(grid.spacing)

    exclusion = structure.radii + solvent.ion_radius
    screening = structure.charges / (1.0 + kappa * exclusion) if kappa else structure.charges

    values = np.zeros(len(points), dtype=np.float64)
    chunk = max(1, _PAIR_CHUNK // max(1, structure.n_atoms))
    for start in range(0, len(points), chunk):
        block = points[start : start + chunk]
        distances = np.linalg.norm(block[:, None, :] - structure.coords[None, :, :], axis=2)
        # A boundary node inside an atom would divide by nothing meaningful.
        # The box is the molecule plus `padding` on every side, so this cannot
        # happen for a structure that fits its own grid; clipping rather than
        # asserting keeps a pathological input from raising here instead of
        # where it can be described.
        np.maximum(distances, 1e-6, out=distances)
        contribution = screening[None, :] / distances
        if kappa:
            contribution *= np.exp(-kappa * (distances - exclusion[None, :]))
        values[start : start + chunk] = prefactor * contribution.sum(axis=1)

    field = np.zeros(grid.shape, dtype=np.float64)
    field[tuple(indices.T)] = values
    return field


def source_term(grid: DebyeGrid, structure: PQRData, solvent: SolventModel) -> FloatArray:
    """The right-hand side 4 pi l_B q, in the units the operator is written in.

    Not scaled by cell volume: `assign_charges` returns the charge *at* a node,
    which is already the integral of the density over that node's control
    volume, so the finite-volume right-hand side needs no further factor. Get
    this wrong by h^3 and the Born energy is off by orders of magnitude in a way
    that looks like a units bug and is a discretization one.
    """
    return 4.0 * np.pi * bjerrum_length_a(solvent.temperature) * assign_charges(grid, structure)
