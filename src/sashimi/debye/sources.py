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

from collections.abc import Sequence

import numpy as np

from sashimi.analytic import debye_length_a
from sashimi.debye.dielectric import bjerrum_length_a
from sashimi.debye.grid import DebyeGrid
from sashimi.errors import InputError
from sashimi.protocol import DIMENSIONS, FloatArray, PQRData, SolventModel

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
    """Dirichlet values on the box face for one state. See `debye_huckel_boundaries`."""
    return debye_huckel_boundaries(grid, structure, [(solvent, homogeneous)])[0]


def single_debye_huckel_solute(structure: PQRData) -> PQRData:
    """The solute as one sphere at its centroid carrying the net charge.

    **APBS's `bcfl sdh`, which is what its focusing buys it.** The multi-atom
    sum this replaces is `O(face nodes x atoms)` — 1.5 billion pairs on serum
    albumin, 36% of a solve, and the only superlinear stage debye has left. A
    single centre is `O(face nodes)`, independent of how many atoms there are.

    **This exists to be measured against, not to be switched on.** APBS can
    afford it because focusing puts its coarse face at 1.7x the *fine box*,
    where the `l >= 1` content a monopole omits has fallen off. debye's face
    sits at `r_max/|R| = 0.89-1.25` — inside the solute's own circumscribing
    sphere above ~2,000 atoms — and no single-centre expansion converges there
    at any order: measured on fas2 the reference-state error runs 18.2% ->
    5.2% -> 4.5% -> 2.2% through octupole, stalling rather than converging.

    So the number this produces is a *baseline*, and ROADMAP.md section 12's M9
    says why it has to exist: `sdh` on debye's existing box already passes the
    milestone's first exit criterion, which is a fact about that criterion. A
    replacement for the multi-atom sum has to beat this, not merely beat the
    thing it replaces.

    The radius is the circumscribing one — the furthest any atom's surface
    reaches from the centroid — so the pseudo-sphere contains the solute it
    stands for, and the Stern layer sits outside it exactly as it does for a
    real atom — `debye_huckel_boundaries` adds `ion_radius` to whatever radius it
    is handed, so the Stern layer needs no special case here.
    """
    centroid = structure.coords.mean(axis=0)
    reach = np.sqrt(((structure.coords - centroid) ** 2).sum(axis=1)) + structure.radii
    return PQRData(
        coords=centroid.reshape(1, DIMENSIONS),
        charges=np.array([structure.charges.sum()], dtype=np.float64),
        radii=np.array([float(reach.max())], dtype=np.float64),
    )


def debye_huckel_boundaries(
    grid: DebyeGrid,
    structure: PQRData,
    states: Sequence[tuple[SolventModel, bool]],
) -> list[FloatArray]:
    """Dirichlet values on the box face, kT/e, zero in the interior — for several
    states over one pass of distances.

    **Why this takes a list.** A solve with `want_energy` needs two of these:
    the solvated state and the uniform-dielectric reference. They differ only in
    a dielectric, a screening length and a prefactor — and they share the
    expensive half, which is the distance from every boundary node to every
    atom. On serum albumin that is 82,050 nodes against 18,242 atoms, **1.5
    billion pairs**, and computing it twice cost 6.4 s of a 45 s solve for
    nothing. Measured separately: 11.82 s for the screened state and 6.37 s for
    the reference, where the reference has no `exp` at all.

    That is the same waste M4 found in `dielectric_faces`, where the reference
    state built a surface and threw it away. Here the arithmetic per state is
    untouched, so the fields come back bit-identical; only the sharing is new.

    Every atom contributes a screened Coulomb tail, evaluated with its own
    ion-exclusion radius:

        phi(r) = sum_i (l_B q_i / eps) * exp(-kappa (d_i - b_i)) / (1 + kappa b_i) / d_i

    With `homogeneous=True` this describes the reference state the solvation
    energy is measured against — solute dielectric everywhere, no mobile ions —
    so it reduces to a plain Coulomb sum. Using the *same* box and the same
    expression for both states is what makes their difference a solvation
    energy rather than two unrelated numbers subtracted.
    """
    recipes = []
    for solvent, homogeneous in states:
        eps = solvent.solute_dielectric if homogeneous else solvent.solvent_dielectric
        kappa = (
            0.0
            if homogeneous or solvent.ionic_strength <= 0.0
            else 1.0
            / debye_length_a(
                solvent.ionic_strength, solvent.solvent_dielectric, solvent.temperature
            )
        )
        exclusion = structure.radii + solvent.ion_radius
        recipes.append(
            (
                bjerrum_length_a(solvent.temperature) / eps,
                kappa,
                exclusion,
                structure.charges / (1.0 + kappa * exclusion) if kappa else structure.charges,
            )
        )

    mask = boundary_mask(grid.shape)
    indices = np.argwhere(mask)
    points = np.asarray(grid.origin) + indices * np.asarray(grid.spacing)

    # The Debye-Huckel tail is a point-charge expression, so it is meaningful
    # only where the box face is clear of the charges — which is what `padding`
    # is for. Refusing is not defensive coding here: this replaced a
    # `np.maximum(distances, 1e-6)` clip whose comment claimed the case could
    # not arise and that something downstream would describe it. Both were
    # wrong. A zero-radius atom at the low corner of its own bounding box with
    # `padding=0` sits exactly *on* a boundary node, is legally inside the grid
    # so `trilinear_weights` does not object, and the clip turned the
    # singularity into 7.1e6 kT/e on that node and solved on. A confident wrong
    # number, from a guard that was documented as unreachable.
    closest = float("inf")
    values = [np.zeros(len(points), dtype=np.float64) for _ in recipes]
    # The bound is in *pairs*, so it holds whatever the atom count is — and it
    # bounds one distance block, which is now shared rather than rebuilt per
    # state. Each state's `contribution` is built and released inside the state
    # loop, so the peak is what it was with one state.
    chunk = max(1, _PAIR_CHUNK // max(1, structure.n_atoms))
    for start in range(0, len(points), chunk):
        block = points[start : start + chunk]
        distances = np.linalg.norm(block[:, None, :] - structure.coords[None, :, :], axis=2)
        closest = min(closest, float(distances.min()))
        if closest < min(grid.spacing):
            raise InputError(
                f"an atom lies {closest:.4g} A from the edge of the box, closer than the "
                f"grid spacing {min(grid.spacing):.4g} A. The boundary values are a "
                "Debye-Huckel tail summed over the atoms, which is only meaningful where "
                "the box face is clear of them; at this distance it is the expression's "
                "own singularity, not a potential. Increase GridSpec.padding — it is the "
                "whole boundary condition for this solver."
            )
        for (prefactor, kappa, exclusion, screening), out in zip(recipes, values, strict=True):
            contribution = screening[None, :] / distances
            if kappa:
                contribution *= np.exp(-kappa * (distances - exclusion[None, :]))
            out[start : start + chunk] = prefactor * contribution.sum(axis=1)

    fields = []
    for out in values:
        field = np.zeros(grid.shape, dtype=np.float64)
        field[tuple(indices.T)] = out
        fields.append(field)
    return fields


def source_term(grid: DebyeGrid, structure: PQRData, solvent: SolventModel) -> FloatArray:
    """The right-hand side 4 pi l_B q, in the units the operator is written in.

    Not scaled by cell volume: `assign_charges` returns the charge *at* a node,
    which is already the integral of the density over that node's control
    volume, so the finite-volume right-hand side needs no further factor. Get
    this wrong by h^3 and the Born energy is off by orders of magnitude in a way
    that looks like a units bug and is a discretization one.
    """
    return 4.0 * np.pi * bjerrum_length_a(solvent.temperature) * assign_charges(grid, structure)
