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

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sashimi.analytic import debye_length_a
from sashimi.debye.dielectric import bjerrum_length_a
from sashimi.debye.grid import DebyeGrid
from sashimi.errors import InputError
from sashimi.protocol import DIMENSIONS, FloatArray, PQRData, SolventModel

__all__ = [
    "DEFAULT_BOUNDARY_PITCH_A",
    "EXACT_FACE_PAIRS",
    "PITCH_CLEARANCE_FRACTION",
    "FaceSampling",
    "assign_charges",
    "boundary_mask",
    "debye_huckel_boundary",
    "interpolate_at_atoms",
    "plan_face_sampling",
    "solute_clearance",
    "source_term",
    "trilinear_weights",
]

# Boundary points are evaluated against every atom, so the intermediate is
# points x atoms. Chunked at a few million entries, which is tens of megabytes
# and keeps a 129^3 box over a 2,000-atom protein from allocating gigabytes.
_PAIR_CHUNK = 4_000_000

# How far apart, in angstroms, the sampled face nodes are. **Stated as a
# distance and not as a node stride**, which is not a cosmetic choice: a stride
# has to divide `n - 1 = 8m`, so a nominal "every 16th node" is really per-axis
# pitches of 15.41 / 7.79 / 12.89 A on serum albumin — three different
# resolutions on the three axes of one face. Measured at matched cost the
# distance-pitched sampler is 1.16x better on the screened face and 1.19x on the
# reference face than the stride that first replaced the exact sum.
#
# **Six and not twelve, and the difference is a gate failure.** Twelve was picked
# from a matched-cost comparison on serum albumin at `padding = 10`. Swept across
# padding it fails: at `padding = 3` a 12 A pitch lands fas2 at r = 0.998798 and
# +0.7536% energy and 1a63 at 0.999400 and +0.6325%, both outside M9's gate,
# where 6 A passes every padding tested on both. It costs nothing to be right —
# the boundary is 1.2-2.5% of a solve at 6 A against 0.3-0.65% at 12 A, so the
# whole saving on offer above 6 A is under two percent of a solve.
DEFAULT_BOUNDARY_PITCH_A = 6.0

# The pitch is also capped at this fraction of the solute's clearance from the
# face. **A fixed distance-pitch scales the wrong way**: shrink the box and the
# face moves closer to the solute, so the field on it varies faster *and* a fixed
# pitch buys fewer samples. Padding is a caller's knob — `protocol.py` validates
# only `padding >= 0` — so without this cap the default is safe only in the range
# it was swept over, which is the shape of defect ROADMAP.md section 7 keeps
# finding. 0.6 reproduces the measured-good 6 A at the default `padding = 10` and
# tightens to 1.8 A at `padding = 3`.
PITCH_CLEARANCE_FRACTION = 0.6

# Below this many (face node x atom) pairs the exact face already costs under a
# millisecond, and striding it would move a recorded answer for no gain. This is
# the line between a molecule and a protein: `ala-gly` is 23,000 pairs and fas2
# is 13.6 million. Every closed-form and small-molecule recording therefore keeps
# the scheme it was recorded under, and only protein-scale cases move — which is
# what makes this change re-recordable in one pass instead of all 58 at once.
EXACT_FACE_PAIRS = 1_000_000


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
    pitch_a: float = DEFAULT_BOUNDARY_PITCH_A,
) -> FloatArray:
    """Dirichlet values on the box face for one state. See `debye_huckel_boundaries`."""
    return debye_huckel_boundaries(grid, structure, [(solvent, homogeneous)], pitch_a=pitch_a)[0]


def single_debye_huckel_solute(structure: PQRData) -> PQRData:
    """The solute as one sphere at its centroid carrying the net charge.

    **APBS's `bcfl sdh`, which is what its focusing buys it.** The multi-atom
    sum this replaces is `O(face nodes x atoms)` — 1.5 billion pairs on serum
    albumin, 43% of a solve, and the only superlinear stage debye has left. A
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


@dataclass(frozen=True)
class FaceSampling:
    """Which face nodes the Debye-Huckel sum is actually evaluated at.

    Per-axis index sets, **shared by all six faces**, which is what makes the
    scheme consistent on the edges where two faces meet. A node on the edge
    between the `x = 0` and `y = 0` faces is reached from either face by
    interpolating along `z` from the same sampled points with the same weights,
    so both faces write the same bits to it. Choose the index sets per face
    instead and that node gets two different values depending on write order.
    """

    indices: tuple[np.ndarray, np.ndarray, np.ndarray]
    exact: bool
    pitch_a: float

    @property
    def label(self) -> str:
        """What `_resolved` records, so `content_address` can tell the two apart."""
        if self.exact:
            return "multiple Debye-Huckel on the box face"
        return f"multiple Debye-Huckel on a {self.pitch_a:g} A strided box face"

    @property
    def n_samples(self) -> int:
        nx, ny, nz = (len(i) for i in self.indices)
        return 2 * (ny * nz + nx * nz + nx * ny)


def _axis_samples(n: int, spacing: float, pitch: float) -> np.ndarray:
    """Node indices along one axis, both endpoints included, ~`pitch` apart.

    Spread by `linspace` and rounded rather than by a fixed step, because a
    fixed step leaves a short last interval where it meets the far endpoint and
    that alone cost 1.08-1.14x in face error when it was measured.
    """
    count = math.ceil((n - 1) * spacing / pitch) + 1
    if count >= n:
        return np.arange(n, dtype=np.intp)
    return np.unique(np.rint(np.linspace(0.0, n - 1, max(2, count))).astype(np.intp))


def plan_face_sampling(
    grid: DebyeGrid, structure: PQRData, pitch_a: float = DEFAULT_BOUNDARY_PITCH_A
) -> FaceSampling:
    """Decide the face sampling for a request, once, for everyone who needs it.

    Called by `debye_huckel_boundaries` to do the work and by the backend's
    `_resolved` to name it, so provenance cannot describe a scheme other than
    the one that ran.
    """
    exact = tuple(np.arange(n, dtype=np.intp) for n in grid.shape)
    nx, ny, nz = grid.shape
    # The closed form rather than `boundary_mask(...).sum()`, which would
    # allocate a 1.6 M-entry bool array on albumin to count its own surface.
    faces = nx * ny * nz - (nx - 2) * (ny - 2) * (nz - 2)
    if pitch_a <= 0.0 or faces * structure.n_atoms <= EXACT_FACE_PAIRS:
        return FaceSampling(indices=exact, exact=True, pitch_a=pitch_a)  # type: ignore[arg-type]

    pitch = min(pitch_a, PITCH_CLEARANCE_FRACTION * solute_clearance(grid, structure))
    indices = tuple(
        _axis_samples(n, h, pitch) for n, h in zip(grid.shape, grid.spacing, strict=True)
    )
    if all(len(i) == n for i, n in zip(indices, grid.shape, strict=True)):
        return FaceSampling(indices=exact, exact=True, pitch_a=pitch)  # type: ignore[arg-type]
    # The *effective* pitch, not the requested one, because this is what
    # `label` reports into provenance: a run capped from 24 A to 1.8 A by a tight
    # box must not record itself as having sampled every 24 A.
    return FaceSampling(indices=indices, exact=False, pitch_a=pitch)  # type: ignore[arg-type]


def _sampled_nodes(shape: tuple[int, int, int], sampling: FaceSampling) -> np.ndarray:
    """(S, 3) indices of the sampled boundary nodes, deduplicated and sorted.

    Sorted lexicographically by `np.unique`, which is the same order as the flat
    key `(i * ny + j) * nz + k`, so `searchsorted` on that key is a valid lookup.
    """
    blocks = []
    for axis in range(DIMENSIONS):
        b, c = (k for k in range(DIMENSIONS) if k != axis)
        gb, gc = np.meshgrid(sampling.indices[b], sampling.indices[c], indexing="ij")
        for position in (0, shape[axis] - 1):
            block = np.empty((gb.size, DIMENSIONS), dtype=np.intp)
            block[:, axis] = position
            block[:, b] = gb.ravel()
            block[:, c] = gc.ravel()
            blocks.append(block)
    return np.unique(np.concatenate(blocks), axis=0)


def _lift(n: int, sample: np.ndarray) -> FloatArray | None:
    """(n x m) matrix carrying values at `sample` to every node, linearly.

    `None` when the samples *are* every node, which the caller takes as an
    identity — not as an optimisation but as a guarantee: an identity matrix
    multiply would be bit-identical anyway, and returning `None` says so without
    asking the reader to convince themselves of it.
    """
    if len(sample) == n:
        return None
    ordinal = np.interp(
        np.arange(n, dtype=np.float64),
        sample.astype(np.float64),
        np.arange(len(sample), dtype=np.float64),
    )
    lower = np.minimum(np.floor(ordinal).astype(np.intp), len(sample) - 2)
    frac = ordinal - lower
    weights = np.zeros((n, len(sample)), dtype=np.float64)
    rows = np.arange(n, dtype=np.intp)
    weights[rows, lower] = 1.0 - frac
    weights[rows, lower + 1] = frac
    return weights


def solute_clearance(grid: DebyeGrid, structure: PQRData) -> float:
    """How far the nearest atom centre sits from the box surface, in angstroms.

    One quantity serving two purposes, which is why it is named rather than
    inlined twice: it is what the refusal below tests, and it is what the face
    pitch is capped against. Both are asking the same question — how close does
    the boundary expression get to the charges it is a far-field approximation
    of — and answering it in two places is how they would drift apart.
    """
    low = np.asarray(grid.origin, dtype=np.float64)
    high = low + (np.asarray(grid.shape, dtype=np.float64) - 1.0) * np.asarray(grid.spacing)
    reach = np.minimum(structure.coords - low, high - structure.coords)
    return float(reach.min()) if reach.size else float("inf")


def _refuse_atoms_near_the_face(grid: DebyeGrid, structure: PQRData) -> None:
    """Refuse a solute whose atoms reach the box face, in `O(atoms)`.

    **This check used to be a side effect of the expensive thing.** It lived
    inside the distance block below as a running minimum over every face node
    against every atom — which is only available while that `O(nodes x atoms)`
    matrix is being built. Sample the face instead and the check evaporates: a
    strided face visits a few hundred of 82,050 nodes and could only catch the
    bad case by luck.

    **And what it protects against is not what its predecessor documented.** The
    comment it replaces described a `np.maximum(distances, 1e-6)` clip turning
    the expression's singularity into 7.1e6 kT/e on one node. Under any
    replacement boundary no singularity returns, because the face value is
    interpolated. What returns is quieter and worse: `_solve_state` zeroes the
    right-hand side on all six faces, so an atom sitting *on* a face has its
    charge assigned entirely to Dirichlet nodes and then discarded — **20.00% of
    a structure's total charge silently missing from the source term**, falling
    as `1 - d/h` to nothing at one full cell. `trilinear_weights` objects only
    when an atom is strictly outside the box, so nothing else sees it.

    Distance is measured from the atom centre to the box surface, which is the
    same quantity the old check minimised over and is a lower bound on the
    distance to the nearest face *node*. Conservative in the safe direction, and
    it costs one pass over the atoms.
    """
    closest = solute_clearance(grid, structure)
    if closest < min(grid.spacing):
        raise InputError(
            f"an atom lies {closest:.4g} A from the edge of the box, closer than the "
            f"grid spacing {min(grid.spacing):.4g} A. The boundary values are a "
            "Debye-Huckel tail summed over the atoms, which is only meaningful where "
            "the box face is clear of them; at this distance it is the expression's "
            "own singularity, not a potential — and an atom this close also loses "
            "charge to the Dirichlet face, which nothing downstream would report. "
            "Increase GridSpec.padding — it is the whole boundary condition for this "
            "solver."
        )


def debye_huckel_boundaries(
    grid: DebyeGrid,
    structure: PQRData,
    states: Sequence[tuple[SolventModel, bool]],
    *,
    pitch_a: float = DEFAULT_BOUNDARY_PITCH_A,
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

    **The sum is evaluated on a strided face and interpolated up.** Every atom at
    every face node is `O(nodes x atoms)` — the only superlinear stage debye has,
    43% of a serum albumin solve and scaling as atoms^1.45 where everything else
    is near-linear. The field it produces is smooth: it is a sum of `1/r` tails
    seen from at least `padding` away, so it has no feature the face lattice can
    resolve and sampling it every ~12 A loses nothing. Measured on albumin the
    boundary goes **12.94 s -> 0.24 s** and the near field comes back at
    r = 1.000000, sign 99.996%, energy +0.068% against the exact sum — better
    than a coarse pre-solve, which ROADMAP.md section 12 priced at 2.0-2.5 s for
    the same accuracy and a second grid hierarchy.

    `pitch_a <= 0` and any case under `EXACT_FACE_PAIRS` evaluate every node, and
    that path is **bit-identical to the exact sum** rather than merely close: the
    per-point arithmetic below is untouched and the interpolation is skipped, not
    applied as an identity.
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

    # The Debye-Huckel tail is a point-charge expression, so it is meaningful
    # only where the box face is clear of the charges — which is what `padding`
    # is for. See `_refuse_atoms_near_the_face` for why this is now an explicit
    # O(atoms) pass rather than a minimum taken over the distance block below.
    _refuse_atoms_near_the_face(grid, structure)

    sampling = plan_face_sampling(grid, structure, pitch_a)
    # **The exact path is the pre-M9 expression, verbatim, and that is not
    # fussiness.** `np.argwhere` returns a transposed view with strides
    # (8, 180240); `_sampled_nodes` returns a C-contiguous (24, 8). The indices
    # are equal element for element and in the same order — but `points`
    # inherits the layout, and numpy's pairwise summation blocks by memory
    # layout, so `norm(..., axis=2)` and `sum(axis=1)` accumulate in a different
    # order and every face node lands one ULP away. Measured on
    # `peptide-molecular` that moved the recorded energy from
    # -218.62772042354118 to -218.62772042354123 on 13,043 of 22,530 nodes.
    #
    # A tolerance-based corpus cannot see that, and neither can a test that
    # compares this function against itself — which is how it survived a green
    # suite. `tests/test_debye_m9.py` now anchors on the literal digits.
    indices = (
        np.argwhere(boundary_mask(grid.shape))
        if sampling.exact
        else _sampled_nodes(grid.shape, sampling)
    )
    points = np.asarray(grid.origin) + indices * np.asarray(grid.spacing)

    values = [np.zeros(len(points), dtype=np.float64) for _ in recipes]
    # The bound is in *pairs*, so it holds whatever the atom count is — and it
    # bounds one distance block, which is now shared rather than rebuilt per
    # state. Each state's `contribution` is built and released inside the state
    # loop, so the peak is what it was with one state.
    chunk = max(1, _PAIR_CHUNK // max(1, structure.n_atoms))
    for start in range(0, len(points), chunk):
        block = points[start : start + chunk]
        distances = np.linalg.norm(block[:, None, :] - structure.coords[None, :, :], axis=2)
        for (prefactor, kappa, exclusion, screening), out in zip(recipes, values, strict=True):
            contribution = screening[None, :] / distances
            if kappa:
                contribution *= np.exp(-kappa * (distances - exclusion[None, :]))
            out[start : start + chunk] = prefactor * contribution.sum(axis=1)

    if sampling.exact:
        fields = []
        for out in values:
            field = np.zeros(grid.shape, dtype=np.float64)
            field[tuple(indices.T)] = out
            fields.append(field)
        return fields

    # Lexicographic order is what `np.unique` returned above, and it is the order
    # this key sorts in, so `searchsorted` finds a sampled node's row.
    shape = grid.shape
    keys = (indices[:, 0] * shape[1] + indices[:, 1]) * shape[2] + indices[:, 2]
    lifts = [_lift(n, s) for n, s in zip(shape, sampling.indices, strict=True)]

    fields = [np.zeros(shape, dtype=np.float64) for _ in recipes]
    for axis in range(DIMENSIONS):
        b, c = (k for k in range(DIMENSIONS) if k != axis)
        lift_b, lift_c = lifts[b], lifts[c]
        gb, gc = np.meshgrid(sampling.indices[b], sampling.indices[c], indexing="ij")
        for position in (0, shape[axis] - 1):
            face: list[np.ndarray] = [np.empty(0, dtype=np.intp)] * DIMENSIONS
            face[axis] = np.full(gb.size, position, dtype=np.intp)
            face[b] = gb.ravel()
            face[c] = gc.ravel()
            rows = np.searchsorted(keys, (face[0] * shape[1] + face[1]) * shape[2] + face[2])
            where = tuple(position if k == axis else slice(None) for k in range(DIMENSIONS))
            for out, field in zip(values, fields, strict=True):
                sampled = out[rows].reshape(gb.shape)
                lifted = sampled if lift_b is None else lift_b @ sampled
                lifted = lifted if lift_c is None else lifted @ lift_c.T
                field[where] = lifted
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
