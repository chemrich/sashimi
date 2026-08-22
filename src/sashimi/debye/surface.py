"""Which points are solute: the two boundaries, behind one oracle.

M1 through M3 needed only a union of spheres, and `dielectric.py` asked for it
directly. M4 adds the solvent-excluded surface, which is not a union of anything
— so the question "is this point inside the solute?" becomes a thing with two
implementations, and this module is where the choice lives. ROADMAP.md section
12 named that seam a milestone early, under M1c: *write against an
inside/distance oracle rather than against spheres, so M4 swaps the oracle.*

**What the solvent-excluded surface is, stated as the construction rather than
as the picture.** Roll a probe of radius `p` over the van der Waals spheres; the
solute is everything the probe cannot touch. A probe centre is legal exactly
where it does not overlap an atom, which is outside the union of spheres
inflated by `p` — call that set A. A point is *solvent* iff some legal probe
covers it, so

    solvent = A dilated by p          solute = everything else

and that is the whole definition. Two consequences worth having in front of you
before reading the code:

- **For a lone convex sphere the two boundaries coincide exactly.** A is
  `|x - c| > r + p`, dilating it by `p` gives `|x - c| > r`, so the solute is the
  sphere itself — no probe, no rolling, no discretization of a re-entrant patch.
  The corpus already asserts this from the other side: `born-ion-molecular` and
  `born-ion-vdw` record -233.9996297277 to the last digit. It falls out of the
  construction here rather than being special-cased, which is the only way to be
  sure it is not being special-cased.
- **A point inside a van der Waals sphere is always solute**, and provably, so
  the construction never has to test it: if `|x - c_i| <= r_i` and `a` is legal
  then `|x - a| >= |a - c_i| - |x - c_i| > (r_i + p) - r_i = p`. That is what
  makes this affordable — the only undecided points are the shell between the
  van der Waals surface and the inflated one, which is `p` thick.

**The probe centres must not live on the lattice, and that is the whole
difficulty of this module.** The obvious implementation dilates the set of
*grid nodes* that are legal probe centres. It is wrong in a way that looks
small and is not: a node just outside the van der Waals surface has a legal
centre within `p` in the continuum, but the nearest legal *node* can be
further, so the node is called solute. Measured on the 3 A Born ion at
h = 0.464: 72 extra solute nodes in a shell at r = 3.009 to 3.080, an effective
radius of 3.0717 against the true 3.0097, and the probe's worth on ALA-GLY
comes out **+11.21%** where APBS reads +5.72%. The bias is one-sided, scales
with `h`, and is worst exactly where the surface is — so it cannot be tuned
away, and a fudged probe radius would only hide it.

What fixes it is a fact about where the nearest legal point is. For `x` inside
the accessible union, the nearest point of A lies **on the boundary of that
union** — so the only probe centres that decide anything are on the
solvent-accessible surface, and that surface has an analytic description. It is
made of three kinds of place, and the nearest point is always on one of them:

| family | the probe rests on | the candidate |
|---|---|---|
| `_radially_reachable` | one atom | project `x` out onto that atom's sphere |
| `_toroidally_reachable` | two atoms | project `x` onto the rim where they meet |
| `_vertex_reachable` | three atoms | the two points tangent to all three |

Each produces an *actual* legal probe centre, checked against the neighbours, so
each can only ever say "solvent" correctly and the union of the three can only
shrink the solute towards the truth. **Together they are exhaustive rather than
dense** — there is no sample count, no tuning constant, and no dependence on
`h` beyond which nodes get asked. An earlier draft sampled candidate centres
over each accessible sphere instead, and the sample count *was* the answer:
ALA-GLY's probe worth ran +3.19% to -1.40% against APBS across 32 to 1024
samples with no plateau, and 256 happened to match the reference, which is
exactly the constant-chosen-to-be-met this project keeps refusing.

Cost is why `_neighbours` exists: every family is built from atoms that overlap,
so the pair and triple loops run over an atom's ~15 neighbours rather than over
the structure.

**Where the incumbents disagree, and which one this follows.** On a sphere plus
a zero-radius charge — the Kirkwood geometry — APBS's two surfaces separate by
0.09% to 0.33% while DelPhi C++'s stay bit-identical. The exact answer is that
they coincide, since a zero-radius atom bounds no volume for a probe to roll
around, so APBS's difference is its own SES discretization and DelPhi is right.
This module reproduces the identity, and `tests/test_debye_m4.py` asserts it
rather than trusting it.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from sashimi.debye import kernel
from sashimi.protocol import (
    DIMENSIONS,
    FloatArray,
    PQRData,
    SolventModel,
    SurfaceModel,
)

__all__ = [
    "ReducedSurface",
    "ball_offsets",
    "dilate",
    "inside_solute",
    "inside_union_of_spheres",
]

# Below this a length is a rounding rather than a direction, so the geometry it
# would define — an axis between two coincident atoms, a radial direction from a
# node sitting exactly on a rim's axis — does not exist. Both cases are real:
# structures carry duplicate atoms, and a node on the axis is a measure-zero
# coincidence that a symmetric fixture hits every time.
DEGENERATE = 1e-12

# A seat is a probe against three atoms, so the atom raising the triple needs two
# higher-numbered neighbours before it can raise one at all.
PAIRS_PER_SEAT = 2

# The working set of one batched rim query, in `(rim, node)` pairs *before* the
# radius test thins them. A memory bound and not an answer:
# `tests/test_debye_m7.py` sweeps it and asserts the mask does not move by a
# single node, because a constant that changes a result is the thing this module
# is most afraid of.
#
# **Counted in pairs rather than in rims, and measured rather than chosen.** The
# first draft bounded a flat 2,000 rims and called it "tens of megabytes" on the
# strength of the *surviving* pair count. A rim count bounds nothing — the pairs
# a rim expands to scale with node density — and the surviving pairs are not the
# working set. Measured on fas2 with `tracemalloc`, peak transients over one
# `inside()`:
#
#     rims=2000, 0.5 A      580 MB      against 13 MB unbatched
#     rims=2000, 1.0 A      348 MB      against  3 MB unbatched
#
# Bounded by pairs instead, the peak is flat in resolution *and* the speed is
# flat in the bound — CPU varies under 1% from 20,000 pairs to 2,000,000 while
# the peak moves 20x, so the whole speed-up is present at the bottom of the
# range and everything above it was cost for nothing:
#
#     50,000 pairs, 0.5 A    41 MB, 18.48 s      2,000,000: 592 MB, 18.6 s
#     50,000 pairs, 1.0 A    37 MB,  6.86 s      2,000,000: 354 MB,  7.0 s
#
# 20,000 halves the memory again at no measurable cost, so there is room here if
# memory ever binds before speed does.
PAIR_BATCH = 50_000


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

    # The compiled path, when `sashimi-electro[fast]` is installed. Same
    # contract as the families: a boolean or, held to a bit-identical mask by
    # `tests/test_debye_kernel.py`.
    if kernel.available():
        kernel.mark_union(axes, coords, radii, mask)
        return mask

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


def _window(axes: list[FloatArray], centre: FloatArray, radius: float) -> list[slice] | None:
    """The index slices covering a sphere, or None if it misses the lattice."""
    window = []
    for axis in range(DIMENSIONS):
        lo = int(np.searchsorted(axes[axis], centre[axis] - radius, side="left"))
        hi = int(np.searchsorted(axes[axis], centre[axis] + radius, side="right"))
        if lo >= hi:
            return None
        window.append(slice(lo, hi))
    return window


def ball_offsets(spacing: tuple[float, float, float], radius: float) -> list[tuple[int, int, int]]:
    """Index offsets whose physical displacement lies within `radius`.

    The structuring element for the dilation, in cells rather than in angstroms,
    computed per axis because debye's lattice is anisotropic — `size_grid`
    turns one requested resolution into three achieved spacings, and a ball
    built from a single `h` would be an ellipsoid in physical space. A cubic
    fixture cannot see that, which is the class of defect
    `tests/test_debye_discretization.py` exists for.
    """
    if radius <= 0.0:
        return [(0, 0, 0)]
    extents = [int(np.floor(radius / h)) for h in spacing]
    offsets = []
    for i in range(-extents[0], extents[0] + 1):
        for j in range(-extents[1], extents[1] + 1):
            for k in range(-extents[2], extents[2] + 1):
                dx, dy, dz = i * spacing[0], j * spacing[1], k * spacing[2]
                if dx * dx + dy * dy + dz * dz <= radius * radius:
                    offsets.append((i, j, k))
    return offsets


def _dilate_within(source: np.ndarray, offsets: list[tuple[int, int, int]]) -> np.ndarray:
    """`source` dilated by the structuring element, as a shifted OR per offset.

    An explicit loop over the ball rather than a distance transform, because the
    ball is small — at a 1.4 A probe and 0.5 A spacing it is about ninety cells
    — and a shifted OR is obviously the dilation it claims to be. An exact
    Euclidean distance transform would be the right tool at a much larger radius
    and is a great deal more code to get wrong for no gain here.
    """
    dilated = np.zeros_like(source)
    shape = source.shape
    for di, dj, dk in offsets:
        src = tuple(
            slice(max(0, -d), shape[axis] - max(0, d)) for axis, d in enumerate((di, dj, dk))
        )
        dst = tuple(
            slice(max(0, d), shape[axis] - max(0, -d)) for axis, d in enumerate((di, dj, dk))
        )
        dilated[dst] |= source[src]
    return dilated


def _grown_box(
    mask: np.ndarray, spacing: tuple[float, float, float], radius: float
) -> tuple[slice, ...] | None:
    """The mask's bounding box, grown by `radius`; None if the mask is empty."""
    where = np.nonzero(mask)
    if len(where[0]) == 0:
        return None
    margin = [int(np.ceil(radius / h)) + 1 for h in spacing]
    return tuple(
        slice(max(0, int(w.min()) - m), min(size, int(w.max()) + m + 1))
        for w, m, size in zip(where, margin, mask.shape, strict=True)
    )


def dilate(mask: np.ndarray, spacing: tuple[float, float, float], radius: float) -> np.ndarray:
    """`mask` grown by a ball of physical radius, on the lattice.

    Works inside the mask's own bounding box grown by the radius, because
    everything this is used for is a molecule inside a box padded by 10 A on
    every side: dilating the full grid would spend most of its time on vacuum
    that cannot change. Nothing outside that window can be reached, so the
    restriction is exact rather than an approximation.
    """
    if radius <= 0.0:
        return mask
    box = _grown_box(mask, spacing, radius)
    if box is None:
        return mask
    grown = mask.copy()
    grown[box] = _dilate_within(mask[box], ball_offsets(spacing, radius))
    return grown


@dataclass(frozen=True)
class _Spheres:
    """The accessible spheres and who overlaps whom: one bundle, built once.

    Every family below needs the same four things, and passing them separately
    made three six-argument functions whose argument order was the only thing
    keeping them straight.
    """

    coords: FloatArray
    inflated: FloatArray  # r_i + probe
    neighbours: list[list[int]]
    probe: float
    # The same lists as `neighbours` with the zero-radius atoms dropped, as
    # arrays: this is what every legality test is taken against, and rebuilding
    # it per feature was a Python loop inside the innermost loop of all three
    # families.
    testable: list[np.ndarray]

    @cached_property
    def sorted_testable_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`testable`, sorted within each atom, for the compiled seat kernel.

        `_probe_seats` sorts each atom's neighbours before pairing them, and the
        sort is load-bearing rather than tidy: it fixes which atom of a triple
        the trilateration frame is raised from. Built here so the kernel is
        handed the same order the reference computes for itself.
        """
        return _ragged_table([np.sort(near) for near in self.testable])

    @cached_property
    def neighbour_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`neighbours` flattened, for the compiled kernels that cannot index a list."""
        return _ragged_table([np.asarray(near, dtype=np.int64) for near in self.neighbours])

    @cached_property
    def testable_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`testable` flattened, for the compiled kernels that cannot index a list.

        Built once per structure rather than once per lattice. `inside()` runs
        sixteen times on a protein solve — once per multigrid level per state —
        and rebuilding this inside it would put back a Python loop over every
        atom per lattice, which is the exact cost the kernel it feeds exists to
        remove. Measured at 18,242 atoms: 8.6 ms a call, so 0.14 s of an 86 s
        solve. Small, and it is the same mistake one level up.
        """
        return _ragged_table(self.testable)

    @classmethod
    def around(cls, structure: PQRData, probe: float) -> _Spheres:
        inflated = structure.radii + probe
        neighbours = _neighbours(structure.coords, inflated)
        testable = [
            np.array([j for j in near if inflated[j] > 0.0], dtype=np.int64) for near in neighbours
        ]
        return cls(structure.coords, inflated, neighbours, probe, testable)


def _neighbours(coords: FloatArray, inflated: FloatArray) -> list[list[int]]:
    """For each atom, the atoms whose accessible spheres it overlaps.

    Every feature of the reduced surface below is built from a pair or a triple
    of *mutually overlapping* spheres, so without this the pair loop is O(N^2)
    and the triple loop O(N^3) — 3 million and 5 billion for barnase, against
    the roughly fifteen neighbours an atom actually has. A uniform bin grid at
    twice the largest accessible radius, so any overlapping sphere is in this
    atom's bin or one of the twenty-six around it.
    """
    count = len(coords)
    lists: list[list[int]] = [[] for _ in range(count)]
    live = [index for index in range(count) if inflated[index] > 0.0]
    if not live:
        return lists

    cell = 2.0 * float(max(inflated[index] for index in live))
    if cell <= 0.0:
        return lists
    origin = coords[live].min(axis=0)
    keys = np.floor((coords - origin) / cell).astype(np.int64)

    # The compiled path, when `sashimi-electro[fast]` is installed. It walks the
    # same twenty-seven bins in the same order and applies the same test, so the
    # lists come back element for element identical — which is what
    # `tests/test_debye_kernel.py` asserts rather than comparing them as sets.
    #
    # `_Bins` over the live atoms alone gives the reference's own origin and
    # membership: its `_origin` is `points.min(axis=0)`, and the points here are
    # exactly `coords[live]`. Its slots address that subset, so they are mapped
    # back to atom numbers before the kernel sees them.
    if kernel.available():
        alive = np.asarray(live, dtype=np.int64)
        bin_index = _Bins(coords[alive], cell)
        flat, offset, counts = kernel.neighbour_lists(
            coords, inflated, keys, bin_index, alive[bin_index.order]
        )
        for index in live:
            start = int(offset[index])
            lists[index] = flat[start : start + int(counts[index])].tolist()
        return lists

    bins: dict[tuple[int, int, int], list[int]] = {}
    for index in live:
        bins.setdefault(tuple(int(v) for v in keys[index]), []).append(index)  # type: ignore[arg-type]

    around = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for index in live:
        key = keys[index]
        for di, dj, dk in around:
            for other in bins.get((int(key[0]) + di, int(key[1]) + dj, int(key[2]) + dk), ()):
                if other == index:
                    continue
                # **Not `np.linalg.norm`, and this is the load-bearing kind of
                # detail.** For a 1-D array numpy computes `sqrt(x.dot(x))`, and
                # `dot` is BLAS — which dispatches a microarchitecture-specific
                # kernel at run time, so the last bit of this distance depends on
                # the CPU the process happens to be running on. Here that lands
                # in a *threshold*: two atoms sitting exactly at the sum of their
                # radii become neighbours on one machine and not on another,
                # which changes the rims, the seats and so the surface. Written
                # out, it is the same three multiplications everywhere.
                offset = coords[other] - coords[index]
                gap = math.sqrt(
                    float(offset[0]) * float(offset[0])
                    + float(offset[1]) * float(offset[1])
                    + float(offset[2]) * float(offset[2])
                )
                if gap < inflated[index] + inflated[other]:
                    lists[index].append(other)
    return lists


def _ragged(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Segment index and within-segment position for a run of variable lengths.

    The one shape that appears everywhere below. Geometry here is ragged in two
    places at once — a rim reaches a different number of nodes, and a node is
    tested against a different number of blockers — and a Python loop over
    either is what M7 exists to remove. Given `[3, 0, 2]` this returns
    `[0, 0, 0, 2, 2]` and `[0, 1, 2, 0, 1]`, so a flat array of every pair can
    be built with one `repeat` and one `arange` and then indexed as if it were
    rectangular.

    Empty segments cost nothing and are not special-cased, which is what lets
    the callers skip a `where` that would otherwise be needed at every use.
    """
    total = int(lengths.sum())
    if not total:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    segment = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    starts = np.cumsum(lengths) - lengths
    return segment, np.arange(total, dtype=np.int64) - np.repeat(starts, lengths)


def _ragged_table(pieces: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A list of variable-length index arrays as flat values, offsets and counts.

    Counts run 0 to 74 on a protein, so this cannot be rectangular. Flat plus
    offsets is the form the compiled kernels index directly — the alternative is
    a list of arrays, which numba reflects at a cost per call that is the whole
    thing those kernels exist to remove.
    """
    count = np.array([len(piece) for piece in pieces], dtype=np.int64)
    arrays = [np.asarray(piece, dtype=np.int64) for piece in pieces]
    flat = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.int64)
    return flat, np.cumsum(count) - count, count


def _blocker_table(
    rims: list[tuple[FloatArray, FloatArray, float, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each rim's blocking atoms, flattened, because every rim has a different set."""
    return _ragged_table([blockers for *_, blockers in rims])


def _batches(weights: np.ndarray, limit: int) -> list[tuple[int, int]]:
    """Consecutive index ranges whose weights sum to under `limit`.

    Used where the work per item is known in advance and varies by orders of
    magnitude, so a fixed count of items would size a batch differently at every
    grid spacing. A single item over the limit gets its own batch rather than an
    exception: the limit bounds a temporary, and refusing to answer would be
    worse than one large allocation.
    """
    if not len(weights):
        return []
    total = np.cumsum(weights)
    ranges = []
    start = 0
    while start < len(weights):
        base = float(total[start - 1]) if start else 0.0
        stop = max(int(np.searchsorted(total, base + limit, side="right")), start + 1)
        ranges.append((start, stop))
        start = stop
    return ranges


def _legal(
    points: FloatArray,
    coords: FloatArray,
    inflated: FloatArray,
    against: np.ndarray,
    exempt: np.ndarray | None = None,
) -> np.ndarray:
    """Which candidate probe centres overlap none of the `against` atoms.

    A candidate sits at exactly `R_i` from the atoms it was constructed from, so
    those are excluded rather than tested — a floating-point comparison against
    the very sphere a point lies on rejects it about half the time, and the
    surface would come out with holes in it wherever a feature was built. Two
    of the three families construct from a fixed atom set and drop it from
    `against`; the third varies per candidate and passes `exempt`, a
    candidate-by-atom mask of the pairs to let through.

    One broadcast over the atoms rather than a loop over them. The loop was the
    innermost thing in the module and it ran a numpy call per neighbour per
    feature — about fifteen times more calls than there was arithmetic to do.
    """
    if not len(points):
        return np.zeros(0, dtype=bool)
    if not len(against):
        return np.ones(len(points), dtype=bool)
    gap = points[:, None, :] - coords[against][None, :, :]
    limits = inflated[against] * inflated[against]
    outside = (gap * gap).sum(axis=2) >= limits[None, :]
    if exempt is not None:
        outside |= exempt
    return np.asarray(outside.all(axis=1))


@dataclass(frozen=True)
class _Nodes:
    """Undecided lattice nodes as a point cloud, with the indices to mark back.

    Families two and three ask "which of these nodes is near this feature"
    thousands of times over one set, so the coordinates are built once here.
    Rebuilding them from an index window per feature — and copying a sub-box in
    and out to mark — is where M4's first implementation spent its time.
    """

    index: tuple[np.ndarray, np.ndarray, np.ndarray]
    points: FloatArray

    @classmethod
    def of(cls, axes: list[FloatArray], mask: np.ndarray) -> _Nodes:
        index = np.nonzero(mask)
        points = np.column_stack([axes[axis][index[axis]] for axis in range(DIMENSIONS)])
        return cls((index[0], index[1], index[2]), points)

    def __len__(self) -> int:
        return len(self.points)

    def mark(self, reachable: np.ndarray, chosen: np.ndarray) -> None:
        reachable[self.index[0][chosen], self.index[1][chosen], self.index[2][chosen]] = True


class _Bins:
    """Points on a uniform bin grid, for repeated ball queries of bounded radius.

    Cell size is a scale rather than a bound: `near` walks the bins its query
    box actually covers, so a query wider than a cell is handled and a query
    narrower than one does not pay for the twenty-six bins around it. Sizing
    the cell near the typical query is therefore a speed choice and not a
    correctness one, which is worth having in a module where a constant that
    changes the answer is the thing to be afraid of.
    """

    def __init__(self, points: FloatArray, cell: float) -> None:
        self.points = points
        self.cell = cell
        self._origin = points.min(axis=0)
        keys = self.keys_of(points)
        self._order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        ordered = keys[self._order]
        starts = np.flatnonzero(np.r_[True, (ordered[1:] != ordered[:-1]).any(axis=1)])
        stops = np.r_[starts[1:], len(ordered)]

        # Occupied bins as sorted integer codes rather than a dict of tuples.
        # `_block` is the innermost thing in the module — 197,000 calls on a
        # 906-atom protein — and a dict keyed by a Python tuple forces the
        # lookup, and therefore the bin walk, to happen one bin at a time in
        # Python. Encoded this way the whole walk is one `searchsorted`.
        self._low_key = keys.min(axis=0)
        self._high_key = keys.max(axis=0)
        extent = self._high_key - self._low_key + 1
        self._strides = np.array([extent[1] * extent[2], extent[2], 1], dtype=np.int64)
        self._codes = self._encode(ordered[starts])
        self._starts = starts.astype(np.int64)
        self._stops = stops.astype(np.int64)

    # The compiled kernel in `kernel.py` walks these same bins, so the structure
    # is named rather than reached into across modules. Read-only by contract:
    # they are the index, not a copy of it.
    @property
    def order(self) -> np.ndarray:
        return self._order

    @property
    def starts(self) -> np.ndarray:
        return self._starts

    @property
    def stops(self) -> np.ndarray:
        return np.asarray(self._stops)

    @property
    def codes(self) -> np.ndarray:
        return self._codes

    @property
    def low_key(self) -> np.ndarray:
        return self._low_key

    @property
    def high_key(self) -> np.ndarray:
        return self._high_key

    @property
    def strides(self) -> np.ndarray:
        return self._strides

    @property
    def bin_origin(self) -> FloatArray:
        return self._origin

    def _encode(self, keys: np.ndarray) -> np.ndarray:
        """Bin keys as one integer each, valid only inside the occupied extent."""
        return np.asarray((keys - self._low_key) @ self._strides)

    def keys_of(self, points: FloatArray) -> np.ndarray:
        return np.floor((points - self._origin) / self.cell).astype(np.int64)

    def _block(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """Every point in the inclusive range of bins from `low` to `high`.

        Clipped to the occupied extent first: a bin outside it holds no points
        by construction, so clipping changes nothing and keeps every code inside
        the range the encoding is valid over.
        """
        low = np.maximum(low, self._low_key)
        high = np.minimum(high, self._high_key)
        if (low > high).any():
            return np.zeros(0, dtype=np.int64)

        span = high - low + 1
        i, j, k = (np.arange(low[axis], high[axis] + 1) for axis in range(3))
        wanted = (
            ((i - self._low_key[0]) * self._strides[0])[:, None, None]
            + ((j - self._low_key[1]) * self._strides[1])[None, :, None]
            + (k - self._low_key[2])[None, None, :]
        ).reshape(int(span.prod()))

        at = np.searchsorted(self._codes, wanted)
        within = at < len(self._codes)
        at, wanted = at[within], wanted[within]
        hit = at[self._codes[at] == wanted]
        starts, stops = self._starts[hit], self._stops[hit]

        lengths = stops - starts
        total = int(lengths.sum())
        if not total:
            return np.zeros(0, dtype=np.int64)
        # Flatten the variable-length ranges without a Python loop: repeat each
        # range's start, then add a running offset that restarts at each range.
        offsets = np.repeat(starts - np.r_[0, np.cumsum(lengths)[:-1]], lengths)
        return np.asarray(self._order[offsets + np.arange(total)])

    def gather(self, key: np.ndarray) -> np.ndarray:
        """Every point in the bin `key` and the twenty-six around it."""
        return self._block(key - 1, key + 1)

    def near_many(self, centres: FloatArray, radii: FloatArray) -> tuple[np.ndarray, np.ndarray]:
        """Flat `(query, point)` pairs for a batch of ball queries at once.

        The same answer as calling `near` once per row and the same bins walked;
        what changes is that the two ragged expansions — a query covers a
        variable number of bins, a bin holds a variable number of points — are
        built with `repeat` rather than with a Python loop around a handful of
        floating-point operations.

        This is the call M7 is about. The rim loop asked it 11,380 times per
        lattice, and on the coarsest multigrid level 3,801 of those calls
        decided fifty-one nodes between them.
        """
        low = np.maximum(self.keys_of(centres - radii[:, None]), self._low_key)
        high = np.minimum(self.keys_of(centres + radii[:, None]), self._high_key)
        span = np.maximum(high - low + 1, 0)

        # Every bin every query covers, as one flat list of (query, bin code).
        query, local = _ragged(span.prod(axis=1))
        if not len(query):
            return query, local
        extent = span[query]
        plane = extent[:, 1] * extent[:, 2]
        wanted = self._encode(
            low[query]
            + np.column_stack([local // plane, local % plane // extent[:, 2], local % extent[:, 2]])
        )

        at = np.searchsorted(self._codes, wanted)
        within = at < len(self._codes)
        at, wanted, query = at[within], wanted[within], query[within]
        hit = self._codes[at] == wanted
        at, query = at[hit], query[hit]

        # Every point in every bin that was hit, as (query, point).
        segment, position = _ragged(self._stops[at] - self._starts[at])
        if not len(segment):
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        found = self._order[self._starts[at][segment] + position]
        query = query[segment]

        gap = self.points[found] - centres[query]
        inside = (gap * gap).sum(axis=1) <= radii[query] * radii[query]
        return query[inside], found[inside]

    def near(self, centre: FloatArray, radius: float) -> np.ndarray:
        """Indices of the points within `radius` of `centre`.

        Bounded by the query box rather than by the bin the centre falls in:
        with the cell at the typical query radius that is two bins on an axis
        instead of three, and it is what lets a query wider than a cell work at
        all.
        """
        corner = np.array([centre - radius, centre + radius])
        low, high = self.keys_of(corner)
        found = self._block(low, high)
        if not len(found):
            return found
        gap = self.points[found] - centre
        return np.asarray(found[(gap * gap).sum(axis=1) <= radius * radius])


def _nodes_in(
    axes: list[FloatArray], window: list[slice], candidates: np.ndarray
) -> tuple[tuple[np.ndarray, ...], FloatArray]:
    """The lattice indices and coordinates of the candidate nodes in a window."""
    local = np.nonzero(candidates)
    points = np.column_stack([axes[axis][window[axis]][local[axis]] for axis in range(DIMENSIONS)])
    return local, points


def _mark(
    reachable: np.ndarray,
    box: tuple[slice, ...],
    local: tuple[np.ndarray, ...],
    chosen: np.ndarray,
) -> None:
    """Set the chosen subset of a window's candidate nodes."""
    marked = reachable[box]
    marked[local[0][chosen], local[1][chosen], local[2][chosen]] = True
    reachable[box] = marked


def _radially_reachable(
    axes: list[FloatArray],
    spheres: _Spheres,
    undecided: np.ndarray,
) -> np.ndarray:
    """Family one: the probe resting on a single atom.

    **The test, and why it is exact rather than a sampling.** Take a shell node
    `x` inside atom `i`'s accessible sphere and project it radially onto that
    sphere: `a = c_i + R_i (x - c_i)/|x - c_i|`, where `R_i = r_i + probe`. Two
    facts make this the whole test wherever the surface is convex:

    - `|x - a| = R_i - |x - c_i| <= probe` **always**, because a shell node is
      outside every van der Waals sphere, so `|x - c_i| > r_i = R_i - probe`.
      The distance condition is free; only legality is in question.
    - So `x` is solvent as soon as `a` lies outside every *other* accessible
      sphere — a probe centred there touches `x` and overlaps nothing.

    For a lone sphere `a` is the radially outward point and the answer is exact:
    the solvent-excluded surface comes back as the van der Waals sphere to the
    node, with no dependence on the lattice and no sample count.
    """
    coords, inflated = spheres.coords, spheres.inflated
    reachable = np.zeros_like(undecided)

    # The compiled path, when `sashimi-electro[fast]` is installed. Same
    # contract as the rim loop below: it answers the identical question and
    # `tests/test_debye_kernel.py` holds it to a bit-identical mask, so the
    # branch is a speed choice and nothing else.
    if kernel.available():
        kernel.decide_radial(
            axes,
            coords,
            inflated,
            spheres.testable_table,
            undecided,
            reachable,
            DEGENERATE,
        )
        return reachable

    for index, (centre, radius) in enumerate(zip(coords, inflated, strict=True)):
        if radius <= 0.0:
            continue
        window = _window(axes, centre, radius)
        if window is None:
            continue
        box = tuple(window)
        candidates = undecided[box] & ~reachable[box]
        if not candidates.any():
            continue

        local, points = _nodes_in(axes, window, candidates)
        offset = points - centre
        distance = np.sqrt((offset**2).sum(axis=1))
        # A node exactly on an atom centre has no radial direction, and cannot
        # be a shell node anyway unless the atom has zero radius.
        usable = distance > DEGENERATE
        pushed = np.zeros_like(points)
        pushed[usable] = centre + radius * offset[usable] / distance[usable, None]

        allowed = np.zeros(len(points), dtype=bool)
        allowed[usable] = _legal(pushed[usable], coords, inflated, spheres.testable[index])
        if allowed.any():
            _mark(reachable, box, local, allowed)
    return reachable


def _toroidally_reachable(
    axes: list[FloatArray],
    surface: ReducedSurface,
    still: np.ndarray,
) -> np.ndarray:
    """Family two: a probe wedged against two atoms at once.

    Where the radial push is blocked, the nearest legal point has slid off the
    open part of one accessible sphere to the rim where two of them meet. That
    rim is the circle `sphere_i ∩ sphere_j`, and the nearest point of a circle
    to a node has a closed form, so this needs no sampling either: project the
    node onto the circle, check the projection overlaps no third atom, and
    check it is within a probe of the node.

    This is the toroidal patch of the classical reduced surface, and it is what
    a groove is made of — the re-entrant surface between two atoms close enough
    that a probe bridges them.

    **This is two thirds of a protein solve, and it is batched rather than
    looped for that reason (M7).** The geometry above is a few dozen
    floating-point operations per rim; the loop that fed it was 11,380
    iterations of about twenty numpy calls each, per lattice, sixteen lattices
    per solve. Measured on `fas2`, the cost per point was 29x worse on the
    coarsest multigrid level than on the finest — the signature of a fixed cost
    per call rather than of arithmetic — and the coarsest level spent 0.42 s to
    decide fifty-one nodes.

    So every stage below runs over a flat array of `(rim, node)` pairs: which
    nodes each rim reaches, the projection onto the circle, and the legality of
    the projections. `PAIR_BATCH` bounds the working set; it is not a sample
    count and it does not appear in an answer.

    **What was given up, and what it cost.** The old loop skipped the nodes an
    earlier rim had already claimed, which is why it had to be a loop. Measured,
    that pruning removes 16% of the pairs — live nodes are 84% of found nodes at
    every level — so it is kept between batches, where it costs nothing, and
    given up inside one. It cannot change an answer either way: `decided` feeds
    a boolean or, so a node claimed by *any* rim is the node claimed by *the
    first*, and that is what makes this rewrite checkable against the last digit
    of an energy rather than against a tolerance.
    """
    spheres = surface.spheres
    coords, inflated = spheres.coords, spheres.inflated
    probe = spheres.probe
    reachable = np.zeros_like(still)
    squared_probe = probe * probe

    rims = surface.rims
    nodes = _Nodes.of(axes, still)
    if not rims or not len(nodes):
        return reachable

    # The rims are geometry and the nodes are one cloud, so both are built
    # before the loop. What is left inside it is the part that genuinely
    # depends on both: projecting a node onto a circle.
    origins = np.array([origin for origin, _, _, _ in rims])
    normals = np.array([normal for _, normal, _, _ in rims])
    ring_radii = np.array([ring_radius for _, _, ring_radius, _ in rims])

    # Every node a rim can decide is within `ring_radius + probe` of its centre,
    # so the typical such distance is the scale to bin on. Swept on fas2 across
    # a quarter to one times the median: the query cost is flat from 0.5x to 1x
    # and only degrades below that, so this is a knob that does not need one.
    reach = ring_radii + probe
    bins = _Bins(nodes.points, float(np.median(reach)))
    decided = np.zeros(len(nodes), dtype=bool)

    # The compiled path, when `sashimi-electro[fast]` is installed. It answers
    # the identical question — `tests/test_debye_kernel.py` asserts the mask is
    # bit-identical on real geometry, and CI runs the numpy path on two legs and
    # this one on the third — so the branch is a speed choice and nothing else.
    # It needs no batch bound: it never materialises the pair lists that the
    # numpy path has to bound.
    if kernel.available():
        kernel.decide_rims(
            nodes.points,
            bins,
            origins,
            normals,
            ring_radii,
            _blocker_table(rims),
            coords,
            inflated,
            probe,
            decided,
        )
        nodes.mark(reachable, decided)
        return reachable

    # What one batch costs, estimated before it is run. `near_many` expands
    # every bin its query box touches, so the pre-filter pair count for a rim is
    # the node density times that box — which is what has to be bounded, since
    # the surviving pairs are a quarter of it and vary with the geometry. The
    # box is padded by the cell because a bin is included whole.
    span = nodes.points.max(axis=0) - nodes.points.min(axis=0)
    volume = float(np.prod(np.maximum(span, bins.cell)))
    density = len(nodes) / volume
    expected = density * (2.0 * reach + bins.cell) ** DIMENSIONS

    for first, last in _batches(expected, PAIR_BATCH):
        batch = slice(first, last)
        rim, node = bins.near_many(origins[batch], reach[batch])
        live = ~decided[node]
        rim, node = rim[live], node[live]
        if not len(rim):
            continue

        owner = rim + first
        offset = nodes.points[node] - origins[owner]
        normal = normals[owner]
        axial = (offset * normal).sum(axis=1)
        radial = offset - axial[:, None] * normal
        length = np.sqrt((radial**2).sum(axis=1))
        usable = length > DEGENERATE
        close = usable & (((length - ring_radii[owner]) ** 2 + axial**2) <= squared_probe)
        if not close.any():
            continue

        rim, owner, node = rim[close], owner[close], node[close]
        projected = (
            origins[owner] + ring_radii[owner][:, None] * radial[close] / length[close, None]
        )

        # Legality stays one call per rim, and that is a measurement rather than
        # a leftover. `_legal` broadcasts a point cloud against an atom set, so
        # it touches 35 atom coordinates and keeps the whole product in cache;
        # the batched form has to *gather* a coordinate per pair, and there are
        # 69 million of those on the finest level. Measured on fas2, batching
        # this stage cost 1.27 s -> 2.47 s, which is most of why the first
        # version of this rewrite came out at 1.000x overall.
        #
        # `near_many` returns its pairs grouped by query, so each rim's rows are
        # already contiguous and the split is a `searchsorted` rather than a
        # sort.
        edges = np.searchsorted(rim, np.arange(last - first + 1))
        for local, (lo, hi) in enumerate(itertools.pairwise(edges)):
            if lo == hi:
                continue
            allowed = _legal(projected[lo:hi], coords, inflated, rims[first + local][3])
            if allowed.any():
                decided[node[lo:hi][allowed]] = True

    nodes.mark(reachable, decided)
    return reachable


def _rims(spheres: _Spheres) -> list[tuple[FloatArray, FloatArray, float, np.ndarray]]:
    """Every reachable circle where two accessible spheres meet, and what blocks it.

    Lattice-independent, like the seats in `_probe_seats` — this is the reduced
    surface itself rather than its discretization.

    **A rim swallowed whole by a third atom is dropped**, and that is most of
    them: 59% on fas2, because an atom overlaps about sixty others once the
    radii carry the probe, and the pair rims buried in the interior vastly
    outnumber the ones on the surface. The test is exact rather than a
    heuristic — a circle's farthest point from a sphere centre has a closed
    form, so a rim is discarded only when *no* point of it is a legal probe
    centre, and a rim with no legal point can decide no node. Without this the
    loop below pairs every buried rim against the nodes near it and then throws
    all of them away in `_legal`.
    """
    coords, inflated, neighbours = spheres.coords, spheres.inflated, spheres.neighbours

    # The compiled path, when `sashimi-electro[fast]` is installed. Unlike the
    # family kernels this one returns *geometry* — a circle three later stages
    # compare against radii — so bit-identity has to be proved rather than
    # arranged, and `tests/test_debye_kernel.py` compares the origins, normals
    # and radii with `array_equal`.
    if kernel.available():
        origins, normals, ring_radii, offsets, counts, blocker_flat = kernel.enumerate_rims(
            coords, inflated, spheres.neighbour_table, spheres.testable_table, DEGENERATE
        )
        return [
            (origins[index], normals[index], float(ring_radii[index]),
             blocker_flat[offsets[index] : offsets[index] + counts[index]])
            for index in range(len(ring_radii))
        ]  # fmt: skip

    rims = []
    for i in range(len(coords)):
        if inflated[i] <= 0.0:
            continue
        for j in neighbours[i]:
            if j <= i or inflated[j] <= 0.0:
                continue
            ring = _rim(coords, inflated, i, j)
            if ring is None:
                continue
            blockers = _blockers(spheres, (i, j), ring)
            if blockers is not None:
                rims.append((*ring, blockers))
    return rims


def _blockers(
    spheres: _Spheres, pair: tuple[int, int], ring: tuple[FloatArray, FloatArray, float]
) -> np.ndarray | None:
    """Which third atoms can cover part of this rim; None if one covers all of it.

    Split `c_m - origin` into its axial and radial parts about the circle's own
    axis. The circle's farthest point from `c_m` is then at
    `hypot(axial, radial + ring_radius)` and its nearest at
    `hypot(axial, radial - ring_radius)`, so one closed form answers both
    questions: inside `R_m` at the far point and the rim is gone entirely,
    outside `R_m` at the near point and this atom can never reject a probe
    centre on the rim.

    Both prunings are exact, and the second is why this returns a set rather
    than a verdict. An atom overlaps about sixty others once the radii carry
    the probe, but only a handful of those reach any given rim — and the
    survivors are the entire argument to `_legal`, which is otherwise the most
    expensive thing in this module.

    Only `i`'s neighbours are tested, and that is exhaustive: a sphere holding
    any point of the rim holds a point at `R_i` from `c_i`, so its centre is
    within `R_m + R_i` of `c_i`.
    """
    i, j = pair
    origin, normal, ring_radius = ring
    against = spheres.testable[i]
    against = against[against != j]
    if not len(against):
        return np.asarray(against)
    gap = spheres.coords[against] - origin
    axial = gap @ normal
    radial = np.sqrt(((gap - axial[:, None] * normal) ** 2).sum(axis=1))
    limits = spheres.inflated[against] * spheres.inflated[against]
    axial_squared = axial * axial
    if np.any(axial_squared + (radial + ring_radius) ** 2 <= limits):
        return None
    return np.asarray(against[axial_squared + (radial - ring_radius) ** 2 < limits])


def _rim(
    coords: FloatArray, inflated: FloatArray, i: int, j: int
) -> tuple[FloatArray, FloatArray, float] | None:
    """Where two accessible spheres meet: centre, axis and radius of the circle."""
    axis_vector = coords[j] - coords[i]
    # Spelled out rather than `np.linalg.norm`, for the reason `_neighbours`
    # records: numpy routes that through BLAS, whose summation order is chosen
    # per CPU at run time. It made this rim's origin and normal differ in the
    # last bit between two GitHub runners of the same operating system.
    separation = math.sqrt(
        float(axis_vector[0]) * float(axis_vector[0])
        + float(axis_vector[1]) * float(axis_vector[1])
        + float(axis_vector[2]) * float(axis_vector[2])
    )
    if separation >= inflated[i] + inflated[j] or separation <= DEGENERATE:
        return None
    if separation <= abs(inflated[i] - inflated[j]):
        return None  # one sphere swallows the other; there is no rim
    normal = axis_vector / separation
    # **Every square here is written as a multiplication, and that is not a
    # style choice.** `x ** 2` on a *scalar* — Python float or numpy float64
    # alike — is a call to the platform's `pow`, and this platform's `pow` is
    # not correctly rounded: measured on fas2, it disagrees with `x * x` by one
    # ulp for 26 of 27,799 separations and 38 of the same set's `along` values.
    # A multiplication is correctly rounded by IEEE 754, so this is the more
    # accurate spelling as well as the reproducible one — the rim radii of a
    # recorded corpus energy should not depend on whose libm ran it. (An
    # *array* `** 2` is safe: numpy fast-paths it to `np.square`, which is the
    # multiplication. Every other square in this module is on an array.)
    along = (separation * separation + inflated[i] * inflated[i] - inflated[j] * inflated[j]) / (
        2.0 * separation
    )
    squared = inflated[i] * inflated[i] - along * along
    if squared <= 0.0:
        return None
    return coords[i] + along * normal, normal, float(np.sqrt(squared))


def _vertex_reachable(
    axes: list[FloatArray],
    surface: ReducedSurface,
    still: np.ndarray,
) -> np.ndarray:
    """Family three: a probe jammed against three atoms, which cannot move at all.

    The last place the nearest legal point can be. Where both a radial push and
    a slide along a rim are blocked, what remains is a point — the probe seated
    in the pocket of three mutually overlapping atoms, touching all three. Three
    spheres meet in at most two points, mirror images through the plane of the
    centres, and both are candidates.

    Small in volume and not optional: without it the construction stays
    conservative exactly at the junctions where three atoms meet, which on a
    real solute is most of the deep surface.

    **The seats do not depend on the lattice**, so they are built as one batch
    and the nodes are asked about once. That split is the whole performance
    story of this family: the per-triple version trilaterated, windowed the
    grid, and copied a sub-box in and out about a hundred thousand times on a
    906-atom protein, at roughly half a millisecond of numpy call overhead each
    for a few dozen floating-point operations of actual geometry.
    """
    reachable = np.zeros_like(still)
    seats = surface.seats
    if not len(seats):
        return reachable
    nodes = _Nodes.of(axes, still)
    if not len(nodes):
        return reachable
    nodes.mark(reachable, _within(nodes.points, seats, surface.probe))
    return reachable


def _probe_seats(spheres: _Spheres) -> FloatArray:
    """Every legal seat: a probe centre touching three atoms and overlapping none.

    The vertex set of the reduced surface. Enumerated once per atom rather than
    once per triple — an atom's triples share both the trilateration frame's
    first centre and the list of atoms their seats must clear, so both become
    single array operations over the whole batch.

    Each triple is raised under its *smallest* member, which is what makes
    "once" true. Legality is still exhaustive under that choice: a seat lies at
    exactly `R_i` from atom `i`, so any sphere `m` containing it has
    `|c_m - c_i| < R_m + R_i` and is therefore already one of `i`'s neighbours.
    The same holds for `j` and `k`, so which of the three raises the triple
    cannot change the answer.
    """
    coords, inflated, neighbours = spheres.coords, spheres.inflated, spheres.neighbours
    count = len(coords)
    overlapping = _overlapping_pairs(neighbours, inflated, count)

    # The compiled path, when `sashimi-electro[fast]` is installed. Geometry
    # again rather than a verdict, so `tests/test_debye_kernel.py` compares the
    # seat coordinates with `array_equal`.
    if kernel.available():
        return kernel.probe_seats(
            coords, inflated, spheres.sorted_testable_table, overlapping, DEGENERATE
        )

    seats = []

    for i in range(count):
        if inflated[i] <= 0.0:
            continue
        near = np.sort(spheres.testable[i])
        near = near[near > i]
        if len(near) < PAIRS_PER_SEAT:
            continue
        first, second = np.triu_indices(len(near), k=1)
        j, k = near[first], near[second]
        # The third pair has to overlap too, or the three spheres leave no seat.
        overlaps = _holds(overlapping, j * count + k)
        j, k = j[overlaps], k[overlaps]
        if not len(j):
            continue

        points, owner = _tangency_points(coords, inflated, i, j, k)
        if not len(points):
            continue
        against = spheres.testable[i]
        exempt = (against[None, :] == j[owner][:, None]) | (against[None, :] == k[owner][:, None])
        legal = _legal(points, coords, inflated, against, exempt)
        if legal.any():
            seats.append(points[legal])

    if not seats:
        return np.zeros((0, DIMENSIONS), dtype=np.float64)
    return np.concatenate(seats)


def _overlapping_pairs(neighbours: list[list[int]], inflated: FloatArray, count: int) -> np.ndarray:
    """Sorted `i * count + j` keys for every overlapping pair with `i < j`."""
    keys = [
        i * count + j
        for i, near in enumerate(neighbours)
        if inflated[i] > 0.0
        for j in near
        if j > i and inflated[j] > 0.0
    ]
    return np.sort(np.array(keys, dtype=np.int64))


def _holds(sorted_keys: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Membership in a sorted key array, without building a set per query."""
    if not len(sorted_keys):
        return np.zeros(len(wanted), dtype=bool)
    at = np.searchsorted(sorted_keys, wanted)
    at = np.minimum(at, len(sorted_keys) - 1)
    return np.asarray(sorted_keys[at] == wanted)


def _tangency_points(
    coords: FloatArray, inflated: FloatArray, i: int, j: np.ndarray, k: np.ndarray
) -> tuple[FloatArray, np.ndarray]:
    """Points at `R_i`, `R_j`, `R_k` from three atom centres, for a batch of triples.

    Trilateration in the frame of the three centres, done for every triple that
    shares the atom `i` at once. Returns the surviving points — at most two per
    triple, mirror images through the plane of the centres — together with the
    index of the triple each came from, since the triples that produce none are
    dropped along the way and the caller needs `j` and `k` back to know which
    spheres a point is allowed to touch.

    Triples drop out at three places, and all three are ordinary rather than
    exceptional: coincident centres have no axis, collinear centres leave a
    circle rather than a pair of points, and three spheres overlapping pairwise
    need not meet at all — which is the common case.
    """
    alive = np.arange(len(j))
    p1 = coords[i]
    r1 = float(inflated[i])

    ex = coords[j] - p1
    span = np.sqrt((ex * ex).sum(axis=1))
    keep = span > DEGENERATE
    alive, ex, span = alive[keep], ex[keep], span[keep]
    ex = ex / span[:, None]

    third = coords[k[alive]] - p1
    along = (ex * third).sum(axis=1)
    ey = third - along[:, None] * ex
    height = np.sqrt((ey * ey).sum(axis=1))
    keep = height > DEGENERATE
    alive, ex, span, along, ey, height = (
        alive[keep],
        ex[keep],
        span[keep],
        along[keep],
        ey[keep],
        height[keep],
    )
    ey = ey / height[:, None]
    ez = np.cross(ex, ey)

    r2, r3 = inflated[j[alive]], inflated[k[alive]]
    x = (r1 * r1 - r2 * r2 + span * span) / (2.0 * span)
    y = (r1 * r1 - r3 * r3 + along * along + height * height - 2.0 * along * x) / (2.0 * height)
    squared = r1 * r1 - x * x - y * y
    keep = squared > 0.0
    alive, ex, ey, ez, x, y = alive[keep], ex[keep], ey[keep], ez[keep], x[keep], y[keep]
    z = np.sqrt(squared[keep])

    base = p1 + x[:, None] * ex + y[:, None] * ey
    points = np.concatenate([base + z[:, None] * ez, base - z[:, None] * ez])
    return points, np.concatenate([alive, alive])


def _within(points: FloatArray, centres: FloatArray, radius: float) -> np.ndarray:
    """Which of `points` lie within `radius` of any of `centres`.

    Grouped by bin rather than asked point by point: the bin a point falls in
    decides which centres can reach it, so points sharing a bin share one
    gather. On a protein that is a few thousand iterations over an undecided
    shell against a hundred thousand seats, where the pairing is quadratic.
    """
    bins = _Bins(centres, radius)

    # The compiled path, when `sashimi-electro[fast]` is installed. It walks the
    # same bins one point at a time — the gather below is what buys the numpy
    # version its speed and what the kernel does not need, since nothing is
    # materialised per bin.
    if kernel.available():
        hit = np.zeros(len(points), dtype=bool)
        kernel.decide_seated(points, bins, centres, radius, hit)
        return hit

    keys = bins.keys_of(points)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    ordered = keys[order]
    starts = np.flatnonzero(np.r_[True, (ordered[1:] != ordered[:-1]).any(axis=1)])
    stops = np.r_[starts[1:], len(ordered)]

    hit = np.zeros(len(points), dtype=bool)
    squared = radius * radius
    for start, stop in zip(starts, stops, strict=True):
        found = bins.gather(ordered[start])
        if not len(found):
            continue
        members = order[start:stop]
        gap = points[members][:, None, :] - centres[found][None, :, :]
        hit[members] = ((gap * gap).sum(axis=2) <= squared).any(axis=1)
    return hit


class ReducedSurface:
    """The probe's reduced surface for one structure, asked about by many lattices.

    **What is geometry and what is discretization**, which is the whole reason
    this is an object rather than a function. Which atoms overlap, where their
    accessible spheres meet, and where a probe can seat against three of them
    are facts about the solute; only *which nodes those features decide* is a
    fact about the lattice. A solve asks the same question of a great many
    lattices — `dielectric_faces` samples three staggered ones, and
    `build_levels` re-discretizes at every multigrid level — so the features are
    built on first use and kept, and each lattice pays only for its own nodes.

    Built lazily, because the lone-sphere case never needs them: a convex
    sphere's solvent-excluded surface is the sphere, `undecided` comes back
    empty, and neither the rims nor the seats are ever touched.
    """

    def __init__(self, structure: PQRData, solvent: SolventModel) -> None:
        self.structure = structure
        self.solvent = solvent
        self.probe = solvent.surface_radius

    @cached_property
    def spheres(self) -> _Spheres:
        return _Spheres.around(self.structure, self.probe)

    @cached_property
    def rims(self) -> list[tuple[FloatArray, FloatArray, float, np.ndarray]]:
        return _rims(self.spheres)

    @cached_property
    def seats(self) -> FloatArray:
        return _probe_seats(self.spheres)

    def inside(self, axes: list[FloatArray]) -> np.ndarray:
        """Boolean mask over the lattice spanned by `axes`: inside the solute.

        The one oracle both boundaries go through. `VAN_DER_WAALS` is the union
        of spheres; `MOLECULAR` rolls a probe over it. Anything else is refused
        before reaching here, by `debye.options.check_surface`.
        """
        structure = self.structure
        inside_vdw = inside_union_of_spheres(axes, structure.coords, structure.radii)
        if self.solvent.surface_model is not SurfaceModel.MOLECULAR or self.probe <= 0.0:
            return inside_vdw  # a zero probe is the van der Waals surface, exactly

        inflated_mask = inside_union_of_spheres(
            axes, structure.coords, structure.radii + self.probe
        )
        undecided = inflated_mask & ~inside_vdw
        if not undecided.any():
            # Every accessible point is already a van der Waals point, so the
            # probe has nowhere to roll into. Not an optimisation: it is the
            # lone-sphere case, and the families below would be work to confirm
            # a mask that cannot change.
            return inside_vdw

        # The three reduced-surface families, cheapest and most productive
        # first. Each witness produces an *actual* legal probe centre, so each
        # can only say "solvent" correctly and the union can only shrink the
        # solute towards the truth. Together they are exhaustive rather than
        # dense: the nearest point of the accessible set to any node lies on the
        # open part of one sphere, on a rim where two meet, or at a seat where
        # three do.
        reachable = _radially_reachable(axes, self.spheres, undecided)
        still = undecided & ~reachable
        if still.any():
            reachable |= _toroidally_reachable(axes, self, still)
            still = undecided & ~reachable
        if still.any():
            reachable |= _vertex_reachable(axes, self, still)

        return np.asarray(inside_vdw | (inflated_mask & ~reachable), dtype=bool)


def inside_solute(
    axes: list[FloatArray],
    structure: PQRData,
    solvent: SolventModel,
) -> np.ndarray:
    """Inside the solute, for a caller with one lattice to ask about.

    Anything asking about several — which is every real solve — should hold a
    `ReducedSurface` instead, so the geometry is built once rather than once
    per lattice.
    """
    return ReducedSurface(structure, solvent).inside(axes)
