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

from dataclasses import dataclass

import numpy as np

from sashimi.protocol import DIMENSIONS, FloatArray, PQRData, SolventModel, SurfaceModel

__all__ = ["ball_offsets", "dilate", "inside_solute", "inside_union_of_spheres"]

# Below this a length is a rounding rather than a direction, so the geometry it
# would define — an axis between two coincident atoms, a radial direction from a
# node sitting exactly on a rim's axis — does not exist. Both cases are real:
# structures carry duplicate atoms, and a node on the axis is a measure-zero
# coincidence that a symmetric fixture hits every time.
DEGENERATE = 1e-12


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

    @classmethod
    def around(cls, structure: PQRData, probe: float) -> _Spheres:
        inflated = structure.radii + probe
        return cls(structure.coords, inflated, _neighbours(structure.coords, inflated), probe)


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
                gap = float(np.linalg.norm(coords[other] - coords[index]))
                if gap < inflated[index] + inflated[other]:
                    lists[index].append(other)
    return lists


def _legal(
    points: FloatArray,
    coords: FloatArray,
    inflated: FloatArray,
    against: list[int],
    skip: tuple[int, ...],
) -> np.ndarray:
    """Which candidate probe centres overlap no atom.

    A candidate sits at exactly `R_i` from the atoms it was constructed from, so
    those are excluded by name rather than tested — a floating-point comparison
    against the very sphere a point lies on rejects it about half the time, and
    the surface would come out with holes in it wherever a feature was built.
    """
    keep = np.ones(len(points), dtype=bool)
    for other in against:
        if other in skip or inflated[other] <= 0.0:
            continue
        limit = inflated[other]
        keep &= ((points - coords[other]) ** 2).sum(axis=1) >= limit * limit
        if not keep.any():
            break
    return keep


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
    coords, inflated, neighbours = spheres.coords, spheres.inflated, spheres.neighbours
    reachable = np.zeros_like(undecided)
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
        allowed[usable] = _legal(pushed[usable], coords, inflated, neighbours[index], skip=(index,))
        if allowed.any():
            _mark(reachable, box, local, allowed)
    return reachable


def _toroidally_reachable(
    axes: list[FloatArray],
    spheres: _Spheres,
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
    """
    coords, inflated, neighbours = spheres.coords, spheres.inflated, spheres.neighbours
    probe = spheres.probe
    reachable = np.zeros_like(still)
    squared_probe = probe * probe

    for i in range(len(coords)):
        if inflated[i] <= 0.0:
            continue
        for j in neighbours[i]:
            if j <= i or inflated[j] <= 0.0:
                continue
            ring = _rim(coords, inflated, i, j)
            if ring is None:
                continue
            origin, normal, ring_radius = ring

            window = _window(axes, origin, ring_radius + probe)
            if window is None:
                continue
            box = tuple(window)
            candidates = still[box] & ~reachable[box]
            if not candidates.any():
                continue

            local, points = _nodes_in(axes, window, candidates)
            offset = points - origin
            axial = offset @ normal
            radial = offset - axial[:, None] * normal
            length = np.sqrt((radial**2).sum(axis=1))
            usable = length > DEGENERATE
            close = usable & (((length - ring_radius) ** 2 + axial**2) <= squared_probe)
            if not close.any():
                continue
            projected = origin + ring_radius * radial[close] / length[close, None]

            allowed = np.zeros(len(points), dtype=bool)
            allowed[close] = _legal(projected, coords, inflated, neighbours[i], skip=(i, j))
            if allowed.any():
                _mark(reachable, box, local, allowed)
    return reachable


def _rim(
    coords: FloatArray, inflated: FloatArray, i: int, j: int
) -> tuple[FloatArray, FloatArray, float] | None:
    """Where two accessible spheres meet: centre, axis and radius of the circle."""
    axis_vector = coords[j] - coords[i]
    separation = float(np.linalg.norm(axis_vector))
    if separation >= inflated[i] + inflated[j] or separation <= DEGENERATE:
        return None
    if separation <= abs(inflated[i] - inflated[j]):
        return None  # one sphere swallows the other; there is no rim
    normal = axis_vector / separation
    along = (separation**2 + inflated[i] ** 2 - inflated[j] ** 2) / (2.0 * separation)
    squared = inflated[i] ** 2 - along**2
    if squared <= 0.0:
        return None
    return coords[i] + along * normal, normal, float(np.sqrt(squared))


def _vertex_reachable(
    axes: list[FloatArray],
    spheres: _Spheres,
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
    """
    coords, inflated, neighbours = spheres.coords, spheres.inflated, spheres.neighbours
    probe = spheres.probe
    reachable = np.zeros_like(still)
    squared_probe = probe * probe
    seen: set[tuple[int, int, int]] = set()

    for i in range(len(coords)):
        if inflated[i] <= 0.0:
            continue
        near = [j for j in neighbours[i] if inflated[j] > 0.0]
        for a in range(len(near)):
            for b in range(a + 1, len(near)):
                j, k = near[a], near[b]
                triple = tuple(sorted((i, j, k)))
                if triple in seen:
                    continue
                seen.add(triple)  # type: ignore[arg-type]
                if k not in neighbours[j]:
                    continue  # the third pair does not overlap, so no seat
                corners = _tangency_points(coords, inflated, i, j, k)
                if corners is None:
                    continue
                keep = _legal(corners, coords, inflated, neighbours[i], skip=(i, j, k))
                corners = corners[keep]
                if not len(corners):
                    continue

                window = _window(
                    axes,
                    corners.mean(axis=0),
                    probe + float(np.linalg.norm(corners - corners.mean(axis=0), axis=1).max()),
                )
                if window is None:
                    continue
                box = tuple(window)
                candidates = still[box] & ~reachable[box]
                if not candidates.any():
                    continue
                local, points = _nodes_in(axes, window, candidates)
                separation = ((points[:, None, :] - corners[None, :, :]) ** 2).sum(axis=2)
                touched = (separation <= squared_probe).any(axis=1)
                if touched.any():
                    _mark(reachable, box, local, touched)
    return reachable


def _tangency_points(
    coords: FloatArray, inflated: FloatArray, i: int, j: int, k: int
) -> FloatArray | None:
    """The (at most two) points at `R_i`, `R_j`, `R_k` from three atom centres.

    Trilateration in the frame of the three centres. Returns None when the
    spheres do not meet — which is the common case, since three atoms
    overlapping pairwise need not leave a seat for the probe.
    """
    p1, p2, p3 = coords[i], coords[j], coords[k]
    r1, r2, r3 = float(inflated[i]), float(inflated[j]), float(inflated[k])

    ex = p2 - p1
    span = float(np.linalg.norm(ex))
    if span <= DEGENERATE:
        return None
    ex = ex / span
    third = p3 - p1
    along = float(ex @ third)
    ey = third - along * ex
    height = float(np.linalg.norm(ey))
    if height <= DEGENERATE:
        return None  # collinear centres leave a circle, not a pair of points
    ey = ey / height
    ez = np.cross(ex, ey)

    x = (r1 * r1 - r2 * r2 + span * span) / (2.0 * span)
    y = (r1 * r1 - r3 * r3 + along * along + height * height - 2.0 * along * x) / (2.0 * height)
    squared = r1 * r1 - x * x - y * y
    if squared <= 0.0:
        return None
    z = float(np.sqrt(squared))
    base = p1 + x * ex + y * ey
    return np.array([base + z * ez, base - z * ez], dtype=np.float64)


def inside_solute(
    axes: list[FloatArray],
    structure: PQRData,
    solvent: SolventModel,
) -> np.ndarray:
    """Boolean mask over the lattice spanned by `axes`: inside the solute.

    The one oracle both boundaries go through. `VAN_DER_WAALS` is the union of
    spheres; `MOLECULAR` rolls a probe over it. Anything else is refused before
    reaching here, by `debye.options.check_surface`.
    """
    inside_vdw = inside_union_of_spheres(axes, structure.coords, structure.radii)
    if solvent.surface_model is not SurfaceModel.MOLECULAR:
        return inside_vdw

    probe = solvent.surface_radius
    if probe <= 0.0:
        return inside_vdw  # a zero probe is the van der Waals surface, exactly

    inflated_mask = inside_union_of_spheres(axes, structure.coords, structure.radii + probe)
    undecided = inflated_mask & ~inside_vdw
    if not undecided.any():
        # Every accessible point is already a van der Waals point, so the probe
        # has nowhere to roll into. Not an optimisation: it is the lone-sphere
        # case, and the families below would be work to confirm a mask that
        # cannot change.
        return inside_vdw

    spheres = _Spheres.around(structure, probe)

    # The three reduced-surface families, cheapest and most productive first.
    # Each witness produces an *actual* legal probe centre, so each can only
    # say "solvent" correctly and the union can only shrink the solute towards
    # the truth. Together they are exhaustive rather than dense: the nearest
    # point of the accessible set to any node lies on the open part of one
    # sphere, on a rim where two meet, or at a seat where three do.
    reachable = _radially_reachable(axes, spheres, undecided)
    still = undecided & ~reachable
    if still.any():
        reachable |= _toroidally_reachable(axes, spheres, still)
        still = undecided & ~reachable
    if still.any():
        reachable |= _vertex_reachable(axes, spheres, still)

    return np.asarray(inside_vdw | (inflated_mask & ~reachable), dtype=bool)
