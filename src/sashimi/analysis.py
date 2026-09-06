"""Questions about a potential map, answered in bytes rather than megabytes.

An agent cannot use a 12 MB grid — it can use "the most negative patch is at
these coordinates" or "this residue sits at -3.2 kT/e". ROADMAP.md §6 argues
this is where the agent-facing value is, and that moving the volume around is
the wrong problem to solve.

Everything here is a pure function of a `PotentialGrid` (plus, sometimes, the
structure). No solver, no subprocess, no I/O — so it is all testable in the
binary-free tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sashimi.protocol import DIMENSIONS, Diagnostics, FloatArray, PotentialGrid, PQRData

DEFAULT_EXCLUSION_MARGIN_A = 1.4
"""Solvent probe radius, in angstroms.

A point at least `r_j + this` from every atom centre is a legal position for a
probe of this radius, so it lies outside the solvent-excluded surface for any
surface model built with that probe or smaller. All three derived-query tools
here reject on it, and they share the name so that stays true.
"""

__all__ = [
    "DEFAULT_EXCLUSION_MARGIN_A",
    "Extremum",
    "ResiduePotential",
    "potential_extrema",
    "potential_in_sphere",
    "residue_potentials",
    "solute_mask",
]


@dataclass(frozen=True)
class Extremum:
    """A local peak in the field, with where it is."""

    position: tuple[float, float, float]  # A
    value: float  # kT/e

    def as_dict(self) -> Diagnostics:
        return {"position": list(self.position), "value_kT_e": self.value}


@dataclass(frozen=True)
class ResiduePotential:
    """Mean potential sampled in the solvent around one residue's atoms.

    The probe counts are not decoration. A residue's mean is taken over
    whatever probe points survived two rejections — points inside a
    neighbouring atom, and points off the map — and a mean over four points
    is a different claim from a mean over forty. `n_probes` is the
    denominator all three of the others share, so a caller can tell the two
    ways of having nothing apart without re-deriving them.

    `value` is None, and `as_dict` omits `mean_kT_e` entirely, when no probe
    survived. That is `potential_in_sphere`'s buried branch in row form: the
    real number or a `KeyError`, never a fabricated one.
    """

    label: str
    value: float | None  # kT/e, or None when no probe point was usable
    n_atoms: int
    n_probes: int  # points offered: len(PROBE_DIRECTIONS) per atom
    n_probes_used: int
    n_probes_excluded_as_solute: int
    n_probes_outside_grid: int
    chain: str | None = None  # from the file, or None when it carried no chain
    segment: int = 1  # 1-based numbering block; >1 only where numbering restarts

    @property
    def unsampled_because(self) -> str | None:
        """Why this residue carries no mean, or None when it carries one.

        Two different nothings, named apart because they ask for opposite
        responses: a buried residue is a true answer about the structure, and
        an off-map residue is a solve that needs a bigger box.
        """
        if self.value is not None:
            return None
        if self.n_probes_outside_grid == 0:
            return (
                f"all {self.n_probes} probe points around this residue fall inside a "
                "neighbouring atom — it is buried, so there is no solvent-side "
                "potential here to average"
            )
        if self.n_probes_excluded_as_solute == 0:
            return (
                f"all {self.n_probes} probe points around this residue fall outside "
                "the map — solve with more padding"
            )
        return (
            f"none of the {self.n_probes} probe points around this residue is usable: "
            f"{self.n_probes_excluded_as_solute} fall inside a neighbouring atom, "
            f"{self.n_probes_outside_grid} outside the map"
        )

    def as_dict(self) -> Diagnostics:
        common: Diagnostics = {
            "residue": self.label,
            "n_atoms": self.n_atoms,
            "n_probes": self.n_probes,
            "n_probes_used": self.n_probes_used,
            "n_probes_excluded_as_solute": self.n_probes_excluded_as_solute,
            "n_probes_outside_grid": self.n_probes_outside_grid,
            "chain": self.chain,
            "segment": self.segment,
        }
        if self.value is None:
            return {**common, "note": self.unsampled_because}
        return {**common, "mean_kT_e": self.value}


def _candidates(
    grid: PotentialGrid,
    exclude_near: PQRData | None,
    exclusion_margin: float,
    exclude_mask: np.ndarray | None,
) -> np.ndarray:
    """Flat indices of the grid points a search may return."""
    if exclude_mask is not None and exclude_near is not None:
        raise ValueError("pass exclude_near or exclude_mask, not both — they can disagree")
    if exclude_mask is None:
        if exclude_near is None:
            return np.arange(grid.values.size)
        exclude_mask = solute_mask(grid, exclude_near, exclusion_margin)
    if exclude_mask.shape != grid.values.shape:
        raise ValueError(
            f"exclude_mask has shape {exclude_mask.shape}, grid has {grid.values.shape}"
        )
    return np.flatnonzero(~exclude_mask.reshape(-1))


def potential_extrema(
    grid: PotentialGrid,
    *,
    n: int = 5,
    most_positive: bool = True,
    min_separation: float = 5.0,
    min_fraction: float = 0.05,
    exclude_near: PQRData | None = None,
    exclusion_margin: float = DEFAULT_EXCLUSION_MARGIN_A,
    exclude_mask: np.ndarray | None = None,
) -> list[Extremum]:
    """The `n` strongest peaks, kept apart by `min_separation` angstroms.

    Naively taking the `n` largest grid points returns `n` neighbours of the
    same peak, which answers nothing — so each accepted extremum suppresses
    everything within `min_separation` of it. That distance is the knob that
    decides what counts as "a different site"; the default is roughly a
    sidechain's reach.

    `min_fraction` then drops anything weaker than that fraction of the
    strongest peak found. Without it a field with one real feature still returns
    `n` results, the rest being well-separated numerical noise at 1e-14 — five
    answers where the truthful answer is one. A caller asking for five peaks is
    asking "show me up to five", not "invent five".

    `exclude_mask` is the same exclusion, already built — what `solute_mask`
    returns. A caller that needs to *report* how much was excluded has to hold
    the mask anyway, and searching both signs rebuilds it twice otherwise, so
    passing it down is cheaper than the argument it replaces rather than merely
    tidier. Passing both is refused: two spellings of one exclusion can
    disagree, and the response would then describe the wrong one.

    `exclude_near` is what makes this answer the question people actually ask.
    The largest magnitudes in any map are the point-charge self-energy
    singularities at the atom centres — on a dipeptide they reach 500 kT/e —
    so an unfiltered search reliably reports "the extrema are at the atoms",
    which is true and useless. Passing the structure masks every point within
    each atom's radius plus `exclusion_margin` (a solvent probe by default),
    leaving the solvent-side features that correspond to binding sites.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if min_separation < 0:
        raise ValueError(f"min_separation must be non-negative, got {min_separation}")
    if not 0.0 <= min_fraction <= 1.0:
        raise ValueError(f"min_fraction must be within [0, 1], got {min_fraction}")

    flat = grid.values.reshape(-1)

    # Rank only the points that are candidates. Sorting the whole array and
    # sentinelling the excluded ones is tempting but fragile: NaN sorts last
    # ascending and therefore *first* once reversed for a descending search.
    candidates = _candidates(grid, exclude_near, exclusion_margin, exclude_mask)
    if candidates.size == 0:
        return []

    order = candidates[np.argsort(flat[candidates])]
    if most_positive:
        order = order[::-1]

    accepted: list[Extremum] = []
    positions: list[FloatArray] = []
    threshold: float | None = None
    for index in order:
        if len(accepted) >= n:
            break
        value = float(flat[index])
        if threshold is None:
            threshold = abs(value) * min_fraction
        elif abs(value) < threshold:
            break  # everything weaker is sorted after this; nothing left to find
        ijk = np.unravel_index(index, grid.values.shape)
        position = grid.origin + np.array(ijk) * grid.spacing
        if any(float(np.linalg.norm(position - taken)) < min_separation for taken in positions):
            continue
        positions.append(position)
        accepted.append(
            Extremum(
                position=(float(position[0]), float(position[1]), float(position[2])),
                value=value,
            )
        )
    return accepted


def solute_mask(grid: PotentialGrid, structure: PQRData, margin: float) -> np.ndarray:
    """True where a grid point lies inside any atom's radius plus `margin`.

    Each atom is tested only against the grid points inside its own bounding
    box. The obvious implementation — evaluate every atom against the whole
    grid — is O(atoms x grid points), which on real input means 1,960 atoms
    against 2.7M points: measured at 64 s for hen lysozyme, three times longer
    than the solve it was analysing. A cutoff sphere spans roughly a dozen cells
    per axis, so restricting to that is ~1000x less arithmetic for an identical
    result.
    """
    shape = grid.values.shape
    origin, spacing = grid.origin, grid.spacing
    mask = np.zeros(shape, dtype=bool)
    upper = np.array(shape)

    for index in range(structure.n_atoms):
        centre = structure.coords[index]
        cutoff = structure.radii[index] + margin

        lo = np.maximum(np.floor((centre - cutoff - origin) / spacing).astype(int), 0)
        hi = np.minimum(np.ceil((centre + cutoff - origin) / spacing).astype(int) + 1, upper)
        if np.any(lo >= hi):
            continue  # this atom's sphere misses the grid entirely

        axes = [origin[a] + np.arange(lo[a], hi[a]) * spacing[a] for a in range(DIMENSIONS)]
        xx, yy, zz = np.meshgrid(*axes, indexing="ij")
        inside = (
            (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (zz - centre[2]) ** 2
        ) <= cutoff**2
        mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] |= inside

    return mask


def _subset(structure: PQRData, indices: np.ndarray) -> PQRData:
    """The atoms at `indices`, as a `PQRData` the mask builder can take.

    Labels and chains are carried along so the result is a well-formed structure
    rather than a bag of coordinates; nothing here reads them, but returning
    something that only half-satisfies the type is how the next caller gets
    surprised. **Both are optional and both are subsetted**: `labels` is empty
    on a structure built in code, and `PQRData.__post_init__` rejects a `chains`
    tuple whose length does not cover the atoms — so passing the parent's
    through unsliced would raise on any real PQR that carries chains.
    """
    picked = [int(i) for i in indices]
    return PQRData(
        coords=structure.coords[indices],
        charges=structure.charges[indices],
        radii=structure.radii[indices],
        labels=tuple(structure.labels[i] for i in picked) if structure.labels else (),
        chains=tuple(structure.chains[i] for i in picked) if structure.chains else (),
    )


def potential_in_sphere(
    grid: PotentialGrid,
    centre: FloatArray,
    radius: float,
    *,
    exclude_near: PQRData | None = None,
    exclusion_margin: float = DEFAULT_EXCLUSION_MARGIN_A,
) -> Diagnostics:
    """Statistics over the grid points inside a sphere — a pocket, say.

    Reports `n_points` so a caller can tell an empty or barely-sampled region
    from a well-covered one; a mean over three points is not a mean.

    **Pass `exclude_near` for any sphere that contains atoms, which is every
    real pocket.** The strongest values in a map are the point-charge
    self-energy singularities at the atom centres — order 500 kT/e on a
    dipeptide — so a mean taken over a pocket without masking the solute is a
    mean of those singularities and not of the field a ligand would feel.
    `potential_extrema` learned this first; the same arithmetic decides a mean,
    and a mean hides it better than a maximum does, because nothing in the
    number looks wrong.

    The margin is the same 1.4 A default: a probe radius past the van der Waals
    surface, so the solvent-side shell a ligand actually occupies survives.
    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    centre = np.asarray(centre, dtype=float).reshape(DIMENSIONS)

    axes = [
        grid.origin[axis] + np.arange(grid.values.shape[axis]) * grid.spacing[axis]
        for axis in range(DIMENSIONS)
    ]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    distance_sq = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (zz - centre[2]) ** 2
    in_sphere = distance_sq <= radius**2
    n_in_sphere = int(in_sphere.sum())

    if not in_sphere.any():
        return {
            "n_points": 0,
            "centre": centre.tolist(),
            "radius_a": radius,
            "solute_masked": exclude_near is not None,
            "note": "no grid points fall inside this sphere",
        }

    inside = in_sphere
    if exclude_near is not None:
        # Only atoms whose own cutoff sphere reaches this query sphere can mask a
        # point in it, and on a protein that is a handful out of thousands. The
        # test is exact — an atom further than `radius + r_i + margin` from the
        # centre cannot contain any point within `radius` of it — so the mask is
        # identical, and `tests/test_analysis.py` pins that against the unfiltered
        # path. Measured on serum albumin at r = 5 A this is the difference
        # between ~0.9 s and a few milliseconds, on a tool an agent calls in a loop.
        reach = radius + exclude_near.radii + exclusion_margin
        near = np.flatnonzero(np.sum((exclude_near.coords - centre) ** 2, axis=1) <= reach**2)
        if near.size:
            inside = in_sphere & ~solute_mask(grid, _subset(exclude_near, near), exclusion_margin)

    n_excluded = n_in_sphere - int(inside.sum())
    common: Diagnostics = {
        "centre": centre.tolist(),
        "radius_a": radius,
        "solute_masked": exclude_near is not None,
        "n_points_in_sphere": n_in_sphere,
        "n_points_excluded_as_solute": n_excluded,
    }

    # Distinct from the empty-sphere case above, and the distinction is the
    # useful part: this sphere is *buried*. Returning a masked mean of nothing
    # would be a divide by zero; returning the unmasked mean would be the exact
    # singularity average this argument exists to prevent.
    if not inside.any():
        return {
            **common,
            "n_points": 0,
            "note": (
                f"all {n_in_sphere} grid points inside this sphere fall inside some "
                f"atom's radius plus the {exclusion_margin} A margin — the sphere is "
                "buried in the solute, so there is no solvent-side potential here to "
                "average. Move the centre outward, or widen the radius past the surface"
            ),
        }

    sampled = grid.values[inside]
    return {
        **common,
        "n_points": int(inside.sum()),
        "min_kT_e": float(sampled.min()),
        "max_kT_e": float(sampled.max()),
        "mean_kT_e": float(sampled.mean()),
        "std_kT_e": float(sampled.std()),
    }


def _golden_spiral(count: int) -> FloatArray:
    """`count` unit vectors spread evenly over the sphere.

    Six axis directions were the first scheme here and are no longer enough,
    for a reason that only appeared once probes started being rejected: a
    buried atom loses whichever of its six directions point into a neighbour,
    and with six there is often nothing left. Measured at the shipped margin,
    six directions leave **20 of 130 residues on 1a63 and 23 of 110 on
    barnase** with no usable probe at all, against 14 and 17 at twenty-six,
    and lift the median surviving probes per residue from 5-7 to 26.

    The spiral rather than a 3x3x3 shell because the axis-aligned set puts
    every probe on a lattice-parallel offset from its atom, which correlates
    the sampling with the grid it is interpolating on. Nothing here measured
    that correlation biting; avoiding it costs one line.
    """
    index = np.arange(count) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / count)
    azimuth = np.pi * (1.0 + 5.0**0.5) * index
    return np.stack(
        [np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)],
        axis=1,
    )


PROBE_DIRECTIONS = _golden_spiral(26)
"""Directions each atom is probed along. Twenty-six; `_golden_spiral` says why."""

_CELL_NEIGHBOURHOOD = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
)


def _buried_probes(
    points: FloatArray, owner: np.ndarray, structure: PQRData, margin: float
) -> np.ndarray:
    """True where a probe point lies inside some *other* atom's radius plus `margin`.

    `owner[p]` is the atom `points[p]` was placed around; an atom never buries
    its own probe. That exemption is not a check that cannot fail: at
    `probe_offset = 0` the probe sits exactly on its own radius, and on real
    coordinates the rounding in `centre + radius * direction` puts 24.1% of
    fas2's probes fractionally inside — measured — so without it a legal call
    returns nothing.

    Cost is why this is not the obvious double loop. Testing every probe
    against every atom is O(probes x atoms), which on serum albumin is 474,292
    probes against 18,242 atoms: measured at 9.8 s for a 20,000-probe
    subsample, so around four minutes at full size — the same shape as the
    64 s `_solute_mask` was written to avoid, on a tool an agent calls in a
    loop. Instead the probes are binned into cells one cutoff wide, and each
    atom is joined against the 27 cells its sphere can reach. The join is
    ragged, so it is expanded with `repeat` rather than looped: 27 vectorised
    passes over the whole structure, no Python loop over atoms at all. That is
    2.3 s on albumin, and it agrees with the naive oracle probe for probe.
    """
    cutoff = structure.radii + margin
    reach = float(cutoff.max())
    if reach <= 0 or len(points) == 0:
        return np.zeros(len(points), dtype=bool)

    lower = points.min(axis=0) - reach
    counts = np.maximum(np.ceil((points.max(axis=0) + reach - lower) / reach).astype(int), 1)
    cell_of = np.clip(((points - lower) / reach).astype(int), 0, counts - 1)
    keys = np.ravel_multi_index(cell_of.T, counts)
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]

    home = np.clip(((structure.coords - lower) / reach).astype(int), 0, counts - 1)
    buried = np.zeros(len(points), dtype=bool)
    atoms = np.arange(structure.n_atoms)

    for offset in _CELL_NEIGHBOURHOOD:
        cells = home + offset
        legal = np.all((cells >= 0) & (cells < counts), axis=1)
        if not legal.any():
            continue
        here = np.ravel_multi_index(cells[legal].T, counts)
        starts = np.searchsorted(ordered, here, side="left")
        ends = np.searchsorted(ordered, here, side="right")
        widths = ends - starts
        if not widths.any():
            continue

        # Ragged gather: `repeat` lays out one slot per (atom, probe) pair, and
        # subtracting each pair's own run start turns a global arange into an
        # offset within its run.
        atom_of_pair = np.repeat(atoms[legal], widths)
        run_start = np.repeat(np.cumsum(widths) - widths, widths)
        within = np.arange(int(widths.sum())) - run_start
        probe_of_pair = order[np.repeat(starts, widths) + within]

        gap = points[probe_of_pair] - structure.coords[atom_of_pair]
        hit = np.einsum("ij,ij->i", gap, gap) < cutoff[atom_of_pair] ** 2
        hit &= owner[probe_of_pair] != atom_of_pair
        buried[probe_of_pair[hit]] = True

    return buried


def residue_potentials(
    grid: PotentialGrid,
    structure: PQRData,
    *,
    probe_offset: float = 2.0,
    exclusion_margin: float = DEFAULT_EXCLUSION_MARGIN_A,
    top: int | None = None,
) -> list[ResiduePotential]:
    """Mean potential in the solvent around each residue, most negative first.

    Sampled at `probe_offset` angstroms *outside* each atom's radius rather
    than at the atom centre: the potential at a point charge is dominated by
    its own self-energy, so atom-centre values report the atom, not its
    environment.

    **Stepping outside the atom's own radius is not the same as stepping into
    solvent**, and until this was measured the difference was the whole
    defect. A probe 2 A beyond atom `i` lands inside atom `j` constantly —
    **55.9% of probes on fas2, 61.4% on 1a63, 63.2% on serum albumin** — so a
    per-residue mean was a majority-interior sample being reported as "its
    environment". `exclude_near` fixed the same arithmetic in
    `potential_extrema` and `potential_in_sphere`; this is the third sibling.

    `exclusion_margin` is what makes the rejection mean solvent rather than
    merely "outside a radius", and 1.4 A is not a fudge factor. The shipped
    surface model is `molecular`, whose interior is the solvent-*excluded*
    volume — strictly larger than the union of van der Waals spheres, because
    the re-entrant crevices between atoms are solute while lying outside every
    radius. Rejecting at the bare radius leaves **21.6% of the survivors on
    fas2 still inside the solute of the map they are reading** (measured
    against `ReducedSurface.inside`, the solver's own oracle), and those
    points carry mean |phi| of 8.75 kT/e against 1.15 for genuine solvent —
    the contamination this function exists to remove, one layer in. A point
    at least `r_j + p` from every atom centre is a legal position for a probe
    of radius `p`, so it is outside the solvent-excluded surface *for any*
    surface model with that probe or smaller. At 1.4 A — the same default the
    two sibling tools take — **0 of the 535 survivors on fas2 are interior**.
    That is a proof rather than a fit, which is why it is the default and why
    it does not need to know which surface the map was solved with.

    It is conservative by construction: it also rejects the genuinely solvent
    shell between the excluded and accessible surfaces. Pass a smaller margin
    to sample closer in, with the understanding that below the solve's own
    probe radius the rejection stops being a proof.

    Residues are grouped as `_residue_groups` describes — by contiguous run,
    because `"<resName> <resSeq>"` is not unique across chains. A residue
    whose probes are *all* rejected keeps its row with `value = None` and a
    note saying which of the two nothings it is, rather than vanishing from a
    list a caller reads as a ranking.

    **What this number still is not.** It is a mean over the solvent points
    that survived, which on a protein is a minority of those offered and sits
    preferentially on the exposed side of each atom. It describes the field a
    solvent-side probe would meet near the residue. It is not a
    surface-area-weighted average, not an interaction energy, and not
    comparable across structures solved on different grids.
    """
    if not structure.labels:
        raise ValueError(
            "structure has no per-atom labels, so residues cannot be grouped; "
            "read it from a PQR that carries them"
        )
    if len(structure.labels) != structure.n_atoms:
        raise ValueError("labels must cover every atom")
    if probe_offset < 0:
        raise ValueError(f"probe_offset must be non-negative, got {probe_offset}")
    if exclusion_margin < 0:
        raise ValueError(f"exclusion_margin must be non-negative, got {exclusion_margin}")

    per_atom_probes = len(PROBE_DIRECTIONS)
    reach = (structure.radii + probe_offset)[:, None, None]
    points = (structure.coords[:, None, :] + reach * PROBE_DIRECTIONS[None, :, :]).reshape(-1, 3)
    owner = np.repeat(np.arange(structure.n_atoms), per_atom_probes)

    sampled = grid.value_at(points)
    outside_grid = np.isnan(sampled)
    # Off the map is charged first, and the order is a real choice. A probe
    # that was never on the grid was not rejected as solute, it was missing --
    # and on a map that does not cover the structure, charging it the other way
    # over-reports masking severalfold on exactly the input where these counts
    # are what tells the caller what went wrong.
    excluded = _buried_probes(points, owner, structure, exclusion_margin) & ~outside_grid
    usable = ~outside_grid & ~excluded

    shape = (structure.n_atoms, per_atom_probes)
    sampled = sampled.reshape(shape)
    outside_grid = outside_grid.reshape(shape)
    excluded = excluded.reshape(shape)
    usable = usable.reshape(shape)

    groups = _residue_groups(structure)
    results = []
    for group, label in zip(groups, _labelled(groups), strict=True):
        indices = np.array(group.indices, dtype=int)
        # Per atom first, then across atoms, so a well-exposed atom does not
        # outvote the rest of its residue by carrying more surviving probes.
        means = [float(sampled[i][usable[i]].mean()) for i in indices if bool(usable[i].any())]
        results.append(
            ResiduePotential(
                label=label,
                value=float(np.mean(means)) if means else None,
                n_atoms=len(group.indices),
                n_probes=len(group.indices) * per_atom_probes,
                n_probes_used=int(usable[indices].sum()),
                n_probes_excluded_as_solute=int(excluded[indices].sum()),
                n_probes_outside_grid=int(outside_grid[indices].sum()),
                chain=group.chain,
                segment=group.segment,
            )
        )
    # Unsampled rows sort last rather than being dropped: `top=N` should spend
    # its N on residues that have a number.
    results.sort(key=lambda r: (r.value is None, r.value if r.value is not None else 0.0))
    return results[:top] if top is not None else results


_RESIDUE_FIELDS = 2  # "<resName> <resSeq>" out of "<resName> <resSeq> <atomName>"


def _residue_of(label: str) -> str:
    """`"ALA 1 CA"` -> `"ALA 1"`. Labels without a sequence number pass through."""
    parts = label.split()
    return " ".join(parts[:_RESIDUE_FIELDS]) if len(parts) >= _RESIDUE_FIELDS else label


def _sequence_number(residue: str) -> int | None:
    """The `58` in `"SER 58"`, or None when it is not a plain integer."""
    parts = residue.split()
    if len(parts) < _RESIDUE_FIELDS:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None  # insertion codes ("58A") and other writers' conventions


@dataclass
class _Group:
    """One residue's atoms, as a contiguous run of the file.

    Unfrozen, unlike everything else here: it is filled in as the run is walked.
    """

    residue: str  # "SER 58"
    chain: str | None
    segment: int
    indices: list[int] = field(default_factory=list)


def _residue_groups(structure: PQRData) -> list[_Group]:
    """Partition atoms into residues, without assuming a chain column exists.

    `"<resName> <resSeq>"` is **not** a residue identifier. Residue 58 of chain
    A and residue 58 of chain B share it, and on serum albumin that collapsed
    1,156 residues to 578 keys, the worst averaging 22 atoms spread over 115 A
    and reporting one number for them.

    The chain would settle it, but it is routinely absent — pdb2pqr drops it
    unless asked, so the structures this tool is pointed at are exactly the ones
    that cannot be keyed that way. What holds regardless is that **one residue's
    atoms are contiguous in the file**, so a new residue begins wherever
    `(resName, resSeq, chain)` differs from the atom before it. That needs no
    chain IDs and invents none.

    Where the file does carry chains they are reported as read. Where it does
    not, a *segment* ordinal stands in: 1-based, incremented at each numbering
    restart. It is an inference from a `resSeq` that failed to advance, not a
    chain ID, and the label spells it `#2` rather than `B` so it cannot be
    mistaken for one. On every multi-chain structure in `tests/data` the two
    agree on the partition; the run is the primitive and the segment is only how
    the group is named.

    The boundary: two *single-residue* chains, adjacent in the file, identically
    numbered and unnamed. The run boundary then coincides with nothing, and only
    the coordinates say there are two — which is why `prepare_structure` asks
    pdb2pqr to keep the chain rather than relying on this. Splitting on distance
    instead would mean choosing a threshold the file gives no basis for.
    """
    chains = structure.chains
    groups: list[_Group] = []
    previous: tuple[str, str | None] | None = None
    previous_number: int | None = None
    segment = 1

    for index in range(structure.n_atoms):
        residue = _residue_of(structure.labels[index])
        chain = (chains[index] or None) if chains else None
        number = _sequence_number(residue)

        restarted = previous is not None and (
            chain != previous[1]
            or (number is not None and previous_number is not None and number < previous_number)
        )
        if restarted:
            segment += 1
        if (residue, chain) != previous:
            groups.append(_Group(residue=residue, chain=chain, segment=segment))
            previous = (residue, chain)
        groups[-1].indices.append(index)
        if number is not None:
            previous_number = number

    return groups


def _labelled(groups: list[_Group]) -> list[str]:
    """Name each group so that no two residues answer to the same string.

    The prefix appears only where there is something to disambiguate. A
    single-chain structure keeps the bare `"SER 58"` it has always reported —
    whether or not the file names its chain, since naming the one chain every
    atom belongs to distinguishes nothing and would churn every label a caller
    has recorded. `ResiduePotential.chain` still reports it.
    """
    multi_chain = len({g.chain for g in groups}) > 1
    multi_segment = len({g.segment for g in groups}) > 1
    labels = []
    for group in groups:
        if multi_chain and group.chain is not None:
            labels.append(f"{group.chain}:{group.residue}")
        elif multi_segment:
            labels.append(f"#{group.segment}:{group.residue}")
        else:
            labels.append(group.residue)

    # A file whose chains interleave can still collide — the same chain and the
    # same number twice, in two runs. Rare enough that no structure in the repo
    # does it, possible enough that the alternative is two rows claiming to be
    # the same residue.
    seen: dict[str, int] = {}
    duplicated = {label for label in labels if labels.count(label) > 1}
    for position, label in enumerate(labels):
        if label in duplicated:
            seen[label] = seen.get(label, 0) + 1
            labels[position] = f"{label}~{seen[label]}"
    return labels
