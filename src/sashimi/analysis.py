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

from dataclasses import dataclass

import numpy as np

from sashimi.protocol import DIMENSIONS, Diagnostics, FloatArray, PotentialGrid, PQRData

__all__ = [
    "Extremum",
    "ResiduePotential",
    "potential_extrema",
    "potential_in_sphere",
    "residue_potentials",
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
    """Mean potential sampled around one residue's atoms."""

    label: str
    value: float  # kT/e
    n_atoms: int
    n_sampled: int  # atoms whose probe points fell inside the grid

    def as_dict(self) -> Diagnostics:
        return {
            "residue": self.label,
            "mean_kT_e": self.value,
            "n_atoms": self.n_atoms,
            "n_sampled": self.n_sampled,
        }


def potential_extrema(
    grid: PotentialGrid,
    *,
    n: int = 5,
    most_positive: bool = True,
    min_separation: float = 5.0,
    min_fraction: float = 0.05,
    exclude_near: PQRData | None = None,
    exclusion_margin: float = 1.4,
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
    if exclude_near is not None:
        candidates = np.flatnonzero(~_solute_mask(grid, exclude_near, exclusion_margin).reshape(-1))
    else:
        candidates = np.arange(flat.size)
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


def _solute_mask(grid: PotentialGrid, structure: PQRData, margin: float) -> np.ndarray:
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


def potential_in_sphere(
    grid: PotentialGrid,
    centre: FloatArray,
    radius: float,
) -> Diagnostics:
    """Statistics over the grid points inside a sphere — a pocket, say.

    Reports `n_points` so a caller can tell an empty or barely-sampled region
    from a well-covered one; a mean over three points is not a mean.
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
    inside = distance_sq <= radius**2

    if not inside.any():
        return {
            "n_points": 0,
            "centre": centre.tolist(),
            "radius_a": radius,
            "note": "no grid points fall inside this sphere",
        }

    sampled = grid.values[inside]
    return {
        "n_points": int(inside.sum()),
        "centre": centre.tolist(),
        "radius_a": radius,
        "min_kT_e": float(sampled.min()),
        "max_kT_e": float(sampled.max()),
        "mean_kT_e": float(sampled.mean()),
        "std_kT_e": float(sampled.std()),
    }


def residue_potentials(
    grid: PotentialGrid,
    structure: PQRData,
    *,
    probe_offset: float = 2.0,
    top: int | None = None,
) -> list[ResiduePotential]:
    """Mean potential near each residue, most negative first.

    Sampled at `probe_offset` angstroms *outside* each atom's radius rather than
    at the atom centre: the potential at a point charge is dominated by its own
    self-energy, so atom-centre values report the atom, not its environment.

    Residues are grouped by the structure's labels. Atoms whose probe points
    fall outside the grid are skipped and counted, so a residue at the box edge
    is visibly under-sampled rather than quietly wrong.
    """
    if not structure.labels:
        raise ValueError(
            "structure has no per-atom labels, so residues cannot be grouped; "
            "read it from a PQR that carries them"
        )
    if len(structure.labels) != structure.n_atoms:
        raise ValueError("labels must cover every atom")

    # Six probe points per atom, on the axes, just outside the vdW radius.
    directions = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float
    )

    grouped: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    sampled_counts: dict[str, int] = {}

    for index in range(structure.n_atoms):
        residue = _residue_of(structure.labels[index])
        counts[residue] = counts.get(residue, 0) + 1

        offset = structure.radii[index] + probe_offset
        probes = structure.coords[index] + directions * offset
        values = grid.value_at(probes)
        usable = values[~np.isnan(values)]
        if usable.size == 0:
            continue
        grouped.setdefault(residue, []).append(float(usable.mean()))
        sampled_counts[residue] = sampled_counts.get(residue, 0) + 1

    results = [
        ResiduePotential(
            label=residue,
            value=float(np.mean(values)),
            n_atoms=counts[residue],
            n_sampled=sampled_counts.get(residue, 0),
        )
        for residue, values in grouped.items()
    ]
    results.sort(key=lambda r: r.value)
    return results[:top] if top is not None else results


_RESIDUE_FIELDS = 2  # "<resName> <resSeq>" out of "<resName> <resSeq> <atomName>"


def _residue_of(label: str) -> str:
    """`"ALA 1 CA"` -> `"ALA 1"`. Labels without a sequence number pass through."""
    parts = label.split()
    return " ".join(parts[:_RESIDUE_FIELDS]) if len(parts) >= _RESIDUE_FIELDS else label
