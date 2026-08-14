"""Sampling a potential map around a sphere, in the directions the error varies over.

One owner for a rule two engines need. `sashimi.corpus` grades one backend's
field against a closed form; `sashimi.validate` grades backends against each
other. Both have to sample the same way or their numbers are not about the same
thing, and a direction set copied into two modules is the parallel list
ROADMAP.md section 7 keeps finding — the kind where sixteen recordings join one
copy and nothing notices.

**Why a set of directions rather than a ray.** The solution around a Born ion is
spherically symmetric; the *error* is not. A sphere discretized on a Cartesian
grid is a staircase, and a staircase has the grid's cubic symmetry, so the
discretization error varies over three direction classes — <100> along the axes,
<110> through the face diagonals, <111> through the body diagonals. Sampling one
ray measures whichever class it happens to lie in, and **which class is worst
depends on the solver**: at 0.25 A on a 3 A sphere APBS is worst along the axes
(1.019%) while DelPhi C++ is worst along the body diagonal (1.890%), reading only
0.736% on the axis a single-ray check would have used.

**Why the radius is in grid cells.** Interpolating across the dielectric
interface is O(1) wrong by construction — the potential is continuous there and
its normal derivative is not, and at eps_s/eps_p ~ 78.5 the gradient jumps by
nearly two orders of magnitude. A sample has to land in a cell that does not
contain the boundary, which `a + k*h` guarantees for every radius and a fixed
fraction of `a` does not.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sashimi.protocol import DIMENSIONS, FloatArray, PotentialGrid

__all__ = [
    "FIELD_DIRECTIONS",
    "FIELD_DIRECTION_NAMES",
    "MIN_CELLS_OUT",
    "errors_by_radius",
    "sample_points",
    "sample_radii",
    "sample_values",
    "worst_error_over_directions",
]

# Samples closer than this to the boundary sit in, or share a stencil corner
# with, the cell the interface passes through.
MIN_CELLS_OUT = 2


def _unit(x: float, y: float, z: float) -> FloatArray:
    vector = np.array([x, y, z], dtype=float)
    return np.asarray(vector / float(np.linalg.norm(vector)), dtype=np.float64)


# The three cubic symmetry classes, two of them with a sign-flipped
# representative. Those two are redundant for a sphere centred on a node — which
# is the point: they are what would catch a grid centred half a cell off, a
# defect that moves every sample equally and so leaves the *worst* error, and
# every tolerance derived from it, untouched.
FIELD_DIRECTIONS: tuple[tuple[str, FloatArray], ...] = (
    ("+x", _unit(1, 0, 0)),
    ("+y", _unit(0, 1, 0)),
    ("+z", _unit(0, 0, 1)),
    ("-x", _unit(-1, 0, 0)),
    ("110", _unit(1, 1, 0)),
    ("011", _unit(0, 1, 1)),
    ("111", _unit(1, 1, 1)),
    ("-1-1-1", _unit(-1, -1, -1)),
)

FIELD_DIRECTION_NAMES: tuple[str, ...] = tuple(name for name, _ in FIELD_DIRECTIONS)


def sample_radii(radius_a: float, spacing: float, cells_out: tuple[int, ...]) -> list[float]:
    """`a + k*h`, the radii at which a field may be compared with a closed form."""
    if not cells_out:
        raise ValueError(
            "cells_out is empty, so there is nothing to sample. A comparison built from "
            "no samples is not a lenient one — every all()-shaped verdict over it reads "
            "as agreement."
        )
    if any(k < MIN_CELLS_OUT for k in cells_out):
        raise ValueError(
            f"cells_out={cells_out} would sample within {MIN_CELLS_OUT} cells of the "
            "boundary, where interpolation crosses the dielectric interface and is "
            "O(1) wrong for every solver"
        )
    return [radius_a + k * spacing for k in cells_out]


def sample_points(centre: FloatArray, radius: float) -> FloatArray:
    """The eight direction samples at one radius, (8, 3) in A."""
    centre = np.asarray(centre, dtype=float).reshape(DIMENSIONS)
    return np.array([centre + direction * radius for _, direction in FIELD_DIRECTIONS])


def sample_values(grid: PotentialGrid, centre: FloatArray, radius: float) -> FloatArray:
    """The eight direction values at one radius, refusing any that fell off the map.

    A NaN is an error rather than a skipped point: `PotentialGrid.value_at`
    returns NaN off-grid precisely so it cannot be mistaken for a measurement,
    and both a silently dropped direction and a `NaN` written into a recording
    report the best of whatever remained.
    """
    values = grid.value_at(sample_points(centre, radius))
    if not np.isfinite(values).all():
        finite = np.isfinite(values)
        outside = [name for name, ok in zip(FIELD_DIRECTION_NAMES, finite, strict=True) if not ok]
        raise ValueError(
            f"the map does not cover r = {radius:.4g} A along {outside}; a field "
            "comparison cannot silently drop the directions that fell off the grid"
        )
    return values


def worst_error_over_directions(
    grid: PotentialGrid,
    centre: FloatArray,
    radius: float,
    exact: float,
) -> tuple[float, str]:
    """Largest relative error at `radius`, and the direction it was found in.

    A NaN — a sample outside the map — is an error rather than a skipped point:
    `PotentialGrid.value_at` returns NaN off-grid precisely so it cannot be
    mistaken for a measurement, and silently dropping it would report the best
    of whatever remained.
    """
    values = sample_values(grid, centre, radius)
    errors = np.abs(values - exact) / abs(exact)
    worst = int(np.argmax(errors))
    return float(errors[worst]), FIELD_DIRECTION_NAMES[worst]


def errors_by_radius(
    grid: PotentialGrid,
    centre: FloatArray,
    radii: list[float],
    exact_at: Callable[[float], float],
) -> tuple[list[float], list[str]]:
    """Worst-direction error at each radius, and where each was found."""
    errors, directions = [], []
    for radius in radii:
        error, direction = worst_error_over_directions(grid, centre, radius, exact_at(radius))
        errors.append(error)
        directions.append(direction)
    return errors, directions
