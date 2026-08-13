"""Physical grid intent -> a Cartesian lattice debye can coarsen.

`GridSpec` has no `dime` because legal multigrid dimensions are an APBS
implementation detail (ROADMAP.md section 4). debye has its own, and it is a
looser one: APBS's mg-auto requires n = c * 2^(l+1) + 1 with l = 4, so its
ladder steps in 32s — 97, 129, 161. debye coarsens as far as the arithmetic
allows rather than a fixed number of times, so its ladder steps in 8s, and the
practical consequence is that it lands nearer the resolution actually asked
for. The Born gate case is the example: 26 A of box at 0.25 A wants 105 points,
which APBS rounds up to 129 (h = 0.203 A) and debye meets exactly (h = 0.25 A).

Nearer is not better here — APBS is solving on a finer grid than it was asked
for, which costs time and buys accuracy — and that is the point of recording
`spacing_achieved` rather than `resolution`. Two backends handed the same
`GridSpec` are not solving on the same lattice, and a comparison that assumes
they are is comparing discretizations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sashimi.errors import GridTooLarge
from sashimi.protocol import DIMENSIONS, Diagnostics, FloatArray, GridSpec, PQRData

__all__ = [
    "LATTICE_STEP",
    "MIN_COARSE_POINTS",
    "DebyeGrid",
    "coarsen",
    "grid_hierarchy",
    "size_grid",
]

# n = m * LATTICE_STEP + 1. The step is what fixes how many times the grid can
# be halved before an axis stops being even, and three levels is the floor this
# guarantees for every grid it produces.
LATTICE_STEP = 8

# Coarsening stops before an axis gets smaller than this. Below ~5 nodes a
# vertex-centred grid has one interior point per axis and the coarse solve is
# no longer approximating the fine operator's smooth modes.
MIN_COARSE_POINTS = 5

# Smallest grid worth solving on at all. Two lattice steps, so every grid this
# module returns can be coarsened at least twice.
MIN_POINTS_PER_AXIS = 2 * LATTICE_STEP + 1


@dataclass(frozen=True)
class DebyeGrid:
    """A uniform Cartesian grid, and what it achieved against what was asked.

    Node (i, j, k) sits at `origin + (i, j, k) * spacing`, so `origin` is the
    corner and not the centre — the same convention `PotentialGrid` uses, which
    is what lets the result be handed back without a translation nobody would
    see.
    """

    shape: tuple[int, int, int]
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    center: tuple[float, float, float]

    @property
    def n_points(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def levels(self) -> int:
        """How many times this grid can be halved, counting the fine grid as one."""
        shape = np.array(self.shape)
        levels = 1
        while np.all((shape - 1) % 2 == 0) and np.all((shape - 1) // 2 + 1 >= MIN_COARSE_POINTS):
            shape = (shape - 1) // 2 + 1
            levels += 1
        return levels

    def node_coordinates(self, axis: int) -> FloatArray:
        """The physical coordinates along one axis, (n,) in A."""
        return self.origin[axis] + self.spacing[axis] * np.arange(self.shape[axis], dtype=float)

    def as_diagnostics(self) -> Diagnostics:
        return {
            "shape": list(self.shape),
            "origin": [round(v, 4) for v in self.origin],
            "spacing_achieved": [round(v, 6) for v in self.spacing],
            "center": [round(v, 4) for v in self.center],
            "n_points": self.n_points,
            "multigrid_levels": self.levels,
        }


def _lattice_ceil(minimum: int) -> int:
    """Smallest n = m * LATTICE_STEP + 1 that is at least `minimum`."""
    n = max(minimum, MIN_POINTS_PER_AXIS)
    steps = -(-(n - 1) // LATTICE_STEP)  # ceiling division
    return steps * LATTICE_STEP + 1


def size_grid(pqr: PQRData, spec: GridSpec) -> DebyeGrid:
    """The lattice a request resolves to, with `max_points` honoured.

    Sized like APBS's fine grid — the molecule's radius-inflated extent plus
    `padding` on each side — because the boundary condition is the same
    approximation: Debye-Huckel evaluated on the box face, which is only as good
    as its distance from the solute. There is no coarse grid and no focusing,
    which is the one place debye is structurally simpler than both incumbents:
    the box is solved directly, so `padding` is the whole boundary story.
    """
    extent = pqr.extent()
    center = pqr.center()
    fglen = extent + 2.0 * spec.padding

    needed = np.ceil(fglen / spec.resolution).astype(int) + 1
    shape = np.array([_lattice_ceil(int(v)) for v in needed])

    # Step the largest axis down until the budget is met, which keeps the grid
    # as isotropic as the budget allows — the same rule `apbs.grid` follows, for
    # the same reason: an anisotropic grid is worse at the axis it starved.
    while int(np.prod(shape)) > spec.max_points:
        axis = int(np.argmax(shape))
        if shape[axis] - LATTICE_STEP < MIN_POINTS_PER_AXIS:
            raise GridTooLarge(
                f"cannot satisfy max_points={spec.max_points:,} for a molecule of extent "
                f"{np.round(extent, 1).tolist()} A with padding={spec.padding} A; "
                f"the smallest grid debye will solve on is "
                f"{'x'.join(str(int(v)) for v in shape)} = {int(np.prod(shape)):,} points. "
                "Raise max_points or reduce padding."
            )
        shape[axis] -= LATTICE_STEP

    spacing = fglen / (shape - 1)
    origin = center - fglen / 2.0
    return DebyeGrid(
        shape=tuple(int(v) for v in shape),  # type: ignore[arg-type]
        origin=tuple(float(v) for v in origin),  # type: ignore[arg-type]
        spacing=tuple(float(v) for v in spacing),  # type: ignore[arg-type]
        center=tuple(float(v) for v in center),  # type: ignore[arg-type]
    )


def coarsen(grid: DebyeGrid) -> DebyeGrid:
    """The next grid up the multigrid hierarchy: every other node, same box.

    The box is preserved exactly — same origin, same extent, twice the spacing —
    so a coarse node coincides with a fine node and the geometry it samples is
    the same geometry. That is what makes re-discretizing the dielectric on each
    level legitimate rather than a second approximation.
    """
    shape = np.array(grid.shape)
    if np.any((shape - 1) % 2 != 0):
        raise ValueError(f"grid {grid.shape} cannot be halved on every axis")
    coarse_shape = (shape - 1) // 2 + 1
    return DebyeGrid(
        shape=tuple(int(v) for v in coarse_shape),  # type: ignore[arg-type]
        origin=grid.origin,
        spacing=tuple(2.0 * s for s in grid.spacing),  # type: ignore[arg-type]
        center=grid.center,
    )


def grid_hierarchy(grid: DebyeGrid) -> list[DebyeGrid]:
    """Fine to coarsest, stopping where `MIN_COARSE_POINTS` says to."""
    levels = [grid]
    for _ in range(grid.levels - 1):
        levels.append(coarsen(levels[-1]))
    return levels


def axis_coordinates(grid: DebyeGrid, *, staggered: int | None = None) -> list[FloatArray]:
    """Per-axis coordinate vectors, for broadcasting rather than meshgrid.

    With `staggered=axis`, that axis carries the face centres between nodes —
    one shorter, offset by half a cell — which is where the dielectric lives.
    Three 1-D vectors broadcast against each other is what keeps a 129^3
    dielectric map from ever materialising three full coordinate arrays.
    """
    coords = []
    for axis in range(DIMENSIONS):
        n = grid.shape[axis] - 1 if axis == staggered else grid.shape[axis]
        offset = 0.5 if axis == staggered else 0.0
        values = grid.origin[axis] + grid.spacing[axis] * (np.arange(n) + offset)
        coords.append(np.asarray(values, dtype=np.float64))
    return coords
