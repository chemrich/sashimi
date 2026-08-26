"""How many band nodes does `_union_gap` get wrong, and by how much a lower bound?

A band node's naive gap is exact iff the foot point it implies -- the radial
projection onto the sphere that won the minimum -- is exposed. Buried foot point
=> the true depth is strictly larger, so the node is somewhere it should not be
in the ramp. This is a lower bound on the wrong set: it does not price how much
larger.
"""

from __future__ import annotations

import sys

import numpy as np

from sashimi.debye.grid import axis_coordinates, size_grid
from sashimi.debye.surface import _union_gap
from sashimi.pqr import parse_pqr
from sashimi.protocol import GridSpec

path = sys.argv[1]
resolution = float(sys.argv[2])
cells = float(sys.argv[3])

pqr = parse_pqr(open(path).read())
grid = size_grid(pqr, GridSpec(resolution=resolution, padding=10.0))
width = cells * float(min(grid.spacing))
print(
    f"{path}  atoms={len(pqr.radii)}  shape={grid.shape} "
    f" spacing={grid.spacing}  width={width:.4f} A"
)

coords, radii = pqr.coords, pqr.radii
total_band = total_wrong = 0
for axis in range(3):
    axes = axis_coordinates(grid, staggered=axis)
    gap = _union_gap(axes, coords, radii)
    band = (gap > -width) & (gap < 0.0)
    idx = np.nonzero(band)
    pts = np.stack([axes[0][idx[0]], axes[1][idx[1]], axes[2][idx[2]]], axis=1)
    # which sphere won
    best = np.full(len(pts), np.inf)
    who = np.zeros(len(pts), dtype=np.int64)
    for i, (c, r) in enumerate(zip(coords, radii)):
        d = np.linalg.norm(pts - c, axis=1) - r
        hit = d < best
        best[hit] = d[hit]
        who[hit] = i
    c = coords[who]
    r = radii[who]
    v = pts - c
    n = np.linalg.norm(v, axis=1)
    foot = c + r[:, None] * v / np.maximum(n, 1e-12)[:, None]
    buried = np.zeros(len(pts), dtype=bool)
    for i, (cc, rr) in enumerate(zip(coords, radii)):
        near = np.linalg.norm(foot - cc, axis=1) < rr - 1e-9
        buried |= near & (who != i)
    total_band += len(pts)
    total_wrong += int(buried.sum())
    print(
        f"  axis {axis}: band faces {len(pts):8d}   foot point buried {int(buried.sum()):8d}"
        f"  ({100.0 * buried.mean():.2f}%)"
    )
print(
    f"TOTAL band faces {total_band}, provably wrong {total_wrong} "
    f"({100.0 * total_wrong / max(total_band, 1):.2f}%)"
)
