"""Prototype: a real two-sided signed distance to a union of spheres.

Outside the union `min_i(|x - c_i| - r_i)` is exact. Inside it is only an upper
bound, because the sphere surface it measures to may be buried under another
sphere. The correct interior value is `-dist(x, U^c)`, and the three reduced
surface families already compute exactly that quantity -- for the *inflated*
spheres, with `probe` as the reach. Rebuild them with a zero inflation and the
band as the reach and they answer the van der Waals question instead.
"""

from __future__ import annotations

import numpy as np

from sashimi.debye import surface as S


class _ZeroProbeSurface:
    """What `_toroidal_distance` and `_vertex_distance` read off a surface."""

    def __init__(self, structure, reach: float) -> None:
        self.structure = structure
        self.probe = reach  # the families use `probe` only as their search reach
        self.spheres = S._Spheres.around(structure, 0.0)
        self.rims = S._rims(self.spheres)
        self.seats = S._probe_seats(self.spheres)
        rims = self.rims
        self.rim_arrays = (
            np.array([o for o, _, _, _ in rims]).reshape(-1, 3),
            np.array([n for _, n, _, _ in rims]).reshape(-1, 3),
            np.array([r for _, _, r, _ in rims]),
        )
        self.blocker_table = S._blocker_table(rims)


def corrected_union_gap(structure, axes, band: float | None = None, reach: float | None = None):
    """`_union_gap` with the interior repaired inside `band`."""
    coords, radii = structure.coords, structure.radii
    gap = S._union_gap(axes, coords, radii)
    # Only nodes the ramp can still observe need a real value: the true depth is
    # never smaller than the naive one, so a node already past the band stays past it.
    region = gap < 0.0
    if band is not None:
        region = region & (gap > -band)
    if not region.any():
        return gap
    look = reach if reach is not None else (band if band is not None else float(radii.max()))
    shim = _ZeroProbeSurface(structure, look)
    far = 4.0 * look
    depth = np.where(region, far, 0.0).astype(np.float64)
    S._radial_distance(axes, shim.spheres, region, depth)
    S._toroidal_distance(axes, shim, region, depth)
    S._vertex_distance(axes, shim, region, depth)
    np.minimum(depth, far, out=depth)
    return np.where(region, -depth, gap)
