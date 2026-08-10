"""Physical grid intent -> legal DelPhi grid parameters.

DelPhi's grid is described by two numbers: `scale`, the number of grid points
per angstrom, and `gsize`, the number of points along each side. The box is
cubic and `gsize` must be odd, so there is exactly one degree of freedom per
axis and no multigrid lattice to respect — the contrast with APBS, whose
`dime` must satisfy n = c*2^(l+1)+1, is the clearest evidence that `GridSpec`
was right to carry `resolution` and `padding` rather than a grid shape.

sashimi always sets `gsize` and `scale` explicitly instead of using DelPhi's
`perfil` (fill the box to a percentage of its extent). Two reasons: `perfil`
makes the resolution a consequence of the molecule's size rather than a
request, and cross-solver validation needs the two backends on grids that were
derived the same way from the same physical intent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sashimi.errors import GridTooLarge
from sashimi.protocol import Diagnostics, GridSpec, PQRData

__all__ = ["DelphiGrid", "odd_gsize", "size_grid"]

MIN_GSIZE = 5  # below this the box holds no interior at all


def odd_gsize(minimum: float) -> int:
    """Smallest odd integer >= `minimum`, floored at `MIN_GSIZE`.

    DelPhi requires an odd grid so that a point sits exactly at the box centre;
    an even grid puts the molecule's centre between samples.
    """
    n = max(MIN_GSIZE, int(np.ceil(minimum)))
    return n if n % 2 == 1 else n + 1


@dataclass(frozen=True)
class DelphiGrid:
    """Everything the parameter file needs, plus what was actually achieved."""

    gsize: int
    scale: float  # grid points per angstrom
    center: tuple[float, float, float]
    box_length: float  # (gsize - 1) / scale, A

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Achieved spacing, A. Cubic by construction, but reported per axis so
        callers can treat every backend's grid the same way."""
        h = 1.0 / self.scale
        return (h, h, h)

    @property
    def n_points(self) -> int:
        return self.gsize**3

    def as_diagnostics(self) -> Diagnostics:
        return {
            "gsize": self.gsize,
            "scale": round(self.scale, 6),
            "center": [round(v, 4) for v in self.center],
            "box_length": round(self.box_length, 4),
            "spacing_achieved": [round(v, 6) for v in self.spacing],
            "n_points": self.n_points,
        }


def size_grid(pqr: PQRData, spec: GridSpec) -> DelphiGrid:
    """Turn a `GridSpec` into a legal DelPhi grid, relaxing resolution if capped.

    The box is cubic, so the longest molecular axis sets the side length for all
    three. That wastes points on an elongated solute compared with APBS's
    per-axis `dime`, and it is DelPhi's constraint rather than a choice made
    here.
    """
    extent = pqr.extent()
    center = pqr.center()

    # The same physical box APBS's fine grid uses: the solute plus padding on
    # both sides, so the two backends see the same geometry.
    needed = float(np.max(extent)) + 2.0 * spec.padding

    gsize = odd_gsize(needed / spec.resolution + 1)

    if gsize**3 > spec.max_points:
        # Coarsen rather than shrink the box: cutting padding would move the
        # boundary condition onto the solute, which changes the physics, while
        # a coarser grid only costs accuracy and is reported as relaxed.
        largest = int(np.floor(spec.max_points ** (1.0 / 3.0)))
        gsize = largest if largest % 2 == 1 else largest - 1
        if gsize < MIN_GSIZE:
            raise GridTooLarge(
                f"cannot satisfy max_points={spec.max_points:,} for a molecule of extent "
                f"{np.round(extent, 1).tolist()} A with padding={spec.padding} A; the "
                f"smallest usable cubic grid is {MIN_GSIZE}^3 = {MIN_GSIZE**3:,} points. "
                "Raise max_points or reduce padding."
            )

    scale = (gsize - 1) / needed
    return DelphiGrid(
        gsize=gsize,
        scale=scale,
        center=(float(center[0]), float(center[1]), float(center[2])),
        box_length=needed,
    )
