"""Physical grid intent -> legal APBS mg-auto parameters.

This is where `GridSpec`'s physics becomes APBS's arithmetic, and it is the
reason `GridSpec` has no `dime`. APBS's multigrid requires dimensions of the
form n = c * 2^(l+1) + 1; with the default 4 levels that is n = 32c + 1, i.e.
33, 65, 97, 129, 161, ... A grid-flexible backend ignores all of this and
honors `resolution` and `padding` directly.

The sizing reimplements the essentials of the classic psize.py: coarse grid at
CFAC x molecular extent to place the focusing boundary far from the solute,
fine grid at extent + 2 x padding, and the smallest legal dime that meets the
resolution target, capped by `max_points`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sashimi.errors import GridTooLarge
from sashimi.protocol import GridSpec, PQRData

__all__ = ["ApbsGrid", "legal_dime", "size_grid", "LEGAL_DIME"]

CFAC = 1.7  # coarse grid / molecular extent, per psize.py
LEVELS = 4  # mg-auto nlev; fixes the legal-dime lattice at 32c + 1

# c = 1..40 -> 33 ... 1281. Beyond this, max_points bites long before dime does.
LEGAL_DIME: tuple[int, ...] = tuple(c * 2 ** (LEVELS + 1) + 1 for c in range(1, 41))


@dataclass(frozen=True)
class ApbsGrid:
    """Everything the mg-auto template needs, plus what was actually achieved."""

    dime: tuple[int, int, int]
    cglen: tuple[float, float, float]
    fglen: tuple[float, float, float]
    center: tuple[float, float, float]
    spacing: tuple[float, float, float]  # fine-grid spacing actually achieved, A

    @property
    def n_points(self) -> int:
        return self.dime[0] * self.dime[1] * self.dime[2]

    def as_diagnostics(self) -> dict:
        return {
            "dime": list(self.dime),
            "cglen": [round(v, 4) for v in self.cglen],
            "fglen": [round(v, 4) for v in self.fglen],
            "center": [round(v, 4) for v in self.center],
            "spacing_achieved": [round(v, 6) for v in self.spacing],
            "n_points": self.n_points,
        }


def legal_dime(minimum: int) -> int:
    """Smallest legal multigrid dimension >= `minimum`."""
    for n in LEGAL_DIME:
        if n >= minimum:
            return n
    return LEGAL_DIME[-1]


def size_grid(pqr: PQRData, spec: GridSpec) -> ApbsGrid:
    extent = pqr.extent()
    center = pqr.center()

    fglen = extent + 2.0 * spec.padding
    # The coarse grid exists only to put the Debye-Huckel boundary far from the
    # solute, so it scales with the fine box rather than being clamped to it.
    # Clamping to `max(CFAC * extent, fglen)` would collapse cglen onto fglen for
    # small solutes, leaving the boundary condition sitting on the fine-grid edge.
    cglen = CFAC * fglen

    # Smallest legal dime meeting the resolution target on each axis.
    needed = np.ceil(fglen / spec.resolution).astype(int) + 1
    dime = np.array([legal_dime(int(v)) for v in needed])

    # Step down the largest axis until the point budget is satisfied. Stepping
    # the largest axis keeps the grid as isotropic as the budget allows.
    while int(np.prod(dime)) > spec.max_points:
        axis = int(np.argmax(dime))
        idx = LEGAL_DIME.index(int(dime[axis]))
        if idx == 0:
            raise GridTooLarge(
                f"cannot satisfy max_points={spec.max_points:,} for a molecule of extent "
                f"{np.round(extent, 1).tolist()} A with padding={spec.padding} A; "
                f"the smallest legal grid is {'x'.join(str(int(v)) for v in dime)} "
                f"= {int(np.prod(dime)):,} points. Raise max_points or reduce padding."
            )
        dime[axis] = LEGAL_DIME[idx - 1]

    spacing = fglen / (dime - 1)
    return ApbsGrid(
        dime=tuple(int(v) for v in dime),  # type: ignore[arg-type]
        cglen=tuple(float(v) for v in cglen),  # type: ignore[arg-type]
        fglen=tuple(float(v) for v in fglen),  # type: ignore[arg-type]
        center=tuple(float(v) for v in center),  # type: ignore[arg-type]
        spacing=tuple(float(v) for v in spacing),  # type: ignore[arg-type]
    )
