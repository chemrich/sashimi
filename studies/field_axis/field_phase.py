"""Arm B with the grid-phase control the near field always needs.

One lattice measures phase. This sweeps the padding, which moves the achieved
spacing and so where the solute falls in its cells, and re-solves the referee on
the same box each time so the comparison never crosses a box.

*Reconstructed 2026-08-26 from `field_real.py` and its recorded output. The
original was written inside a `git worktree` and went with it; see
`studies/README.md`. `results/field_phase_alagly.txt` is the output it produced
and this reproduces it.*

Produces ROADMAP.md section 12, "The field axis, measured", Arm B's five-padding
`ala-gly` table. Run from the repository root.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from sashimi.debye.backend import DebyeSolver
from sashimi.debye.grid import axis_coordinates, size_grid
from sashimi.debye.options import DebyeOptions
from sashimi.debye.surface import _union_gap
from sashimi.pqr import read_pqr
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, SolventModel, SurfaceModel

path, coarse_res, fine_res, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
paddings = [float(x) for x in sys.argv[5].split(",")]
WIDTHS = (0.0, 0.25, 0.5, 0.75, 1.0, 2.0)
CAP = 40_000_000
structure = read_pqr(path)
solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR)


def solve(res: float, w: float, padding: float):
    spec = GridSpec(resolution=res, padding=padding, max_points=CAP)
    request = FiniteDifferenceRequest(
        structure=structure, solvent=solvent, grid=spec, want_potential=True, want_energy=False
    )
    return DebyeSolver(options=DebyeOptions(dielectric_smoothing=w)).solve(request).potential


rows = []
print(f"{'pad':>6} {'h':>7} " + " ".join(f"{'w=' + str(w):>15}" for w in WIDTHS[1:]))
for padding in paddings:
    grid = size_grid(structure, GridSpec(resolution=coarse_res, padding=padding, max_points=CAP))
    axes = axis_coordinates(grid)
    h = float(max(grid.spacing))
    distance = _union_gap(axes, structure.coords, structure.radii + solvent.surface_radius)
    mask = (distance >= 2.0) & (distance <= 3.0)
    index = np.nonzero(mask)
    points = np.stack([axes[a][index[a]] for a in range(3)], axis=1)
    refs = {w: solve(fine_res, w, padding).value_at(points) for w in (0.0, 0.5)}
    errs = {}
    for w in WIDTHS:
        got = solve(coarse_res, w, padding).value_at(points)
        errs[w] = {rw: float(np.sqrt(np.mean((got - refs[rw]) ** 2))) for rw in refs}
    cells = [
        " ".join(f"{errs[w][rw] / errs[0.0][rw]:6.2f}x" for rw in (0.0, 0.5)) for w in WIDTHS[1:]
    ]
    rows.append(
        {
            "padding": padding,
            "h": h,
            "nodes": len(points),
            "errs": {str(k): v for k, v in errs.items()},
        }
    )
    print(f"{padding:6.2f} {h:7.4f} " + " ".join(f"{c:>15}" for c in cells))
    sys.stdout.flush()

with open(out, "w") as handle:
    json.dump(rows, handle)
print("\nover phase, ratio to hard (referee hard | referee ramp):")
for w in WIDTHS[1:]:
    for rw in (0.0, 0.5):
        v = np.array([r["errs"][str(w)][rw] / r["errs"]["0.0"][rw] for r in rows])
        print(
            f"  w={w:4.2f} referee w={rw}: worst {v.max():5.2f}x  "
            f"median {np.median(v):5.2f}x  best {v.min():5.2f}x"
        )
