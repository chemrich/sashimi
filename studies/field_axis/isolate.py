"""Two variables, separated: interpolation, and how fine the referee is.

At paddings where `size_grid` nests exactly at ratio 2, the res-0.25 referee can
be read by INDEX (exact) or by `value_at` (interpolated). Comparing those two
isolates interpolation. Comparing the interpolated 0.25 against the interpolated
0.12 isolates the referee's own resolution.
"""

import sys

import numpy as np

from sashimi.debye.backend import DebyeSolver
from sashimi.debye.grid import axis_coordinates, size_grid
from sashimi.debye.options import DebyeOptions
from sashimi.debye.surface import _union_gap
from sashimi.pqr import read_pqr
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, SolventModel, SurfaceModel

WIDTHS = (0.25, 0.5, 0.75, 1.0)
CAP = 40_000_000
structure = read_pqr("tests/data/ala-gly.pqr")
solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR)


def solve(res, w, pad):
    spec = GridSpec(resolution=res, padding=pad, max_points=CAP)
    r = FiniteDifferenceRequest(
        structure=structure, solvent=solvent, grid=spec, want_potential=True, want_energy=False
    )
    return DebyeSolver(options=DebyeOptions(dielectric_smoothing=w)).solve(r).potential


for pad in (11.0, 13.0):
    coarse = size_grid(structure, GridSpec(resolution=0.5, padding=pad, max_points=CAP))
    axes = axis_coordinates(coarse)
    d = _union_gap(axes, structure.coords, structure.radii + solvent.surface_radius)
    idx = np.nonzero((d >= 2.0) & (d <= 3.0))
    pts = np.stack([axes[a][idx[a]] for a in range(3)], axis=1)

    fine25 = {rw: solve(0.25, rw, pad) for rw in (0.0, 0.5)}
    fine12 = {rw: solve(0.12, rw, pad) for rw in (0.0, 0.5)}
    refs = {
        "0.25 index": {rw: fine25[rw].values[::2, ::2, ::2][idx] for rw in (0.0, 0.5)},
        "0.25 interp": {rw: fine25[rw].value_at(pts) for rw in (0.0, 0.5)},
        "0.12 interp": {rw: fine12[rw].value_at(pts) for rw in (0.0, 0.5)},
    }
    # how much interpolation alone moves the referee's values
    for rw in (0.0, 0.5):
        a, b = refs["0.25 index"][rw], refs["0.25 interp"][rw]
        print(
            f"pad {pad}: referee w={rw}, index vs interp RMS {np.sqrt(np.mean((a - b) ** 2)):.6f} "
            f"(referee's own magnitude RMS {np.sqrt(np.mean(a**2)):.4f})"
        )

    coarse_vals = {w: solve(0.5, w, pad).values[idx] for w in (0.0, *WIDTHS)}
    print(f"  {'referee':>14} " + " ".join(f"{'w=' + str(w):>12}" for w in WIDTHS))
    for name, ref in refs.items():
        for rw in (0.0, 0.5):
            base = float(np.sqrt(np.mean((coarse_vals[0.0] - ref[rw]) ** 2)))
            cells = [
                f"{float(np.sqrt(np.mean((coarse_vals[w] - ref[rw]) ** 2))) / base:11.2f}x"
                for w in WIDTHS
            ]
            print(f"  {name:>14} w={rw} " + " ".join(cells))
    sys.stdout.flush()
    print()
