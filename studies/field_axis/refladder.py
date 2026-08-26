"""Does the verdict settle as the referee refines? A referee ladder at one padding."""

import sys

import numpy as np

from sashimi.debye.backend import DebyeSolver
from sashimi.debye.grid import axis_coordinates, size_grid
from sashimi.debye.options import DebyeOptions
from sashimi.debye.surface import _union_gap
from sashimi.pqr import read_pqr
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, SolventModel, SurfaceModel

WIDTHS = (0.25, 0.5, 0.75, 1.0)
CAP = 60_000_000
PAD = 11.0
structure = read_pqr("tests/data/ala-gly.pqr")
solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR)


def solve(res, w):
    spec = GridSpec(resolution=res, padding=PAD, max_points=CAP)
    r = FiniteDifferenceRequest(
        structure=structure, solvent=solvent, grid=spec, want_potential=True, want_energy=False
    )
    return DebyeSolver(options=DebyeOptions(dielectric_smoothing=w)).solve(r).potential


coarse = size_grid(structure, GridSpec(resolution=0.5, padding=PAD, max_points=CAP))
axes = axis_coordinates(coarse)
d = _union_gap(axes, structure.coords, structure.radii + solvent.surface_radius)
idx = np.nonzero((d >= 2.0) & (d <= 3.0))
pts = np.stack([axes[a][idx[a]] for a in range(3)], axis=1)
coarse_vals = {w: solve(0.5, w).values[idx] for w in (0.0, *WIDTHS)}

print(
    f"referee ladder at padding {PAD}, coarse h={float(max(coarse.spacing)):.4f}, "
    f"{len(pts)} shell nodes"
)
print(
    f"{'referee res':>12} {'scheme':>7} "
    + " ".join(f"{'w=' + str(w):>10}" for w in WIDTHS)
    + "   cross-scheme spread"
)
for res in (0.25, 0.15, 0.12, 0.10, 0.09):
    g = size_grid(structure, GridSpec(resolution=res, padding=PAD, max_points=CAP))
    if int(np.prod(g.shape)) > CAP:
        print(f"  {res}: {np.prod(g.shape) / 1e6:.1f}M points, skipped")
        continue
    vals = {rw: solve(res, rw).value_at(pts) for rw in (0.0, 0.5)}
    spread = float(np.sqrt(np.mean((vals[0.0] - vals[0.5]) ** 2)))
    for rw in (0.0, 0.5):
        base = float(np.sqrt(np.mean((coarse_vals[0.0] - vals[rw]) ** 2)))
        cells = [
            f"{float(np.sqrt(np.mean((coarse_vals[w] - vals[rw]) ** 2))) / base:9.2f}x"
            for w in WIDTHS
        ]
        tail = f"   {spread:.5f} (hard err {base:.5f})" if rw == 0.0 else ""
        print(f"  {res:10.3f} {'hard' if rw == 0 else 'ramp':>7} " + " ".join(cells) + tail)
    sys.stdout.flush()
