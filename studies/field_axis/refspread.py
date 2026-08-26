"""Split the referee spread into its two axes: resolution, and scheme."""

import numpy as np

from sashimi.debye.backend import DebyeSolver
from sashimi.debye.grid import axis_coordinates, size_grid
from sashimi.debye.options import DebyeOptions
from sashimi.debye.surface import _union_gap
from sashimi.pqr import read_pqr
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, SolventModel, SurfaceModel

structure = read_pqr("tests/data/ala-gly.pqr")
solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR)
CAP = 40_000_000


def solve(res, w):
    spec = GridSpec(resolution=res, padding=10.0, max_points=CAP)
    r = FiniteDifferenceRequest(
        structure=structure, solvent=solvent, grid=spec, want_potential=True, want_energy=False
    )
    return DebyeSolver(options=DebyeOptions(dielectric_smoothing=w)).solve(r).potential


grid = size_grid(structure, GridSpec(resolution=0.5, padding=10.0, max_points=CAP))
axes = axis_coordinates(grid)
d = _union_gap(axes, structure.coords, structure.radii + solvent.surface_radius)
mask = (d >= 2.0) & (d <= 3.0)
idx = np.nonzero(mask)
pts = np.stack([axes[0][idx[0]], axes[1][idx[1]], axes[2][idx[2]]], axis=1)
print("shell nodes", len(pts))

vals = {}
for res in (0.20, 0.15, 0.12, 0.10):
    for w in (0.0, 0.5):
        vals[(res, w)] = solve(res, w).value_at(pts)
        print(f"  solved {res} w={w}")


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


print("\nSAME SCHEME, refining the referee (should shrink toward zero):")
for w in (0.0, 0.5):
    for a, b in ((0.20, 0.15), (0.15, 0.12), (0.12, 0.10)):
        print(f"  w={w}: {a} vs {b}: {rms(vals[(a, w)], vals[(b, w)]):.5f}")
print("\nSAME RESOLUTION, the two schemes (shrinks only if they share a limit):")
for res in (0.20, 0.15, 0.12, 0.10):
    print(f"  res {res}: hard vs ramp: {rms(vals[(res, 0.0)], vals[(res, 0.5)]):.5f}")
