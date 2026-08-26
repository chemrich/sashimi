import time

import numpy as np

from sashimi.debye.backend import DebyeSolver
from sashimi.debye.grid import axis_coordinates, size_grid
from sashimi.debye.options import DebyeOptions
from sashimi.debye.surface import _union_gap
from sashimi.pqr import read_pqr
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, SolventModel, SurfaceModel

s = read_pqr("tests/data/ala-gly.pqr")
solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR)
CAP = 40_000_000


def solve(res, w):
    spec = GridSpec(resolution=res, padding=10.0, max_points=CAP)
    r = FiniteDifferenceRequest(
        structure=s, solvent=solvent, grid=spec, want_potential=True, want_energy=False
    )
    t = time.process_time()
    p = DebyeSolver(options=DebyeOptions(dielectric_smoothing=w)).solve(r).potential
    return p, time.process_time() - t


grid = size_grid(s, GridSpec(resolution=0.5, padding=10.0, max_points=CAP))
axes = axis_coordinates(grid)
d = _union_gap(axes, s.coords, s.radii + solvent.surface_radius)
idx = np.nonzero((d >= 2.0) & (d <= 3.0))
pts = np.stack([axes[0][idx[0]], axes[1][idx[1]], axes[2][idx[2]]], axis=1)
for fine in (0.25, 0.2):
    ref, tr = solve(fine, 0.5)
    rv = ref.value_at(pts)
    out = []
    total = tr
    for w in (0.0, 0.5, 1.0):
        got, tc = solve(0.5, w)
        total += tc
        out.append(float(np.sqrt(np.mean((got.value_at(pts) - rv) ** 2))))
    print(
        f"referee res={fine} (ramp scheme, {tr:.1f}s): hard {out[0]:.5f}  "
        f"w=0.5 {out[1]:.5f} ({out[1] / out[0]:.2f}x)  w=1.0 {out[2]:.5f} ({out[2] / out[0]:.2f}x)"
        f"   total {total:.1f}s"
    )
