"""Arm B of the field axis: the ramp on a real solute, against a refined referee.

No closed form exists above two atoms, so the referee is the same solver on a
much finer lattice -- run in BOTH schemes, because a fine hard referee inherits
hard's construction and a fine ramp referee inherits the ramp's. A verdict is
only reported where it holds at both ends of that bracket.

The shell is a fixed PHYSICAL band outside the solvent-accessible surface, so
the same shell is graded at every coarse resolution. `_union_gap` on the
probe-inflated spheres is exact outside the SAS-union, and a point outside the
SAS is outside the SES by at least that distance -- so `lo >= 2 * h` guarantees
the repo's own two-cells-out rule at every resolution graded.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from sashimi.debye.backend import DebyeSolver
from sashimi.debye.grid import axis_coordinates, size_grid
from sashimi.debye.options import DebyeOptions
from sashimi.debye.surface import _union_gap
from sashimi.pqr import read_pqr
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, SolventModel, SurfaceModel

path = sys.argv[1]
coarse_res = [float(x) for x in sys.argv[2].split(",")]
fine_res = [float(x) for x in sys.argv[3].split(",")]
out_path = sys.argv[4]
padding = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
widths = (0.0, 0.25, 0.5, 0.75, 1.0, 2.0)
SHELL_LO, SHELL_HI = 2.0, 3.0
CAP = 40_000_000

structure = read_pqr(path)
solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR)


def solve(res, w):
    spec = GridSpec(resolution=res, padding=padding, max_points=CAP)
    request = FiniteDifferenceRequest(
        structure=structure,
        solvent=solvent,
        grid=spec,
        want_potential=True,
        want_energy=False,
    )
    t = time.process_time()
    answer = DebyeSolver(options=DebyeOptions(dielectric_smoothing=w)).solve(request)
    return answer.potential, time.process_time() - t


def shell_points(res):
    grid = size_grid(structure, GridSpec(resolution=res, padding=padding, max_points=CAP))
    axes = axis_coordinates(grid)
    h = float(max(grid.spacing))
    assert 2.0 * h <= SHELL_LO, f"shell {SHELL_LO} A is under two cells of {h:.4f}"
    d = _union_gap(axes, structure.coords, structure.radii + solvent.surface_radius)
    mask = (d >= SHELL_LO) & (d <= SHELL_HI)
    idx = np.nonzero(mask)
    pts = np.stack([axes[0][idx[0]], axes[1][idx[1]], axes[2][idx[2]]], axis=1)
    return pts, h, grid.shape


referees = {}
for res in fine_res:
    for w in (0.0, 0.5):
        pot, secs = solve(res, w)
        referees[(res, w)] = pot
        print(f"referee res={res} w={w}: {secs:.0f}s  shape={pot.values.shape}")
        sys.stdout.flush()

rows = []
for res in coarse_res:
    pts, h, shape = shell_points(res)
    print(f"\ncoarse res={res} h={h:.4f} shape={shape} shell nodes={len(pts)}")
    ref_values = {k: v.value_at(pts) for k, v in referees.items()}
    for k, v in ref_values.items():
        assert np.isfinite(v).all(), f"referee {k} does not cover the shell"
    # how far apart the referees are, per pair -- the ambiguity of the yardstick
    keys = sorted(referees)
    spread = max(
        float(np.sqrt(np.mean((ref_values[a] - ref_values[b]) ** 2)))
        for i, a in enumerate(keys)
        for b in keys[i + 1 :]
    )
    errs = {}
    for w in widths:
        pot, secs = solve(res, w)
        got = pot.value_at(pts)
        assert np.isfinite(got).all()
        e = {
            str(k): dict(
                rms=float(np.sqrt(np.mean((got - ref_values[k]) ** 2))),
                p95=float(np.percentile(np.abs(got - ref_values[k]), 95)),
                mx=float(np.max(np.abs(got - ref_values[k]))),
            )
            for k in keys
        }
        errs[w] = e
        rows.append(dict(res=res, w=w, h=h, nodes=len(pts), secs=secs, errors=e, spread=spread))
        print(f"  w={w:4.2f} " + "  ".join(f"{k}: rms {e[str(k)]['rms']:.4f}" for k in keys))
        sys.stdout.flush()
    print(f"  referee spread (RMS, worst pair): {spread:.4f}")
    print("  ratios against hard (rms):")
    for k in keys:
        base = errs[0.0][str(k)]["rms"]
        gaps = [abs(errs[w][str(k)]["rms"] - base) for w in widths if w > 0]
        print(
            f"    referee {k}: "
            + "  ".join(f"w={w}:{errs[w][str(k)]['rms'] / base:.3f}x" for w in widths if w > 0)
            + f"   | discriminable: spread {spread:.4f} vs smallest gap {min(gaps):.4f}"
            f" -> {'YES' if spread < 0.5 * min(gaps) else 'NO'}"
        )

json.dump(rows, open(out_path, "w"))
