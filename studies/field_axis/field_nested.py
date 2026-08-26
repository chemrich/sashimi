"""The interpolation control: the same comparison on exactly nested lattices.

At paddings 9/11/13/15/17 with coarse 0.5 A and fine 0.25 A, `size_grid` returns
`n_fine - 1 = 2 * (n_coarse - 1)` on every axis over the same box -- so every
coarse node IS a fine node and `value_at` is exact there. If the verdict is an
artefact of trilinear interpolation off the referee, it dies here.
"""

import sys

import numpy as np

from sashimi.debye.backend import DebyeSolver
from sashimi.debye.grid import axis_coordinates, size_grid
from sashimi.debye.options import DebyeOptions
from sashimi.debye.surface import _union_gap
from sashimi.pqr import read_pqr
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, SolventModel, SurfaceModel

WIDTHS = (0.0, 0.25, 0.5, 0.75, 1.0, 2.0)
PADDINGS = (9.0, 11.0, 13.0, 15.0, 17.0)
CAP = 40_000_000
structure = read_pqr("tests/data/ala-gly.pqr")
solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR)


def solve(res, w, pad):
    spec = GridSpec(resolution=res, padding=pad, max_points=CAP)
    r = FiniteDifferenceRequest(
        structure=structure, solvent=solvent, grid=spec, want_potential=True, want_energy=False
    )
    return DebyeSolver(options=DebyeOptions(dielectric_smoothing=w)).solve(r).potential


print(f"{'pad':>5} {'h':>7} {'nodes':>7} " + " ".join(f"{'w=' + str(w):>13}" for w in WIDTHS[1:]))
rows = []
for pad in PADDINGS:
    coarse = size_grid(structure, GridSpec(resolution=0.5, padding=pad, max_points=CAP))
    fine = size_grid(structure, GridSpec(resolution=0.25, padding=pad, max_points=CAP))
    step = [(f - 1) // (c - 1) for f, c in zip(fine.shape, coarse.shape)]
    assert step == [2, 2, 2], step
    axes = axis_coordinates(coarse)
    d = _union_gap(axes, structure.coords, structure.radii + solvent.surface_radius)
    mask = (d >= 2.0) & (d <= 3.0)
    idx = np.nonzero(mask)
    # Read the referee by INDEX, not by interpolation: coarse node (i,j,k) is
    # fine node (2i, 2j, 2k) exactly.
    refs = {}
    for rw in (0.0, 0.5):
        v = solve(0.25, rw, pad).values
        refs[rw] = v[::2, ::2, ::2][idx]
    # a hard check that the two lattices really coincide
    fa = axis_coordinates(fine)
    for a in range(3):
        assert np.allclose(fa[a][::2], axes[a], rtol=0, atol=1e-12), f"axis {a} not nested"
    errs = {}
    for w in WIDTHS:
        got = solve(0.5, w, pad).values[idx]
        errs[w] = {rw: float(np.sqrt(np.mean((got - refs[rw]) ** 2))) for rw in refs}
    rows.append((pad, errs))
    cells = [
        " ".join(f"{errs[w][rw] / errs[0.0][rw]:5.2f}x" for rw in (0.0, 0.5)) for w in WIDTHS[1:]
    ]
    print(
        f"{pad:5.1f} {float(max(coarse.spacing)):7.4f} {int(mask.sum()):7d} "
        + " ".join(f"{c:>13}" for c in cells)
    )
    sys.stdout.flush()

print("\nover the five nested paddings (referee hard | referee ramp):")
for w in WIDTHS[1:]:
    for rw in (0.0, 0.5):
        v = np.array([e[w][rw] / e[0.0][rw] for _, e in rows])
        print(
            f"  w={w:4.2f} referee w={rw}: worst {v.max():5.2f}x "
            f" median {np.median(v):5.2f}x  best {v.min():5.2f}x"
        )
