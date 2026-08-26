"""Does the energy improve on the same fixture whose field degrades?

The other half of ROADMAP.md section 12's "the two axes disagree on one fixture":
four rungs on `ala-gly`'s molecular surface, five widths, on the same lattices the
field was graded on. All five columns converge on the same limit, and against it
the ramp at `w = 0.5` is 4.9x closer than the hard assignment at h = 0.4545 and
8.8x closer at 0.2432.

`max_points` is raised explicitly: the default 161**3 clamps the finest rung
silently and the ladder would read as converged when it is not.

Run from the repository root.

*Reconstructed 2026-08-26 from `results/energy_ladder_mol.txt`; the original went
with a removed `git worktree`. See `studies/README.md`.*
"""

from __future__ import annotations

import numpy as np

from sashimi.debye.backend import DebyeSolver
from sashimi.debye.grid import size_grid
from sashimi.debye.options import DebyeOptions
from sashimi.pqr import read_pqr
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, SolventModel, SurfaceModel

structure = read_pqr("tests/data/ala-gly.pqr")
CAP = 40_000_000
LADDER = (1.0, 0.5, 0.25, 0.125)
WIDTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


def energy(res: float, w: float) -> float:
    request = FiniteDifferenceRequest(
        structure=structure,
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
        grid=GridSpec(resolution=res, padding=10.0, max_points=CAP),
        want_potential=False,
    )
    answer = DebyeSolver(options=DebyeOptions(dielectric_smoothing=w)).solve(request)
    return float(answer.energy_kj_mol)


rows = {}
print(
    f"{'req':>6} {'achieved':>9} {'points':>11} " + " ".join(f"{'w=' + str(w):>11}" for w in WIDTHS)
)
for res in LADDER:
    grid = size_grid(structure, GridSpec(resolution=res, padding=10.0, max_points=CAP))
    h = float(np.prod(np.asarray(grid.spacing, float)) ** (1 / 3))
    rows[res] = [energy(res, w) for w in WIDTHS]
    print(
        f"{res:6.3f} {h:9.4f} {int(np.prod(grid.shape)):11d} "
        + " ".join(f"{e:11.4f}" for e in rows[res])
    )

fine = rows[LADDER[-1]]
print("\ndistance from each column's own finest rung, and from the ramp's:")
ref = fine[WIDTHS.index(0.5)]
for res in LADDER:
    print(
        f"  {res:5.3f}: "
        + "  ".join(f"w={w}: {rows[res][i] - ref:+9.4f}" for i, w in enumerate(WIDTHS))
    )
