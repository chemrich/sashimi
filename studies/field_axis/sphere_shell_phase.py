"""The control that separates geometry from statistic.

Arm A grades the sphere by worst-direction error at one radius, phase-swept, and
says the ramp helps. Arm B grades `ala-gly` by shell RMS against a refined
referee, phase-swept, and says it hurts. This runs **Arm B's statistic on Arm A's
geometry**, phase-swept, against the exact Born potential -- so the only thing
left that could explain a disagreement is the geometry.

Produces ROADMAP.md section 12, "The field axis, measured", Arm A's shell-RMS
table. Run from the repository root.

*Reconstructed 2026-08-26 from `results/sphere_shell_phase.txt`; the original went
with a removed `git worktree`. See `studies/README.md`.*
"""

from __future__ import annotations

import sys

import numpy as np

from sashimi.analytic import born_potential
from sashimi.debye.backend import DebyeSolver
from sashimi.debye.grid import axis_coordinates, size_grid
from sashimi.debye.options import DebyeOptions
from sashimi.debye.surface import _union_gap
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, PQRData, SolventModel, SurfaceModel

WIDTHS = (0.0, 0.25, 0.5, 0.75, 1.0, 2.0)
PADDINGS = tuple(np.round(np.arange(10.0, 15.01, 0.25), 3))
solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR, ionic_strength=0.0)

for radius in (2.0, 3.0):
    structure = PQRData(coords=np.zeros((1, 3)), charges=np.array([1.0]), radii=np.array([radius]))
    table = {w: [] for w in WIDTHS}
    for padding in PADDINGS:
        spec = GridSpec(resolution=0.5, padding=float(padding))
        grid = size_grid(structure, spec)
        axes = axis_coordinates(grid)
        distance = _union_gap(axes, structure.coords, structure.radii + solvent.surface_radius)
        index = np.nonzero((distance >= 2.0) & (distance <= 3.0))
        points = np.stack([axes[a][index[a]] for a in range(3)], axis=1)
        exact = np.array(
            [
                born_potential(x, 1.0, solvent.solvent_dielectric, solvent.temperature)
                for x in np.linalg.norm(points, axis=1)
            ]
        )
        for w in WIDTHS:
            request = FiniteDifferenceRequest(
                structure=structure,
                solvent=solvent,
                grid=spec,
                want_potential=True,
                want_energy=False,
            )
            got = (
                DebyeSolver(options=DebyeOptions(dielectric_smoothing=w))
                .solve(request)
                .potential.value_at(points)
            )
            table[w].append(float(np.sqrt(np.mean((got - exact) ** 2))))
    hard = np.array(table[0.0])
    print(f"a={radius}, {len(PADDINGS)} paddings, shell RMS against the exact Born potential")
    print(
        f"  {'w':>5} {'worst':>9} {'median':>9} {'best':>9} | ratio to hard: worst / median / best"
    )
    for w in WIDTHS:
        v = np.array(table[w])
        r = v / hard
        print(
            f"  {w:5.2f} {v.max():9.5f} {np.median(v):9.5f} {v.min():9.5f} | "
            f"{r.max():5.2f}x {np.median(r):5.2f}x {r.min():5.2f}x"
        )
    sys.stdout.flush()
