"""Arm B's statistic on Arm A's geometry, at one lattice, against the exact answer.

Arm A grades the sphere by worst-direction error at one radius; Arm B grades a
real solute by shell RMS against a refined referee. They disagreed about the
ramp. This runs Arm B's statistic on the Born sphere, where the reference is
exact, so the two arms differ only in geometry.

The answer is that at one lattice it is mixed and lattice-dependent, which is why
`sphere_shell_phase.py` sweeps it. Produces the working number behind ROADMAP.md
section 12's "the two summaries disagree". Run from the repository root.

*Reconstructed 2026-08-26 from `results/sphere_shell.txt`; the original went with
a removed `git worktree`. See `studies/README.md`.*
"""

from __future__ import annotations

import numpy as np

from sashimi.analytic import born_potential
from sashimi.debye.backend import DebyeSolver
from sashimi.debye.grid import axis_coordinates, size_grid
from sashimi.debye.options import DebyeOptions
from sashimi.debye.surface import _union_gap
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, PQRData, SolventModel, SurfaceModel

WIDTHS = (0.0, 0.25, 0.5, 0.75, 1.0, 2.0)
solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR, ionic_strength=0.0)

for radius in (2.0, 3.0):
    structure = PQRData(coords=np.zeros((1, 3)), charges=np.array([1.0]), radii=np.array([radius]))
    for res in (1.0, 0.5):
        spec = GridSpec(resolution=res, padding=10.0)
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
        line = []
        base = None
        for w in WIDTHS:
            request = FiniteDifferenceRequest(
                structure=structure,
                solvent=solvent,
                grid=spec,
                want_potential=True,
                want_energy=False,
            )
            answer = DebyeSolver(options=DebyeOptions(dielectric_smoothing=w)).solve(request)
            got = answer.potential.value_at(points)
            rms = float(np.sqrt(np.mean((got - exact) ** 2)))
            if base is None:
                base = rms
            line.append(f"w={w}:{rms:.5f}({rms / base:.2f}x)")
        print(
            f"a={radius} res={res} h={float(max(grid.spacing)):.4f} "
            f"nodes={len(points)}  " + "  ".join(line)
        )
