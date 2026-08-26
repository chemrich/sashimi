"""Does each scheme converge to the EXACT field, or only to its own limit?

`shared_limit.py` finds that on `ala-gly` the difference between the two schemes
does not extrapolate to zero: a model with a non-zero limit fits 3.6x better than
`delta = C h^p`, and its exponent agrees with each scheme's own convergence order
where the shared-limit model's does not. That is a claim about consistency and it
cannot be settled without a reference that is neither scheme.

On a Born sphere there is one. `born_potential` is exact, so this asks the
question directly: does `||phi_X(h) - phi_exact||` go to zero for **both** X, or
does one of them flatten at a floor?

If one flattens, that scheme is not a consistent discretization of this problem
on the field -- which is the O(1)-at-the-interface class section 12 already
records for hard midpoint assignment, and which is what interface methods exist
for. If both go to zero, `shared_limit.py`'s non-zero limit is an artefact of
`ala-gly`'s geometry rather than of either scheme.

The sphere makes the solvent-excluded surface identical to the van der Waals one,
so this says nothing about re-entrant geometry. It is not meant to: consistency
is a property of the scheme, and a scheme that is inconsistent on a sphere is
inconsistent.

Run from the repository root.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from sashimi.analytic import born_potential
from sashimi.debye.backend import DebyeSolver
from sashimi.debye.grid import axis_coordinates, size_grid
from sashimi.debye.options import DebyeOptions
from sashimi.debye.surface import _union_gap
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, PQRData, SolventModel, SurfaceModel

RADIUS = 3.0
PADDING = 11.0
CAP = 40_000_000
LADDER = (0.5, 0.35, 0.25, 0.18, 0.14, 0.11)
WIDTHS = (0.0, 0.25, 0.5, 1.0)
SHELL_LO, SHELL_HI = 2.0, 3.0

structure = PQRData(coords=np.zeros((1, 3)), charges=np.array([1.0]), radii=np.array([RADIUS]))
solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR, ionic_strength=0.0)

reference = size_grid(structure, GridSpec(resolution=0.5, padding=PADDING, max_points=CAP))
axes = axis_coordinates(reference)
distance = _union_gap(axes, structure.coords, structure.radii + solvent.surface_radius)
index = np.nonzero((distance >= SHELL_LO) & (distance <= SHELL_HI))
shell = np.stack([axes[a][index[a]] for a in range(3)], axis=1)
exact = np.array(
    [
        born_potential(x, 1.0, solvent.solvent_dielectric, solvent.temperature)
        for x in np.linalg.norm(shell, axis=1)
    ]
)
print(f"a = {RADIUS} A, {len(shell)} shell nodes, {SHELL_LO}-{SHELL_HI} A outside the SAS")
print(f"exact potential over the shell: RMS {float(np.sqrt(np.mean(exact**2))):.5f} kT/e\n")

table: dict[float, list[float]] = {w: [] for w in WIDTHS}
spacings: list[float] = []
for res in LADDER:
    grid = size_grid(structure, GridSpec(resolution=res, padding=PADDING, max_points=CAP))
    spacings.append(float(np.prod(np.asarray(grid.spacing, float)) ** (1 / 3)))
    for w in WIDTHS:
        request = FiniteDifferenceRequest(
            structure=structure,
            solvent=solvent,
            grid=GridSpec(resolution=res, padding=PADDING, max_points=CAP),
            want_potential=True,
            want_energy=False,
        )
        got = (
            DebyeSolver(options=DebyeOptions(dielectric_smoothing=w))
            .solve(request)
            .potential.value_at(shell)
        )
        table[w].append(float(np.sqrt(np.mean((got - exact) ** 2))))

print(f"  {'h':>8} " + " ".join(f"{'w=' + str(w):>12}" for w in WIDTHS))
for i, h in enumerate(spacings):
    print(f"  {h:8.5f} " + " ".join(f"{table[w][i]:12.6f}" for w in WIDTHS))
print()
print("local order of the error against the exact answer -- a scheme that")
print("converges shows a positive order that does not collapse toward zero:")
for w in WIDTHS:
    orders = [
        np.log(a / b) / np.log(ha / hb)
        for (ha, a), (hb, b) in pairwise(list(zip(spacings, table[w], strict=True)))
    ]
    print(f"  w={w:4.2f}: " + "  ".join(f"{o:6.2f}" for o in orders))
