"""Do the ramp and the hard assignment converge to the SAME field?

ROADMAP.md section 12's field-axis verdict rests on a premise it states as a
theorem and never measured: the ramp's band is `w*h` and vanishes with `h`, so
the two schemes discretize one continuum problem and share a limit. If they do
not, the comparison is not *bounded*, it is **meaningless** -- one of the two is
converging to the wrong field and no amount of refereeing debye against debye
would say which.

Section 12 already flags the tension. Each scheme's own successive differences
fit an order near 1.7-1.8; the cross-scheme difference decays at more like 0.8.
Under `phi_X(h) = L + A_X h^p` with a shared `L` those two orders must agree, so
either the fields converge more slowly than each ladder suggests or they approach
limits that differ.

**This needs no referee at all**, which is what makes it worth doing first: at
each `h` the two schemes solve on the *identical* lattice, so
`delta(h) = ||phi_ramp(h) - phi_hard(h)||` over a fixed physical shell involves no
cross-lattice comparison and no reference. The question is only whether
`delta(h) -> 0`.

Two controls, because a norm that fails to reach zero is exactly what a broken
measurement channel also looks like:

- **`w -> 0`.** At `w = 1e-5` the ramp is bit-identical to the hard branch
  (section 12 records `fas2` returning the same digits at 1e-5 and 5e-5), so
  `delta` must be exactly 0.0 at every rung. This tests the channel end to end
  including the shell, the reader and the norm.
- **A uniform dielectric.** With `solute == solvent` there is no interface, the
  ramp has nothing to blend, and `delta` must again be identically zero -- on a
  lattice, source and shell that are otherwise untouched.

`delta` is also split into the part that is a constant offset over the shell and
the part that is a change of shape, because those have different causes and a
constant offset between two solutions of the same equation with the same total
charge would be its own finding.

Run from the repository root.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from sashimi.debye.backend import DebyeSolver
from sashimi.debye.grid import axis_coordinates, size_grid
from sashimi.debye.options import DebyeOptions
from sashimi.debye.surface import _union_gap
from sashimi.pqr import read_pqr
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, SolventModel, SurfaceModel

PADDING = 11.0
CAP = 40_000_000
LADDER = (0.5, 0.35, 0.25, 0.18, 0.14, 0.11)
WIDTH = 0.5
SHELL_LO, SHELL_HI = 2.0, 3.0

structure = read_pqr("tests/data/ala-gly.pqr")


def solve(res: float, width: float, solvent: SolventModel, points: np.ndarray) -> np.ndarray:
    spec = GridSpec(resolution=res, padding=PADDING, max_points=CAP)
    request = FiniteDifferenceRequest(
        structure=structure, solvent=solvent, grid=spec, want_potential=True, want_energy=False
    )
    answer = DebyeSolver(options=DebyeOptions(dielectric_smoothing=width)).solve(request)
    got = answer.potential.value_at(points)
    assert np.isfinite(got).all(), f"res={res} w={width} does not cover the shell"
    return got


def achieved(res: float) -> tuple[float, tuple[int, ...]]:
    grid = size_grid(structure, GridSpec(resolution=res, padding=PADDING, max_points=CAP))
    return float(np.prod(np.asarray(grid.spacing, float)) ** (1 / 3)), grid.shape


# The shell is fixed in PHYSICAL space, chosen once on a reference lattice, so
# every rung reads the same points. `_union_gap` on the probe-inflated spheres is
# exact outside the solvent-accessible surface, and a point outside the SAS is
# outside the solute by at least that distance -- so SHELL_LO clears the
# two-coarse-cell rule at the coarsest rung with room to spare.
solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR)
reference = size_grid(structure, GridSpec(resolution=0.5, padding=PADDING, max_points=CAP))
axes = axis_coordinates(reference)
distance = _union_gap(axes, structure.coords, structure.radii + solvent.surface_radius)
index = np.nonzero((distance >= SHELL_LO) & (distance <= SHELL_HI))
shell = np.stack([axes[a][index[a]] for a in range(3)], axis=1)
coarsest_h = float(max(reference.spacing))
assert 2.0 * coarsest_h <= SHELL_LO, f"shell is inside two cells of {coarsest_h:.4f}"
print(f"{len(shell)} shell nodes, {SHELL_LO}-{SHELL_HI} A outside the accessible surface")
print(f"coarsest h = {coarsest_h:.4f}, so the shell is {SHELL_LO / coarsest_h:.1f} cells out\n")


def report(label: str, rows: list[tuple[float, float, float, float]]) -> None:
    print(f"{label}")
    print(f"  {'h':>8} {'points':>11} {'delta':>12} {'offset':>12} {'shape':>12}")
    for h, points, total, offset, shape in rows:
        print(f"  {h:8.5f} {points:11.0f} {total:12.6f} {offset:12.6f} {shape:12.6f}")
    if len(rows) >= 3:
        print(f"  {'local order of delta:':>34} ", end="")
        orders = []
        for (h0, _, d0, _, _), (h1, _, d1, _, _) in pairwise(rows):
            orders.append(np.log(d0 / d1) / np.log(h0 / h1) if d1 > 0 and d0 > 0 else float("nan"))
        print("  ".join(f"{o:6.2f}" for o in orders))
    print()


def measure(solvent_model: SolventModel, width: float, label: str) -> list:
    rows = []
    for res in LADDER:
        h, shape = achieved(res)
        hard = solve(res, 0.0, solvent_model, shell)
        ramp = solve(res, width, solvent_model, shell)
        diff = ramp - hard
        offset = float(np.mean(diff))
        rows.append(
            (
                h,
                float(np.prod(shape)),
                float(np.sqrt(np.mean(diff**2))),
                abs(offset),
                float(np.sqrt(np.mean((diff - offset) ** 2))),
            )
        )
    report(label, rows)
    return rows


print("CONTROL 1 -- w = 1e-5, where the ramp is bit-identical to hard")
measure(solvent, 1e-5, "  delta must be exactly zero at every rung")

print("CONTROL 2 -- uniform dielectric, where there is no interface to blend")
measure(
    SolventModel(surface_model=SurfaceModel.MOLECULAR, solute_dielectric=78.54),
    WIDTH,
    "  delta must be exactly zero at every rung",
)

print(f"THE MEASUREMENT -- w = {WIDTH}, molecular surface")
rows = measure(solvent, WIDTH, f"  delta(h) = ||phi_ramp - phi_hard|| at w = {WIDTH}")

print("Richardson on delta itself, over every window of three (Refinement's guard applies):")
from sashimi.invariants import Refinement  # noqa: E402

for i in range(len(rows) - 2):
    window = rows[i : i + 3]
    grade = Refinement(
        backend="delta",
        spacings=tuple(r[0] for r in window),
        energies=tuple(r[2] for r in window),
    )
    verdict = "converging" if grade.converging else "REFUSED by the guard"
    print(
        f"  h = {window[0][0]:.4f} / {window[1][0]:.4f} / {window[2][0]:.4f}: "
        f"{verdict}, limit {grade.limit:+.6f}, order {grade.order:5.2f}"
    )
