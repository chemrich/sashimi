"""Why debye and DelPhi agree on the Kirkwood field to five significant figures.

ROADMAP.md section 12 recorded the observation and did not explain it: debye and
the DelPhi backend agree to 2.7e-5 relative on the recorded probes while APBS
sits ~0.4% from both, and "two codes sharing no source agreeing that closely is
either a shared discretization convention or a shared ancestry in one". Section
12's referee work assumes they are independent.

They are independent. What they share is the **lattice**, and the near-field
observable is dominated by where it falls -- the effect section 12 already
records as grid phase. This script measures all three parts:

1. Two hypotheses about the *physics*, both refuted. Charge assignment: debye
   assigns trilinearly, so APBS at `chgm spl0` should meet it -- it does not, it
   gets worse. Boundary condition: debye keeps the exact multi-atom sum, so APBS
   at `bcfl mdh` should meet it -- it moves 0.006%.

2. The lattices. debye needs `n = 8m + 1`, DelPhi needs an odd `gsize`, APBS
   needs `2^k + 1`. At the corpus's 0.25 A both debye and DelPhi land on 105
   exactly and APBS cannot. At 0.203125 all three land on 129 -- and APBS still
   differs fivefold, because its *origin* differs, so the charge sits at a
   different sub-cell position.

3. The agreement is a property of the shared lattice, not of the codes: at
   resolutions where debye's and DelPhi's point-count rules disagree, their
   errors diverge by a factor of three.

Needs APBS and a DelPhi binary; debye is in process.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from sashimi import backends
from sashimi.apbs import ApbsSolver
from sashimi.apbs.options import ApbsOptions
from sashimi.corpus import MANIFEST, AnalyticField, _analytic_field_summary
from sashimi.protocol import GridSpec

RADIUS = 3.0
CELLS = (2, 4, 8)
CASE = "kirkwood-vdw-09"
OFFSET = 2.7  # d/a = 0.9
LADDER = (0.25, 0.203125, 0.20, 0.1875)


def error_at(case, solver, request):
    reference = AnalyticField(
        radius_a=RADIUS, charge_e=1.0, cells_out=CELLS, rtol=1.0, offset_a=OFFSET
    )
    probed = dataclasses.replace(case, analytic_field=reference)
    result = solver.solve(request)
    summary = _analytic_field_summary(probed, result)
    return summary["worst_sample"]["relative_error"], result.potential


def main() -> None:
    case = next(c for c in MANIFEST if c.name == CASE)
    charge = case.structure().coords[int(np.argmax(np.abs(case.structure().charges)))]

    print("1. Two physics hypotheses, both refuted (d/a = 0.9, worst sample)\n")
    knobs = [
        ("apbs chgm spl4", ApbsSolver(options=ApbsOptions(charge_discretization="spl4"))),
        ("apbs chgm spl0", ApbsSolver(options=ApbsOptions(charge_discretization="spl0"))),
        ("apbs bcfl mdh", ApbsSolver(options=ApbsOptions(boundary_condition="mdh"))),
        ("debye", backends.solver_for("debye")[0]),
        ("delphi", backends.solver_for("delphi")[0]),
    ]
    for label, solver in knobs:
        err, _ = error_at(case, solver, case.request())
        print(f"   {label:16s} {err * 100:9.4f}%")
    print("\n   spl0 is APBS's trilinear option, which is what debye uses, and it is")
    print("   worse. mdh is debye's boundary condition, and it moves nothing.\n")

    print(f"2. The lattices, and where the charge at x={charge[0]} falls in a cell\n")
    print(
        f"   {'h asked':>9} {'backend':8s} {'shape':>15s} {'h got':>10s} {'origin x':>10s} "
        f"{'phase x':>8s} {'worst':>10s}"
    )
    for h in LADDER:
        for name in ("apbs", "debye", "delphi"):
            solver, _ = backends.solver_for(name)
            request = dataclasses.replace(
                case.request(),
                grid=GridSpec(
                    resolution=h, padding=case.grid.padding, max_points=case.grid.max_points
                ),
            )
            err, grid = error_at(case, solver, request)
            origin, spacing = np.asarray(grid.origin), np.asarray(grid.spacing)
            phase = float(((charge[0] - origin[0]) / spacing[0]) % 1.0)
            print(
                f"   {h:9.6f} {name:8s} {grid.values.shape!s:>15s} {spacing[0]:10.6f} "
                f"{origin[0]:10.4f} {phase:8.4f} {err * 100:9.4f}%"
            )
        print()


if __name__ == "__main__":
    main()
