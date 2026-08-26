"""Why APBS's Kirkwood field error is large, and what it is a function of.

Attaching `kirkwood_potential` to the corpus measured APBS at 20.96% on the
d/a = 0.9 rung against debye's 1.86% -- an eleven-fold spread between two codes
solving the same geometry on the same grid. This asks whether that is a property
of the *field* or of *where it is sampled*.

`a + k*h` was designed to clear the dielectric interface. An off-centre charge
adds a second thing to clear -- itself -- and at d/a = 0.9 the near pole lands
0.71 A from a point charge on APBS's achieved 0.203 A lattice. Walking `k` out
separates the two explanations: a field defect would not care how far the sample
sits from the charge, and a discretized point charge would care a great deal.

Reproduces the ROADMAP §12 ladder under "The field reference stops being a Born
ion". Needs APBS installed; debye is in-process.
"""

from __future__ import annotations

import dataclasses

from sashimi import backends
from sashimi.corpus import MANIFEST, AnalyticField, _analytic_field_summary
from sashimi.errors import SashimiError

RADIUS = 3.0
LADDERS = ((2, 4, 8), (4, 8), (6, 12))
CASES = ("kirkwood-vdw-07", "kirkwood-vdw-09")
BACKENDS = ("apbs", "debye")


def main() -> None:
    print(
        f"{'case':22s} {'cells_out':12s} {'backend':8s} {'worst':>8} {'dir':>7} "
        f"{'r':>7} {'gap to charge':>14}"
    )
    for name in CASES:
        case = next(c for c in MANIFEST if c.name == name)
        offset = RADIUS * int(name.split("-")[-1]) / 10.0
        for cells in LADDERS:
            reference = AnalyticField(
                radius_a=RADIUS,
                charge_e=1.0,
                cells_out=cells,
                rtol=1.0,  # this study reports the error; it does not grade it
                offset_a=offset,
            )
            probed = dataclasses.replace(case, analytic_field=reference)
            for backend in BACKENDS:
                try:
                    solver, _family = backends.solver_for(backend)
                    summary = _analytic_field_summary(probed, solver.solve(probed.request()))
                except SashimiError as exc:
                    print(f"{name:22s} {cells!s:12s} {backend:8s} SKIPPED {exc}")
                    continue
                worst = summary["worst_sample"]
                # The +x pole is the closest sample to the charge, so the
                # innermost radius minus the offset is the gap that matters.
                gap = min(summary["radii_a"]) - offset
                print(
                    f"{name:22s} {cells!s:12s} {backend:8s} "
                    f"{summary['max_relative_error'] * 100:7.3f}% {worst['direction']:>7} "
                    f"{worst['radius_a']:7.3f} {gap:13.3f} A"
                )
    print()
    print("APBS walks down steeply with the gap and debye does not move, and")
    print("debye's worst sample is on the *far* pole at d/a = 0.9 -- so its")
    print("near-pole error is smaller still at a shorter gap. The spread is")
    print("charge proximity, not the field.")


if __name__ == "__main__":
    main()
