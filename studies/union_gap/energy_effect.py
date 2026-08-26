"""What repairing the interior of `_union_gap` does to a real vdW ramp solve."""

from __future__ import annotations

import sys
import time

sys.path.insert(
    0,
    "studies/union_gap",
)

from vdw_gap_fix import corrected_union_gap

from sashimi.debye import surface as S
from sashimi.debye.backend import DebyeSolver
from sashimi.debye.options import DebyeOptions
from sashimi.pqr import parse_pqr
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, SolventModel, SurfaceModel

structure = parse_pqr(open(sys.argv[1]).read())
resolution = float(sys.argv[2])
width = float(sys.argv[3])

original = S.ReducedSurface.signed_gap


def patched(self, axes, band=None):
    if self.solvent.surface_model is not SurfaceModel.MOLECULAR or self.probe <= 0.0:
        return corrected_union_gap(self.structure, axes, band=band)
    return original(self, axes, band=band)


def solve(smoothing: float) -> tuple[float, float]:
    solver = DebyeSolver(options=DebyeOptions(dielectric_smoothing=smoothing))
    request = FiniteDifferenceRequest(
        structure=structure,
        solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS),
        grid=GridSpec(resolution=resolution, padding=10.0),
        want_potential=False,
    )
    start = time.process_time()
    answer = solver.solve(request).energy_kj_mol
    return float(answer), time.process_time() - start


hard, t_hard = solve(0.0)
ramp, t_ramp = solve(width)
S.ReducedSurface.signed_gap = patched
fixed, t_fixed = solve(width)
fixed_hard, _ = solve(0.0)
S.ReducedSurface.signed_gap = original

print(f"structure {sys.argv[1]}  atoms={len(structure.radii)}  res={resolution}  w={width}")
print(f"  hard                {hard:16.6f} kJ/mol   {t_hard:6.2f} s")
print(f"  ramp, naive gap     {ramp:16.6f} kJ/mol   {t_ramp:6.2f} s")
print(f"  ramp, repaired gap  {fixed:16.6f} kJ/mol   {t_fixed:6.2f} s")
print(f"  hard, patched (control, must equal hard): {fixed_hard:16.6f}")
print(
    f"  repair moves the ramp by {fixed - ramp:+.6f} kJ/mol "
    f"({100.0 * (fixed - ramp) / abs(ramp):+.4f}%)"
)
print(
    f"  ramp-minus-hard offset: naive {ramp - hard:+.4f}, repaired {fixed - hard:+.4f} "
    f"({100.0 * (fixed - ramp) / abs(ramp - hard):+.2f}% of the offset)"
)
