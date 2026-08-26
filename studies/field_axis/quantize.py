"""Separate the ramp's two halves: does the sub-cell information buy the answer?

Quantize the solute fraction to exactly 0.5 on every band face -- keep the fact
that a face is in the band, throw away where in it -- and see how much of the
ramp's pose-dispersion benefit and how much of its accuracy survives.

**This is the experiment that did not reproduce, and it is checked in for that
reason.** A previous session recorded it recovering 2.9x of a 4.3x dispersion
benefit and only 6% of the accuracy, which would have made stability and accuracy
separable by construction. Re-run on `ala-gly` van der Waals it recovers 23.6% of
the accuracy and 28.2% of the dispersion at 0.5 A -- proportional, not separated
-- and is *worse than hard on both* at 1.0 A. The claim is withdrawn and appears
nowhere in ROADMAP.md; `results/quantize.txt` is what it actually returns.

"Do not gate an interface scheme on pose dispersion" still stands, on Q0's
reasoning that a spread sees only the phase-dependent half. Not on this.

Run from the repository root.

*Reconstructed 2026-08-26 from `results/quantize.txt`; the original went with a
removed `git worktree`. See `studies/README.md`.*
"""

from __future__ import annotations

import numpy as np

from sashimi.debye import dielectric as D
from sashimi.debye.backend import DebyeSolver
from sashimi.debye.options import DebyeOptions
from sashimi.invariants import posed
from sashimi.pqr import read_pqr
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, PQRData, SolventModel, SurfaceModel

structure = read_pqr("tests/data/ala-gly.pqr")
MODEL = SurfaceModel.VAN_DER_WAALS
original = np.clip

# The ala-gly van der Waals ladder's plateau, from `studies/union_gap/ladder.py`
# at 19.75 M points.
LIMIT = -216.4596


def quantized_clip(a, lo, hi):
    """Stand in for the ramp's `np.clip`, flattening the interior of the band."""
    out = original(a, lo, hi)
    return np.where((out > 0.0) & (out < 1.0), 0.5, out)


def energy(struct: PQRData, w: float, res: float, quantize: bool = False) -> float:
    request = FiniteDifferenceRequest(
        structure=struct,
        solvent=SolventModel(surface_model=MODEL),
        grid=GridSpec(resolution=res, padding=10.0, max_points=40_000_000),
        want_potential=False,
    )
    if quantize:
        D.np.clip = quantized_clip
    try:
        answer = DebyeSolver(options=DebyeOptions(dielectric_smoothing=w)).solve(request)
        return float(answer.energy_kj_mol)
    finally:
        D.np.clip = original


for res in (1.0, 0.5):
    hard = energy(structure, 0.0, res)
    ramp = energy(structure, 0.5, res)
    quant = energy(structure, 0.5, res, quantize=True)
    disp = {}
    for label, w, q in (("hard", 0.0, False), ("ramp", 0.5, False), ("quant", 0.5, True)):
        e = [energy(posed(structure, i, spacing=res), w, res, quantize=q) for i in range(12)]
        disp[label] = float(np.std(e, ddof=1) / abs(np.mean(e)))
    span = abs(hard - LIMIT) - abs(ramp - LIMIT)
    got = abs(hard - LIMIT) - abs(quant - LIMIT)
    dspan = disp["hard"] - disp["ramp"]
    dgot = disp["hard"] - disp["quant"]
    print(f"res={res}: hard {hard:9.4f}  ramp {ramp:9.4f}  quantized {quant:9.4f}")
    print(
        f"   from the limit: hard {abs(hard - LIMIT):7.4f}  ramp {abs(ramp - LIMIT):7.4f}  "
        f"quantized {abs(quant - LIMIT):7.4f}   -> quantized recovers "
        f"{100 * got / span:5.1f}% of the accuracy"
    )
    print(
        f"   dispersion: hard {100 * disp['hard']:.3f}%  ramp {100 * disp['ramp']:.3f}%  "
        f"quantized {100 * disp['quant']:.3f}%   -> recovers {100 * dgot / dspan:5.1f}% of the "
        f"benefit ({disp['hard'] / disp['ramp']:.2f}x available, "
        f"{disp['hard'] / disp['quant']:.2f}x taken)"
    )
