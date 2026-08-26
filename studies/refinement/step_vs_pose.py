"""Is the last Richardson step below the pose noise, on the ladders the gate reads?

The step spans two rungs and inherits the phase error of both, so it is compared
against the pose spread at the COARSER of the two -- the one that dominates.
"""

from __future__ import annotations

import json

import numpy as np

from sashimi.debye.backend import DebyeSolver
from sashimi.debye.grid import size_grid
from sashimi.debye.options import DebyeOptions
from sashimi.invariants import posed
from sashimi.pqr import read_pqr
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, SolventModel, SurfaceModel

LADDER = (1.0, 0.5, 0.25)
POSES = 12
structure = read_pqr("tests/data/ala-gly.pqr")


def energy(struct, model, res, w):
    r = FiniteDifferenceRequest(
        structure=struct,
        solvent=SolventModel(surface_model=model),
        grid=GridSpec(resolution=res, padding=10.0),
        want_potential=False,
    )
    return float(DebyeSolver(options=DebyeOptions(dielectric_smoothing=w)).solve(r).energy_kj_mol)


def achieved(res):
    g = size_grid(structure, GridSpec(resolution=res, padding=10.0))
    return float(np.prod(np.asarray(g.spacing, float)) ** (1 / 3))


out = []
for model in (SurfaceModel.VAN_DER_WAALS, SurfaceModel.MOLECULAR):
    for w in (0.0, 0.25, 0.5):
        rungs = [energy(structure, model, res, w) for res in LADDER]
        d_last = rungs[1] - rungs[2]
        # the pose spread at the COARSE rung of that difference
        coarse = LADDER[1]
        spread = [
            energy(posed(structure, i, spacing=coarse), model, coarse, w) for i in range(POSES)
        ]
        std = float(np.std(spread, ddof=1))
        ptp = float(np.ptp(spread))
        mean = float(np.mean(spread))
        out.append(
            dict(
                model=str(model),
                w=w,
                rungs=rungs,
                d_last=d_last,
                std=std,
                ptp=ptp,
                disp=std / abs(mean),
                ratio_std=abs(d_last) / std,
                ratio_ptp=abs(d_last) / ptp,
                achieved=[achieved(r) for r in LADDER],
            )
        )
        print(
            f"{model} w={w}: |d_last|={abs(d_last):8.4f} kJ  pose "
            f"std={std:7.4f} ({100 * std / abs(mean):.3f}%) "
            f"ptp={ptp:7.4f}  |d_last|/std={abs(d_last) / std:6.2f}x "
            f" /ptp={abs(d_last) / ptp:6.2f}x"
        )
json.dump(
    out,
    open(
        "studies/refinement/results/step_vs_pose.json",
        "w",
    ),
)
