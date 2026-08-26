import sys
import time

from sashimi.debye.backend import DebyeSolver
from sashimi.debye.options import DebyeOptions
from sashimi.pqr import read_pqr
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, SolventModel, SurfaceModel

res = float(sys.argv[1])
path = sys.argv[2]
s = read_pqr(path)


def solve(w):
    r = FiniteDifferenceRequest(
        structure=s,
        solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS),
        grid=GridSpec(resolution=res, padding=10.0),
        want_potential=False,
    )
    t = time.process_time()
    e = DebyeSolver(options=DebyeOptions(dielectric_smoothing=w)).solve(r).energy_kj_mol
    return float(e), time.process_time() - t


solve(0.0)
solve(0.5)  # warm the JIT
hard, ramp = [], []
for _ in range(3):  # interleaved, so drift hits both
    e0, t0 = solve(0.0)
    hard.append(t0)
    e1, t1 = solve(0.5)
    ramp.append(t1)
print(
    f"{path} res={res}  hard {min(hard):6.2f}s  ramp {min(ramp):6.2f}s  "
    f"ratio {min(ramp) / min(hard):.2f}x   E_hard {e0:.6f}  E_ramp {e1:.6f}"
)
