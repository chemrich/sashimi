import math
import time

import numpy as np

from sashimi.constants import ANGSTROM, BOLTZMANN, ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY
from sashimi.corpus import kirkwood_pqr
from sashimi.debye.backend import DebyeSolver
from sashimi.debye.grid import size_grid
from sashimi.debye.options import DebyeOptions
from sashimi.pqr import parse_pqr
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, SolventModel, SurfaceModel

KT = BOLTZMANN * 298.15 / ELEMENTARY_CHARGE
PRE = ELEMENTARY_CHARGE / (4 * math.pi * VACUUM_PERMITTIVITY * ANGSTROM) / KT


def exact_A(n, d, ep=1.0, es=78.54):
    return PRE * (2 * n + 1) / (n * ep + (n + 1) * es) * d**n


def legmat(N, x):
    P = np.zeros((N + 1, len(x)))
    P[0] = 1.0
    if N >= 1:
        P[1] = x
    for k in range(1, N):
        P[k + 1] = ((2 * k + 1) * x * P[k] - k * P[k - 1]) / (k + 1)
    return P


NT, NPHI, NMAX = 24, 16, 4
mu, w = np.polynomial.legendre.leggauss(NT)
ph = 2 * np.pi * np.arange(NPHI) / NPHI
P = legmat(NMAX, mu)
st = np.sqrt(1 - mu**2)


def shell(r):
    return np.array(
        [[r * mu[i], r * st[i] * np.cos(p), r * st[i] * np.sin(p)] for i in range(NT) for p in ph]
    )


def proj(v, r):
    az = v.reshape(NT, NPHI).mean(axis=1)
    return [(2 * n + 1) / 2 * r ** (n + 1) * float(np.sum(w * P[n] * az)) for n in range(NMAX + 1)]


a, d = 3.0, 1.5
pqr = parse_pqr(kirkwood_pqr(a, d))
solv = SolventModel(
    solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.VAN_DER_WAALS
)
MP = 20_000_000
for req in (0.9, 0.45, 0.21):
    spec = GridSpec(resolution=req, padding=10.0, max_points=MP)
    g = size_grid(pqr, spec)
    h = max(g.spacing)
    print("request %.3f -> shape %s h=%.7f" % (req, g.shape, h))
    for wsm in (0.0, 0.5):
        t = time.perf_counter()
        res = DebyeSolver(options=DebyeOptions(dielectric_smoothing=wsm)).solve(
            FiniteDifferenceRequest(
                structure=pqr, solvent=solv, grid=spec, want_energy=False, want_potential=True
            )
        )
        dt = time.perf_counter() - t
        out = []
        for k in (2, 4, 8):
            r = a + k * h
            A = proj(res.potential.value_at(shell(r)), r)
            out.append(
                "k=%d e0=%+.2e e1=%+.2e e2=%+.2e"
                % (k, A[0] / exact_A(0, d) - 1, A[1] / exact_A(1, d) - 1, A[2] / exact_A(2, d) - 1)
            )
        print("   w=%.2f %.2fs " % (wsm, dt) + " | ".join(out))
