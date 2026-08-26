import numpy as np

from sashimi.debye.backend import DebyeSolver
from sashimi.debye.options import DebyeOptions
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, PQRData, SolventModel, SurfaceModel

for radius in (2.0, 3.0):
    for res in (0.5, 0.25):
        for w in (0.0, 0.25, 0.5):
            s = PQRData(coords=np.zeros((1, 3)), charges=np.array([1.0]), radii=np.array([radius]))
            r = FiniteDifferenceRequest(
                structure=s,
                solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS, ionic_strength=0.0),
                grid=GridSpec(resolution=res, padding=10.0),
                want_potential=False,
            )
            e = DebyeSolver(options=DebyeOptions(dielectric_smoothing=w)).solve(r).energy_kj_mol
            print(f"a={radius} h={res} w={w}: {e!r}")
