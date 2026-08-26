"""Arm A of the field axis: the ramp against the Born closed form, over grid phase.

Fixed physical sample radius, swept padding. Grid phase swings every FD backend's
near field 5-21x, so a single lattice measures phase; a sweep measures the scheme.
Reports worst AND median over phase, because M1c's lesson is that the swing ratio
flatters a change that only raises the floor.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from sashimi.analytic import born_potential
from sashimi.debye.backend import DebyeSolver
from sashimi.debye.grid import size_grid
from sashimi.debye.options import DebyeOptions
from sashimi.field import MIN_CELLS_OUT, worst_error_over_directions
from sashimi.protocol import FiniteDifferenceRequest, GridSpec, PQRData, SolventModel, SurfaceModel

RADII = (2.0, 3.0)
WIDTHS = (0.0, 0.25, 0.5, 0.75, 1.0, 2.0)
PADDINGS = tuple(np.round(np.linspace(10.0, 20.0, 21), 3))
RESOLUTION = 0.5
OUT_A = 1.5  # the physical distance outside the sphere that every solve is read at

rows = []
for radius in RADII:
    structure = PQRData(coords=np.zeros((1, 3)), charges=np.array([1.0]), radii=np.array([radius]))
    solvent = SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS, ionic_strength=0.0)
    exact = born_potential(radius + OUT_A, 1.0, solvent.solvent_dielectric, solvent.temperature)
    for padding in PADDINGS:
        spec = GridSpec(resolution=RESOLUTION, padding=float(padding))
        grid = size_grid(structure, spec)
        h = float(min(grid.spacing))
        if MIN_CELLS_OUT * h > OUT_A:
            print(
                f"skip a={radius} pad={padding}: {OUT_A:.3f} "
                f"A is under {MIN_CELLS_OUT} cells of {h:.4f}"
            )
            continue
        half = radius + padding
        if half - (radius + OUT_A) < (radius + OUT_A):
            print(f"skip a={radius} pad={padding}: sample does not clear the box")
            continue
        for width in WIDTHS:
            request = FiniteDifferenceRequest(
                structure=structure,
                solvent=solvent,
                grid=spec,
                want_potential=True,
                want_energy=False,
            )
            answer = DebyeSolver(options=DebyeOptions(dielectric_smoothing=width)).solve(request)
            error, direction = worst_error_over_directions(
                answer.potential, np.zeros(3), radius + OUT_A, exact
            )
            rows.append(
                {
                    "radius": radius,
                    "padding": float(padding),
                    "width": width,
                    "h": h,
                    "a_over_h": radius / h,
                    "error": error,
                    "direction": direction,
                }
            )
        print(
            f"a={radius} pad={padding:5.2f} h={h:.4f} a/h={radius / h:6.3f}  "
            + "  ".join(f"w={r['width']}:{100 * r['error']:.3f}%" for r in rows[-len(WIDTHS) :])
        )
        sys.stdout.flush()

json.dump(rows, open(sys.argv[1], "w"))
print()
print(f"{'a':>4} {'w':>5} {'worst':>9} {'median':>9} {'best':>9} {'swing':>7}")
for radius in RADII:
    for width in WIDTHS:
        e = np.array([r["error"] for r in rows if r["radius"] == radius and r["width"] == width])
        if not len(e):
            continue
        print(
            f"{radius:4.1f} {width:5.2f} {100 * e.max():8.3f}% {100 * np.median(e):8.3f}% "
            f"{100 * e.min():8.3f}% {e.max() / e.min():6.2f}x"
        )
