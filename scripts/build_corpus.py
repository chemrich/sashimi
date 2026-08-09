"""Regenerate the golden corpus.

Run this only after a deliberate, reviewed change to the solver or its inputs:
`uv run python scripts/build_corpus.py`. A diff in the output means the numbers
moved, which is exactly what `tests/test_corpus.py` exists to catch — so
regenerating to make a failing test pass defeats the point.

This is the phase 0/1 slice of what PLAN.md section 7 describes. The full
`sashimi corpus build` / `corpus verify` CLI arrives in phase 3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sashimi.apbs import ApbsSolver, discover_apbs
from sashimi.protocol import GridSpec, PQRData, SolventModel

CORPUS = Path(__file__).resolve().parent.parent / "tests" / "corpus" / "born-sashimi.json"

# Classic Born setup: vacuum reference, no mobile ions.
SOLVENT = SolventModel(
    solvent_dielectric=78.54,
    solute_dielectric=1.0,
    ionic_strength=0.0,
    surface_method="smol",
    surface_radius=1.4,
    temperature=298.15,
)
RESOLUTIONS = (0.5, 0.25)
PADDING = 10.0
PROBE_X = (3.75, 4.5, 5.25, 6.0, 7.5)  # all >= 1.25a; see PLAN.md section 7


def born_ion() -> PQRData:
    return PQRData(
        coords=np.zeros((1, 3)),
        charges=np.array([1.0]),
        radii=np.array([3.0]),
        labels=("ION 1 I",),
    )


def build() -> dict[str, Any]:
    binary = discover_apbs()
    pqr = born_ion()
    probes = np.array([[x, 0.0, 0.0] for x in PROBE_X])

    cases: dict[str, Any] = {}
    for resolution in RESOLUTIONS:
        result = ApbsSolver().solve_lpbe(
            pqr,
            GridSpec(resolution=resolution, padding=PADDING),
            SOLVENT,
            compute_energy=True,
        )
        values = result.potential.value_at(probes)
        cases[f"res={resolution}"] = {
            "grid_spec": {"resolution": resolution, "padding": PADDING},
            "dime": result.diagnostics["dime"],
            "spacing": result.diagnostics["spacing_achieved"],
            "energy_kj_mol": result.energy_kj_mol,
            "potential_at_x": {str(x): float(v) for x, v in zip(PROBE_X, values, strict=True)},
        }

    return {
        "case": "born-ion",
        "description": (
            "Born ion (+1e, 3 A) solved through sashimi's own GridSpec/SolventModel. "
            "Regenerate with `uv run python scripts/build_corpus.py` after a "
            "deliberate, reviewed change; a diff here means the numbers moved."
        ),
        "solvent_model": {
            "solvent_dielectric": SOLVENT.solvent_dielectric,
            "solute_dielectric": SOLVENT.solute_dielectric,
            "ionic_strength": SOLVENT.ionic_strength,
            "surface_method": SOLVENT.surface_method,
            "surface_radius": SOLVENT.surface_radius,
            "temperature": SOLVENT.temperature,
        },
        "reference_backend": binary.label,
        "cases": cases,
    }


def main() -> None:
    doc = build()
    CORPUS.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {CORPUS.relative_to(Path.cwd())} against {doc['reference_backend']}")  # noqa: T201
    for name, case in doc["cases"].items():
        print(f"  {name}: {case['energy_kj_mol']:.6f} kJ/mol")  # noqa: T201


if __name__ == "__main__":
    main()
