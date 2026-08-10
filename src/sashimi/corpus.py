"""The golden corpus: a fixed manifest, built once and verified forever after.

This is a first-class deliverable, not a test artifact. `build` runs every case
in the manifest and records a compact summary — grid geometry, energy, potential
statistics, and the potential at pinned probe points. `verify` re-runs the same
manifest against any `Solver` and reports what moved.

Day one it is a regression net for sashimi itself, and for the system APBS that
no lockfile pins any more. The day debye exists, `verify(DebyeSolver())` is its
acceptance test, with APBS ground truth already baked in and no APBS
installation required to run it.

Cases start from PQR, never PDB. Preparation is pdb2pqr's business and it has
its own version; starting from a checked-in PQR means a corpus diff implicates
the solver rather than the structure-prep pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sashimi.pqr import parse_pqr, read_pqr
from sashimi.protocol import (
    DIMENSIONS,
    FiniteDifferenceRequest,
    FloatArray,
    GridSpec,
    PotentialGrid,
    PQRData,
    SolventModel,
    Solver,
)

__all__ = [
    "MANIFEST",
    "Case",
    "Discrepancy",
    "Tolerances",
    "build_case",
    "build_manifest",
    "load_summary",
    "summary_path",
    "verify_case",
    "verify_manifest",
    "write_summary",
]

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "data"
CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "corpus"

# A single charged sphere with a closed-form answer. Defined in code rather than
# as a file so the canonical analytic case cannot drift on disk.
BORN_ION_PQR = "ATOM      1  I   ION     1       0.000   0.000  0.000  1.00  3.00\n"

N_PROBES = 50
PROBE_SEED = 20260809  # pinned; probe placement must never move between builds
PROBE_INSET = 0.6  # sample the middle 60% of the box, away from boundary effects


@dataclass(frozen=True)
class Case:
    """One reproducible solve. Everything that affects the numbers lives here."""

    name: str
    description: str
    source: str  # "born-ion" for the built-in, else a filename in tests/data
    grid: GridSpec
    solvent: SolventModel
    compute_energy: bool = True

    def request(self) -> FiniteDifferenceRequest:
        """The case as a solver request. Every case is finite-difference today;
        a BEM case would build a `BoundaryElementRequest` here instead."""
        return FiniteDifferenceRequest(
            structure=self.structure(),
            solvent=self.solvent,
            grid=self.grid,
            want_energy=self.compute_energy,
            want_potential=True,
        )

    def structure(self) -> PQRData:
        if self.source == "born-ion":
            return parse_pqr(BORN_ION_PQR)
        path = DATA_DIR / self.source
        if not path.is_file():
            raise FileNotFoundError(f"corpus case {self.name!r} needs {path}")
        return read_pqr(path)


@dataclass(frozen=True)
class Tolerances:
    """How far a backend may drift before `verify` calls it a change.

    Energies are an integrated scalar and reproduce tightly across platforms.
    Pointwise potentials are interpolated off the grid and carry more float
    noise, and near a zero crossing a relative tolerance is meaningless — hence
    the absolute floor.
    """

    energy_rtol: float = 1e-4
    potential_rtol: float = 1e-3
    potential_atol: float = 1e-4
    stats_rtol: float = 1e-3
    geometry_atol: float = 1e-9


@dataclass
class Discrepancy:
    case: str
    field: str
    expected: float | list[int]
    actual: float | list[int]
    detail: str = ""

    def __str__(self) -> str:
        suffix = f" ({self.detail})" if self.detail else ""
        return f"{self.case}: {self.field} expected {self.expected}, got {self.actual}{suffix}"


MANIFEST: tuple[Case, ...] = (
    Case(
        name="born-ion-coarse",
        description="Born ion, +1e on a 3 A sphere, vacuum reference. Closed form exists.",
        source="born-ion",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(solute_dielectric=1.0, ionic_strength=0.0),
    ),
    Case(
        name="born-ion-fine",
        description="Born ion at 2x resolution. Pairs with the coarse case to show convergence.",
        source="born-ion",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(solute_dielectric=1.0, ionic_strength=0.0),
    ),
    Case(
        name="born-ion-salt",
        description="Born ion in 150 mM 1:1 salt. Exercises the ion-declaration path.",
        source="born-ion",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(solute_dielectric=1.0, ionic_strength=0.15),
    ),
    Case(
        name="peptide-default",
        description="ALA-GLY dipeptide at physiological salt with sashimi's defaults.",
        source="ala-gly.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(),
    ),
    Case(
        name="peptide-low-dielectric",
        description="Same peptide with a harder solute interior; catches dielectric plumbing.",
        source="ala-gly.pqr",
        grid=GridSpec(resolution=0.6, padding=8.0),
        solvent=SolventModel(solute_dielectric=1.0, ionic_strength=0.0),
    ),
)


def probe_points(origin: FloatArray, spacing: FloatArray, shape: tuple[int, ...]) -> FloatArray:
    """Deterministic sample points inside a grid.

    Seeded and derived only from grid geometry, so the same case always probes
    the same coordinates. Points are kept off the boundary, where the smoothed
    dielectric surface makes values sensitive to details that are not the
    solver's arithmetic.
    """
    extent = (np.asarray(shape) - 1) * spacing
    centre = origin + extent / 2
    half = extent * PROBE_INSET / 2
    rng = np.random.default_rng(PROBE_SEED)
    offsets = rng.uniform(-1.0, 1.0, size=(N_PROBES, DIMENSIONS)) * half
    return np.asarray(centre + offsets, dtype=np.float64)


def build_case(solver: Solver[FiniteDifferenceRequest], case: Case) -> dict[str, Any]:
    """Solve one case and reduce it to a checkable summary."""
    result = solver.solve(case.request())
    grid = result.potential
    if not isinstance(grid, PotentialGrid):
        raise TypeError(
            f"corpus case {case.name!r} expected a volumetric potential, got "
            f"{type(grid).__name__}. A surface-returning backend needs its own case shape."
        )
    points = probe_points(grid.origin, grid.spacing, grid.shape)
    stats = grid.stats()

    return {
        "name": case.name,
        "description": case.description,
        "source": case.source,
        "backend": result.provenance.backend,
        "grid_spec": {
            "resolution": case.grid.resolution,
            "padding": case.grid.padding,
            "max_points": case.grid.max_points,
        },
        "solvent_model": _solvent_dict(case.solvent),
        "resolved_parameters": result.provenance.resolved_parameters,
        "geometry": {
            "shape": list(grid.shape),
            "origin": [float(v) for v in grid.origin],
            "spacing": [float(v) for v in grid.spacing],
        },
        "energy_kj_mol": result.energy_kj_mol,
        "potential_stats_kT_e": {key: float(stats[key]) for key in ("min", "max", "mean", "std")},
        "probes": {
            "seed": PROBE_SEED,
            "points": [[float(c) for c in p] for p in points],
            "values_kT_e": [float(v) for v in grid.value_at(points)],
        },
    }


def _solvent_dict(solvent: SolventModel) -> dict[str, Any]:
    return {
        "solvent_dielectric": solvent.solvent_dielectric,
        "solute_dielectric": solvent.solute_dielectric,
        "ionic_strength": solvent.ionic_strength,
        "ion_radius": solvent.ion_radius,
        "temperature": solvent.temperature,
        "surface_model": solvent.surface_model.value,
        "surface_radius": solvent.surface_radius,
    }


def summary_path(case: Case, directory: Path | None = None) -> Path:
    return (directory or CORPUS_DIR) / f"{case.name}.json"


def write_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")


def load_summary(case: Case, directory: Path | None = None) -> dict[str, Any]:
    path = summary_path(case, directory)
    if not path.is_file():
        raise FileNotFoundError(f"no recorded summary for {case.name!r} at {path}")
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def build_manifest(
    solver: Solver[FiniteDifferenceRequest],
    cases: tuple[Case, ...] = MANIFEST,
    directory: Path | None = None,
) -> list[Path]:
    written = []
    for case in cases:
        path = summary_path(case, directory)
        write_summary(build_case(solver, case), path)
        written.append(path)
    return written


def verify_case(
    solver: Solver[FiniteDifferenceRequest],
    case: Case,
    recorded: dict[str, Any],
    tolerances: Tolerances = Tolerances(),  # noqa: B008 — frozen dataclass
) -> list[Discrepancy]:
    """Re-solve a case and diff it against what was recorded."""
    found: list[Discrepancy] = []
    fresh = build_case(solver, case)

    if fresh["geometry"]["shape"] != recorded["geometry"]["shape"]:
        # Geometry drift invalidates every pointwise comparison below, so stop.
        return [
            Discrepancy(
                case.name,
                "geometry.shape",
                recorded["geometry"]["shape"],
                fresh["geometry"]["shape"],
                "grid sizing changed; pointwise comparison skipped",
            )
        ]

    for key in ("origin", "spacing"):
        a = np.array(recorded["geometry"][key])
        b = np.array(fresh["geometry"][key])
        if not np.allclose(a, b, atol=tolerances.geometry_atol):
            found.append(Discrepancy(case.name, f"geometry.{key}", a.tolist(), b.tolist()))

    expected_energy, actual_energy = recorded["energy_kj_mol"], fresh["energy_kj_mol"]
    if (expected_energy is None) != (actual_energy is None):
        found.append(Discrepancy(case.name, "energy_kj_mol", expected_energy, actual_energy))
    elif (
        expected_energy is not None
        and actual_energy is not None
        and not _close(actual_energy, expected_energy, tolerances.energy_rtol, 0.0)
    ):
        found.append(
            Discrepancy(
                case.name,
                "energy_kj_mol",
                expected_energy,
                actual_energy,
                f"{_relative(actual_energy, expected_energy):.3%} off",
            )
        )

    for key, expected in recorded["potential_stats_kT_e"].items():
        actual = fresh["potential_stats_kT_e"][key]
        if not _close(actual, expected, tolerances.stats_rtol, tolerances.potential_atol):
            found.append(Discrepancy(case.name, f"potential_stats.{key}", expected, actual))

    found.extend(_verify_probes(case, recorded, fresh, tolerances))
    return found


def _verify_probes(
    case: Case,
    recorded: dict[str, Any],
    fresh: dict[str, Any],
    tolerances: Tolerances,
) -> list[Discrepancy]:
    expected = np.array(recorded["probes"]["values_kT_e"])
    actual = np.array(fresh["probes"]["values_kT_e"])
    if expected.shape != actual.shape:
        return [
            Discrepancy(case.name, "probes.count", len(expected), len(actual), "probe set changed")
        ]

    off = ~np.isclose(
        actual, expected, rtol=tolerances.potential_rtol, atol=tolerances.potential_atol
    )
    if not off.any():
        return []

    worst = int(np.argmax(np.abs(actual - expected)))
    return [
        Discrepancy(
            case.name,
            "probes.values_kT_e",
            float(expected[worst]),
            float(actual[worst]),
            f"{int(off.sum())}/{len(expected)} probes differ; worst at index {worst}",
        )
    ]


def verify_manifest(
    solver: Solver[FiniteDifferenceRequest],
    cases: tuple[Case, ...] = MANIFEST,
    tolerances: Tolerances = Tolerances(),  # noqa: B008 — frozen dataclass
    directory: Path | None = None,
) -> list[Discrepancy]:
    found: list[Discrepancy] = []
    for case in cases:
        found.extend(verify_case(solver, case, load_summary(case, directory), tolerances))
    return found


def _close(actual: float, expected: float, rtol: float, atol: float) -> bool:
    return bool(np.isclose(actual, expected, rtol=rtol, atol=atol))


def _relative(actual: float, expected: float) -> float:
    return abs(actual - expected) / abs(expected) if expected else float("inf")
