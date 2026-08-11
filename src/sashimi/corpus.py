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
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from sashimi.analytic import born_solvation_energy
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
    "TIER_ORDER",
    "AnalyticReference",
    "BackendReference",
    "Case",
    "CaseTier",
    "Discrepancy",
    "RecordedReference",
    "Reference",
    "Tolerances",
    "build_case",
    "build_manifest",
    "cases_for_tier",
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


def born_ion_pqr(radius: float, charge: float = 1.0) -> str:
    """One sphere at the origin, as PQR text.

    Generated rather than checked in, for the same reason `BORN_ION_PQR` is a
    literal: the cases with closed forms are the only ones that can prove a
    backend is *right* rather than merely unchanged, and a file on disk is a
    thing that can drift out from under them.
    """
    return f"ATOM      1  I   ION     1       0.000   0.000  0.000 {charge:5.2f} {radius:5.2f}\n"


N_PROBES = 50
PROBE_SEED = 20260809  # pinned; probe placement must never move between builds
PROBE_INSET = 0.6  # sample the middle 60% of the box, away from boundary effects


class CaseTier(StrEnum):
    """How much of the corpus to run, because not all of it can run every push.

    Cumulative: `STANDARD` includes `FAST`, `FULL` includes both. The split is
    wall time, not importance — a 16,000-atom solve is 15 s per backend per
    platform, and a corpus that made every push wait for it would be a corpus
    people turn off.
    """

    FAST = "fast"  # seconds in total; every push
    STANDARD = "standard"  # ~2 minutes; every push
    FULL = "full"  # tens of minutes; nightly or on demand


TIER_ORDER: tuple[CaseTier, ...] = (CaseTier.FAST, CaseTier.STANDARD, CaseTier.FULL)


@dataclass(frozen=True)
class AnalyticReference:
    """A closed-form answer for a case, where one exists.

    The corpus's recorded summaries answer "has this backend changed?" to 1e-4.
    They cannot answer "is it right?" — a backend can reproduce a wrong number
    forever, and four of the five original cases had no independent check at all.
    Where geometry admits a closed form, this carries it, and `verify_case`
    checks both: tight against the recording, loose against the physics.

    `rtol` is per-case because the gap is discretization, not arithmetic: it
    shrinks as the grid refines, and a Born ion at 0.5 A is legitimately 2.4%
    from exact where the same case at 0.125 A is 0.4%.
    """

    energy_kj_mol: float
    rtol: float
    source: str  # how the number was derived, for the summary


@dataclass(frozen=True)
class Case:
    """One reproducible solve. Everything that affects the numbers lives here."""

    name: str
    description: str
    source: str  # "born-ion" for the built-in, else a filename in tests/data
    grid: GridSpec
    solvent: SolventModel
    compute_energy: bool = True
    tier: CaseTier = CaseTier.FAST
    analytic: AnalyticReference | None = None

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
        if self.source in SYNTHETIC:
            return parse_pqr(SYNTHETIC[self.source])
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


class Reference(Protocol):
    """Where the numbers a case is checked against come from.

    `verify_case` never cared whether its reference was loaded from disk or
    produced by another backend — the comparison is identical either way. Making
    that explicit is what turns cross-solver validation (ROADMAP.md §8) into a
    second implementation of this protocol rather than a second comparison
    engine.
    """

    @property
    def label(self) -> str:
        """How this reference names itself in a discrepancy report."""
        ...

    def summary_for(self, case: Case) -> dict[str, Any]:
        """The recorded or freshly-computed summary for `case`."""
        ...


@dataclass(frozen=True)
class RecordedReference:
    """The checked-in golden summaries. Answers "has this backend changed?"."""

    directory: Path | None = None

    @property
    def label(self) -> str:
        return "recorded corpus"

    def summary_for(self, case: Case) -> dict[str, Any]:
        return load_summary(case, self.directory)


@dataclass(frozen=True)
class BackendReference:
    """Another live solver. Answers "do these two backends agree right now?".

    This is `sashimi validate` in embryo; phase 7 adds the CLI around it.
    """

    solver: Solver[FiniteDifferenceRequest]
    name: str = "reference backend"

    @property
    def label(self) -> str:
        return self.name

    def summary_for(self, case: Case) -> dict[str, Any]:
        return build_case(self.solver, case)


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


# Synthetic structures, keyed by the `source` a case names. Every one has a
# closed-form answer; see `sashimi.analytic`.
SYNTHETIC: dict[str, str] = {
    "born-ion": BORN_ION_PQR,
    **{f"born-ion-r{r:g}": born_ion_pqr(r) for r in (1.0, 2.0, 4.0, 6.0)},
    "born-ion-negative": born_ion_pqr(3.0, -1.0),
    "born-ion-divalent": born_ion_pqr(3.0, 2.0),
}


def _born(
    radius: float, charge: float = 1.0, solute_dielectric: float = 1.0, *, rtol: float
) -> AnalyticReference:
    """A Born case's closed form, computed from CODATA constants rather than quoted.

    `rtol` is measured, not chosen: it is roughly twice the discretization error
    APBS 3.4.1 actually shows on that geometry. Loose enough to survive a
    platform, tight enough that a unit error or a factor of two cannot hide.
    """
    return AnalyticReference(
        energy_kj_mol=born_solvation_energy(radius, charge, solute_dielectric, 78.54),
        rtol=rtol,
        source=f"Born: q={charge:g}e, a={radius:g} A, eps_p={solute_dielectric:g}",
    )


MANIFEST: tuple[Case, ...] = (
    Case(
        name="born-ion-coarse",
        description="Born ion, +1e on a 3 A sphere, vacuum reference. Closed form exists.",
        source="born-ion",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(solute_dielectric=1.0, ionic_strength=0.0),
        analytic=_born(3.0, rtol=0.015),  # measured 0.619%
    ),
    Case(
        name="born-ion-fine",
        description="Born ion at 2x resolution. Pairs with the coarse case to show convergence.",
        source="born-ion",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(solute_dielectric=1.0, ionic_strength=0.0),
        analytic=_born(3.0, rtol=0.008),  # measured 0.278%; the pair's whole point
    ),
    Case(
        name="born-ion-salt",
        description="Born ion in 150 mM 1:1 salt. Exercises the ion-declaration path.",
        source="born-ion",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(solute_dielectric=1.0, ionic_strength=0.15),
        # Deliberately no analytic reference. The Debye-Huckel screening term
        # depends on an ion-exclusion convention the backends do not share:
        # APBS's ionic contribution is -0.688 kJ/mol here and DelPhi's is
        # -0.496, both reporting `polar-solvation`. Pinning either as "the"
        # closed form would encode one code's convention as physics.
        # `sashimi.analytic.screened_born_solvation_energy` records the details.
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
    # --- the analytic sweep -------------------------------------------------
    #
    # Every case below has an exact answer, so each one asks whether the solver
    # is *right* rather than whether it has *changed*. Together they sweep the
    # three parameters the Born expression depends on — radius, charge, solute
    # dielectric — which turns a single agreeing number into a functional form
    # that has to agree. A missing factor of two passes one case and fails eight.
    Case(
        name="born-ion-r1-coarse",
        description=(
            "1 A sphere at 0.5 A spacing: two grid points across the ion. The "
            "case that documents where the discretization gives up — 5.1% off, "
            "and the corpus says so rather than pretending otherwise."
        ),
        source="born-ion-r1",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(solute_dielectric=1.0, ionic_strength=0.0),
        analytic=_born(1.0, rtol=0.08),  # measured 5.086%
    ),
    Case(
        name="born-ion-r1-fine",
        description="The same undersized ion at 0.25 A: 5.1% becomes 3.2%, converging.",
        source="born-ion-r1",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(solute_dielectric=1.0, ionic_strength=0.0),
        analytic=_born(1.0, rtol=0.055),  # measured 3.197%
    ),
    Case(
        name="born-ion-r2",
        description="2 A sphere. Error falls to 0.8% once the ion spans a few grid points.",
        source="born-ion-r2",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(solute_dielectric=1.0, ionic_strength=0.0),
        analytic=_born(2.0, rtol=0.02),  # measured 0.796%
    ),
    Case(
        name="born-ion-r4",
        description="4 A sphere; with 3 A and 6 A this is the radius arm of the sweep.",
        source="born-ion-r4",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(solute_dielectric=1.0, ionic_strength=0.0),
        analytic=_born(4.0, rtol=0.015),  # measured 0.601%
    ),
    Case(
        name="born-ion-r6",
        description="6 A sphere, the best-resolved of the sweep at 0.46%.",
        source="born-ion-r6",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(solute_dielectric=1.0, ionic_strength=0.0),
        analytic=_born(6.0, rtol=0.015),  # measured 0.461%
    ),
    Case(
        name="born-ion-negative",
        description=(
            "-1e on the same 3 A sphere. Solvation goes as q^2, so this must "
            "return the +1e energy exactly; a sign handled wrongly anywhere in "
            "the charge pipeline shows up here and nowhere else in the corpus."
        ),
        source="born-ion-negative",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(solute_dielectric=1.0, ionic_strength=0.0),
        analytic=_born(3.0, -1.0, rtol=0.015),  # measured 0.619%, same as +1e
    ),
    Case(
        name="born-ion-divalent",
        description="+2e: the q^2 scaling, which must be 4x the +1e energy and is.",
        source="born-ion-divalent",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(solute_dielectric=1.0, ionic_strength=0.0),
        analytic=_born(3.0, 2.0, rtol=0.015),  # measured 0.619%
    ),
    Case(
        name="born-ion-solute-eps2",
        description="Solute dielectric 2, the protein-interior value. Exercises 1/eps_p - 1/eps_s.",
        source="born-ion",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(solute_dielectric=2.0, ionic_strength=0.0),
        analytic=_born(3.0, 1.0, 2.0, rtol=0.015),  # measured 0.598%
    ),
    Case(
        name="born-ion-solute-eps4",
        description="Solute dielectric 4; with eps_p 1 and 2 this is the dielectric arm.",
        source="born-ion",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(solute_dielectric=4.0, ionic_strength=0.0),
        analytic=_born(3.0, 1.0, 4.0, rtol=0.015),  # measured 0.571%
    ),
)


def cases_for_tier(tier: CaseTier, cases: tuple[Case, ...] = MANIFEST) -> tuple[Case, ...]:
    """Every case at or below `tier`, in manifest order."""
    allowed = set(TIER_ORDER[: TIER_ORDER.index(tier) + 1])
    return tuple(case for case in cases if case.tier in allowed)


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

    analytic: dict[str, Any] | None = None
    if case.analytic is not None and result.energy_kj_mol is not None:
        exact = case.analytic.energy_kj_mol
        analytic = {
            "energy_kj_mol": exact,
            "rtol": case.analytic.rtol,
            "source": case.analytic.source,
            # Recorded so a summary diff shows convergence moving, not just the
            # energy: this is the number that says whether the solver is right,
            # and it is the only one in the file that is not self-referential.
            "relative_error": abs(result.energy_kj_mol - exact) / abs(exact),
        }

    return {
        "name": case.name,
        "description": case.description,
        "source": case.source,
        "tier": case.tier.value,
        "backend": result.provenance.backend,
        "analytic": analytic,
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
    reference: Reference | dict[str, Any],
    tolerances: Tolerances = Tolerances(),  # noqa: B008 — frozen dataclass
) -> list[Discrepancy]:
    """Re-solve a case and diff it against a reference.

    Accepts a bare summary dict as well as a `Reference`, because that is what
    the comparison actually needs and it keeps ad-hoc use simple.
    """
    recorded = reference if isinstance(reference, dict) else reference.summary_for(case)
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

    found.extend(_verify_analytic(case, fresh))
    found.extend(_verify_probes(case, recorded, fresh, tolerances))
    return found


def _verify_analytic(case: Case, fresh: dict[str, Any]) -> list[Discrepancy]:
    """Check the solver against the closed form, not against its own past.

    Deliberately compares `fresh` to the physics rather than to the recording.
    Every other check here would pass forever on a backend that was wrong from
    the first build; this is the only one that would not.
    """
    if case.analytic is None or fresh.get("analytic") is None:
        return []

    error = fresh["analytic"]["relative_error"]
    if error <= case.analytic.rtol:
        return []
    return [
        Discrepancy(
            case.name,
            "analytic.energy_kj_mol",
            case.analytic.energy_kj_mol,
            fresh["energy_kj_mol"],
            f"{error:.3%} from the closed form, tolerance {case.analytic.rtol:.3%} "
            f"({case.analytic.source})",
        )
    ]


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
    reference: Reference | None = None,
) -> list[Discrepancy]:
    """Verify every case against a reference, the recorded corpus by default.

    Passing a `BackendReference` here is cross-solver validation over the whole
    manifest — the same call, a different reference.
    """
    against = reference if reference is not None else RecordedReference(directory)
    found: list[Discrepancy] = []
    for case in cases:
        found.extend(verify_case(solver, case, against, tolerances))
    return found


def _close(actual: float, expected: float, rtol: float, atol: float) -> bool:
    return bool(np.isclose(actual, expected, rtol=rtol, atol=atol))


def _relative(actual: float, expected: float) -> float:
    return abs(actual - expected) / abs(expected) if expected else float("inf")
