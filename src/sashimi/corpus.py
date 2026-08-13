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

from sashimi.analytic import born_solvation_energy, kirkwood_solvation_energy
from sashimi.pqr import parse_pqr, read_pqr
from sashimi.protocol import (
    DIMENSIONS,
    FiniteDifferenceRequest,
    FloatArray,
    GridSpec,
    Potential,
    PotentialGrid,
    PQRData,
    SolventModel,
    Solver,
    SolverFamily,
    SurfaceModel,
    SurfacePotential,
    System,
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


def kirkwood_pqr(radius: float, offset: float, charge: float = 1.0) -> str:
    """A sphere with its charge off-centre, as two atoms.

    The dielectric boundary is one uncharged sphere of `radius`; the charge is a
    second atom of *zero* radius sitting `offset` from the centre, which adds no
    volume and so leaves the boundary a clean sphere. That is exactly Kirkwood's
    geometry, and it is only expressible because a PQR separates charge from
    radius.
    """
    return (
        f"ATOM      1  C   SPH     1       0.000   0.000  0.000  0.00 {radius:5.2f}\n"
        f"ATOM      2  Q   SPH     1     {offset:7.3f}   0.000  0.000 {charge:5.2f}  0.00\n"
    )


N_PROBES = 50
PROBE_SEED = 20260809  # pinned; probe placement must never move between builds
PROBE_INSET = 0.6  # sample the middle 60% of the box, away from boundary effects


class CaseTier(StrEnum):
    """How much of the corpus to run, because not all of it can run every push.

    Cumulative: `STANDARD` includes `FAST`, `FULL` includes both. The split is
    wall time, not importance — a corpus that made every push wait for it is a
    corpus people turn off.

    Membership is assigned from *measured* per-case cost rather than by
    intuition, which had already drifted once: the 0.25 A cases look small
    because their solutes are, and a Kirkwood sphere at that spacing is 5.2 s
    against a Born ion's 0.47 s at 0.5 A. That put 52 seconds of work in a tier
    whose contract says "seconds". Totals as measured on APBS 3.4.1, osx-arm64:
    `fast` 18 s, `standard` 129 s cumulative, `full` 270 s cumulative.

    The cost is APBS's, which makes it a statement about this backend and no
    other. A boundary-element solver's cost is its mesh: `fas2-molecular` is
    `standard` and meshes in 48 s where `ion-protein-complex-molecular` is a
    third the atoms and takes 450 s. `tests/test_tabipb_solver.py` therefore
    names what it re-verifies per push rather than reading a tier from here.
    """

    FAST = "fast"  # ~10 s in total; `pytest` verifies this one
    STANDARD = "standard"  # ~90 s cumulative; a dedicated CI step per push
    FULL = "full"  # everything; nightly or on demand


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
    mesh_density: float = 2.0  # vertices per square angstrom, for BEM backends

    def request(self) -> FiniteDifferenceRequest:
        """The case as a finite-difference request."""
        return FiniteDifferenceRequest(
            structure=self.structure(),
            solvent=self.solvent,
            grid=self.grid,
            want_energy=self.compute_energy,
            want_potential=True,
        )

    def system(self, *, want_potential: bool = True) -> System:
        """The same case as a family-agnostic question.

        The corpus was finite-difference by construction: `Case` recorded grid
        geometry and `request()` built one request type, so `corpus --backend gb`
        refused and a curated set of physically meaningful systems was
        usable by exactly one backend. `System` is the seam phase 7 built for
        cross-family validation, and it is the same seam this needs — a case is a
        physical question, and which dialect it is asked in is the backend's
        business.
        """
        return System(
            structure=self.structure(),
            solvent=self.solvent,
            grid=self.grid,
            mesh_density=self.mesh_density,
            want_energy=self.compute_energy,
            want_potential=want_potential,
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
    # `str` because some differences are not numeric: which solver family
    # produced a recording, or whether a field was there at all.
    expected: float | list[int] | str
    actual: float | list[int] | str
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
    **{f"kirkwood-{int(f * 10):02d}": kirkwood_pqr(3.0, 3.0 * f) for f in (0.3, 0.5, 0.7, 0.9)},
}


def _kirkwood(offset_fraction: float, *, rtol: float) -> AnalyticReference:
    """The off-centre closed form for a 3 A sphere, computed not quoted."""
    return AnalyticReference(
        energy_kj_mol=kirkwood_solvation_energy(3.0, 3.0 * offset_fraction, 1.0, 1.0, 78.54),
        rtol=rtol,
        source=f"Kirkwood: q=1e at d/a={offset_fraction:g} in a 3 A sphere, eps_p=1",
    )


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
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
        analytic=_born(3.0, rtol=0.015),  # measured 0.619%
    ),
    Case(
        name="born-ion-fine",
        description="Born ion at 2x resolution. Pairs with the coarse case to show convergence.",
        source="born-ion",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
        analytic=_born(3.0, rtol=0.008),  # measured 0.278%; the pair's whole point
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="born-ion-salt",
        description="Born ion in 150 mM 1:1 salt. Exercises the ion-declaration path.",
        source="born-ion",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0,
            ionic_strength=0.15,
            surface_model=SurfaceModel.SMOOTHED_MOLECULAR,
        ),
        # Deliberately no analytic reference. The Debye-Huckel screening term
        # depends on an ion-exclusion convention the backends do not share:
        # APBS's ionic contribution is -0.688 kJ/mol here and DelPhi's is
        # -0.496, both reporting `polar-solvation`. Pinning either as "the"
        # closed form would encode one code's convention as physics.
        # `sashimi.analytic.screened_born_solvation_energy` records the details.
    ),
    Case(
        name="peptide-default",
        description=(
            "ALA-GLY dipeptide at physiological salt on the smoothed molecular "
            "surface. Named for the defaults it was built from; the surface "
            "default moved to `molecular` on 2026-08-13 and this case kept its "
            "question, so the pair with `peptide-molecular` measures the switch."
        ),
        source="ala-gly.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
    ),
    Case(
        name="peptide-low-dielectric",
        description="Same peptide with a harder solute interior; catches dielectric plumbing.",
        source="ala-gly.pqr",
        grid=GridSpec(resolution=0.6, padding=8.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
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
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
        analytic=_born(1.0, rtol=0.08),  # measured 5.086%
    ),
    Case(
        name="born-ion-r1-fine",
        description="The same undersized ion at 0.25 A: 5.1% becomes 3.2%, converging.",
        source="born-ion-r1",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
        analytic=_born(1.0, rtol=0.055),  # measured 3.197%
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="born-ion-r2",
        description="2 A sphere. Error falls to 0.8% once the ion spans a few grid points.",
        source="born-ion-r2",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
        analytic=_born(2.0, rtol=0.02),  # measured 0.796%
    ),
    Case(
        name="born-ion-r4",
        description="4 A sphere; with 3 A and 6 A this is the radius arm of the sweep.",
        source="born-ion-r4",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
        analytic=_born(4.0, rtol=0.015),  # measured 0.601%
    ),
    Case(
        name="born-ion-r6",
        description="6 A sphere, the best-resolved of the sweep at 0.46%.",
        source="born-ion-r6",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
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
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
        analytic=_born(3.0, -1.0, rtol=0.015),  # measured 0.619%, same as +1e
    ),
    Case(
        name="born-ion-divalent",
        description="+2e: the q^2 scaling, which must be 4x the +1e energy and is.",
        source="born-ion-divalent",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
        analytic=_born(3.0, 2.0, rtol=0.015),  # measured 0.619%
    ),
    Case(
        name="born-ion-solute-eps2",
        description="Solute dielectric 2, the protein-interior value. Exercises 1/eps_p - 1/eps_s.",
        source="born-ion",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=2.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
        analytic=_born(3.0, 1.0, 2.0, rtol=0.015),  # measured 0.598%
    ),
    Case(
        name="born-ion-solute-eps4",
        description="Solute dielectric 4; with eps_p 1 and 2 this is the dielectric arm.",
        source="born-ion",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=4.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
        analytic=_born(3.0, 1.0, 4.0, rtol=0.015),  # measured 0.571%
    ),
    # --- real chemistry -----------------------------------------------------
    #
    # Structures vendored from the APBS examples; see
    # tests/data/apbs-examples/PROVENANCE.md for where each came from, why it is
    # here, and why the energies APBS's own READMEs publish for them are a
    # different quantity from the one sashimi reports.
    #
    # None carries an analytic reference — none has a closed form. What they add
    # is the axis the Born sweep cannot cover at all: real charge distributions,
    # real geometry, and the chance for a bug that only shows up above three
    # atoms. Every genuine defect this project has found came from a structure
    # like these rather than from the fixtures (section 12).
    Case(
        name="methanol",
        description=(
            "3-atom neutral, with a 0.2 A hydrogen — the smallest radius in the "
            "corpus and the kind of atom Generalized Born cannot divide by."
        ),
        source="apbs-examples/methanol.pqr",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=2.0,
            solvent_dielectric=78.00,
            ionic_strength=0.0,
            temperature=300.0,
            surface_model=SurfaceModel.SMOOTHED_MOLECULAR,
        ),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="methoxide",
        description="The methanol anion: a -1e solute of two atoms. Pairs with methanol.",
        source="apbs-examples/methoxide.pqr",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=2.0,
            solvent_dielectric=78.00,
            ionic_strength=0.0,
            temperature=300.0,
            surface_model=SurfaceModel.SMOOTHED_MOLECULAR,
        ),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="acetic-acid",
        description="Neutral acid, 8 atoms. With acetate this is an ionization pair.",
        source="apbs-examples/acetic-acid.pqr",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=2.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="acetate",
        description="Its conjugate base at -1e; the charged half of the ionization pair.",
        source="apbs-examples/acetate.pqr",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=2.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="fas2",
        description=(
            "906 atoms, and the only non-integer net charge in the corpus "
            "(+4.053 e) — which is what a real forcefield assignment looks like."
        ),
        source="apbs-examples/fas2.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="barstar",
        description="1,403 atoms at -5e: the most negatively charged case here.",
        source="apbs-examples/barstar.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="barnase",
        description="Barstar's binding partner at +2e; a charge-complementary pair.",
        source="apbs-examples/barnase.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="lysozyme",
        description=(
            "Hen lysozyme, 1,960 atoms at +8e. In the corpus at last: it is the "
            "structure that surfaced the 64-second extrema scan, the DX writer's "
            "first real consumer, and both Generalized Born mistakes."
        ),
        source="apbs-examples/2LZT-ASP66.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="protein-rna",
        description=(
            "2,065 atoms. The only nucleic acid in the corpus, so the only case "
            "where phosphate backbone charges are exercised at all."
        ),
        source="apbs-examples/1a63.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="carbonic-anhydrase",
        description="2,482 atoms, the largest case; 13 s, which is why it is not standard.",
        source="apbs-examples/hca.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
        tier=CaseTier.FULL,
    ),
    # --- charge placement ---------------------------------------------------
    #
    # The Born ion is symmetric in every way a solver could be wrong about
    # direction, so it cannot catch a mistake in *where* a charge is. Kirkwood's
    # series is the same sphere with the charge moved off centre, and every term
    # above the monopole — the whole multipole structure of the reaction field —
    # only exists once it moves.
    Case(
        name="kirkwood-03",
        description="Charge at 0.3 of the way to the boundary of a 3 A sphere.",
        source="kirkwood-03",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
        analytic=_kirkwood(0.3, rtol=0.005),  # measured 0.097%
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="kirkwood-05",
        description="Halfway out. The series is well past the monopole here.",
        source="kirkwood-05",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
        analytic=_kirkwood(0.5, rtol=0.012),  # measured 0.473%
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="kirkwood-07",
        description="0.7 out, where the reaction field is dominated by high multipoles.",
        source="kirkwood-07",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
        analytic=_kirkwood(0.7, rtol=0.005),  # measured 0.114%
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="kirkwood-09",
        description=(
            "0.3 A from the dielectric boundary at 0.25 A spacing — 7.7% out, and "
            "the case that records where charge placement stops being resolvable."
        ),
        source="kirkwood-09",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR
        ),
        analytic=_kirkwood(0.9, rtol=0.12),  # measured 7.678%
        tier=CaseTier.STANDARD,
    ),
    # --- surface models -----------------------------------------------------
    #
    # Until these, every case in the corpus was `smoothed-molecular`, which is
    # APBS's alone. Two consequences, both bad: the single largest modelling
    # choice in the calculation — worth 25.7% on a dipeptide (section 5) — was
    # untested, and *no corpus case could ever be verified against another
    # backend*, which undercuts the corpus's stated job as debye's acceptance
    # gate. Every case below runs on a model at least two backends support.
    Case(
        name="born-ion-molecular",
        description=(
            "The Born ion on the molecular surface, which DelPhi and TABI-PB can "
            "also solve. Rolling a probe over a lone sphere cannot carve a "
            "re-entrant surface, so the boundary is the sphere and Born still holds."
        ),
        source="born-ion",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.MOLECULAR
        ),
        analytic=_born(3.0, rtol=0.05),  # measured 2.36%
    ),
    Case(
        name="born-ion-vdw",
        description=(
            "The same ion with the probe collapsed. For one sphere this must "
            "return the molecular answer exactly — the two surfaces coincide, and "
            "no other case in the corpus can catch a probe applied where it "
            "should not be."
        ),
        source="born-ion",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.VAN_DER_WAALS
        ),
        analytic=_born(3.0, rtol=0.05),
    ),
    Case(
        name="peptide-molecular",
        description="ALA-GLY on the molecular surface: the cross-backend workhorse.",
        source="ala-gly.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
    ),
    Case(
        name="peptide-vdw",
        description="ALA-GLY with no probe. Against `peptide-molecular` this is the 25.7%.",
        source="ala-gly.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS),
    ),
    Case(
        name="methanol-molecular",
        description="A 3-atom solute where the probe genuinely changes the boundary.",
        source="apbs-examples/methanol.pqr",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=2.0,
            solvent_dielectric=78.00,
            ionic_strength=0.0,
            temperature=300.0,
            surface_model=SurfaceModel.MOLECULAR,
        ),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="acetate-molecular",
        description="A charged small molecule on a shared surface model.",
        source="apbs-examples/acetate.pqr",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=2.0, ionic_strength=0.0, surface_model=SurfaceModel.MOLECULAR
        ),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="lysozyme-molecular",
        description="Hen lysozyme where all four backends can be asked the same question.",
        source="apbs-examples/2LZT-ASP66.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
        tier=CaseTier.FULL,
    ),
    Case(
        name="barnase-vdw",
        description="A protein-scale van der Waals boundary; the surface TABI-PB refuses.",
        source="apbs-examples/barnase.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS),
        tier=CaseTier.FULL,
    ),
    # --- the solvent, swept -------------------------------------------------
    Case(
        name="peptide-no-salt",
        description="ALA-GLY at zero ionic strength; the low end of the salt arm.",
        source="ala-gly.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(ionic_strength=0.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
    ),
    Case(
        name="peptide-high-salt",
        description="500 mM: a Debye length of 4.3 A, well inside the solute's own size.",
        source="ala-gly.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(ionic_strength=0.5, surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
    ),
    Case(
        name="peptide-cold",
        description=(
            "277 K. Temperature enters both the Boltzmann factor and the kT/e "
            "the potential is reported in, and a solver that reads it in the "
            "wrong unit still returns a plausible number — which is exactly how "
            "DelPhi's Celsius parameter cost a day (section 12)."
        ),
        source="ala-gly.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(temperature=277.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
    ),
    Case(
        name="peptide-low-solvent-dielectric",
        description="Solvent dielectric 4, a membrane interior rather than water.",
        source="ala-gly.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(solvent_dielectric=4.0, surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
    ),
    # --- charge states, binding partners, and scale --------------------------
    #
    # What a corpus of one-structure-per-chemistry cannot test: the same solute
    # in two charge states, the same protein with and without its ligand, and
    # what happens when the atom count outruns the grid guardrail.
    Case(
        name="aspartate-residue",
        description="A single aspartate at -1e. 12 atoms — a residue, not a molecule.",
        source="apbs-examples/ASP66.pqr",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
    ),
    Case(
        name="lysozyme-protonated",
        description=(
            "2LZT with Asp66 protonated: +9e against `lysozyme`'s +8e, the same "
            "1,960 atoms otherwise. One titratable proton, which is the quantity "
            "a pKa calculation is a difference of."
        ),
        source="apbs-examples/2LZT-ASH66.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="lysozyme-deleted-residue",
        description=(
            "The same structure with Asp66 removed rather than protonated — also "
            "+9e, so it isolates geometry from charge against `lysozyme-protonated`."
        ),
        source="apbs-examples/2LZT-noASP66.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="fkbp-apo",
        description="FKBP with an empty binding site; pairs with `fkbp-dmso`.",
        source="apbs-examples/fkbp-apo.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="fkbp-dmso",
        description=(
            "The same protein with a DMSO molecule bound — ten more atoms and "
            "the same net charge. A binding energy is the difference of these "
            "two, which is what most real users of a PB solver actually want."
        ),
        source="apbs-examples/fkbp-dmso.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="ion-protein-complex",
        description=(
            "260 atoms carrying +21.69 e — by far the most charged solute here, "
            "and small enough that the charge is not spread thin."
        ),
        source="apbs-examples/ion-protein-complex.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
        tier=CaseTier.FULL,
    ),
    Case(
        name="hca-complex",
        description=(
            "Carbonic anhydrase with acetazolamide bound, and the only "
            "*net-neutral* protein in the corpus. Everything else has a monopole "
            "to dominate its solvation energy; this one does not."
        ),
        source="apbs-examples/hca-complex.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
        tier=CaseTier.FULL,
    ),
    Case(
        name="actin-monomer",
        description="5,877 atoms at -12e; where the grid guardrail starts relaxing resolution.",
        source="apbs-examples/actin-monomer.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
        tier=CaseTier.FULL,
    ),
    Case(
        name="acetylcholinesterase",
        description=(
            "8,279 atoms, the largest in the corpus. It asks for 0.5 A and is "
            "given coarser, because `max_points` caps the grid rather than the "
            "atom count — which is why 8,000 atoms costs 15 s and not an hour."
        ),
        source="apbs-examples/mache.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
        tier=CaseTier.FULL,
    ),
    Case(
        name="fas2-fine",
        description=(
            "906 atoms at 0.35 A. The corpus tests convergence only on a single "
            "sphere otherwise; this is the same claim on a real charge distribution."
        ),
        source="apbs-examples/fas2.pqr",
        grid=GridSpec(resolution=0.35, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
        tier=CaseTier.FULL,
    ),
    # --- the shared surface, widened ----------------------------------------
    #
    # The surface-model section above made *five* cases askable by more than one
    # backend, out of fifty. That is the corpus's own stated job — debye's
    # acceptance gate, and the reference tier the approximate one is measured
    # against — resting on five questions, one of which has a single atom.
    #
    # Every case below is a molecular-surface sibling of a case the corpus
    # already has on `smoothed-molecular`, so each one carries an axis that was
    # previously APBS-only into the set every backend can answer: the salt and
    # temperature sweeps, the dielectric arm of the analytic sweep, both halves
    # of an ionization pair, a binding pair, a nucleic acid, and the sign of the
    # charge at protein scale. They add no new machinery and no new structures —
    # only questions that more than one solver can be asked.
    Case(
        name="peptide-molecular-no-salt",
        description=(
            "ALA-GLY at zero ionic strength on the shared surface. With "
            "`peptide-molecular` and `-high-salt` this is the salt arm as all "
            "three solver families see it — screening is where an analytic tier "
            "and a boundary-element one part company with a grid."
        ),
        source="ala-gly.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR, ionic_strength=0.0),
    ),
    Case(
        name="peptide-molecular-high-salt",
        description="500 mM on the shared surface: the high end of the cross-family salt arm.",
        source="ala-gly.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR, ionic_strength=0.5),
    ),
    Case(
        name="peptide-molecular-cold",
        description=(
            "277 K on the shared surface. `peptide-cold` makes this point on a "
            "model only APBS supports, which is the wrong place for it: reading "
            "temperature in the wrong unit is a *cross-backend* bug, and the one "
            "that cost a day when DelPhi turned out to want Celsius (section 12)."
        ),
        source="ala-gly.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR, temperature=277.0),
    ),
    Case(
        name="born-ion-molecular-eps2",
        description=(
            "The dielectric arm of the analytic sweep, on a surface more than "
            "one backend supports. The closed form holds for any solute "
            "dielectric, so this asks whether 1/eps_p - 1/eps_s is right rather "
            "than merely unchanged — the only such question the shared set had "
            "at eps_p other than 1."
        ),
        source="born-ion",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=2.0, ionic_strength=0.0, surface_model=SurfaceModel.MOLECULAR
        ),
        # Measured 2.263% for APBS and 3.093% for Generalized Born, which is the
        # same 3.093% it shows on `born-ion-molecular` at eps_p = 1: the OBC2
        # offset is a property of the method, not of the dielectric, and holding
        # across both cases is evidence the dielectric factor itself is right.
        analytic=_born(3.0, 1.0, 2.0, rtol=0.05),
    ),
    Case(
        name="methoxide-molecular",
        description=(
            "The methanol anion on the shared surface. With `methanol-molecular` "
            "it is an ionization pair every backend can be asked — the neutral "
            "and charged halves of one chemistry, where a solvation model that "
            "is only right about monopoles shows it."
        ),
        source="apbs-examples/methoxide.pqr",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=2.0,
            solvent_dielectric=78.00,
            ionic_strength=0.0,
            temperature=300.0,
            surface_model=SurfaceModel.MOLECULAR,
        ),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="aspartate-residue-molecular",
        description=(
            "One aspartate at -1e, 12 atoms, on the shared surface: between "
            "`acetate-molecular`'s eight atoms and the peptide's twenty, which "
            "is the size range where a boundary-element mesher stops being able "
            "to answer. Measured: TABI-PB aborts on this one immediately, on "
            "`stoul: no conversion` *after* reporting the surface built, where "
            "acetate instead runs past its 600 s timeout. Same size class, two "
            "unrelated mechanisms, so neither is 'too small' as a rule."
        ),
        source="apbs-examples/ASP66.pqr",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="fas2-molecular",
        description=(
            "906 atoms carrying +4.053 e on the shared surface. The only "
            "non-integer net charge any backend but APBS can be handed."
        ),
        source="apbs-examples/fas2.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="acetic-acid-molecular",
        description="The neutral half of the acetate ionization pair, on the shared surface.",
        source="apbs-examples/acetic-acid.pqr",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=2.0, ionic_strength=0.0, surface_model=SurfaceModel.MOLECULAR
        ),
        tier=CaseTier.FULL,
    ),
    Case(
        name="barstar-molecular",
        description=(
            "-5e on 1,403 atoms, on the shared surface. Every other protein a "
            "second backend can take is positively charged, so this is the sign "
            "of the charge at protein scale rather than on a sphere."
        ),
        source="apbs-examples/barstar.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
        tier=CaseTier.FULL,
    ),
    Case(
        name="fkbp-apo-molecular",
        description=(
            "FKBP with an empty site, on the shared surface; pairs with `fkbp-dmso-molecular`."
        ),
        source="apbs-examples/fkbp-apo.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
        tier=CaseTier.FULL,
    ),
    Case(
        name="fkbp-dmso-molecular",
        description=(
            "The bound form on the shared surface. A binding energy is the "
            "difference of these two, and until now it was a difference only "
            "APBS could take — which makes it the one quantity most users "
            "actually want and the one the corpus could not check across tiers."
        ),
        source="apbs-examples/fkbp-dmso.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
        tier=CaseTier.FULL,
    ),
    Case(
        name="protein-rna-molecular",
        description=(
            "2,065 atoms with a phosphate backbone, on the shared surface. No "
            "backend but APBS had ever been handed a nucleic acid by the corpus, "
            "and a radius set assigned per element is exactly the thing that has "
            "no phosphorus entry until someone tries."
        ),
        source="apbs-examples/1a63.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
        tier=CaseTier.FULL,
    ),
    Case(
        name="ion-protein-complex-molecular",
        description=(
            "260 atoms at +21.69 e on the shared surface: the most charged "
            "solute in the corpus, and a united-atom structure with no hydrogens "
            "at all, which is a second radius dialect (see `hca-molecular`)."
        ),
        source="apbs-examples/ion-protein-complex.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
        tier=CaseTier.FULL,
    ),
    Case(
        name="hca-molecular",
        description=(
            "Carbonic anhydrase, 2,482 atoms, on the shared surface — and the "
            "case that says where the approximate tier's default is wrong. It is "
            "a polar-hydrogen structure, so its heavy-atom radii carry the volume "
            "of hydrogens that are not in the file; Generalized Born substitutes "
            "an all-atom set and over-solvates by 21%, against 2-4% on every "
            "all-atom protein here. Measured, not assumed: the file's own radii "
            "give 7.3% on this structure and 26-33% on the all-atom ones."
        ),
        source="apbs-examples/hca.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
        tier=CaseTier.FULL,
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


def build_case(
    solver: Solver[Any],
    case: Case,
    family: SolverFamily = SolverFamily.FINITE_DIFFERENCE,
) -> dict[str, Any]:
    """Solve one case and reduce it to a checkable summary.

    The summary's shape follows what the backend actually returns, which is what
    lets a curated case be recorded for a solver that has no grid. A volumetric
    answer carries geometry, statistics and pinned probes; a surface answer
    carries its vertex count and value statistics; an analytic one carries the
    energy and nothing else, because it computed nothing else.

    `family` is the dialect to ask in, not a claim about the solver: handing an
    FD backend `ANALYTIC` would simply drop the grid from the request.
    """
    # An analytic backend has no field to sample — not "does not implement one",
    # but has none, which is what places it in that family at all. Asking anyway
    # would make every corpus build fail on a refusal that is the correct answer.
    want_potential = family is not SolverFamily.ANALYTIC
    result = solver.solve(case.system(want_potential=want_potential).request_for(family))

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

    summary: dict[str, Any] = {
        "name": case.name,
        "description": case.description,
        "source": case.source,
        "tier": case.tier.value,
        "family": family.value,
        "backend": result.provenance.backend,
        "analytic": analytic,
        "grid_spec": {
            "resolution": case.grid.resolution,
            "padding": case.grid.padding,
            "max_points": case.grid.max_points,
        },
        "solvent_model": _solvent_dict(case.solvent),
        "resolved_parameters": result.provenance.resolved_parameters,
        "energy_kj_mol": result.energy_kj_mol,
    }
    return summary | _potential_summary(result.potential)


def _potential_summary(potential: Potential | None) -> dict[str, Any]:
    """Whatever there is to record about the field, which may be nothing."""
    if isinstance(potential, PotentialGrid):
        points = probe_points(potential.origin, potential.spacing, potential.shape)
        stats = potential.stats()
        return {
            "geometry": {
                "shape": list(potential.shape),
                "origin": [float(v) for v in potential.origin],
                "spacing": [float(v) for v in potential.spacing],
            },
            "potential_stats_kT_e": {
                key: float(stats[key]) for key in ("min", "max", "mean", "std")
            },
            "probes": {
                "seed": PROBE_SEED,
                "points": [[float(c) for c in p] for p in points],
                "values_kT_e": [float(v) for v in potential.value_at(points)],
            },
        }
    if isinstance(potential, SurfacePotential):
        # No probes: the vertices are the mesher's choice and move when it is
        # rebuilt, so pinned coordinates would compare nothing. Statistics over
        # the surface are what survives a remesh.
        stats = potential.stats()
        return {
            "surface": {
                "n_vertices": potential.n_vertices,
                "potential_stats_kT_e": {
                    key: float(stats[key]) for key in ("min", "max", "mean", "std")
                },
            }
        }
    return {}


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
    solver: Solver[Any],
    case: Case,
    reference: Reference | dict[str, Any],
    tolerances: Tolerances = Tolerances(),  # noqa: B008 — frozen dataclass
    family: SolverFamily = SolverFamily.FINITE_DIFFERENCE,
) -> list[Discrepancy]:
    """Re-solve a case and diff it against a reference.

    Accepts a bare summary dict as well as a `Reference`, because that is what
    the comparison actually needs and it keeps ad-hoc use simple.

    What gets compared follows what both sides recorded: a grid answer is
    checked on geometry, statistics and probes, a surface answer on its vertex
    count and statistics, and an analytic one on the energy alone. Comparing a
    recording of one shape against a solve of another is refused rather than
    partially attempted — that is a backend swap, not a drift.
    """
    recorded = reference if isinstance(reference, dict) else reference.summary_for(case)
    found: list[Discrepancy] = []
    fresh = build_case(solver, case, family)

    recorded_family = recorded.get("family", SolverFamily.FINITE_DIFFERENCE.value)
    if recorded_family != fresh["family"]:
        return [
            Discrepancy(
                case.name,
                "family",
                recorded_family,
                fresh["family"],
                "the recording came from a different solver family; nothing else is comparable",
            )
        ]

    if "geometry" in recorded or "geometry" in fresh:
        if "geometry" not in recorded or "geometry" not in fresh:
            return [
                Discrepancy(
                    case.name,
                    "geometry",
                    "present" if "geometry" in recorded else "absent",
                    "present" if "geometry" in fresh else "absent",
                    "one side returned a volumetric field and the other did not",
                )
            ]
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

    for key, expected in recorded.get("potential_stats_kT_e", {}).items():
        actual = fresh["potential_stats_kT_e"][key]
        if not _close(actual, expected, tolerances.stats_rtol, tolerances.potential_atol):
            found.append(Discrepancy(case.name, f"potential_stats.{key}", expected, actual))

    found.extend(_verify_surface(case, recorded, fresh, tolerances))
    found.extend(_verify_analytic(case, fresh))
    if "probes" in recorded and "probes" in fresh:
        found.extend(_verify_probes(case, recorded, fresh, tolerances))
    return found


def _verify_surface(
    case: Case,
    recorded: dict[str, Any],
    fresh: dict[str, Any],
    tolerances: Tolerances,
) -> list[Discrepancy]:
    """Compare a boundary-element answer, which has no grid to compare.

    The vertex count is checked exactly: a remesh that moves it means the
    triangulation changed, and every statistic below is then a statistic of a
    different surface. That is worth reporting rather than absorbing, because
    the mesher's version is part of a TABI-PB result's identity.
    """
    if "surface" not in recorded or "surface" not in fresh:
        return []

    found: list[Discrepancy] = []
    if recorded["surface"]["n_vertices"] != fresh["surface"]["n_vertices"]:
        return [
            Discrepancy(
                case.name,
                "surface.n_vertices",
                recorded["surface"]["n_vertices"],
                fresh["surface"]["n_vertices"],
                "the mesh changed; its statistics describe a different surface",
            )
        ]

    for key, expected in recorded["surface"]["potential_stats_kT_e"].items():
        actual = fresh["surface"]["potential_stats_kT_e"][key]
        if not _close(actual, expected, tolerances.stats_rtol, tolerances.potential_atol):
            found.append(Discrepancy(case.name, f"surface.potential_stats.{key}", expected, actual))
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
