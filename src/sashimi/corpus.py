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
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from sashimi.analytic import (
    born_solvation_energy,
    kirkwood_solvation_energy,
    screened_born_potential,
)
from sashimi.field import (
    FIELD_DIRECTION_NAMES,
    FIELD_DIRECTIONS,
    sample_radii,
    sample_values,
)
from sashimi.pqr import parse_pqr, read_pqr
from sashimi.protocol import (
    DIMENSIONS,
    AccuracyTier,
    FiniteDifferenceRequest,
    FloatArray,
    GridSpec,
    Potential,
    PotentialGrid,
    PQRData,
    SolventModel,
    Solver,
    SolveResult,
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

    **`per_backend_rtol` exists because one tolerance per case is set by the
    worst backend that runs it.** On a sharp boundary APBS is 2.36% from the
    Born closed form where DelPhi C++ is 0.0006%, so a shared tolerance wide
    enough for APBS cannot catch a second solver being percent-wrong — and that
    is the tolerance a future backend's acceptance gate would inherit. Keys are
    matched as a *prefix* of the recorded backend identity (`delphicpp` matches
    `delphicpp-8.6` across versions, and does not match `pydelphi`), because
    that identity is what the summary carries and what accuracy actually
    belongs to.

    **`gated` is how a case can be recorded without being judged.** Kirkwood at
    d/a = 0.9 puts the charge 0.3 A inside the boundary, where the continuum
    model's reaction field diverges: APBS reports 6.40/9.85/9.05% under
    refinement and DelPhi 26.67/4.29/7.50%, both non-monotonic. That is worth
    recording and worth nobody gating on. Without this flag the only way to
    express it is an `rtol` slack enough to absorb 27%, which is a check that
    cannot fail wearing the costume of one that can.
    """

    energy_kj_mol: float
    rtol: float
    source: str  # how the number was derived, for the summary
    per_backend_rtol: tuple[tuple[str, float], ...] = ()
    gated: bool = True

    def rtol_for(self, backend: str) -> float:
        """The tolerance that applies to this backend's answer.

        A tuple of pairs rather than a mapping so the reference stays hashable,
        which `Case` being frozen would otherwise quietly lose.
        """
        for prefix, tolerance in self.per_backend_rtol:
            if backend.startswith(prefix):
                return tolerance
        return self.rtol


@dataclass(frozen=True)
class AnalyticField:
    """A closed-form *potential* to check a solver's field against.

    Every other analytic check in this corpus is on the energy — one integrated
    scalar — and the field is compared only against its own recording, so a
    backend wrong in the field from its first build stays wrong and passes
    forever. That is the half a consumer actually displays: protean colours a
    surface with a potential, and `sashimi.gb` exists because it could not.

    **Sampling is in grid cells beyond the boundary, not in fractions of the
    radius**, and the difference is not pedantry. Interpolating across the
    dielectric interface is O(1) wrong by construction — phi is continuous there
    but its normal derivative is not, and at eps_s/eps_p ~ 78.5 the gradient
    jumps by nearly two orders of magnitude — so a sample must land in a cell
    that does not contain the interface. A fixed `1.05a` does not guarantee
    that: at a = 1 A and 0.25 A spacing the sample sits *inside* the straddling
    cell, and at a = 2 A a cell corner lands exactly on the boundary. `a + k*h`
    holds for every radius, and `k >= 2` keeps the whole stencil clear.

    Sampling *on* the interface is left alone deliberately. Every shipped solver
    is ~100% wrong there and that is not a defect any of them can fix; it is an
    ill-posed question about a grid.

    **Under salt there are two branches and the sampling rule survives both.**
    `screened_born_potential` is Poisson between the dielectric boundary and the
    Stern radius `a + ion_radius`, screened beyond it. A sample may land on the
    Stern radius — `born-ion-vdw-salt` puts DelPhi's **second** one exactly
    there, at r = 5.0 A on its h = 0.5 lattice with `cells_out=(2, 4, 8)` —
    and that is fine where a sample on the *dielectric* boundary is not: eps is
    the same on both sides of the Stern radius, so phi and its first derivative
    are continuous and only the second jumps. The interpolation error there is
    O(h^2), not the O(1) that makes the dielectric interface ill-posed.

    **The sample is a sphere's worth of directions, not one ray**, and that was
    the second parameter this rule turned out to be conditioned on. It shipped
    sampling `centre + r*x_hat` alone, which reads as arbitrary-but-harmless for
    a spherically symmetric problem — and is not, because the *error* is not
    spherically symmetric. A sphere on a Cartesian grid is a staircase, and the
    staircase has the grid's cubic symmetry: at 0.25 A on `born-ion-vdw-fine`,
    two cells out, DelPhi C++ reads +0.736% along an axis and -1.890% along the
    body diagonal, while APBS reads +1.019% and -0.674%. **Which ray is worst
    depends on the backend**, so a single ray recorded APBS's true worst case and
    understated DelPhi's by a factor of 2.6.
    """

    radius_a: float  # the dielectric boundary, A
    charge_e: float
    cells_out: tuple[int, ...]  # k, in achieved grid cells beyond the boundary
    rtol: float
    per_backend_rtol: tuple[tuple[str, float], ...] = ()

    def rtol_for(self, backend: str) -> float:
        """The tolerance that applies to this backend's field, by identity prefix."""
        for prefix, tolerance in self.per_backend_rtol:
            if backend.startswith(prefix):
                return tolerance
        return self.rtol

    def sample_radii(self, spacing: float) -> list[float]:
        return sample_radii(self.radius_a, spacing, self.cells_out)

    def exact_at(self, radii: Sequence[float], solvent: SolventModel) -> list[float]:
        """The closed form, taking every parameter from the case's own solvent.

        **Reading the solvent rather than restating it is the whole point of this
        method**, and it is what makes salt safe to add. Carrying a private
        `solvent_dielectric` — as this did — meant a case could be paired with a
        field reference describing different physics and nothing would say so.
        The concrete instance was salt: `born_potential` is unscreened, and
        attaching it to a 0.15 M case would report about 30% two cells out and
        about half eight cells out as a solver defect — 29.7% and 47.9% on
        APBS's achieved spacing, 31.1% and 52.6% on DelPhi C++'s, since the
        samples are at `a + k*h`.

        M3 closes that by making the reference salt-aware instead of refusing:
        `screened_born_potential` reduces to `born_potential` exactly at zero
        ionic strength, so every pre-existing case is byte-identical and a salted
        one is described correctly. That is lesson 1 of the guards file — an
        illegal state made unrepresentable rather than guarded against — and the
        refusal it replaces was reachable only by writing a case the manifest
        would then have to remember not to write.

        `ion_radius` reaches the expression the same way, which matters more than
        it looks: it is the Stern radius, and debye's `screening_nodes` switches
        its Boltzmann term on at exactly `radius + ion_radius` for the same
        reason. A case that set one and not the other would compare two different
        exclusion conventions.

        **`temperature` was being defaulted rather than read, which is the
        `solvent_dielectric` defect one parameter along**, and no test could have
        seen it: all ten field cases sit at 298.15 K, so the ten recordings are
        byte-identical across this change. It is fixed here because the fix is
        the same one — take every parameter from the case — and because
        `peptide-cold` exists precisely to catch a solver reading a temperature
        in the wrong unit, so a *reference* that ignores temperature would have
        been the mirror image of that trap.
        """
        return [
            screened_born_potential(
                r,
                self.charge_e,
                solvent.solvent_dielectric,
                solvent.temperature,
                radius_a=self.radius_a,
                ionic_strength=solvent.ionic_strength,
                ion_radius=solvent.ion_radius,
            )
            for r in radii
        ]


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
    analytic_field: AnalyticField | None = None
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


def _per_backend_rtol(
    delphi_rtol: float | None, debye_rtol: float | None
) -> tuple[tuple[str, float], ...]:
    """The tight-tolerance pairs, built in one place because the keys matter.

    `rtol_for` matches these as a *prefix* of the recorded backend identity, so a
    mistyped key does not fail — it falls through to the shared tolerance, which
    is set by the least accurate backend on the case. A tight tolerance that
    quietly stops applying is the `per_backend_rtol` instance in the guards file,
    and two copies of these strings is two chances at it.
    """
    pairs: list[tuple[str, float]] = []
    if delphi_rtol is not None:
        pairs.append(("delphicpp", delphi_rtol))
    if debye_rtol is not None:
        pairs.append(("debye", debye_rtol))
    return tuple(pairs)


def _kirkwood(
    offset_fraction: float,
    *,
    rtol: float,
    delphi_rtol: float | None = None,
    debye_rtol: float | None = None,
    gated: bool = True,
) -> AnalyticReference:
    """The off-centre closed form for a 3 A sphere, computed not quoted.

    `debye_rtol` follows `_born`'s convention rather than this function's other
    two: it carries ROADMAP.md section 12's **milestone bar**, not twice a
    measurement. M2 holds debye to 1.5% at every gated rung, decided 2026-08-14
    by Charlie over the shared tolerance. The shared one would have been a bar
    debye meets by construction — it reproduces APBS's discretization, and APBS
    is what sets it — which is section 7's check that cannot fail.
    """
    return AnalyticReference(
        energy_kj_mol=kirkwood_solvation_energy(3.0, 3.0 * offset_fraction, 1.0, 1.0, 78.54),
        rtol=rtol,
        source=f"Kirkwood: q=1e at d/a={offset_fraction:g} in a 3 A sphere, eps_p=1",
        per_backend_rtol=_per_backend_rtol(delphi_rtol, debye_rtol),
        gated=gated,
    )


def _born(
    radius: float,
    charge: float = 1.0,
    solute_dielectric: float = 1.0,
    *,
    rtol: float,
    delphi_rtol: float | None = None,
    debye_rtol: float | None = None,
) -> AnalyticReference:
    """A Born case's closed form, computed from CODATA constants rather than quoted.

    `rtol` is measured, not chosen: it is roughly twice the discretization error
    APBS 3.4.1 actually shows on that geometry. Loose enough to survive a
    platform, tight enough that a unit error or a factor of two cannot hide.

    `delphi_rtol` is the same convention applied to the C++ flavour, which on a
    sharp boundary is three to four orders of magnitude closer to exact than
    APBS. Without it the shared tolerance — which has to accommodate APBS —
    would let DelPhi drift by percent and call it agreement.

    `debye_rtol` is where the convention is deliberately *not* applied. It
    carries ROADMAP.md section 12's milestone bar rather than twice a
    measurement: M1 says the Born ion within 1% at 0.25 A, so the fine case is
    0.01 against a measured 0.853% and not the 0.017 that doubling would give.
    A milestone tolerance set from what the solver already does is a milestone
    that cannot be failed — which is the shape section 7 keeps finding, and it
    is worth one comment to say the difference is intended.
    """
    return AnalyticReference(
        energy_kj_mol=born_solvation_energy(radius, charge, solute_dielectric, 78.54),
        rtol=rtol,
        source=f"Born: q={charge:g}e, a={radius:g} A, eps_p={solute_dielectric:g}",
        per_backend_rtol=_per_backend_rtol(delphi_rtol, debye_rtol),
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
        analytic=_born(3.0, rtol=0.05, delphi_rtol=0.001),  # measured 2.357% / 0.001%
        analytic_field=AnalyticField(
            radius_a=3.0,
            charge_e=1.0,
            cells_out=(2, 4, 8),
            # Across all eight directions: 3.186% APBS, 0.789% DelPhi. The old
            # numbers here were +x only, and DelPhi's 0.150% was the ray it
            # happens to be best on — its face diagonals are 5.3x worse.
            rtol=0.064,
            per_backend_rtol=(("delphicpp", 0.016),),
        ),
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
        # debye: measured 1.576%, twice it is 0.032. The gate M1 names is on the
        # fine case below; this one is its convergence partner, and a pair only
        # states convergence if both ends are held.
        analytic=_born(3.0, rtol=0.05, delphi_rtol=0.001, debye_rtol=0.032),
        analytic_field=AnalyticField(
            radius_a=3.0,
            charge_e=1.0,
            cells_out=(2, 4, 8),
            # Across all eight directions: 3.186% APBS, 0.789% DelPhi. The old
            # numbers here were +x only, and DelPhi's 0.150% was the ray it
            # happens to be best on — its face diagonals are 5.3x worse.
            rtol=0.064,
            per_backend_rtol=(("delphicpp", 0.016),),
        ),
    ),
    # --- the sharp-boundary ladder ------------------------------------------
    #
    # ROADMAP.md section 12, M0. Fifteen of the eighteen closed-form cases above
    # sit on `smoothed-molecular` — APBS's harmonic averaging, which no other
    # backend implements and which a clean-room solver has no reason to. That
    # made the whole analytic sweep unusable for grading anything but APBS, and
    # all four Kirkwood cases were on the wrong side of it, so the milestone
    # "the Kirkwood cases within their measured tolerances" could not be met by
    # construction. These are the same physics on a boundary every backend can
    # build.
    #
    # They cost more than their smoothed twins to be right about: smoothing *is*
    # APBS's discretization-error reduction, so APBS is roughly four times worse
    # here — 2.36% against 0.62% on the same ion at the same spacing — while
    # DelPhi C++, which resolves the sphere exactly, is 0.0006%. That spread is
    # why `per_backend_rtol` exists: a shared tolerance wide enough for APBS
    # cannot notice DelPhi drifting by percent.
    Case(
        name="born-ion-molecular-r1",
        description=(
            "1 A sphere on a sharp boundary, where the smoothed twin is 5.1% out "
            "and this is 4.1%. The radius arm has to start where the "
            "discretization is visibly failing, or it only measures the easy end."
        ),
        source="born-ion-r1",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.MOLECULAR
        ),
        analytic=_born(1.0, rtol=0.09, delphi_rtol=0.001),  # measured 4.101% / 0.001%
        analytic_field=AnalyticField(
            radius_a=1.0,
            charge_e=1.0,
            cells_out=(2, 4, 8),
            rtol=0.035,  # all directions: 1.165% APBS / 1.727% DelPhi (axes worst)
        ),
    ),
    Case(
        name="born-ion-molecular-r2",
        description="2 A sphere; with 1, 3, 4 and 6 this is the radius arm on a sharp boundary.",
        source="born-ion-r2",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.MOLECULAR
        ),
        analytic=_born(2.0, rtol=0.05, delphi_rtol=0.001),  # measured 2.254% / 0.001%
        analytic_field=AnalyticField(
            radius_a=2.0,
            charge_e=1.0,
            cells_out=(2, 4, 8),
            # All directions: 4.744% APBS (axes) / 2.521% DelPhi (body diagonal,
            # 1.6x its +x reading).
            rtol=0.095,
            per_backend_rtol=(("delphicpp", 0.051),),
        ),
    ),
    Case(
        name="born-ion-molecular-r4",
        description=(
            "4 A sphere, and the best-resolved of the sharp radius arm at 0.42% "
            "— better than 6 A, which is grid alignment rather than physics and "
            "is the reason the arm is a sweep and not one point."
        ),
        source="born-ion-r4",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.MOLECULAR
        ),
        analytic=_born(4.0, rtol=0.01, delphi_rtol=0.001),  # measured 0.421% / 0.007%
        analytic_field=AnalyticField(
            radius_a=4.0,
            charge_e=1.0,
            cells_out=(2, 4, 8),
            rtol=0.029,  # all directions: 1.412% APBS / 1.239% DelPhi
        ),
    ),
    Case(
        name="born-ion-molecular-r6",
        description="6 A sphere, the long end of the sharp radius arm.",
        source="born-ion-r6",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.MOLECULAR
        ),
        analytic=_born(6.0, rtol=0.015, delphi_rtol=0.001),  # measured 0.667% / 0.001%
        analytic_field=AnalyticField(
            radius_a=6.0,
            charge_e=1.0,
            cells_out=(2, 4, 8),
            # All directions: 1.531% APBS / 1.902% DelPhi. Both were under 1%
            # on +x alone; the body diagonal is where this radius is worst, and
            # the old 0.02 would now fail. That is the result change M1a is.
            rtol=0.038,
        ),
    ),
    # --- the van der Waals field arm, for M1b -------------------------------
    #
    # ROADMAP.md section 12 M1b grades debye's field against the best reference
    # solver installed, and debye builds only the van der Waals boundary — so
    # before these, the whole milestone rested on two cases at one radius. These
    # are the extremes of the radius arm above, on the surface debye can answer,
    # which is the cheapest way to make the gate span a = 1 to 6 A.
    #
    # **On the reference legs these mostly are duplicates of their `molecular`
    # twins, and the first version of this comment claimed otherwise.** Checked
    # against the recordings: `born-ion-vdw-r1` is numerically identical to
    # `born-ion-molecular-r1` on both APBS and DelPhi in every key but the
    # surface model and `resolved_parameters`, and `-r6` is identical on DelPhi.
    # Only `-r6` on APBS moves, by 0.19%. That is the physics — a probe cannot
    # carve a re-entrant surface out of a lone sphere — and `srad 0` against
    # `srad 1.4` only builds a visibly different dielectric map once the grid is
    # fine enough to see it, which `born-ion-vdw-fine` measures at 0.25 A as
    # 0.787% against 0.621%.
    #
    # They earn their place for debye, which cannot build `molecular` at all and
    # would otherwise have M1b resting on two cases at one radius. They do not
    # earn it by adding an independent reference-tier result, and a reader of a
    # corpus diff should not be told they do.
    Case(
        name="born-ion-vdw-r1",
        description=(
            "1 A sphere on van der Waals: the small end of the field arm debye "
            "is graded on, and the radius where the sampling rule is tightest."
        ),
        source="born-ion-r1",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.VAN_DER_WAALS
        ),
        analytic=_born(1.0, rtol=0.09, delphi_rtol=0.001),  # measured 4.101% / 0.001%
        analytic_field=AnalyticField(
            radius_a=1.0,
            charge_e=1.0,
            cells_out=(2, 4, 8),
            # All directions: 1.165% APBS / 1.727% DelPhi. debye reads 9.871%
            # here — the worst it is anywhere on the corpus, and the case that
            # makes M1b's gate about the small-radius near field.
            rtol=0.035,
        ),
    ),
    Case(
        name="born-ion-vdw-r6",
        description="6 A sphere on van der Waals: the long end of the same arm.",
        source="born-ion-r6",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.VAN_DER_WAALS
        ),
        analytic=_born(6.0, rtol=0.018, delphi_rtol=0.001),  # measured 0.856% / 0.001%
        analytic_field=AnalyticField(
            radius_a=6.0,
            charge_e=1.0,
            cells_out=(2, 4, 8),
            # All directions: 1.877% APBS / 1.902% DelPhi, and debye 1.894% —
            # all three on h = 0.5 A, which is what agreement looks like when
            # nothing about the grid separates them.
            rtol=0.038,
        ),
    ),
    Case(
        name="born-ion-molecular-negative",
        description=(
            "-1e on a sharp boundary. Solvation goes as q^2, so this must return "
            "the +1e energy exactly on any surface model; the smoothed twin "
            "cannot show that the sign path is right for a sharp one."
        ),
        source="born-ion-negative",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.MOLECULAR
        ),
        analytic=_born(3.0, -1.0, rtol=0.05, delphi_rtol=0.001),  # measured 2.357% / 0.001%
    ),
    Case(
        name="born-ion-molecular-divalent",
        description="+2e on a sharp boundary: the q^2 scaling, four times the +1e energy.",
        source="born-ion-divalent",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.MOLECULAR
        ),
        analytic=_born(3.0, 2.0, rtol=0.05, delphi_rtol=0.001),  # measured 2.357% / 0.001%
    ),
    Case(
        name="born-ion-molecular-eps4",
        description=(
            "Solute dielectric 4 on a sharp boundary; with eps_p 1 and 2 this "
            "completes the dielectric arm on a surface debye can build."
        ),
        source="born-ion",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=4.0, ionic_strength=0.0, surface_model=SurfaceModel.MOLECULAR
        ),
        analytic=_born(3.0, 1.0, 4.0, rtol=0.045, delphi_rtol=0.001),  # measured 2.101% / 0.014%
    ),
    Case(
        name="born-ion-molecular-fine",
        description=(
            "The sharp-boundary convergence pair with `born-ion-molecular`: "
            "2.36% at 0.5 A becomes 0.62% at 0.25 A. M1's claim is that the "
            "error falls monotonically under refinement, and a single case "
            "cannot state it."
        ),
        source="born-ion",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.MOLECULAR
        ),
        analytic=_born(3.0, rtol=0.013, delphi_rtol=0.001),  # measured 0.621% / 0.001%
        analytic_field=AnalyticField(
            radius_a=3.0,
            charge_e=1.0,
            cells_out=(2, 4, 8),
            # All directions: 1.107% APBS / 1.891% DelPhi, both on the body
            # diagonal, against 0.827% / 0.736% on +x alone.
            rtol=0.038,
        ),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="born-ion-vdw-fine",
        description=(
            "The same pair on van der Waals, which is the surface debye climbs "
            "first. It also shows what the coarse pair hides: `born-ion-vdw` and "
            "`born-ion-molecular` agree exactly at 0.5 A and do not here — "
            "0.787% against 0.621% — because `srad 0` and `srad 1.4` build "
            "different dielectric maps of the same boundary."
        ),
        source="born-ion",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.VAN_DER_WAALS
        ),
        # This is M1's gate case. debye measures 0.853% and is held to 1.0%,
        # the milestone's own number rather than twice the measurement.
        analytic=_born(3.0, rtol=0.016, delphi_rtol=0.001, debye_rtol=0.01),
        analytic_field=AnalyticField(
            radius_a=3.0,
            charge_e=1.0,
            cells_out=(2, 4, 8),
            # All directions: 1.050% APBS / 1.891% DelPhi. This is the case
            # ROADMAP.md section 12 quotes for M1b's bar, and the reason that bar
            # has to be re-argued: neither reference solver clears 1% at k = 2.
            rtol=0.038,
        ),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="kirkwood-molecular-03",
        description="Charge placement on a sharp boundary: d/a = 0.3, the gentlest rung.",
        source="kirkwood-03",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.MOLECULAR
        ),
        analytic=_kirkwood(0.3, rtol=0.018, delphi_rtol=0.003),  # measured 0.844% / 0.097%
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="kirkwood-molecular-05",
        description="d/a = 0.5 on a sharp boundary; the middle rung of M2's ladder.",
        source="kirkwood-05",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.MOLECULAR
        ),
        analytic=_kirkwood(0.5, rtol=0.019, delphi_rtol=0.005),  # measured 0.902% / 0.205%
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="kirkwood-molecular-07",
        description=(
            "d/a = 0.7, the last rung anything reproduces. APBS is 3.8% here "
            "against 0.11% on the smoothed twin, which is the sharp boundary "
            "costing what it costs rather than a solver misbehaving."
        ),
        source="kirkwood-07",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.MOLECULAR
        ),
        analytic=_kirkwood(0.7, rtol=0.08, delphi_rtol=0.01),  # measured 3.800% / 0.416%
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="kirkwood-molecular-09",
        description=(
            "d/a = 0.9: recorded, deliberately not gated. The charge sits 0.3 A "
            "inside the boundary, where the continuum reaction field diverges, "
            "and both reference codes get *worse* under refinement and "
            "non-monotonically — APBS 6.40/9.85/9.05% and DelPhi "
            "26.67/4.29/7.50% at 0.5/0.25/0.125 A. Gating a new solver on a "
            "number no shipped solver reproduces would be a check that cannot "
            "fail; recording it says where the method gives up, which is worth "
            "having. The closed form is not in doubt: the series is converged to "
            "the last digit by 100 terms."
        ),
        source="kirkwood-09",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.MOLECULAR
        ),
        # `rtol` is unused while `gated=False`, and is deliberately set *below*
        # what either code achieves — APBS 9.85%, DelPhi 4.29% — so that
        # ungating this case fails loudly. The first version set it to the
        # measured APBS error, which both codes then passed: a tolerance that
        # says "nobody should judge this" while quietly judging it green is the
        # vacuous check this case exists to avoid.
        analytic=_kirkwood(0.9, rtol=0.01, gated=False),  # measured 9.848% / 4.288%
        tier=CaseTier.STANDARD,
    ),
    # M2's rungs, on the one boundary debye builds.
    #
    # **These exist because M0 dropped them and the reason it gave does not hold
    # for the solver M2 grades.** M0 budgeted "one fewer case than the first
    # draft, which also spent a case on a `van-der-waals` Kirkwood", reasoning
    # that another sphere geometry re-measures what the existing rungs already
    # measure. True of APBS and DelPhi, which build both surfaces — and false of
    # debye, whose `SUPPORTED_SURFACES` is `van-der-waals` alone, so every
    # Kirkwood rung in the corpus was on a surface it refuses by name. M2's exit
    # criterion was unreachable by construction, which is the same shape as the
    # `smoothed-molecular` gap M0 itself was created to close, one surface along.
    #
    # **The relabel is not an identity, which is worth knowing before assuming
    # it.** For the *Born* geometry it is: `born-ion-molecular` and
    # `born-ion-vdw` record -233.9996297277 to the last digit, because the
    # solvent-excluded surface of an isolated convex sphere is the sphere. Add
    # Kirkwood's zero-radius charge atom and APBS's two surfaces separate by
    # ~0.24% (d/a = 0.3: -253.191 against -253.792), while DelPhi's stay
    # bit-identical. So these carry their own measured tolerances rather than
    # inheriting the molecular twins'.
    Case(
        name="kirkwood-vdw-03",
        description="Charge placement on a van der Waals boundary: d/a = 0.3, M2's gentlest rung.",
        source="kirkwood-03",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.VAN_DER_WAALS
        ),
        # measured 1.083% / 0.097% / 1.047%; debye's 1.5% is M2's bar, not 2x its own
        analytic=_kirkwood(0.3, rtol=0.022, delphi_rtol=0.003, debye_rtol=0.015),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="kirkwood-vdw-05",
        description="d/a = 0.5 on a van der Waals boundary; the middle rung of M2's ladder.",
        source="kirkwood-05",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.VAN_DER_WAALS
        ),
        # measured 1.239% / 0.205% / 1.254%
        analytic=_kirkwood(0.5, rtol=0.025, delphi_rtol=0.005, debye_rtol=0.015),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="kirkwood-vdw-07",
        description=(
            "d/a = 0.7 on a van der Waals boundary, M2's hardest gated rung. The "
            "charge sits 0.9 A inside the interface, and this is where the "
            "shared tolerance stops describing the solvers: APBS reads 3.896% "
            "where DelPhi reads 0.416% and debye 1.328%."
        ),
        source="kirkwood-07",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.VAN_DER_WAALS
        ),
        # measured 3.896% / 0.416% / 1.328%. debye's 1.5% is *stricter than APBS
        # manages here*, which is what makes M2 a claim rather than a formality.
        analytic=_kirkwood(0.7, rtol=0.08, delphi_rtol=0.01, debye_rtol=0.015),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="kirkwood-vdw-09",
        description=(
            "d/a = 0.9 on a van der Waals boundary: recorded, deliberately not "
            "gated, for the same reason as its molecular twin — no shipped "
            "solver reproduces it. APBS 9.854%, DelPhi 4.288%, debye 8.280%."
        ),
        source="kirkwood-09",
        grid=GridSpec(resolution=0.25, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.VAN_DER_WAALS
        ),
        # Set below what any code achieves, so ungating fails loudly rather than
        # passing vacuously — the trap the molecular twin's comment records.
        analytic=_kirkwood(0.9, rtol=0.01, gated=False),  # measured 9.854% / 4.288% / 8.280%
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="born-ion-molecular-salt",
        description=(
            "Screening on a sharp boundary at physiological salt. No closed "
            "form, for the reason `born-ion-salt` states: the two backends "
            "disagree 39% on the mobile-ion term because they do not share an "
            "ion-exclusion convention, and pinning either would encode a "
            "convention as physics. With the case below it this is M3's arm."
        ),
        source="born-ion",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.15, surface_model=SurfaceModel.MOLECULAR
        ),
    ),
    Case(
        name="born-ion-molecular-high-salt",
        description=(
            "0.5 M on the same sphere. The pair states the direction screening "
            "moves the energy, which is what M3 is gated on — a relationship "
            "between two recordings rather than a number either has to hit."
        ),
        source="born-ion",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.5, surface_model=SurfaceModel.MOLECULAR
        ),
    ),
    # --- the salt arm on the surface debye builds ---------------------------
    #
    # ROADMAP.md section 12, M3, and the third occurrence of one class. The two
    # cases above are `molecular`; debye's `SUPPORTED_SURFACES` is
    # `van-der-waals` alone, so M3's arm named cases debye refuses *by name* —
    # exactly as M0 found for the closed forms and M2 for Kirkwood. **A case
    # added for coverage of the incumbents is not automatically coverage of the
    # candidate**, and this is the third milestone to have paid for it.
    #
    # **Deliberately no `analytic` energy reference, and the reason is
    # measured.** The ionic contribution is 0.3% of the total where
    # discretization is 1.6%, so a closed-form check on the *total* cannot see
    # the salt: every mutation of debye's screening tried at M3 — dropping the
    # Boltzmann term outright, moving the Stern layer to the probe radius,
    # removing it, using kappa for kappa^2 — leaves the total within -1.40% to
    # -1.76% of `screened_born_solvation_energy`, and APBS alone needs 2.4%. An
    # `AnalyticReference` here would be the purest check that cannot fail in the
    # corpus. What *does* discriminate is G(I) - G(0) across this pair and its
    # zero-salt sibling `born-ion-vdw`, where the same four mutations read
    # -28.8%, +4.9%, +18.2% and +63.4% against debye's +0.10%.
    #
    # The **field** is a different matter and does carry a closed form: under
    # salt `AnalyticField` describes the Stern layer and the screened bulk,
    # which is a 29-51% correction to the unscreened expression over the sampled
    # radii, so these are the cases that would catch a solver ignoring salt in
    # the quantity protean displays.
    Case(
        name="born-ion-vdw-salt",
        description=(
            "Physiological salt on the boundary a sharp-boundary solver builds. "
            "The `molecular` pair above cannot be answered by one, and M3 needed "
            "an arm that could. On this surface the ionic term is also clean "
            "where the probe-based ones are not: APBS reproduces it to 0.03% at "
            "0.5, 0.25 and 0.125 A, against a 9-13% swing on `molecular` and "
            "`smoothed-molecular`."
        ),
        source="born-ion",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.15, surface_model=SurfaceModel.VAN_DER_WAALS
        ),
        analytic_field=AnalyticField(
            radius_a=3.0,
            charge_e=1.0,
            cells_out=(2, 4, 8),
            rtol=0.091,  # measured, all eight directions: APBS 4.506%
            per_backend_rtol=(("delphicpp", 0.037),),  # 1.807%
        ),
    ),
    Case(
        name="born-ion-vdw-high-salt",
        description=(
            "0.5 M on the same sphere, a Debye length of 4.30 A against the "
            "sphere's own 3 A. With the case above and `born-ion-vdw` this is a "
            "three-point arm, and M3 is gated on the relationship between the "
            "points rather than on any one of them."
        ),
        source="born-ion",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(
            solute_dielectric=1.0, ionic_strength=0.5, surface_model=SurfaceModel.VAN_DER_WAALS
        ),
        analytic_field=AnalyticField(
            radius_a=3.0,
            charge_e=1.0,
            cells_out=(2, 4, 8),
            rtol=0.109,  # measured: APBS 5.403%
            per_backend_rtol=(("delphicpp", 0.033),),  # 1.620%
        ),
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
        description=(
            "ALA-GLY with no probe. Against `peptide-molecular` this is the "
            "25.7%; against the two cases below it, it is the 0.15 M rung of a "
            "real-structure salt arm on the surface debye builds. It has carried "
            "`SolventModel`'s 0.15 M default since it was written, which is why "
            "debye's screening was exercised at M1 without being graded."
        ),
        source="ala-gly.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS),
    ),
    Case(
        name="peptide-vdw-no-salt",
        description=(
            "The zero-salt end of the sharp-boundary salt arm; with `peptide-vdw` "
            "it is what G(I) - G(0) is measured across on a real solute."
        ),
        source="ala-gly.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(ionic_strength=0.0, surface_model=SurfaceModel.VAN_DER_WAALS),
    ),
    Case(
        name="peptide-vdw-high-salt",
        description=(
            "500 mM, and the corpus's record of where the ionic term stops being "
            "a monopole. ALA-GLY is net neutral, so its screening is of a dipole: "
            "the three reference-tier codes spread over 22% here (debye -0.196, "
            "APBS -0.212, DelPhi C++ -0.174 at 0.15 M) where on every net-charged "
            "solute measured at M3 they agree to 1.4%. The spread is stable "
            "across 0.5/0.35/0.25/0.2 A, so it is a convention difference and not "
            "grid noise, and no closed form exists to say which is right — so M3 "
            "records this and gates on the sphere."
        ),
        source="ala-gly.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(ionic_strength=0.5, surface_model=SurfaceModel.VAN_DER_WAALS),
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
        analytic=_born(3.0, 1.0, 2.0, rtol=0.05, delphi_rtol=0.001),  # 2.263% / 0.010%
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
    # --- the probe's worth, at protein scale (M4) ----------------------------
    #
    # M4 is gated on `(E_molecular - E_vdw)/|E_vdw|` — the quantity the
    # solvent-excluded surface *alone* decides — and until these cases existed
    # the corpus could not state it above twenty atoms. Every closed-form case
    # is blind to it by construction: a lone convex sphere's solvent-excluded
    # surface is that sphere, so a solver implementing `molecular` as "return
    # the van der Waals answer" passed all eighteen of them exactly. ALA-GLY
    # was the only multi-atom structure carrying both surfaces, and the probe is
    # worth 4% there against 17-35% on a protein — so the peptide was not
    # standing in for this, it was a different question.
    #
    # Each case below is the missing half of a pair whose other half the corpus
    # already had. No new structures and no new machinery: the point is only
    # that both surfaces are now recorded for the same solute, which is what
    # makes the difference between them a thing `verify` can regress on.
    #
    # Measured across all three reference-tier backends when these were added
    # (2026-08-15): debye lands strictly between DelPhi C++ and APBS on every
    # one, two to eight times closer to APBS than DelPhi is, and the ordering
    # holds down a 0.7-0.35 A refinement ladder on fas2. ROADMAP.md section 12's
    # M4 carries the table.
    Case(
        name="barnase-molecular",
        description=(
            "The molecular half of `barnase-vdw`. Their difference is the "
            "probe's worth on a 1,730-atom protein: +29.5% for APBS, where the "
            "same quantity on ALA-GLY is +5.6%."
        ),
        source="apbs-examples/barnase.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
        tier=CaseTier.FULL,
    ),
    Case(
        name="fas2-vdw",
        description=(
            "The van der Waals half of `fas2-molecular`, and the cheapest "
            "protein-scale pair in the corpus — which is why it is `standard` "
            "where the rest of this section is `full`."
        ),
        source="apbs-examples/fas2.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS),
        tier=CaseTier.STANDARD,
    ),
    Case(
        name="lysozyme-vdw",
        description=(
            "The van der Waals half of `lysozyme-molecular`. Hen lysozyme "
            "carries the largest probe worth in the set (+35.8% for APBS), so "
            "it is where a construction that under-fills re-entrant volume "
            "would show first."
        ),
        source="apbs-examples/2LZT-ASP66.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS),
        tier=CaseTier.FULL,
    ),
    Case(
        name="barstar-vdw",
        description="The van der Waals half of `barstar-molecular`; barnase's binding partner.",
        source="apbs-examples/barstar.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS),
        tier=CaseTier.FULL,
    ),
    Case(
        name="protein-rna-vdw",
        description=(
            "The van der Waals half of `protein-rna-molecular`. The one pair "
            "here that is not all protein: a nucleic acid packs differently, and "
            "its probe worth (+18.4%) sits with fas2's rather than with the "
            "similarly sized proteins' +27% to +36%."
        ),
        source="apbs-examples/1a63.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS),
        tier=CaseTier.FULL,
    ),
    Case(
        name="hca-vdw",
        description=(
            "The van der Waals half of `hca-molecular`, and the largest solute "
            "in this section at 2,482 atoms."
        ),
        source="apbs-examples/hca.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS),
        tier=CaseTier.FULL,
    ),
    Case(
        name="fkbp-apo-vdw",
        description=(
            "The van der Waals half of `fkbp-apo-molecular`. With the hca pair "
            "this is the second ligand-binding site in the section, and the "
            "probe's worth is where a bound ligand changes the surface most."
        ),
        source="apbs-examples/fkbp-apo.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS),
        tier=CaseTier.FULL,
    ),
    Case(
        name="fkbp-dmso-vdw",
        description=(
            "The van der Waals half of `fkbp-dmso-molecular`, so the apo/holo "
            "pair is complete on both surfaces. Binding DMSO moves the probe's "
            "worth by less than half a point (+27.5% to +27.9% for APBS), which "
            "is the point: a ligand in a pocket changes the surface far less "
            "than folding does."
        ),
        source="apbs-examples/fkbp-dmso.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS),
        tier=CaseTier.FULL,
    ),
    Case(
        name="ion-protein-complex-vdw",
        description=(
            "The van der Waals half of `ion-protein-complex-molecular`, and the "
            "case that says where the probe's worth stops being gateable. At "
            "260 atoms dominated by explicit ions there is almost no re-entrant "
            "volume, so the probe is worth about 0.5% — comparable to each "
            "code's own discretization error rather than far above it, where "
            "the protein pairs sit. Recorded and not judged, for the same "
            "reason M2 records Kirkwood's ninth rung: a quantity smaller than "
            "the lattice noise around it grades the lattice."
        ),
        source="apbs-examples/ion-protein-complex.pqr",
        grid=GridSpec(resolution=0.5, padding=10.0),
        solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS),
        tier=CaseTier.FULL,
    ),
)


def cases_for_tier(tier: CaseTier, cases: tuple[Case, ...] = MANIFEST) -> tuple[Case, ...]:
    """Every case at or below `tier`, in manifest order."""
    allowed = set(TIER_ORDER[: TIER_ORDER.index(tier) + 1])
    return tuple(case for case in cases if case.tier in allowed)


SURFACES_IN_A_PAIR = (SurfaceModel.VAN_DER_WAALS, SurfaceModel.MOLECULAR)


def surface_pairs(cases: tuple[Case, ...] = MANIFEST) -> tuple[tuple[Case, Case], ...]:
    """Every (van der Waals, molecular) pair asking one question but for the probe.

    Two cases pair when their solute, grid and solvent are identical except for
    `surface_model`, so the difference between their energies is attributable to
    the solvent-excluded surface and to nothing else. That is the quantity M4 is
    gated on — see `probe_worth`.

    **Derived rather than listed.** A case added with a sibling joins the M4
    gate without anyone remembering to update a constant, which is the second
    lesson in this project's guards file: a check keyed on a hand-maintained
    list stops covering the thing it was written for the first time someone
    extends the manifest and forgets.
    """
    keyed: dict[tuple[str, GridSpec, SolventModel], dict[SurfaceModel, Case]] = {}
    for case in cases:
        model = case.solvent.surface_model
        if model not in SURFACES_IN_A_PAIR:
            continue
        # Normalising the surface model is what makes the rest of the solvent
        # the key: two cases collide here exactly when the probe is the only
        # thing between them.
        key = (case.source, case.grid, replace(case.solvent, surface_model=SurfaceModel.MOLECULAR))
        keyed.setdefault(key, {})[model] = case
    return tuple(
        (found[SurfaceModel.VAN_DER_WAALS], found[SurfaceModel.MOLECULAR])
        for found in keyed.values()
        if len(found) == len(SURFACES_IN_A_PAIR)
    )


def probe_worth(vdw: dict[str, Any], molecular: dict[str, Any]) -> float:
    """`(E_molecular - E_vdw)/|E_vdw|` as a percentage, from two recorded summaries.

    What rolling the probe is worth, and the one quantity the solvent-excluded
    surface alone decides. Every closed-form case in the corpus is blind to it
    — a lone convex sphere's solvent-excluded surface *is* that sphere — so this
    is the only thing that distinguishes a real reduced-surface construction
    from a solver that answers `molecular` by returning the van der Waals
    number.

    Reads recordings rather than solving, so the M4 gate needs no binary
    installed: the incumbents' halves are files in the repository and only the
    candidate has to run.
    """
    before = vdw["energy_kj_mol"]
    after = molecular["energy_kj_mol"]
    if before is None or after is None:
        raise ValueError("both halves of a surface pair must record an energy")
    return float(100.0 * (after - before) / abs(before))


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
            # The tolerance that *applies to this backend*, which is the one a
            # reader of the file needs; the case's shared value is in the
            # manifest and is not what judged this answer.
            "rtol": case.analytic.rtol_for(result.provenance.backend),
            "source": case.analytic.source,
            # Recorded so a summary diff shows convergence moving, not just the
            # energy: this is the number that says whether the solver is right,
            # and it is the only one in the file that is not self-referential.
            "relative_error": abs(result.energy_kj_mol - exact) / abs(exact),
        }

    # Omitted rather than recorded as null when there is nothing to say, the way
    # a gridless summary omits `geometry`: shape follows the answer, and a
    # rebuild should not write seventy files of `"analytic_field": null` for the
    # sake of schema symmetry. CLAUDE.md asks that a corpus diff read as a
    # result change, which it cannot do buried under schema noise. Spread in
    # place rather than merged afterwards, so the key keeps its position and a
    # rebuild does not reorder every file that has one.
    field = _analytic_field_summary(case, result)

    summary: dict[str, Any] = {
        "name": case.name,
        "description": case.description,
        "source": case.source,
        "tier": case.tier.value,
        "family": family.value,
        "backend": result.provenance.backend,
        "analytic": analytic,
        **({} if field is None else {"analytic_field": field}),
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


def _analytic_field_summary(case: Case, result: SolveResult) -> dict[str, Any] | None:
    """The solver's potential against the closed form, where a case asks for it.

    Returns None rather than an empty block when there is nothing to say: a
    backend that produced no volumetric field has no field to be graded, which
    is a fact about its family and not a failure. `SurfacePotential` is excluded
    for the same reason — a boundary-element answer lives *on* the interface,
    which is the one place this measurement is meaningless.
    """
    reference = case.analytic_field
    if reference is None or not isinstance(result.potential, PotentialGrid):
        return None

    grid = result.potential
    # The coarsest axis, so the samples clear the interface on every axis rather
    # than on the average of them.
    spacing = float(np.max(grid.spacing))
    radii = reference.sample_radii(spacing)
    centre = case.structure().center()

    exact = np.array(reference.exact_at(radii, case.solvent))
    # Through `sashimi.field`, which refuses a sample that fell off the map
    # rather than recording a bare NaN: `json.dumps` writes `NaN` happily, it is
    # not valid JSON per spec, and the discrepancy would be reported as "nan%".
    # Every current case clears the box, so this is about the next one.
    values = np.array(
        [sample_values(grid, centre, radius) for radius in radii]
    )  # [radius][direction]
    errors = np.abs(values - exact[:, None]) / np.abs(exact)[:, None]

    worst = np.unravel_index(int(np.argmax(errors)), errors.shape)
    return {
        "spacing_used_a": spacing,
        "cells_out": list(reference.cells_out),
        "radii_a": [float(r) for r in radii],
        "directions": list(FIELD_DIRECTION_NAMES),
        "exact_kT_e": [float(v) for v in exact],
        # [radius][direction]. Nested rather than flat because the whole point of
        # this block is that the two axes are not interchangeable.
        "values_kT_e": [[float(v) for v in row] for row in values],
        "relative_errors": [[float(e) for e in row] for row in errors],
        "max_relative_error": float(np.max(errors)),
        "worst_sample": {
            "direction": FIELD_DIRECTION_NAMES[worst[1]],
            "radius_a": float(radii[worst[0]]),
            "cells_out": reference.cells_out[worst[0]],
            "relative_error": float(errors[worst]),
        },
        "rtol": reference.rtol_for(result.provenance.backend),
        "source": (
            f"Born potential: q={reference.charge_e:g}e outside a "
            f"{reference.radius_a:g} A sphere, sampled at a + k*h along "
            f"{len(FIELD_DIRECTIONS)} directions"
        ),
    }


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
    found.extend(_verify_analytic_field(case, fresh))
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


def _accuracy_tier(backend: str) -> AccuracyTier:
    """The tier of the backend that produced a summary, from its recorded identity.

    Matched by prefix, because a summary carries `gb-obc2-1` where the registry
    knows `gb`. An identity from no registered backend is treated as a reference
    tier: refusing to grade something merely because it is unrecognised is the
    wrong default for a check whose whole purpose is to catch a wrong answer.
    """
    from sashimi.backends import reports  # noqa: PLC0415 — import cost off the hot path

    # Longest prefix wins, so registering a backend whose name extends an
    # existing one — `gb2` alongside `gb` — cannot silently inherit the other's
    # tier. Registry order is not a safe tiebreak for a correctness check.
    matches = sorted(
        (r for r in reports() if backend.startswith(r.name)), key=lambda r: -len(r.name)
    )
    if matches:
        return AccuracyTier(matches[0].accuracy_tier)
    return AccuracyTier.REFERENCE


def _verify_analytic(case: Case, fresh: dict[str, Any]) -> list[Discrepancy]:
    """Check the solver against the closed form, not against its own past.

    Deliberately compares `fresh` to the physics rather than to the recording.
    Every other check here would pass forever on a backend that was wrong from
    the first build; this is the only one that would not.
    """
    if case.analytic is None or fresh.get("analytic") is None:
        return []

    # Recorded, deliberately not judged. The summary still carries the closed
    # form and the deviation from it, so the number is visible and diffable;
    # what is withheld is a verdict the physics cannot support.
    if not case.analytic.gated:
        return []

    # Nor does a closed form judge an approximation. These tolerances are
    # measured discretization error — how far a solver that *discretizes the
    # equation* lands from exact — and `AccuracyTier.APPROXIMATE` names a
    # backend that does not discretize it at all. Generalized Born is 14.6% from
    # Born on a 1 A sphere and that is the method, not a defect; holding it to
    # APBS's 4.1% would say the corpus had found a bug. It was only ever passing
    # because the sharp-boundary tolerances used to be loose enough to swallow
    # it. What grades the approximate tier is its recorded deviation from the
    # reference tier, in `tests/test_corpus_gb.py`, and its reduction to the
    # closed forms in `tests/test_gb_reference.py` — where that claim is exact
    # rather than approximate.
    if _accuracy_tier(fresh["backend"]) is AccuracyTier.APPROXIMATE:
        return []

    error = fresh["analytic"]["relative_error"]
    tolerance = case.analytic.rtol_for(fresh["backend"])
    if error <= tolerance:
        return []
    return [
        Discrepancy(
            case.name,
            "analytic.energy_kj_mol",
            case.analytic.energy_kj_mol,
            fresh["energy_kj_mol"],
            f"{error:.3%} from the closed form, tolerance {tolerance:.3%} ({case.analytic.source})",
        )
    ]


def _verify_analytic_field(case: Case, fresh: dict[str, Any]) -> list[Discrepancy]:
    """Check the solver's *potential* against the closed form.

    The counterpart to `_verify_analytic`, and the axis the corpus was missing:
    an energy is one integrated scalar, and a solver can integrate to the right
    number while the field it hands a viewer is wrong where the viewer reads it.

    The approximate tier is skipped for the same reason it is skipped there, and
    a backend that returned no volumetric field simply has nothing recorded.
    """
    reference = case.analytic_field
    if reference is None or fresh.get("analytic_field") is None:
        return []
    if _accuracy_tier(fresh["backend"]) is AccuracyTier.APPROXIMATE:
        return []

    field = fresh["analytic_field"]
    error = field["max_relative_error"]
    tolerance = reference.rtol_for(fresh["backend"])
    if error <= tolerance:
        return []
    worst = field["worst_sample"]
    return [
        Discrepancy(
            case.name,
            "analytic_field.max_relative_error",
            tolerance,
            error,
            f"{error:.3%} from the closed-form potential at r = {worst['radius_a']:.4g} A "
            f"({worst['cells_out']} cells beyond a {reference.radius_a:g} A boundary) "
            f"along {worst['direction']}, tolerance {tolerance:.3%}",
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
