"""M1b: debye's field, graded against the best solver already installed.

ROADMAP.md section 12 M1b. The bar is **not** a round number, and the reason is
worth stating once. debye reproduces DelPhi C++'s discretization to three
decimal places, so any bar of the form "no worse than the worst incumbent" is
one it meets by construction rather than by merit — a check that cannot fail, in
section 7's sense, wearing the costume of a milestone. Against the *best*
incumbent at each sample it is a real measurement: agreeing with DelPhi does not
put a solver within a factor of APBS.

Every backend is sampled at the **same physical radii**, taken from the coarsest
grid in the comparison. The corpus's own field check deliberately does not do
this — it samples `a + k*h` on each backend's own achieved spacing, because
every sample must clear *its* interface cell — and the consequence is that a
coarser grid is sampled further out, where the error is smaller. Comparing those
numbers across backends reads a sampling difference as an accuracy difference.

These tests need reference solvers to grade against, so unlike the rest of
debye's suite they are marked and they skip on a bare machine. That is the
honest shape of "as good as the incumbents": without an incumbent there is no
claim to check.

**They pin the reference set to APBS *and* DelPhi C++ rather than grading
against whatever is installed, and that is not caution — it is required for the
verdict to exist.** The first draft fell back to APBS alone when DelPhi was
absent, and `born-ion-vdw` flips from 5.24x to 1.60x under that fallback: the
sample radii come from the coarsest grid in the comparison, so dropping DelPhi's
h = 0.5 A moves them, and the yardstick changes from DelPhi's 0.789% to APBS's
2.453%. Both moves favour debye. A milestone whose verdict depends on what
happens to be installed is not a gate, and CI would have found this the hard
way: the strict xfails would have XPASSed on the `apbs-only` leg.

`grade_field` itself stays flexible — grading against whatever you have is a
legitimate thing for a library to do. It is the *claim* that has to pin its
incumbents.
"""

from __future__ import annotations

import pytest

from sashimi.apbs import ApbsSolver
from sashimi.corpus import MANIFEST, Case
from sashimi.debye import DebyeSolver
from sashimi.delphi import DelphiSolver
from sashimi.delphi.discover import DelphiFlavour, discover_delphi
from sashimi.delphi.options import SUPPORTED_SURFACES as DELPHI_SURFACES
from sashimi.protocol import PotentialGrid, SolveResult, SurfaceModel
from sashimi.validate import (
    DEFAULT_FIELD_FACTOR,
    BackendRun,
    Incomparable,
    grade_field,
)

# The van der Waals field cases, which are the ones debye can answer at all,
# split by what actually predicts the verdict: **a/h, the number of cells across
# the sphere's radius.** Not resolution, and not radius — `born-ion-vdw-r6` at
# h = 0.5 A is at parity while `born-ion-vdw` at the same h is 5.2x off, and the
# difference between them is that one sphere is twelve cells across and the
# other six.
#
#   born-ion-vdw-r6    a/h = 12   1.01 / 1.02 / 1.03x   parity
#   born-ion-vdw-fine  a/h = 12   1.77 / 1.00 / 1.01x   parity
#   born-ion-vdw       a/h =  6   5.24 / 4.81 / 3.74x   outside
#   born-ion-vdw-r1    a/h =  2   8.64 / 3.61 / 1.55x   outside
#
# So debye's near field degrades faster than the incumbents' as the sphere stops
# being resolved, and DelPhi C++ holds up where it does not. That is the shape of
# an interface-handling gap rather than a discretization error — which is what
# ROADMAP.md section 10's referee tier exists for.
AT_PARITY = ("born-ion-vdw-r6", "born-ion-vdw-fine")
UNDER_RESOLVED = ("born-ion-vdw", "born-ion-vdw-r1")


def case_named(name: str) -> Case:
    return next(case for case in MANIFEST if case.name == name)


def require_delphi_that_builds_van_der_waals() -> None:
    """Skip unless the installed DelPhi can build the boundary debye climbs.

    **The `delphi` marker names a program, not a capability**, and the two
    flavours differ exactly here: pyDelPhi has no van der Waals surface — its
    `vdw` method rolls the probe, so it gives the molecular one, and the
    `prbrad=0` that would produce a real vdW boundary aborts inside numba. A
    marked test asking for that surface therefore *fails* rather than skips
    wherever pyDelPhi is the flavour present.

    CI found it: the `full` leg has a "Verify the pyDelPhi flavour actually ran"
    step that re-runs `-m delphi` with `SASHIMI_DELPHI_PATH` pointed at
    pyDelPhi, and three of these tests failed there while the main run was
    green. ROADMAP.md section 12 already names the C++ flavour the touchstone
    for precisely this reason.

    Read from `SUPPORTED_SURFACES`, which is the module that owns the mapping,
    rather than testing `flavour is CPP`: the question is what the binary can
    build, and a flavour name is a proxy for it that can drift.
    """
    flavour = discover_delphi().flavour
    if SurfaceModel.VAN_DER_WAALS not in DELPHI_SURFACES[flavour]:
        pytest.skip(
            f"{flavour.value} cannot build a van der Waals boundary, so it cannot be "
            "an incumbent for a milestone measured on one"
        )


def graded(case: Case, factor: float = DEFAULT_FIELD_FACTOR):
    """Solve the case through both incumbents and debye, and grade debye.

    The reference set is fixed, not discovered. See the module docstring: with
    DelPhi omitted, `born-ion-vdw` reads 1.60x instead of 5.24x.
    """
    require_delphi_that_builds_van_der_waals()
    request = case.request()
    solvers: list[tuple[str, object]] = [
        ("apbs", ApbsSolver()),
        ("delphicpp", DelphiSolver()),
        ("debye", DebyeSolver()),
    ]

    runs = []
    for name, solver in solvers:
        result: SolveResult = solver.solve(request)  # type: ignore[attr-defined]
        assert isinstance(result.potential, PotentialGrid)
        runs.append(BackendRun.from_result(name, result, request))

    reference = case.analytic_field
    assert reference is not None
    return grade_field(
        runs,
        candidate="debye",
        centre=case.structure().center(),
        radius_a=reference.radius_a,
        charge_e=reference.charge_e,
        solvent_dielectric=case.solvent.solvent_dielectric,
        cells_out=reference.cells_out,
        factor=factor,
    )


@pytest.mark.apbs
@pytest.mark.delphi
@pytest.mark.parametrize("name", AT_PARITY)
def test_debye_is_within_a_factor_of_the_best_incumbent_where_the_sphere_is_resolved(name):
    """M1b's exit criterion, on the cases that meet it.

    Twelve cells across the radius, and debye is at 1.77x the best reference at
    the closest sample and at parity further out — which is what "as good as the
    incumbents" looks like when the grid resolves the interface.
    """
    grade = graded(case_named(name))

    assert grade.agrees, grade.summary()
    assert max(grade.ratios) < DEFAULT_FIELD_FACTOR


@pytest.mark.apbs
@pytest.mark.delphi
@pytest.mark.parametrize("name", UNDER_RESOLVED)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "M1b is not met where the sphere is under-resolved: debye is 5.24x the best "
        "reference on born-ion-vdw (a/h = 6) and 8.64x on born-ion-vdw-r1 (a/h = 2), "
        "against DelPhi C++'s 0.789% and APBS's 1.079% on grids no finer than debye's. "
        "Not the grid — at APBS's exact spacing debye reads 3.212% against APBS's "
        "3.186% — but the near-interface handling. Strict, so that closing the gap "
        "turns this red and says so rather than passing quietly."
    ),
)
def test_debye_is_within_a_factor_of_the_best_incumbent_where_it_is_not(name):
    grade = graded(case_named(name))
    assert grade.agrees, grade.summary()


@pytest.mark.apbs
@pytest.mark.delphi
def test_the_grade_samples_every_backend_at_the_same_radii():
    """The property that makes a cross-backend field number mean anything.

    Without it a coarse grid is sampled further from the boundary than a fine
    one and reads a smaller error for it: on `born-ion-vdw` the corpus samples
    DelPhi at r = 4.0 A and APBS at r = 3.81 A, because each uses its own
    achieved spacing.
    """
    case = case_named("born-ion-vdw")
    grade = graded(case)

    assert len(grade.radii_a) == len(case.analytic_field.cells_out)  # type: ignore[union-attr]
    # The radii come from the coarsest grid, so every backend's sample clears
    # its own interface cell by at least `cells_out` of its own spacing.
    for radius, cells in zip(grade.radii_a, case.analytic_field.cells_out, strict=True):  # type: ignore[union-attr]
        assert radius == pytest.approx(3.0 + cells * grade.spacing_used_a)
    assert set(grade.errors) >= {"apbs", "debye"}


@pytest.mark.apbs
def test_a_field_grade_refuses_a_candidate_with_nothing_to_grade_it_against():
    """An approximation is not a yardstick for a discretization."""
    case = case_named("born-ion-vdw")
    request = case.request()
    result = DebyeSolver().solve(request)
    runs = [BackendRun.from_result("debye", result, request)]

    with pytest.raises(Incomparable, match="at least"):
        grade_field(
            runs,
            candidate="debye",
            centre=case.structure().center(),
            radius_a=3.0,
            charge_e=1.0,
            solvent_dielectric=case.solvent.solvent_dielectric,
        )


@pytest.mark.apbs
def test_a_field_grade_refuses_backends_asked_different_questions():
    """Surface model changes the field, so a spread across it is not a solver gap."""
    case = case_named("born-ion-vdw")
    molecular = case_named("born-ion-molecular")

    runs = [
        BackendRun.from_result("apbs", ApbsSolver().solve(case.request()), case.request()),
        BackendRun.from_result(
            "debye-on-another-surface",
            ApbsSolver().solve(molecular.request()),
            molecular.request(),
        ),
    ]
    with pytest.raises(Incomparable, match="surface model"):
        grade_field(
            runs,
            candidate="apbs",
            centre=case.structure().center(),
            radius_a=3.0,
            charge_e=1.0,
            solvent_dielectric=case.solvent.solvent_dielectric,
        )


def test_only_one_delphi_flavour_can_be_an_incumbent_here():
    """Why these tests skip on pyDelPhi, asserted from the map that decides it.

    Needs no binary: `SUPPORTED_SURFACES` is a static declaration, so this holds
    the reason in place on every machine including the bare one. If pyDelPhi
    ever grows a real van der Waals surface this fails, which is the right time
    to revisit `require_delphi_that_builds_van_der_waals`.
    """
    assert SurfaceModel.VAN_DER_WAALS in DELPHI_SURFACES[DelphiFlavour.CPP]
    assert SurfaceModel.VAN_DER_WAALS not in DELPHI_SURFACES[DelphiFlavour.PYDELPHI]
