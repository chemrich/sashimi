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

These tests need a reference solver to grade against, so unlike the rest of
debye's suite they are marked and they skip on a bare machine. That is the
honest shape of "as good as the incumbents": without an incumbent there is no
claim to check.
"""

from __future__ import annotations

import pytest

from sashimi.apbs import ApbsSolver
from sashimi.corpus import MANIFEST, Case
from sashimi.debye import DebyeSolver
from sashimi.delphi import DelphiSolver
from sashimi.delphi.discover import discover_delphi
from sashimi.errors import BackendUnavailable
from sashimi.protocol import PotentialGrid, SolveResult
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


def delphi_available() -> bool:
    try:
        discover_delphi()
    except BackendUnavailable:
        return False
    return True


def graded(case: Case, factor: float = DEFAULT_FIELD_FACTOR):
    """Solve the case through every reference backend present, and grade debye."""
    request = case.request()
    solvers: list[tuple[str, object]] = [("apbs", ApbsSolver()), ("debye", DebyeSolver())]
    if delphi_available():
        solvers.insert(1, ("delphicpp", DelphiSolver()))

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
