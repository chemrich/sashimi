"""M1b: debye's field, graded against the best solver already installed.

ROADMAP.md section 12 M1b. The bar is **not** a round number, and the reason is
worth stating once. debye reproduces DelPhi C++'s discretization to three decimal
places, so any bar of the form "no worse than the worst incumbent" is one it
meets by construction rather than by merit — a check that cannot fail, in section
7's sense, wearing the costume of a milestone. Against the *best* incumbent at
each sample it is a real measurement: agreeing with DelPhi does not put a solver
within a factor of APBS.

**This file previously recorded a gap that was not there, and the correction is
the most useful thing in it.** M1b's first measurement reported debye at 5.24x
the best reference on `born-ion-vdw` and 8.64x on `born-ion-vdw-r1`, concluded
that debye's near-interface handling fell behind DelPhi's where the sphere was
under-resolved, and pointed the next milestone at the Matched Interface and
Boundary literature. All of that was an artifact of **grid phase**.

The comparison sampled every backend at the same physical radii — which it had to
— but left each one on the lattice its own grid sizing happened to choose. It
graded debye at a/h = 6.46 against DelPhi at a/h = 6.00. Holding the sample
radius fixed at r = 4 A on the 3 A sphere and varying only the spacing across
0.43-0.50 A, the worst-direction error swings

    APBS        0.585% .. 3.915%   (6.7x)
    DelPhi C++  0.763% .. 3.837%   (5.0x)
    debye       0.773% .. 4.101%   (5.3x)

and on the 1 A sphere APBS alone spans 21x. The error collapses wherever a/h
nears an integer: the discretized cavity is a staircase, and its shape changes
discretely as face centres cross the sphere, so at low a/h a single face flipping
is a large fraction of the boundary. **Every finite-difference backend here does
this.** It is a property of hard midpoint dielectric assignment, not of debye.

At the spacings two backends both land on, debye is at parity with both:

    debye vs DelPhi C++   0.994-1.013x over 11 shared spacings (born-ion-vdw)
                          1.000-1.062x over 13 shared spacings (r1)
    debye vs APBS         0.871-1.116x over 16 shared spacings (born-ion-vdw)
                          1.037-1.617x over 18 shared spacings (r1)

so debye tracks both incumbents at matched spacing. The shipped gate, which pins
one lattice per case rather than joining wherever backends coincide, reads

    born-ion-vdw-r6    a/h = 12   1.00 / 1.01 / 1.01x
    born-ion-vdw-fine  a/h = 12   1.00 / 1.01 / 1.00x
    born-ion-vdw       a/h =  6   1.00 / 1.05 / 1.01x
    born-ion-vdw-r1    a/h =  2   1.69 / 1.19 / 1.04x

so 1.69x at worst, against a bar of 2, and M1b is met on all four.

Three controls, because the claim is that a recorded measurement was wrong:
reading grid **nodes** directly with no interpolation shows the same swing, so it
is the solver's field and not `sashimi.field`'s sampling; the error is
anisotropic across the cubic direction classes, which is the staircase's
signature and not a monopole defect; and at fixed h = 0.5 changing the padding
from 3 to 13 A moves debye's error only 0.7735% -> 0.7916%, so the box is not
doing the work that the spacing is.

`grade_field` now refuses a comparison across lattices outright — see
`sashimi.validate.check_same_lattice`. It is refused rather than annotated
because there is no honest way to read the number: the per-backend spacings were
already reported in `notes`, they were visible, and the verdict was wrong anyway.

**These tests pin the reference set to APBS *and* DelPhi C++ rather than grading
against whatever is installed, and that is not caution — it is required for the
verdict to exist.** An earlier draft fell back to APBS alone when DelPhi was
absent, which moved both the sample radii and the yardstick in debye's favour. A
milestone whose verdict depends on what happens to be installed is not a gate.
Each case now also pins the **padding**, for the same class of reason: it is what
puts all three backends on one lattice, and the grade is meaningless otherwise.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sashimi.apbs import ApbsSolver
from sashimi.corpus import MANIFEST, Case
from sashimi.debye import DebyeSolver
from sashimi.delphi import DelphiSolver
from sashimi.delphi.discover import DelphiFlavour, discover_delphi
from sashimi.delphi.options import SUPPORTED_SURFACES as DELPHI_SURFACES
from sashimi.field import sample_values
from sashimi.protocol import (
    AccuracyTier,
    Equation,
    PotentialGrid,
    SolveResult,
    SurfaceModel,
)
from sashimi.validate import (
    DEFAULT_FIELD_FACTOR,
    BackendRun,
    Incomparable,
    check_same_lattice,
    grade_field,
)

# case -> the padding that puts APBS, DelPhi C++ and debye on **one** lattice,
# and the a/h it produces. Not free parameters: each was found by asking the
# three backends' own `size_grid` what spacing a padding resolves to and keeping
# the ones where all three agree to better than 1e-6 relative. They agree
# exactly, because each lands on a binary fraction of the same box.
#
# a/h is the number that predicts the error's size, so the cases are chosen to
# span it — 12, 12, 6, 2 — rather than to span resolution or radius. That is what
# the first measurement got wrong: `born-ion-vdw-r6` and `born-ion-vdw` sit at
# the *same* h = 0.5 A and differ only in how many cells cross the radius.
#
# **The padding also has to keep the outermost sample clear of the box face**,
# which a review caught after the lattice fix and which three of the four
# original choices failed. The boundary condition lives on that face, and a
# sample near it inflates **APBS** specifically — the reference — which flatters
# debye. Measured at fixed lattice, worst-direction error at r = a + 8h against
# the margin expressed in units of that radius:
#
#   margin/r_out   APBS           DelPhi C++     debye
#   0.14           0.637%         0.118%         0.111%    <- born-ion-vdw @ 5.0
#   0.60           0.301-0.413%   0.233-0.370%   0.234-0.380%
#   >= 1.29        0.119-0.396%   converged      converged
#
# So the paddings below are the smallest common lattice clearing
# `MIN_BOX_MARGIN_RATIO`, and `grade_field` now refuses anything that does not —
# the old values are the mutation that reddens it.
COMMON_LATTICE = {
    "born-ion-vdw-r6": (18.0, 12.0),
    "born-ion-vdw-fine": (9.0, 12.0),
    "born-ion-vdw": (13.0, 6.0),
    "born-ion-vdw-r1": (15.0, 2.0),
}

# What each case's padding was before the box-margin finding, kept because it is
# the mutation that makes the new guard fire. `fine` is absent: its 9.0 already
# cleared the margin, which is why the table above shows it converged.
CONTAMINATED_PADDING = {
    "born-ion-vdw-r6": 10.0,
    "born-ion-vdw": 5.0,
    "born-ion-vdw-r1": 7.0,
}


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


def request_for(case: Case, padding: float | None = None):
    base = case.request()
    if padding is None:
        return base
    return dataclasses.replace(base, grid=dataclasses.replace(base.grid, padding=padding))


def graded(case: Case, padding: float | None = None, factor: float = DEFAULT_FIELD_FACTOR):
    """Solve the case through both incumbents and debye, and grade debye.

    The reference set is fixed, not discovered, and so is the padding. See the
    module docstring for both.
    """
    require_delphi_that_builds_van_der_waals()
    request = request_for(case, padding)
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
        cells_out=reference.cells_out,
        factor=factor,
    )


@pytest.mark.apbs
@pytest.mark.delphi
@pytest.mark.parametrize("name", sorted(COMMON_LATTICE))
def test_debye_is_within_a_factor_of_the_best_incumbent(name):
    """M1b's exit criterion, on every case — including the two it was thought to fail.

    These two were `strict` xfails carrying a 5.24x and an 8.64x, and closing the
    gap was supposed to turn them red. It did, but not by improving debye: the
    gap was the comparison, and the ratios above were between two different
    lattices. Graded on one lattice, all four cases pass.
    """
    case = case_named(name)
    padding, expected_cells = COMMON_LATTICE[name]
    grade = graded(case, padding=padding)

    # The padding is load-bearing, so assert it bought what it was chosen for
    # rather than trusting that it still does.
    assert grade.cells_across_radius == pytest.approx(expected_cells, rel=1e-6)
    assert grade.agrees, grade.summary()
    assert max(grade.ratios) < DEFAULT_FIELD_FACTOR


@pytest.mark.apbs
@pytest.mark.delphi
def test_a_field_grade_refuses_two_backends_on_different_lattices():
    """The check whose absence produced M1b's wrong verdict, with the mutation that reddens it.

    ROADMAP.md section 7's rule, applied: a guard lands with the mutation that
    makes it fire. The mutation here is the *original measurement* — grade at the
    corpus case's own padding of 10 A, where debye's grid sizing lands on
    h = 0.4643 and DelPhi's on h = 0.5000. That is exactly what M1b did, and it
    is what reported 5.24x.
    """
    require_delphi_that_builds_van_der_waals()
    case = case_named("born-ion-vdw")
    request = request_for(case)  # the case's own padding — the mismatching one
    runs = [
        BackendRun.from_result(name, solver.solve(request), request)
        for name, solver in (
            ("apbs", ApbsSolver()),
            ("delphicpp", DelphiSolver()),
            ("debye", DebyeSolver()),
        )
    ]
    spacings = {float(max(run.potential.spacing)) for run in runs}  # type: ignore[union-attr]
    assert len(spacings) > 1, f"the mutation did not mutate: every backend got {spacings}"

    with pytest.raises(Incomparable, match="differing lattices"):
        grade_field(
            runs,
            candidate="debye",
            centre=case.structure().center(),
            radius_a=case.analytic_field.radius_a,  # type: ignore[union-attr]
            charge_e=case.analytic_field.charge_e,  # type: ignore[union-attr]
        )


def test_the_near_field_error_depends_on_grid_phase_more_than_on_the_solver():
    """The finding underneath the correction, pinned so it cannot be quietly lost.

    Two debye solves of the same physical question, differing only in the lattice
    they land on, at the *same* physical sample radius. If this ever stops
    holding — because debye grows a dielectric treatment that varies continuously
    with h instead of in steps — this test is the one that should be revisited
    first, and it should be revisited rather than deleted: the number it pins is
    the reason `check_same_lattice` exists.

    Needs no binary. debye is the whole experiment, which is the point: the
    incumbents show the same swing, so nothing here is about which solver runs.
    """
    case = case_named("born-ion-vdw")
    centre = case.structure().center()
    exact = case.analytic_field.exact_at([4.0], case.solvent)[0]  # type: ignore[union-attr]

    errors = {}
    for padding in (5.0, 10.0):  # -> h = 0.5000 (a/h = 6.00) and h = 0.4643 (a/h = 6.46)
        result = DebyeSolver().solve(request_for(case, padding))
        assert isinstance(result.potential, PotentialGrid)
        values = sample_values(result.potential, centre, 4.0)
        errors[float(max(result.potential.spacing))] = max(
            abs(v - exact) / abs(exact) for v in values
        )

    assert len(errors) == 2, f"both paddings landed on one lattice: {errors}"
    worst, best = max(errors.values()), min(errors.values())
    # Measured at 4.14% against 0.77%. Asserted loosely, because the claim is
    # "phase dominates", not "phase is exactly 5.4x" — but a factor of three is
    # already far larger than the 1.62x that separates these three solvers.
    assert worst / best > 3.0, (
        f"grid phase moved debye's near-field error by only {worst / best:.2f}x "
        f"({errors}); if that is now small, the comparison across lattices that "
        "check_same_lattice refuses may no longer be the trap it was"
    )


def grid_at(spacing, origin=(0.0, 0.0, 0.0), n=41):
    """A `PotentialGrid` with a chosen lattice. Values are irrelevant to these checks."""
    return PotentialGrid(
        values=np.zeros((n, n, n)),
        origin=np.asarray(origin, float),
        spacing=np.asarray(spacing, float),
    )


def run_on(name: str, grid: PotentialGrid) -> BackendRun:
    return BackendRun(
        name=name,
        energy_kj_mol=None,
        energy_term=None,
        surface_model=SurfaceModel.VAN_DER_WAALS,
        equation=Equation.LINEAR,
        potential=grid,
    )


def test_the_lattice_check_refuses_differing_spacing_and_offset_needing_no_binary():
    """Both halves of `check_same_lattice`, reached directly. **Runs on the bare leg.**

    The binary-marked test above exercises the spacing half through real
    backends, which is the honest end-to-end version and which *skips* wherever
    APBS or DelPhi is absent — so on CI's pure-Python leg the whole refusal path
    was uncovered. And the **offset** half could not fire at all through any
    shipped backend: all three sizers place `origin = centre - fglen/2` on an odd
    node count, so the solute lands exactly on a node and the fractional offsets
    are identically zero. That is ROADMAP.md section 7's guard-that-cannot-fire,
    and the fix that section prescribes is to construct the state directly.
    """
    centre = np.array([10.0, 10.0, 10.0])

    same = [run_on("a", grid_at((0.5, 0.5, 0.5))), run_on("b", grid_at((0.5, 0.5, 0.5)))]
    assert check_same_lattice(same, centre) == pytest.approx(0.5)

    differing = [run_on("a", grid_at((0.5, 0.5, 0.5))), run_on("b", grid_at((0.4, 0.4, 0.4)))]
    with pytest.raises(Incomparable, match="differing lattices"):
        check_same_lattice(differing, centre)

    # Same spacing, solute half a cell over: a different staircase.
    shifted = [
        run_on("a", grid_at((0.5, 0.5, 0.5), origin=(0.0, 0.0, 0.0))),
        run_on("b", grid_at((0.5, 0.5, 0.5), origin=(0.25, 0.0, 0.0))),
    ]
    with pytest.raises(Incomparable, match="differently within a cell"):
        check_same_lattice(shifted, centre)


def test_the_lattice_check_accepts_two_identical_anisotropic_grids():
    """A non-cubic solute gives anisotropic spacing, and that is not a mismatch.

    `apbs.grid.size_grid` returns [0.4672, 0.4393, 0.4004] on `peptide-vdw`. The
    first version of this check took a single max-minus-min over an (n_grids, 3)
    array, which folds a grid's own anisotropy into the cross-backend spread — so
    two **byte-identical** anisotropic lattices were refused, with a message that
    printed one `h` per backend and contradicted itself. The comparison has to be
    per axis, and the returned spacing has to be the coarsest axis because
    `sample_radii` must clear the interface cell on all three.
    """
    centre = np.array([10.0, 10.0, 9.0])
    grid = grid_at((0.5, 0.5, 0.45))
    runs = [run_on("a", grid), run_on("b", grid)]

    assert check_same_lattice(runs, centre) == pytest.approx(0.5)


def test_a_field_grade_refuses_samples_that_do_not_clear_the_box():
    """The box-margin guard, with the paddings that produced the finding as its mutation.

    Needs no binary: debye alone, solved twice, at the padding each case carried
    before the margin was measured. The guard is one-sided in a way worth
    restating — it is **APBS** that a near-face sample inflates, and inflating the
    *reference* raises the yardstick and flatters the candidate.
    """
    for name, padding in CONTAMINATED_PADDING.items():
        case = case_named(name)
        request = request_for(case, padding)
        reference = case.analytic_field
        assert reference is not None
        runs = [
            BackendRun.from_result("debye", DebyeSolver().solve(request), request),
            BackendRun.from_result("also-debye", DebyeSolver().solve(request), request),
        ]
        with pytest.raises(Incomparable, match="clears the outermost sample"):
            grade_field(
                runs,
                candidate="debye",
                centre=case.structure().center(),
                radius_a=reference.radius_a,
                charge_e=reference.charge_e,
                cells_out=reference.cells_out,
            )


def test_a_field_grade_refuses_a_lone_backend():
    """One map is not a comparison. Needs no binary, and is marked accordingly.

    This was marked `apbs` while solving only with debye, so it skipped on the
    bare leg — the one configuration this repo most wants a pure-Python refusal
    path covered in.
    """
    case = case_named("born-ion-vdw")
    request = case.request()
    runs = [BackendRun.from_result("debye", DebyeSolver().solve(request), request)]

    with pytest.raises(Incomparable, match="at least"):
        grade_field(
            runs, candidate="debye", centre=case.structure().center(), radius_a=3.0, charge_e=1.0
        )


def test_a_field_grade_refuses_an_approximation_as_the_yardstick():
    """ "An approximation is not a yardstick for a discretization" — reached, not just written.

    The branch cannot be hit by any shipped backend: `sashimi.gb` returns no
    field and TABI-PB returns a `SurfacePotential`, so both are filtered out
    before the reference-tier count is taken. It was therefore a guard that
    could not fire, and the test that claimed to cover it asserted on the
    *count* message instead. Constructing the run directly is what reaches it,
    and it is the shape a future in-process approximate FD backend would have.
    """
    case = case_named("born-ion-vdw")
    request = case.request()
    candidate = BackendRun.from_result("debye", DebyeSolver().solve(request), request)
    approximate = dataclasses.replace(
        candidate, name="an-approximation", accuracy_tier=AccuracyTier.APPROXIMATE
    )

    with pytest.raises(Incomparable, match="approximation is not a yardstick"):
        grade_field(
            [candidate, approximate],
            candidate="debye",
            centre=case.structure().center(),
            radius_a=3.0,
            charge_e=1.0,
        )


def test_a_field_grade_refuses_a_closed_form_describing_different_physics():
    """The dielectric and the temperature must be the runs', not the caller's.

    Both shift the Born potential by a factor common to every backend, and the
    verdict is a *ratio* of errors — so a common offset drives every ratio
    towards 1.0 and the grade passes. `corpus.AnalyticField.exact_at` refuses
    exactly this on ionic strength; these are the other two axes, and they were
    caller arguments nothing cross-checked until a review asked.
    """
    case = case_named("born-ion-vdw")
    request = case.request()
    run = BackendRun.from_result("debye", DebyeSolver().solve(request), request)
    assert run.solvent_dielectric == case.solvent.solvent_dielectric
    assert run.temperature == case.solvent.temperature

    for attribute, value, message in (
        ("solvent_dielectric", 40.0, "solvent dielectric"),
        ("temperature", 277.0, "temperature"),
    ):
        other = dataclasses.replace(run, name="other", **{attribute: value})  # type: ignore[arg-type]
        with pytest.raises(Incomparable, match=message):
            grade_field(
                [run, other],
                candidate="debye",
                centre=case.structure().center(),
                radius_a=3.0,
                charge_e=1.0,
            )


def test_a_field_grade_refuses_a_sample_inside_the_interface_cell():
    """`sashimi.field` owns the sampling rule, so `grade_field` has to go through it.

    It computed `a + k*h` inline, which bypassed the `MIN_CELLS_OUT` guard that
    module exists to hold — `cells_out=(1,)` would have sampled the cell the
    dielectric interface passes through and reported an O(1) interpolation
    error as a solver gap. `cells_out=()` was worse: no ratios at all, and
    `agrees` is an `all()`, so nothing to check reads as agreement.
    """
    case = case_named("born-ion-vdw")
    request = case.request()
    runs = [
        BackendRun.from_result("debye", DebyeSolver().solve(request), request),
        BackendRun.from_result("apbs-like", DebyeSolver().solve(request), request),
    ]
    kwargs = {
        "candidate": "debye",
        "centre": case.structure().center(),
        "radius_a": 3.0,
        "charge_e": 1.0,
    }

    with pytest.raises(ValueError, match="cells of the boundary"):
        grade_field(runs, cells_out=(1,), **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nothing to sample"):
        grade_field(runs, cells_out=(), **kwargs)  # type: ignore[arg-type]


@pytest.mark.apbs
def test_a_field_grade_refuses_backends_asked_different_questions():
    """Surface model changes the field, so a spread across it is not a solver gap."""
    case = case_named("born-ion-vdw")
    molecular = case_named("born-ion-molecular")

    runs = [
        BackendRun.from_result("apbs", ApbsSolver().solve(case.request()), case.request()),
        BackendRun.from_result(
            "apbs-on-another-surface",
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
