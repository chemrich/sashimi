"""Quality where the corpus has no ground truth, which is everywhere above two atoms.

Every closed form in the manifest is a one- or two-atom solute — 37 energies and
12 fields, all Born ions and Kirkwood spheres. **Thirty-two cases sit above 500
atoms and not one has a reference answer.** So above a peptide the corpus grades
agreement, and when the reference-tier backends spread 10.4% on a 1,156-residue
protein nothing in the suite can say which is closer to right.

These two checks close that, not with a better reference but by asserting what
the answer must satisfy whatever it is.

**The `gb` control is what makes the pose metric trustworthy.** An analytic
method has no lattice to fall out of phase with, so its spread must be zero —
and it is, to one ulp, where the grid solvers read 0.4-1.4%.

**Nothing here skips on a solver failure.** Only `BackendUnavailable` skips. An
earlier draft caught `SashimiError`, which swallowed `SolverCrash` too — and it
immediately hid a real finding: TABI-PB solves the unrotated peptide and
**crashes on every rotated pose of it**, which these tests discovered and that
`except` reported as "unavailable here". A green skip that is really a crash is
the failure this repository keeps recording.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sashimi import backends
from sashimi.errors import BackendUnavailable, UnsupportedRequest
from sashimi.invariants import grade_charge_scaling, grade_pose_spread, posed
from sashimi.pqr import read_pqr
from sashimi.protocol import (
    AccuracyTier,
    GridSpec,
    PQRData,
    SolventModel,
    SurfaceModel,
    System,
)
from sashimi.validate import Backend

# Small enough that every backend can run the sweep on every push, and
# `molecular` because it is the one surface all five share.
PEPTIDE = "tests/data/ala-gly.pqr"
RESOLUTION = 0.5

# The linear equation makes this exact, so the only tolerance needed is the one
# a backend's *reporting* imposes. Measured across all five: APBS 0 to 2.4e-13,
# pyDelPhi 0 to 6.2e-8, TABI-PB 2.5e-10, debye and gb exactly 0 — and DelPhi C++
# 3.0e-5, which is printed precision rather than solver error, since it reports
# two decimals in kT. The bar is three times the worst of those.
CHARGE_SCALING_TOLERANCE = 1e-4

# Twelve poses because the estimator has to be a statistic: the *range* over five
# read 3.01% and 0.60% for debye on two draws of one structure, where the
# standard deviation over twelve splits into halves agreeing within a factor of
# 1.4. `dispersion` is asserted on; `relative` is only ever quoted.
POSES = 12

# debye's own pose dispersion on this case, and a band that catches a real
# regression without going red on a quiet day. Measured on `ala-gly` at 0.5 A.
#
# **Deliberately a recorded value rather than a relational bar**, which is the
# opposite of what M1b and M4 chose and needs saying why. A relational gate here
# would read "no worse than N times the worst reference-tier backend installed",
# and that bar *moves with the machine*: with APBS present the worst is 0.80% and
# debye passes at 3x; with only DelPhi, the worst is 0.35% and the same
# unchanged debye goes red. A gate whose verdict depends on which binaries a
# contributor happens to have is not a gate. The cross-backend comparison is
# recorded in ROADMAP section 12 instead, where it is a measurement and not a
# threshold.
DEBYE_POSE_DISPERSION = 0.0091
DEBYE_POSE_BAND = 0.003


def _system(path: str = PEPTIDE, resolution: float = RESOLUTION) -> System:
    return System(
        structure=read_pqr(path),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
        grid=GridSpec(resolution=resolution, padding=10.0),
        want_energy=True,
        want_potential=False,
    )


def _backend(name: str) -> Backend:
    solver, family = backends.solver_for(name)
    return Backend(name=name, solver=solver, family=family)


def _is_reference(name: str) -> bool:
    """Read the tier from the registry's report, which is where it lives.

    An earlier draft asked `getattr(solver, "accuracy_tier", REFERENCE)`. No
    solver object carries that attribute, so the default always won and `gb` —
    an approximation — was graded as a reference backend. The tier is on
    `BackendReport` and on `SolveResult.provenance`, and `validate.py` reads the
    latter; nothing puts it on the solver.
    """
    report = next(r for r in backends.reports() if r.name == name)
    return report.accuracy_tier == AccuracyTier.REFERENCE.value


@pytest.mark.parametrize("name", sorted(backends.names()))
def test_scaling_every_charge_scales_the_energy_by_the_square(name: str):
    """`E(lam*q) == lam**2 * E(q)`, exactly, for every family.

    The linearized equation is linear in the charge and the dielectric map does
    not depend on it, so this is an identity rather than an approximation — and
    it holds for a boundary-element and an analytic method too, since a
    Generalized Born radius is a property of the geometry.

    **It is the check that would have caught the bug section 7 records costing a
    year.** `format_pqr` wrote minimum-width fields, a four-character residue
    name shifted every column after it, and DelPhi solved on charges that were
    not in the file — returning -865,205 kJ/mol for acetate against APBS's -197.
    Any mis-assignment of charge breaks the square.
    """
    try:
        grade = grade_charge_scaling(_backend(name), _system(), factor=2.0)
    except BackendUnavailable as exc:
        pytest.skip(f"{name} is not installed here: {exc}")
    except UnsupportedRequest as exc:
        pytest.skip(f"{name} refuses this system by design: {exc}")
    assert grade.error < CHARGE_SCALING_TOLERANCE, (
        f"{grade.backend} broke E ~ q^2 by {grade.error:.3e}: "
        f"{grade.scaled_energy} against an expected {grade.expected}"
    )


def test_an_analytic_backend_is_exactly_pose_invariant():
    """The control the whole pose metric rests on.

    `gb` discretizes nothing, so a rigid motion cannot change its answer beyond
    floating point. If this ever reads a real number the metric is measuring the
    harness — the poses, the centroid, the rotation — rather than the solvers.

    **Not bitwise, and the difference is the point.** Rotating coordinates
    perturbs their last bits, so the measured dispersion is ~2.6e-16, about one
    ulp, against 0.4% to 1.4% for the grid solvers. Twelve orders of magnitude is
    the margin; asserting exact equality would only make this fragile against a
    numpy release.
    """
    grade = grade_pose_spread(_backend("gb"), _system(), poses=POSES)
    assert grade.dispersion < 1e-12, f"gb moved under rigid motion: {grade.energies}"


# TABI-PB solves this peptide as given and aborts on every rotated pose of it —
# `terminating due to uncaught exception`, exit -6, from the mesher rather than
# from sashimi. Marked strict so that a future TABI-PB release fixing it turns
# this red and the marker gets removed, rather than the xfail quietly outliving
# the defect. It is a real backend limitation, found by this file.
POSE_ROTATION_XFAIL = {
    "tabipb": "TABI-PB aborts on a rotated solute; see ROADMAP.md section 12",
}


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            n,
            marks=(
                [pytest.mark.xfail(strict=True, reason=POSE_ROTATION_XFAIL[n])]
                if n in POSE_ROTATION_XFAIL
                else []
            ),
        )
        for n in sorted(backends.names())
    ],
)
def test_every_backend_answers_a_rigidly_moved_solute(name: str):
    """Rotating a molecule must not stop a solver working.

    Recorded rather than gated per backend — an incumbent's discretization error
    is not sashimi's to hold a line on. What *is* asserted is that the backend
    answers at all, and that is not a formality: TABI-PB solves this peptide as
    given and crashes on every rotated pose of it, which this test found.
    """
    try:
        grade = grade_pose_spread(_backend(name), _system(), poses=4)
    except BackendUnavailable as exc:
        pytest.skip(f"{name} is not installed here: {exc}")
    except UnsupportedRequest as exc:
        pytest.skip(f"{name} refuses this system by design: {exc}")
    assert len(grade.energies) == 4
    assert all(e < 0 for e in grade.energies), "a solvation energy should be negative here"
    assert grade.dispersion < 0.5, (
        f"{grade.backend} moved {grade.dispersion:.1%} under rigid motion"
    )


def test_debye_pose_dispersion_stays_where_it_was_measured():
    """debye's own discretization error, as a regression band.

    See `DEBYE_POSE_DISPERSION` for why this is recorded rather than graded
    against the incumbents: a relational bar would move with which binaries the
    machine has, and go red on an unchanged debye.
    """
    grade = grade_pose_spread(_backend("debye"), _system(), poses=POSES)
    assert grade.dispersion == pytest.approx(DEBYE_POSE_DISPERSION, abs=DEBYE_POSE_BAND), (
        f"debye's pose dispersion moved to {grade.dispersion:.4%} from a recorded "
        f"{DEBYE_POSE_DISPERSION:.4%}"
    )


def test_the_tier_filter_reads_the_registry_and_not_the_solver():
    """`gb` is an approximation and must be classified as one.

    The guard on a no-op: `getattr(solver, "accuracy_tier", REFERENCE)` returns
    the default for every backend, so a filter written that way silently treats
    the analytic tier as a reference solver.
    """
    assert not _is_reference("gb")
    assert _is_reference("debye")
    assert any(_is_reference(n) for n in backends.names())


def test_a_pose_is_a_rotation_and_not_a_reflection():
    """The determinant correction in `posed` has to be covered, and distances do not cover it.

    A reflection preserves every distance — including every distance from the
    centroid — so a test built on those passes with the correction deleted, and
    for a chiral solute a reflection is a different molecule. This asserts the
    full pairwise distance matrix *and* that the motion is orientation-preserving,
    by checking a signed volume rather than a length.
    """
    structure = read_pqr(PEPTIDE)
    assert np.array_equal(posed(structure, 0, spacing=RESOLUTION).coords, structure.coords)

    def pairwise(coords):
        return np.sort(np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1).ravel())

    def chirality(coords):
        centred = coords - coords.mean(axis=0)
        return float(np.linalg.det(centred[:3]))

    for index in (1, 2, 3):
        moved = posed(structure, index, spacing=RESOLUTION).coords
        assert not np.array_equal(moved, structure.coords), f"pose {index} did not move"
        assert np.allclose(pairwise(moved), pairwise(structure.coords)), "not a rigid motion"
        assert np.sign(chirality(moved)) == np.sign(chirality(structure.coords)), (
            f"pose {index} reflected the solute instead of rotating it"
        )


def test_the_translation_scales_with_the_grid():
    """`POSE_SHIFT_CELLS` is in cells, so the same pose means the same thing at any spacing.

    A fixed shift in angstroms was the first draft, and it was sub-spacing only
    at 1.0 A: at the 0.5 A these tests use it spanned two whole cells, which is
    not the "different place within a cell" the metric is defined as.
    """
    structure = read_pqr(PEPTIDE)
    fine = posed(structure, 1, spacing=0.25).coords - structure.coords
    coarse = posed(structure, 1, spacing=1.0).coords - structure.coords
    # Same rotation, so the difference between the two is purely the translation.
    assert np.linalg.norm(coarse.mean(axis=0)) > np.linalg.norm(fine.mean(axis=0))


def test_a_sub_cell_translation_changes_nothing_because_the_box_follows_it():
    """The half of `posed` that does no work, pinned so nobody re-derives its rationale.

    `POSE_SHIFT_CELLS` used to be documented as probing grid phase. It cannot:
    `size_grid` builds the box from `pqr.center()` and `pqr.extent()`, so moving
    the solute moves the lattice with it and every atom keeps its position
    relative to its own nodes. **The rotation is what varies the phase**, by
    changing the bounding box and therefore the lattice.

    Asserted as an equality rather than a tolerance because it is an identity of
    the construction, not a numerical accident — and the contrast is asserted
    too, so a change that quietly made rotation inert as well would not slip
    through a test that only said "translation does nothing".
    """
    base = read_pqr(PEPTIDE)
    solver, family = backends.solver_for("debye")

    def energy(structure: PQRData) -> float:
        system = replace(_system(), structure=structure)
        answer = solver.solve(system.request_for(family)).energy_kj_mol
        assert answer is not None
        return float(answer)

    still = energy(base)
    for cells in (0.25, 0.5, 0.75):
        moved = replace(base, coords=base.coords + cells * RESOLUTION)
        assert energy(moved) == pytest.approx(still, abs=1e-9)

    rng = np.random.default_rng(7)
    turn, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    turn = turn * np.sign(np.linalg.det(turn))
    centre = base.coords.mean(axis=0)
    rotated = replace(base, coords=(base.coords - centre) @ turn.T + centre)
    assert abs(energy(rotated) - still) > 0.1, (
        "the rotation moved the energy by less than 0.1 kJ/mol, so this fixture "
        "no longer exercises the phase variation the pose spread is built on"
    )


def _born(radius: float) -> PQRData:
    return PQRData(coords=np.zeros((1, 3)), charges=np.array([1.0]), radii=np.array([radius]))


def _born_system(radius: float, spacing: float) -> System:
    return System(
        structure=_born(radius),
        solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS, ionic_strength=0.0),
        grid=GridSpec(resolution=spacing, padding=10.0),
        want_energy=True,
        want_potential=False,
    )


@pytest.mark.parametrize(
    ("radius", "ladder", "tolerance"),
    [(3.0, (1.0, 0.5, 0.25), 0.006), (2.0, (0.5, 0.25, 0.125), 0.005)],
)
def test_richardson_finds_a_limit_that_is_already_known(radius, ladder, tolerance):
    """The extrapolator graded where the answer is not in question.

    `grade_refinement` reports a limit no backend can vouch for, so its accuracy
    has to be established somewhere the truth is independent — which is the Born
    sphere, whose solvation energy is a closed form. Measured: the limit lands
    0.08-0.48% from exact where the ladder converges. That is the claim this
    instrument is allowed to make — half a percent, not a gold standard.

    *This used to add "and beats the finest rung it was built from **every
    time**". It does not: re-measured over 24 Born windows, three of the sixteen
    survivors land no closer than their own finest rung — `r=3, w=0.25` at
    0.949% against that rung's 0.892%, `r=3, w=0.5` at 0.596% against 0.306%,
    and `r=3, w=1.0` at 1.275% against 0.415%. The assertion below still holds
    for these two ladders and is kept as a per-case bar; what is withdrawn is
    the universal.*
    """
    from sashimi.analytic import born_solvation_energy  # noqa: PLC0415
    from sashimi.invariants import grade_refinement  # noqa: PLC0415

    backend = _backend("debye")
    system = _born_system(radius, ladder[0])
    grade = grade_refinement(backend, system, spacings=ladder)
    # The closed form has to be evaluated at the *solvent model's* dielectrics,
    # not the function's defaults: `solute_dielectric` defaults to 1.0 there and
    # is 2.0 here, which is a factor of two in the answer rather than a rounding.
    exact = born_solvation_energy(
        radius_a=radius,
        charge_e=1.0,
        solute_dielectric=system.solvent.solute_dielectric,
        solvent_dielectric=system.solvent.solvent_dielectric,
    )

    assert grade.converging, f"the ladder {grade.energies} is not in the asymptotic regime"
    assert abs(grade.limit - exact) / abs(exact) < tolerance
    assert abs(grade.limit - exact) < abs(grade.energies[-1] - exact), (
        "the extrapolation is no better than the finest rung it was built from, "
        "so it is not earning the two extra solves"
    )


def test_a_ladder_outside_the_asymptotic_regime_is_refused_rather_than_fitted():
    """The guard that makes the limit safe to quote.

    A 2 A sphere on a ladder starting at 1 A is under-resolved, and its energies
    do not descend — they read -172.6, -177.0, -171.7. Richardson on that
    returns a limit 19.6% from exact, which is worse than every rung it was
    built from. `converging` is what stops a caller believing it, and it is
    checked here rather than assumed because ROADMAP.md section 12 records that
    nothing converges monotonically at `d/a >= 0.5` on a sharp boundary.
    """
    from sashimi.invariants import grade_refinement  # noqa: PLC0415

    grade = grade_refinement(_backend("debye"), _born_system(2.0, 1.0), spacings=(1.0, 0.5, 0.25))
    assert not grade.converging, f"expected a non-monotone ladder, got {grade.energies}"


def test_converging_refuses_the_two_shapes_that_produced_impossible_limits():
    """The half-guard that shipped, and the two disasters it let through.

    `converging` bounded only the *magnitude* of successive differences for as
    long as it existed. A width sweep on 2026-08-24 found it labelling a `fas2`
    ladder `converging=True` whose Richardson limit was **+2902 kJ/mol** — a
    positive polar solvation energy, which linear response forbids for any
    solute at `eps_solvent > eps_solute`.

    Both shapes below are held as synthetic ladders rather than as solves. The
    numbers are the *scheme* — a ratio and a sign — not a recorded answer, so
    this cannot drift with a backend and cannot become a per-platform digit
    anchor the way ROADMAP.md section 12 records an absolute energy doing.

    Measured over 24 Born-sphere windows where the exact answer is known, these
    two clauses catch **different** disasters and neither subsumes the other:
    269.688% at a shrink ratio of 1.0327 with matching signs, and 7.078% at a
    ratio of 8.75 with opposite ones.
    """
    from sashimi.invariants import MIN_SHRINKAGE, Refinement  # noqa: PLC0415

    spacings = (1.1988, 0.7552, 0.4757)

    def grade(energies: tuple[float, float, float]) -> Refinement:
        return Refinement(backend="debye", spacings=spacings, energies=energies)

    # Shape one: differences share a sign and shrink, but by only 3%. This is
    # the `fas2` w=2.0 ladder's shape, and the amplifier it licenses is 30.6x.
    barely = grade((-1000.0, -1900.0, -2771.0))
    steps = barely._differences
    assert steps[0] * steps[1] > 0.0, "this fixture is meant to have matching signs"
    # Near 1 is the property; the exact digit is not. `fas2` read 1.0327 and
    # this reads 1.0333, and pinning either would be a fitted constant standing
    # in for "the corrections hardly moved".
    assert 1.0 < abs(steps[0] / steps[1]) < 1.05
    assert not barely.converging, "a ladder whose corrections barely shrink is not a fit"

    # Shape two: differences reverse sign — a zigzag, which refutes
    # `E(h) = L + C h^p` outright rather than fitting it badly.
    #
    # **Deliberately shrinking by 9x, far clear of `MIN_SHRINKAGE`, so this
    # isolates the sign clause.** The first draft of this fixture used `fas2`'s
    # own w=2.0 energies, whose ratio is 1.1377 — under the floor, so the shrink
    # clause rejected them whatever their sign did and the sign clause was never
    # exercised. The Born set's 7.078% disaster is the shape that matters here:
    # a ratio of 8.75 with opposite signs, which only the sign test refuses.
    zigzag = grade((-1000.0, -1900.0, -1800.0))
    steps = zigzag._differences
    assert abs(steps[0]) > abs(steps[1]) > 0.0, "the old magnitude-only test passed this"
    assert abs(steps[0] / steps[1]) > MIN_SHRINKAGE, (
        "this fixture must clear the shrink floor, or it does not test the sign clause"
    )
    assert steps[0] * steps[1] < 0.0, "this fixture is meant to oscillate"
    assert not zigzag.converging, "an oscillating ladder is what the docstring always claimed"

    # And a ladder that is genuinely converging still passes, so the guard has
    # not simply been tightened into rejecting everything.
    clean = grade((-200.0, -208.0, -210.0))
    assert abs(clean._differences[0] / clean._differences[1]) > MIN_SHRINKAGE
    assert clean.converging


def test_the_limit_is_a_function_of_the_energies_and_not_the_spacings():
    """`step` cancels out of the extrapolation, so only `order` ever reads it.

    `order` is `log|d_prev / d_last| / log(step)` and `limit` divides by
    `step ** order`, which is therefore identically `|d_prev / d_last|` whatever
    `step` was. The Richardson correction is `d_last / (ratio - 1)` — three
    energies and nothing else.

    Worth a test rather than a comment because the opposite is the natural
    assumption, and this repository acted on it: `_achieved_spacing` exists
    because "Richardson divides by the refinement ratio, so the ratio has to be
    the real one", and ROADMAP.md section 12 records M8a's orders moving
    0.10-0.16 when the convention was corrected. Both are true of `order`.
    Neither is true of `limit`, and a reader who believes otherwise will look
    for a bias here that is not here.

    Asserted with `==` on these three triples, which do agree exactly — but the
    identity is exact in **real** arithmetic and not in IEEE double, because
    `x ** (log r / log x)` reintroduces a rounding. Over 20,000 randomised
    accepted ladders two spacing triples return a different `limit` float on
    **0.66%** of them, worst relative difference 7e-15. So this pins the
    identity on a fixed set rather than asserting a universal, and a caller
    comparing two limits should carry a tolerance.
    """
    from sashimi.invariants import Refinement  # noqa: PLC0415

    energies = (-236.5184, -227.8202, -220.5902)
    limits = {
        Refinement(backend="debye", spacings=spacings, energies=energies).limit
        for spacings in (
            (0.8695, 0.4545, 0.2432),  # achieved: ratios 1.913 and 1.869
            (1.0, 0.5, 0.25),  # requested: exactly 2
            (5.0, 0.4545, 0.2432),  # absurd, and it must not matter
        )
    }
    assert len(limits) == 1, f"the limit moved with the spacings: {sorted(limits)}"

    # And it is the closed form the docstring names.
    first, second = energies[0] - energies[1], energies[1] - energies[2]
    by_hand = energies[-1] - second / (abs(first / second) - 1.0)
    assert limits.pop() == by_hand

    # The orders, by contrast, do move — otherwise the paragraph above would be
    # describing a distinction with no difference.
    orders = {
        round(Refinement(backend="debye", spacings=spacings, energies=energies).order, 6)
        for spacings in ((0.8695, 0.4545, 0.2432), (1.0, 0.5, 0.25))
    }
    assert len(orders) == 2


def test_min_shrinkage_bounds_how_far_the_extrapolation_can_move():
    """Why `converging` needs no noise floor beside it.

    A clause refusing ladders whose steps sit below the backend's own phase
    noise was proposed on the strength of "the guard still accepts fits whose
    steps are below the pose spread". It is not needed, and the reason is the
    identity above: with `step ** order == |d_prev / d_last|`,

        |limit - E_finest| = |d_last| / (ratio - 1) <= |d_last| / (MIN_SHRINKAGE - 1)

    so a ladder this guard accepts cannot move the answer by more than **four
    times its own last difference**. A step at the noise floor moves the limit
    by at most four noise units; there is no unbounded amplifier left to catch.

    How tight the bound is, from the shrink ratios recorded beside
    `MIN_SHRINKAGE`: the nearest keep, 1.3968, licenses an amplifier of
    **2.5202** where the rejected 1.0327 would have licensed **30.58**. Over
    50,000 randomised accepted ladders the worst observed is **3.9877**, so the
    bound is attained rather than generous.

    *A second argument — that the surviving windows' relative last steps run
    0.0247% to 2.7903% continuously, with the four most accurate fits holding
    four of the five smallest steps — is real and is **not** reproducible from
    anything checked in: the 24-window generator lives in a scratch directory,
    not the repository. It is recorded in ROADMAP.md as what it is.*

    `MIN_SHRINKAGE` is what makes this hold, so the bound is asserted at the
    boundary as well as inside it.
    """
    from sashimi.invariants import MIN_SHRINKAGE, Refinement  # noqa: PLC0415

    # **The bound is a literal, and that is the whole point of this test.** The
    # first draft computed it as `1 / (MIN_SHRINKAGE - 1)` and drew its ladders
    # from the same constant, so both sides moved together and the assertion
    # held for *any* value of it — verified by mutation: at `MIN_SHRINKAGE = 1.05`,
    # where the true bound is 20x, that version stayed green. It was the shape
    # `sashimi-guards-that-guard-nothing` is about, in a test written to close a
    # question about guards.
    bound = 4.0
    spacings = (1.1988, 0.7552, 0.4757)

    assert bound >= 1.0 / (MIN_SHRINKAGE - 1.0), (
        f"MIN_SHRINKAGE = {MIN_SHRINKAGE} licenses an amplifier of "
        f"{1.0 / (MIN_SHRINKAGE - 1.0):.4g}, over the {bound:g} this test and "
        "the comment beside the constant both assert"
    )

    # Ratios fixed in the test rather than derived from the constant, and
    # spanning both sides of it, so relaxing the guard admits the small ones and
    # they violate the literal bound.
    accepted = 0
    for ratio in (1.05, 1.1, 1.2, MIN_SHRINKAGE, 1.4, 2.0, 8.0, 64.0):
        last = -3.0
        grade = Refinement(
            backend="debye",
            spacings=spacings,
            energies=(-200.0 - last * ratio - last, -200.0 - last, -200.0),
        )
        if not grade.converging:
            continue
        accepted += 1
        moved = abs(grade.limit - grade.energies[-1])
        assert moved <= bound * abs(last) * (1.0 + 1e-9), (
            f"ratio {ratio}: the guard accepts it and the limit moves "
            f"{moved / abs(last):.4g} last-differences, over the bound of {bound:g}"
        )
    assert accepted >= 5, f"only {accepted} ladders were accepted, so this grades almost nothing"

    # And the bound is attained at the floor rather than slack, which is the
    # argument for the constant: measured over 50,000 randomised accepted
    # ladders the worst amplifier is 3.9877.
    tight = Refinement(
        backend="debye",
        spacings=spacings,
        energies=(-200.0 + 3.0 * MIN_SHRINKAGE + 3.0, -200.0 + 3.0, -200.0),
    )
    assert abs(tight.limit - tight.energies[-1]) == pytest.approx(bound * 3.0, rel=1e-9)
