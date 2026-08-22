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
    0.08-0.48% from exact where the ladder converges, and **beats the finest
    rung it was built from every time** (0.45-0.81%). That is the claim this
    instrument is allowed to make — half a percent, not a gold standard.
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
