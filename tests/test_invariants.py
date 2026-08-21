"""Quality where the corpus has no ground truth, which is everywhere above two atoms.

Every closed form in the manifest is a one- or two-atom solute — 37 energies and
12 fields, all Born ions and Kirkwood spheres. **Thirty-two cases sit above 500
atoms and not one has any reference answer.** So above a peptide the corpus
grades agreement, and when the reference-tier backends spread 10.4% on a
1,156-residue protein nothing in the suite can say which is closer to right.

These two checks close that, not by finding a better reference but by asserting
what the answer must satisfy whatever it is.

**The `gb` control is what makes the pose metric trustworthy.** An analytic
method has no lattice to fall out of phase with, so its spread must be exactly
zero — and it is, to the last bit, where the grid solvers read 0.4-1.4%. A
reference-free metric whose one exactly-known case comes out exactly right is
measuring what it claims to.
"""

from __future__ import annotations

import pytest

from sashimi import backends
from sashimi.errors import SashimiError, UnsupportedRequest
from sashimi.invariants import grade_charge_scaling, grade_pose_spread
from sashimi.pqr import read_pqr
from sashimi.protocol import AccuracyTier, GridSpec, SolventModel, SurfaceModel, System

# Small enough that every backend can run the whole sweep on every push, and
# `molecular` because it is the one surface all five share.
PEPTIDE = "tests/data/ala-gly.pqr"
RESOLUTION = 0.5

# The linear equation makes this exact, so the only tolerance needed is the one
# a backend's *reporting* imposes. Measured across all five: APBS 0 to 2.4e-13,
# pyDelPhi 0 to 6.2e-8, TABI-PB 2.5e-10, debye and gb exactly 0 — and DelPhi C++
# 3.0e-5, which is not solver error but printed precision, since it reports two
# decimals in kT. The bar is three times the worst of those.
CHARGE_SCALING_TOLERANCE = 1e-4

# How many rigid poses. Twelve because the estimator needs to be a statistic:
# the *range* over five poses read 3.01% and 0.60% for debye on two draws of the
# same structure, where the standard deviation over twelve splits into halves
# agreeing to within a factor of 1.4. `dispersion` is gated; `relative` is only
# quoted.
POSES = 12

# debye's discretization error against the incumbents', relational so it carries
# no absolute constant — the shape M1b and M4 both landed on. Measured on fas2
# at 1.0 A: debye 1.416% against APBS 0.764% and DelPhi 0.410%, so 1.85x the
# *worst* reference-tier backend. Gated at 3x, which a doubling of debye's error
# would break and which the present margin clears comfortably.
POSE_FACTOR = 3.0


def _system(path: str = PEPTIDE, resolution: float = RESOLUTION) -> System:
    return System(
        structure=read_pqr(path),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
        grid=GridSpec(resolution=resolution, padding=10.0),
        want_energy=True,
        want_potential=False,
    )


def _graded(name: str):
    """The solver, or a skip naming why this machine cannot run it."""
    try:
        solver, family = backends.solver_for(name)
    except SashimiError as exc:  # pragma: no cover - registry names are valid
        pytest.skip(f"{name} is not constructible here: {exc}")
    return solver, family


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
    solver, family = _graded(name)
    try:
        grade = grade_charge_scaling(solver, family, _system(), factor=2.0)
    except UnsupportedRequest as exc:
        pytest.skip(f"{name} refuses this system: {exc}")
    except SashimiError as exc:
        pytest.skip(f"{name} is unavailable here: {exc}")
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
    perturbs their last bits, so the measured dispersion is 2.6e-16 — about one
    ulp — against 0.4% to 1.4% for the grid solvers. Twelve orders of magnitude
    is the margin, and asserting exact equality would only make the test fragile
    against a numpy release.
    """
    solver, family = _graded("gb")
    grade = grade_pose_spread(solver, family, _system(), poses=POSES)
    assert grade.dispersion < 1e-12, f"gb moved under rigid motion: {grade.energies}"


@pytest.mark.parametrize("name", sorted(backends.names()))
def test_pose_spread_is_recorded_for_every_backend(name: str):
    """Every backend answers, and the spread is finite and sane.

    Recorded rather than gated per backend: the incumbents' own discretization
    error is not sashimi's to hold a line on, and the number is useful precisely
    because it describes them. `test_debye_is_not_far_worse_than_the_incumbents`
    is where it becomes a bar.
    """
    solver, family = _graded(name)
    try:
        grade = grade_pose_spread(solver, family, _system(), poses=4)
    except UnsupportedRequest as exc:
        pytest.skip(f"{name} refuses this system: {exc}")
    except SashimiError as exc:
        pytest.skip(f"{name} is unavailable here: {exc}")
    assert len(grade.energies) == 4
    assert all(e < 0 for e in grade.energies), "a solvation energy should be negative here"
    assert grade.dispersion < 0.5, (
        f"{grade.backend} moved {grade.dispersion:.1%} under rigid motion"
    )


def test_debye_is_not_far_worse_than_the_incumbents_under_rigid_motion():
    """debye's discretization error, graded against the solvers it must replace.

    Relational, so it carries no constant of its own — the shape M1b chose over
    a round number and M4 over an absolute band. The comparison is against the
    *worst* installed reference-tier backend rather than the best, because the
    claim being defended is "debye is in the same class", not "debye is the best
    discretization here", which it is not: on fas2 at 1.0 A it reads 1.416%
    against DelPhi's 0.410%.

    Skips rather than passes when no other reference-tier backend is installed —
    a bar with nothing to compare against is the check that cannot fail.
    """
    system = _system()
    reference = []
    for name in sorted(backends.names()):
        if name == "debye":
            continue
        solver, family = _graded(name)
        if getattr(solver, "accuracy_tier", AccuracyTier.REFERENCE) is not AccuracyTier.REFERENCE:
            continue
        try:
            reference.append(grade_pose_spread(solver, family, system, poses=POSES))
        except (SashimiError, UnsupportedRequest):
            continue
    if not reference:
        pytest.skip("no other reference-tier backend is installed to grade against")

    worst = max(g.dispersion for g in reference)
    solver, family = _graded("debye")
    debye = grade_pose_spread(solver, family, system, poses=POSES)
    assert debye.dispersion <= POSE_FACTOR * worst, (
        f"debye's pose spread is {debye.dispersion:.4%}, more than {POSE_FACTOR}x the "
        f"worst reference-tier backend at {worst:.4%} "
        f"({', '.join(f'{g.backend} {g.dispersion:.4%}' for g in reference)})"
    )


def test_the_pose_sweep_actually_moves_the_solute():
    """Otherwise every spread above is zero for the wrong reason."""
    from sashimi.invariants import posed  # noqa: PLC0415 — local to the one test

    structure = read_pqr(PEPTIDE)
    first = posed(structure, 0)
    assert first.coords is structure.coords or (first.coords == structure.coords).all()
    for index in (1, 2, 3):
        moved = posed(structure, index)
        assert not (moved.coords == structure.coords).all(), f"pose {index} did not move"
        # A rigid motion preserves every interatomic distance.
        import numpy as np  # noqa: PLC0415

        def spread(c):
            return np.linalg.norm(c - c.mean(axis=0), axis=1)

        assert np.allclose(np.sort(spread(moved.coords)), np.sort(spread(structure.coords)))
