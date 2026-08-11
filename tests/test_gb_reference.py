"""Generalized Born against a real Poisson-Boltzmann solver.

`tests/test_gb.py` pins the method against its own closed form and against
quadrature, neither of which needs a binary. This file asks the different
question: does the approximation land where an approximation should, measured
against the thing it approximates? It needs APBS, so it is marked for it.

**This is the file that would have caught both mistakes made while building the
backend**, and neither was visible from the physics tests:

- The surface model. Descreening integrates over van der Waals spheres, so
  `van-der-waals` is the intuitive declaration, and it is wrong: the OBC
  rescaling exists to carry the union of spheres onto the solvent-excluded
  volume, and the parameters were fit against Poisson-Boltzmann on the
  *molecular* surface. Declared the intuitive way, hen lysozyme sat 31% out.
- The radii. pdb2pqr emits Lennard-Jones radii, including zero for hydroxyl
  hydrogens; Generalized Born needs the mbondi set it was fit with. Handed
  pdb2pqr's, the same protein sat 35% out.

Both look like "the approximation is imprecise" and are neither.
"""

from __future__ import annotations

import dataclasses

import pytest

from sashimi.apbs import ApbsSolver
from sashimi.corpus import MANIFEST, Case
from sashimi.errors import UnsupportedRequest
from sashimi.gb import GbSolver
from sashimi.protocol import (
    AccuracyTier,
    FiniteDifferenceRequest,
    SolveRequest,
    SurfaceModel,
)
from sashimi.validate import (
    DEFAULT_APPROXIMATION_TOLERANCE,
    Backend,
    SolverFamily,
    System,
    validate_system,
)

pytestmark = pytest.mark.apbs

# Measured against the APBS/DelPhi consensus on the molecular surface: 1.89,
# 2.77, 2.04, 7.10 and 6.75% across the corpus, and 1.48% on hen lysozyme. The
# gate is `DEFAULT_APPROXIMATION_TOLERANCE`, which is set from those numbers.
MAX_DEVIATION = DEFAULT_APPROXIMATION_TOLERANCE


def backends() -> list[Backend]:
    return [
        Backend("apbs", ApbsSolver(), SolverFamily.FINITE_DIFFERENCE),
        Backend("gb", GbSolver(), SolverFamily.ANALYTIC),
    ]


def system_for(case: Case, surface: SurfaceModel = SurfaceModel.MOLECULAR) -> System:
    """A corpus case as a `System`, on the one surface both tiers can answer.

    The corpus runs on `smoothed-molecular`, which is APBS-only — the same
    reason `sashimi validate` has to choose a model rather than take the
    default.
    """
    return System(
        structure=case.structure(),
        solvent=dataclasses.replace(case.solvent, surface_model=surface),
        grid=case.grid,
        want_energy=True,
        want_potential=False,
    )


def peptide() -> Case:
    return next(case for case in MANIFEST if case.name == "peptide-default")


@pytest.mark.parametrize("case", MANIFEST, ids=lambda c: c.name)
def test_the_approximation_lands_where_an_approximation_should(case):
    """Not "GB agrees with APBS" — the point is that it is allowed not to."""
    comparison = validate_system(system_for(case), backends())

    deviation = comparison.approximation_deviation["gb"]
    assert deviation <= MAX_DEVIATION, (
        f"{case.name}: GB is {deviation:.2%} from APBS, beyond the "
        f"{MAX_DEVIATION:.0%} an approximation is allowed. A wrong surface "
        "model costs 31% and the wrong radii 35%; imprecision costs single digits."
    )


def test_the_approximation_does_not_enter_the_reference_spread():
    """The property the tier partition exists for, on real backends."""
    with_gb = validate_system(system_for(peptide()), backends())

    # One reference backend, so there is no spread to report — and the
    # approximation is still measured rather than silently dropped.
    assert with_gb.energy_spread is None
    assert "gb" in with_gb.approximation_deviation
    assert [r.name for r in with_gb.runs if r.accuracy_tier is AccuracyTier.REFERENCE] == ["apbs"]


def test_gb_is_faster_than_the_solver_it_triages_for():
    """The whole argument for the tier: seconds instead of a minute.

    Asserted as an ordering rather than a ratio — 35x on hen lysozyme, but a
    dipeptide is dominated by process startup on both sides and CI machines vary.
    """
    system = system_for(peptide())

    fd = system.request_for(SolverFamily.FINITE_DIFFERENCE)
    # `request_for` is typed to the base request, so the FD backend needs the
    # narrowing that `validate_system` does by construction. GB needs none.
    assert isinstance(fd, FiniteDifferenceRequest)

    gb = GbSolver().solve(system.request_for(SolverFamily.ANALYTIC))
    apbs = ApbsSolver().solve(fd)

    assert gb.provenance.wall_seconds is not None
    assert apbs.provenance.wall_seconds is not None
    assert gb.provenance.wall_seconds < apbs.provenance.wall_seconds


def test_a_mismatched_surface_is_refused_rather_than_approximated():
    """Being an approximation buys latitude on accuracy, not on the question.

    `smoothed-molecular` is sashimi's default and APBS-only. GB declines it at
    the door — the same refusal DelPhi makes — rather than substituting the
    molecular surface and reporting a number that is 2,000 times corpus
    tolerance away from what was asked for.
    """
    system = system_for(peptide(), SurfaceModel.SMOOTHED_MOLECULAR)

    with pytest.raises(UnsupportedRequest, match="molecular surface"):
        validate_system(system, backends())


def test_gb_answers_the_base_request_either_family_produces():
    """A third family, and the protocol needed no new request type."""
    system = system_for(peptide())

    request = system.request_for(SolverFamily.ANALYTIC)
    assert type(request) is SolveRequest  # not FD, not BEM: the base

    result = GbSolver().solve(request)
    assert result.energy_kj_mol is not None
    # ...and it also accepts an FD request, because `Solver` is contravariant.
    fd = system.request_for(SolverFamily.FINITE_DIFFERENCE)
    assert isinstance(fd, FiniteDifferenceRequest)
    assert GbSolver().solve(fd).energy_kj_mol == pytest.approx(result.energy_kj_mol)
