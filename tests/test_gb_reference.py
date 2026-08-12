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
from sashimi.gb import GbOptions, GbSolver
from sashimi.gb.options import GbRadii
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

    Goes through `Case.system()` rather than rebuilding one: this file used to
    assemble the fields by hand, which is a second copy of a construction that
    silently stops matching the moment `Case` grows a field — `mesh_density`
    already did. Overriding the surface stays here, because choosing a shared
    model is the *comparison's* business and not the case's; the corpus runs on
    `smoothed-molecular`, which is APBS-only.
    """
    return dataclasses.replace(
        case.system(want_potential=False),
        solvent=dataclasses.replace(case.solvent, surface_model=surface),
    )


def peptide() -> Case:
    return next(case for case in MANIFEST if case.name == "peptide-default")


# GB is handed the same structure as APBS but not the same radii: it substitutes
# mbondi, which is what the method was parameterized with. That is a small
# correction when the input is an AMBER Lennard-Jones set — what pdb2pqr emits,
# and so what sashimi's own prep produces — and a large one when the PQR already
# carries a considered radius set from somewhere else. Measured against APBS:
#
#   case                 mbondi   as-given   input radii
#   barnase               1.65%     55.68%   AMBER-like
#   protein-rna           3.89%    115.72%   AMBER-like (as-given flips the sign)
#   lysozyme (2LZT)      13.45%      8.62%   PARSE-like
#   carbonic-anhydrase   21.13%      7.34%   PARSE-like
#   methanol             28.21%     19.15%   3 atoms; one radius moves 0.2 -> 1.2 A
#
# So neither setting is universally right, mbondi is right for the inputs sashimi
# itself produces, and what this file gates is approximation error — which needs
# the two solvers to have been given comparable solutes.
LIKE_FOR_LIKE = ("peptide-default", "acetic-acid", "acetate", "fas2", "barstar")

# Where they were not. The gap here is the radius set, not the method, and the
# test below demonstrates that rather than asserting it.
FOREIGN_RADIUS_SET = ("methanol", "lysozyme")


def case_named(name: str) -> Case:
    return next(c for c in MANIFEST if c.name == name)


@pytest.mark.parametrize("case", [case_named(n) for n in LIKE_FOR_LIKE], ids=lambda c: c.name)
def test_the_approximation_lands_where_an_approximation_should(case):
    """Not "GB agrees with APBS" — the point is that it is allowed not to."""
    comparison = validate_system(system_for(case), backends())

    deviation = comparison.approximation_deviation["gb"]
    assert deviation <= MAX_DEVIATION, (
        f"{case.name}: GB is {deviation:.2%} from APBS, beyond the "
        f"{MAX_DEVIATION:.0%} an approximation is allowed. A wrong surface "
        "model costs 31% and the wrong radii 35%; imprecision costs single digits."
    )


@pytest.mark.parametrize("name", FOREIGN_RADIUS_SET)
def test_a_foreign_radius_set_shows_up_as_deviation_it_did_not_cause(name: str):
    """Demonstrates the cause rather than pinning the number.

    These structures arrive with radii from a set mbondi is not, so GB's
    substitution has it solving a measurably different solute from the one APBS
    was given. If that is really what the deviation is, handing GB the
    structure's own radii must shrink it — and it does. The reverse holds for
    AMBER-like inputs, which is why `MBONDI` remains the default: on those,
    `AS_GIVEN` reaches 55% and can return a positive solvation energy.
    """
    case = case_named(name)
    structure = case.structure()
    solvent = dataclasses.replace(case.solvent, surface_model=SurfaceModel.MOLECULAR)
    request = SolveRequest(structure=structure, solvent=solvent, want_potential=False)

    reference = ApbsSolver().solve(
        FiniteDifferenceRequest(
            structure=structure, solvent=solvent, grid=case.grid, want_potential=False
        )
    )
    assert reference.energy_kj_mol is not None
    exact = reference.energy_kj_mol

    def deviation(radii: GbRadii) -> float:
        energy = GbSolver(GbOptions(radii=radii)).solve(request).energy_kj_mol
        assert energy is not None
        return abs(energy - exact) / abs(exact)

    assert deviation(GbRadii.AS_GIVEN) < deviation(GbRadii.MBONDI)


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
