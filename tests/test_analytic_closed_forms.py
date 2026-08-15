"""The closed forms against each other, with no solver in the room.

`tests/test_analytic.py` checks `sashimi.analytic` against APBS and is marked
accordingly, so until M3 nothing exercised these expressions where they can be
checked *internally* — and internal checks are what catch a transcription error,
because a solver that disagrees by 2% could be either party's fault.

It also holds the one pin nothing had: `debye_length_a` feeds both salted
expressions *and* debye's Boltzmann term *and* its boundary data, so every check
that grades the solver against the closed form shares it and cannot see it being
wrong.

The identity that matters is between the two salted expressions.
`screened_born_solvation_energy` and `screened_born_potential` are two
consequences of one pair of matching conditions at the Stern radius, derived
independently in their own docstrings, and they are related exactly:

    dG_ion = 1/2 q phi_shift

where `phi_shift` is the constant by which the screened potential sits below the
unscreened one inside the Stern layer. Getting one right and the other wrong is
the failure this file exists to make impossible.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from sashimi.analytic import (
    born_potential,
    born_solvation_energy,
    debye_length_a,
    screened_born_potential,
    screened_born_solvation_energy,
)
from sashimi.constants import AVOGADRO, BOLTZMANN, JOULES_PER_KJ
from sashimi.gb.energy import debye_kappa
from sashimi.protocol import SolventModel

RADIUS = 3.0
CHARGE = 1.0
EPS_S = 78.54
EPS_P = 1.0
TEMPERATURE = 298.15
ION_RADIUS = 2.0

SALTS = (0.05, 0.15, 0.5, 1.0)


def kt_per_mol_kj(temperature: float) -> float:
    """kJ/mol per kT, for turning a potential in kT/e into an energy."""
    return BOLTZMANN * temperature * AVOGADRO / JOULES_PER_KJ


def test_the_debye_length_is_the_textbook_value():
    """7.86 A at 0.15 M, the number `debye_length_a`'s own docstring names.

    **Nothing asserted this until M3's review asked, and by then everything
    depended on it.** kappa flows from this one function into both salted closed
    forms, `debye/dielectric.py`'s Boltzmann coefficient and `debye/sources.py`'s
    Dirichlet data — so a wrong kappa moves the solver and its reference by the
    same factor and every no-binary check in this file stays green. That is the
    common-mode failure the guards file keeps finding: a verdict computed as a
    ratio or a difference hides an error in the term both sides share.

    Only `test_the_ionic_term_agrees_with_a_backend_that_computes_kappa_itself`
    would have caught it, and that one needs APBS. Two pins here, both cheap:
    the textbook value, and agreement with `sashimi.gb`'s independent
    implementation of the same formula — which existed, and was pinned, the
    whole time.
    """
    assert debye_length_a(0.15, EPS_S, TEMPERATURE) == pytest.approx(7.86, abs=0.02)
    assert debye_length_a(0.0) == math.inf

    # A second implementation of kappa^2 = 2 N_A e^2 I / (eps0 eps_s k T) has
    # shipped in `sashimi.gb` since phase 7. Two copies is a duplication worth
    # noting; until one goes, their agreement is a free cross-check.
    for ionic_strength in SALTS:
        gb_kappa = debye_kappa(
            SolventModel(
                ionic_strength=ionic_strength,
                solvent_dielectric=EPS_S,
                temperature=TEMPERATURE,
            )
        )
        assert 1.0 / gb_kappa == pytest.approx(
            debye_length_a(ionic_strength, EPS_S, TEMPERATURE), rel=1e-12
        )


@pytest.mark.parametrize("ionic_strength", SALTS)
def test_the_two_salted_forms_are_the_same_matching_condition(ionic_strength):
    """The energy is half the charge times the potential shift it causes.

    Inside the Stern layer `screened_born_potential` is the unscreened Born
    potential minus a constant. That constant is the mobile ions' contribution
    to the reaction potential at the charge, so half of q times it is their
    contribution to the solvation energy — which is what
    `screened_born_solvation_energy` adds to the Born term. Nothing here shares
    an expression with the thing it checks: one was derived by matching at the
    Stern radius, the other by the Debye-Huckel energy integral.
    """
    inside = RADIUS + ION_RADIUS / 2.0  # strictly within the Stern layer
    shift_kt_e = screened_born_potential(
        inside,
        CHARGE,
        EPS_S,
        TEMPERATURE,
        radius_a=RADIUS,
        ionic_strength=ionic_strength,
        ion_radius=ION_RADIUS,
    ) - born_potential(inside, CHARGE, EPS_S, TEMPERATURE)

    from_potential = 0.5 * CHARGE * shift_kt_e * kt_per_mol_kj(TEMPERATURE)
    from_energy = screened_born_solvation_energy(
        RADIUS,
        CHARGE,
        EPS_P,
        EPS_S,
        ionic_strength=ionic_strength,
        temperature=TEMPERATURE,
        ion_radius=ION_RADIUS,
    ) - born_solvation_energy(RADIUS, CHARGE, EPS_P, EPS_S)

    assert from_potential == pytest.approx(from_energy, rel=1e-12)


@pytest.mark.parametrize("ionic_strength", SALTS)
def test_the_shift_is_constant_across_the_whole_stern_layer(ionic_strength):
    """It is a constant, not a slowly varying function that looks like one.

    The inner branch is `1/r` plus a constant, and the identity above samples it
    at exactly one radius — so a shift that drifted with r would satisfy that
    test at the sampled point and be wrong everywhere else.
    """
    shifts = [
        screened_born_potential(
            RADIUS + fraction * ION_RADIUS,
            CHARGE,
            EPS_S,
            TEMPERATURE,
            radius_a=RADIUS,
            ionic_strength=ionic_strength,
            ion_radius=ION_RADIUS,
        )
        - born_potential(RADIUS + fraction * ION_RADIUS, CHARGE, EPS_S, TEMPERATURE)
        for fraction in (0.01, 0.25, 0.5, 0.75, 1.0)
    ]
    assert shifts[0] < 0.0  # screening lowers the potential
    for shift in shifts[1:]:
        assert shift == pytest.approx(shifts[0], rel=1e-12)


@pytest.mark.parametrize("ionic_strength", SALTS)
def test_the_branches_meet_at_the_stern_radius(ionic_strength):
    """Continuous in value and in slope, which is what makes a sample there legal.

    eps is the same on both sides of the Stern radius, so only the second
    derivative jumps. `sashimi.corpus.AnalyticField` relies on this: a sample may
    land on `a + ion_radius` where one on the *dielectric* boundary is O(1)
    wrong for every solver.
    """
    exclusion = RADIUS + ION_RADIUS

    def phi(r):
        return screened_born_potential(
            r,
            CHARGE,
            EPS_S,
            TEMPERATURE,
            radius_a=RADIUS,
            ionic_strength=ionic_strength,
            ion_radius=ION_RADIUS,
        )

    step = 1e-6
    # The residual here is the O(step) variation of a continuous function across
    # 2e-6 A, not a jump: phi' is about -0.1 kT/e per A, so 5.7e-7 relative is
    # what continuity looks like at this step. Measured, for scale: dropping the
    # flux condition and matching phi alone — which is the plausible way to get
    # this wrong, since it loses exactly the 1/(1 + kappa b) factor — puts a
    # **63.6%** step at the Stern radius at 0.15 M.
    assert phi(exclusion - step) == pytest.approx(phi(exclusion + step), rel=1e-5)
    inner_slope = (phi(exclusion - step) - phi(exclusion - 2 * step)) / step
    outer_slope = (phi(exclusion + 2 * step) - phi(exclusion + step)) / step
    assert inner_slope == pytest.approx(outer_slope, rel=1e-4)


def test_zero_salt_is_the_unscreened_expression_exactly():
    """Not approximately: the corpus's ten pre-existing field cases depend on it.

    `AnalyticField` stopped refusing salted cases by calling the screened form
    unconditionally, which is only safe if it reduces to the old expression bit
    for bit at zero ionic strength.
    """
    for r in (3.5, 5.0, 7.0, 12.0):
        assert screened_born_potential(
            r, CHARGE, EPS_S, TEMPERATURE, radius_a=RADIUS, ionic_strength=0.0
        ) == born_potential(r, CHARGE, EPS_S, TEMPERATURE)


def test_the_far_field_decays_as_a_screened_coulomb_tail():
    """Beyond the Stern radius the log of r*phi is linear in r with slope -kappa.

    What this establishes is that the outer branch uses kappa *consistently* —
    the decay constant is the same quantity the matching condition's
    `1 + kappa b` is built from. It says nothing about whether kappa is right,
    because the expected slope here comes from `debye_length_a` too;
    `test_the_debye_length_is_the_textbook_value` is what pins that, and it has
    to be a separate test for exactly that reason.
    """
    ionic_strength = 0.15
    kappa = 1.0 / debye_length_a(ionic_strength, EPS_S, TEMPERATURE)
    radii = (6.0, 8.0, 10.0, 14.0)
    logs = [
        math.log(
            r
            * screened_born_potential(
                r,
                CHARGE,
                EPS_S,
                TEMPERATURE,
                radius_a=RADIUS,
                ionic_strength=ionic_strength,
                ion_radius=ION_RADIUS,
            )
        )
        for r in radii
    ]
    for (r0, l0), (r1, l1) in pairwise(zip(radii, logs, strict=True)):
        assert (l1 - l0) / (r1 - r0) == pytest.approx(-kappa, rel=1e-12)


def test_stronger_salt_screens_harder_at_every_radius():
    """Monotone in ionic strength, which no single-salt check can state."""
    for r in (4.0, 5.0, 8.0):
        values = [
            screened_born_potential(
                r,
                CHARGE,
                EPS_S,
                TEMPERATURE,
                radius_a=RADIUS,
                ionic_strength=ionic_strength,
                ion_radius=ION_RADIUS,
            )
            for ionic_strength in (0.0, *SALTS)
        ]
        assert values == sorted(values, reverse=True), (
            f"the potential at r = {r} A does not fall monotonically with salt: {values}"
        )
