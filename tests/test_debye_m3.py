"""M3: debye in salt, graded on how the energy moves with ionic strength.

ROADMAP.md section 12 M3. The milestone's criterion is deliberately a
*relationship between two recordings* rather than a number either has to hit,
and this file is where that phrasing stops being a stylistic preference and
becomes the only thing that works.

**Why the total energy cannot be the gate, measured rather than argued.** On the
van der Waals Born ion the ionic contribution is 0.3% of the solvation energy
where discretization is 1.6%. Mutating debye's screening four ways — deleting
the Boltzmann term outright, moving the Stern layer to the solvent probe,
removing it, using kappa where kappa^2 belongs — moves the total across a range
of -1.40% to -1.76% against the closed form, while APBS on the same case needs
2.4%. So a tolerance loose enough for the incumbents cannot tell a solver that
models salt from one that ignores it. The same four mutations move
`G(I) - G(0)` by -28.8%, +4.9%, +18.2% and +63.4% against debye's +0.10%.

**The bar is 2%, decided 2026-08-14 by Charlie**, on both halves of the gate. It
is twenty times what debye reads and it still sits below the *subtlest* of those
mutations — a Stern layer at the probe radius, 4.86% — which is what
`test_the_bar_rejects_a_stern_layer_at_the_probe_radius` demonstrates by running
it rather than by citing it.

**Two halves, because the closed form shares a definition with debye.**
`screening_nodes` takes kappa from `sashimi.analytic.debye_length_a`, which is
also what `screened_born_solvation_energy` uses, so grading debye against that
expression cannot catch a wrong kappa — ROADMAP.md section 12 flagged exactly
this at M1. APBS computes kappa inside its own C and lands on the same number,
so the cross-backend half is the one that closes it. Both are held to 2%.

**What is not gated, and why.** Once the solute is net neutral the ionic term is
dipole screening rather than monopole screening, an order of magnitude smaller,
and the three reference-tier codes spread over 22% with no closed form to
referee them. That is recorded in `peptide-vdw-high-salt`'s description and in
the last test here, in the same shape M2 used for its non-monotonic rungs.
"""

from __future__ import annotations

import dataclasses
from itertools import pairwise

import numpy as np
import pytest

from sashimi.analytic import screened_born_solvation_energy
from sashimi.apbs import ApbsSolver
from sashimi.corpus import MANIFEST, Case
from sashimi.debye import DebyeSolver
from sashimi.field import sample_values
from sashimi.pqr import read_pqr
from sashimi.protocol import GridSpec, PotentialGrid, SolventModel, SurfaceModel

# The arm: one zero-salt case and two salted ones on the surface debye builds.
ZERO_SALT = "born-ion-vdw"
SALTED = ("born-ion-vdw-salt", "born-ion-vdw-high-salt")

M3_BAR = 0.02

# What a Stern layer at the solvent probe rather than the ion radius reads, at
# 0.15 M. The bar has to sit below this or it grades nothing about the
# convention `screening_nodes`'s docstring goes out of its way to name.
PROBE_STERN_LAYER_ERROR = 0.0486


def case_named(name: str) -> Case:
    return next(case for case in MANIFEST if case.name == name)


def solve_energy(case: Case, **solvent_overrides) -> float:
    """The case's solvation energy from debye, with the solvent optionally varied."""
    request = case.request()
    if solvent_overrides:
        request = dataclasses.replace(
            request, solvent=dataclasses.replace(request.solvent, **solvent_overrides)
        )
    energy = DebyeSolver().solve(request).energy_kj_mol
    assert energy is not None
    return energy


def ionic_contribution(name: str, **solvent_overrides) -> float:
    """G(I) - G(0) for a salted case, both states on that case's own grid."""
    case = case_named(name)
    salted = solve_energy(case, **solvent_overrides)
    unsalted = solve_energy(case, ionic_strength=0.0, **solvent_overrides)
    return salted - unsalted


def exact_ionic_contribution(case: Case) -> float:
    """The same difference from the closed form, taking the case's own solvent.

    Deliberately has **no** `ion_radius` override, though the mutation test below
    wants one on the solver side. A helper that let the reference be moved to
    match a mutated solver is a ready-made way to grade a mutation against its
    own physics, which is the thing that test exists to avoid — so the two sides
    are asymmetric on purpose.
    """
    solvent = case.solvent
    kwargs = {
        "solute_dielectric": solvent.solute_dielectric,
        "solvent_dielectric": solvent.solvent_dielectric,
        "temperature": solvent.temperature,
        "ion_radius": solvent.ion_radius,
    }
    radius = float(case.structure().radii[0])
    charge = float(case.structure().charges[0])
    salted = screened_born_solvation_energy(
        radius, charge, ionic_strength=solvent.ionic_strength, **kwargs
    )
    unsalted = screened_born_solvation_energy(radius, charge, ionic_strength=0.0, **kwargs)
    return salted - unsalted


@pytest.mark.parametrize("name", SALTED)
def test_debye_reproduces_the_screened_born_ionic_term(name):
    """M3's exit criterion. Needs no binary — the reference is a closed form.

    Measured: +0.10% at 0.15 M and +0.14% at 0.5 M, against a 2% bar.
    """
    case = case_named(name)
    got = ionic_contribution(name)
    want = exact_ionic_contribution(case)
    error = (got - want) / abs(want)

    assert abs(error) <= M3_BAR, (
        f"{name}: debye's ionic contribution is {got:.4f} kJ/mol against a "
        f"closed-form {want:.4f}, {error:.3%} out and past the {M3_BAR:.1%} "
        "ROADMAP.md section 12 M3 holds it to."
    )


def test_the_bar_rejects_a_stern_layer_at_the_probe_radius():
    """The mutation that says the bar grades something, run rather than cited.

    `screening_nodes` excludes mobile ions inside `radius + ion_radius` and its
    docstring says, in as many words, that this is *not* `radius + probe`. That
    distinction is worth exactly 4.86% at 0.15 M, so a bar above it would leave
    the most plausible convention error in this module undetected — and every
    total-energy check in the corpus already does, at 1.59% against 2.4%.

    Held as a live solve rather than as the recorded constant, because a
    constant in a comment is what drifted between M1b's gate and its
    justification. Needs no binary.
    """
    case = case_named("born-ion-vdw-salt")
    got = ionic_contribution(case.name, ion_radius=1.4)
    # Judged against the *shipped* convention's closed form: a mutation compared
    # against its own physics is a mutation that grades itself.
    want = exact_ionic_contribution(case)
    error = abs(got - want) / abs(want)

    assert error == pytest.approx(PROBE_STERN_LAYER_ERROR, abs=0.005), (
        f"a probe-radius Stern layer now reads {error:.3%}, not the "
        f"{PROBE_STERN_LAYER_ERROR:.2%} the M3 bar was set against"
    )
    assert error > M3_BAR, (
        f"the {M3_BAR:.1%} bar admits a Stern layer at the probe radius "
        f"({error:.3%}), so it does not grade the ion-exclusion convention"
    )


def test_a_total_energy_check_could_not_see_the_salt():
    """Why M3 gates a difference, and not the number the corpus records.

    debye's *zero-salt* answer, compared against the *salted* closed form,
    is 1.27% out — well inside the 5% shared tolerance `born-ion-vdw` carries
    because APBS needs 2.4% there. So an `AnalyticReference` on a salted case
    would pass for a solver with no Boltzmann term at all. That is the reason
    the two salted cases carry an `analytic_field` and no `analytic`, and it is
    the measurement behind ROADMAP.md section 7's refusal rather than the
    convention argument, which is a different point about DelPhi.

    Needs no binary.
    """
    case = case_named("born-ion-vdw-salt")
    salt_blind = solve_energy(case, ionic_strength=0.0)
    want = screened_born_solvation_energy(
        3.0,
        1.0,
        case.solvent.solute_dielectric,
        case.solvent.solvent_dielectric,
        ionic_strength=case.solvent.ionic_strength,
        temperature=case.solvent.temperature,
        ion_radius=case.solvent.ion_radius,
    )
    error = abs(salt_blind - want) / abs(want)

    zero_salt = case_named(ZERO_SALT).analytic
    assert zero_salt is not None
    assert error < zero_salt.rtol, (
        "a salt-blind answer no longer passes the shared energy tolerance, so a "
        "total-energy analytic check on a salted case may now be worth having — "
        "revisit ROADMAP.md section 12 M3 rather than deleting this test"
    )


@pytest.mark.apbs
@pytest.mark.parametrize("name", SALTED)
def test_the_ionic_term_agrees_with_a_backend_that_computes_kappa_itself(name):
    """The half the closed form cannot supply: kappa from someone else's code.

    debye takes kappa from `sashimi.analytic.debye_length_a`, which is what the
    closed form uses too, so the test above shares a definition with the thing
    it grades and a wrong kappa would move both together. APBS computes it
    inside its own C from its own constants. Measured: 0.13% at 0.15 M and
    0.22% at 0.5 M.

    No shared lattice is pinned here, unlike M1b's field gate, and that is a
    measurement rather than an oversight: `G(I) - G(0)` moves 0.13% across
    paddings 5 to 30 A and under 1% across spacings 0.5 / 0.35 / 0.25 / 0.2 A,
    both far below this bar. The near field is the quantity grid phase moves by
    5-21x; the ionic term is not.
    """
    case = case_named(name)

    def apbs_energy(ionic: float) -> float:
        request = case.request()
        request = dataclasses.replace(
            request, solvent=dataclasses.replace(request.solvent, ionic_strength=ionic)
        )
        energy = ApbsSolver().solve(request).energy_kj_mol
        assert energy is not None
        return float(energy)

    reference = apbs_energy(case.solvent.ionic_strength) - apbs_energy(0.0)
    got = ionic_contribution(name)
    error = (got - reference) / abs(reference)

    assert abs(error) <= M3_BAR, (
        f"{name}: debye's ionic contribution is {got:.4f} kJ/mol against APBS's "
        f"{reference:.4f}, {error:.3%} apart and past the {M3_BAR:.1%} bar"
    )


def test_screening_deepens_the_solvation_energy_across_the_whole_arm():
    """The direction, on the sphere and on a real solute. Needs no binary.

    Cheap, and the one statement in this file that a sign error cannot survive
    while still landing near a plausible magnitude. Both arms are three points,
    because two points state a difference and three state a trend.
    """
    for arm in (
        (ZERO_SALT, *SALTED),
        ("peptide-vdw-no-salt", "peptide-vdw", "peptide-vdw-high-salt"),
    ):
        energies = [solve_energy(case_named(name)) for name in arm]
        strengths = [case_named(name).solvent.ionic_strength for name in arm]
        assert strengths == sorted(strengths), f"{arm} is not in ionic-strength order"
        assert all(later < earlier for earlier, later in pairwise(energies)), (
            f"solvation does not deepen monotonically with salt across {arm}: "
            f"{[f'{e:.4f}' for e in energies]}"
        )


def test_the_screening_adds_no_discretization_error_of_its_own():
    """Recorded, and gated only where the discretization error actually lives.

    The relative field error grows with salt — debye reads 4.47% two cells out
    at zero salt and 7.61% at 0.5 M — which reads as the screening being harder
    to resolve. It is not: the *absolute* error at that sample is 0.0812,
    0.0805 and 0.0798 kT/e across the arm, on one lattice at one radius, and
    what changed is the potential being divided by. The agreement loosens
    further out, where the screened solution has decayed and the errors are
    a tenth the size, so the claim is held only at the sample nearest the
    boundary.

    Needs no binary. This is the field analogue of the energy gate above, and
    the reason both exist is that M1c's dielectric spike moved one of these
    axes and not the other.
    """
    baseline: float | None = None
    for name in (ZERO_SALT, *SALTED):
        case = case_named(name)
        result = DebyeSolver().solve(case.request())
        assert isinstance(result.potential, PotentialGrid)
        reference = case.analytic_field
        assert reference is not None
        spacing = float(np.mean(result.potential.spacing))
        radius = reference.sample_radii(spacing)[0]  # two cells out
        exact = reference.exact_at([radius], case.solvent)[0]
        centre = case.structure().coords[0]
        values = sample_values(result.potential, centre, radius)
        worst = float(np.max(np.abs(values - exact)))
        if baseline is None:
            baseline = worst
            continue
        assert abs(worst - baseline) / baseline < 0.05, (
            f"{name}: the absolute field error two cells out is {worst:.6f} kT/e "
            f"against {baseline:.6f} at zero salt on the same lattice, so the "
            "Boltzmann term has started contributing discretization error of its own"
        )


def test_the_neutral_solute_is_recorded_and_not_judged():
    """Where the three reference codes stop agreeing, and nobody can referee it.

    Every net-charged solute measured at M3 has debye and APBS within 1.4% on
    the ionic term — the sphere at 0.13%, two spheres 20 A apart at 1.4%,
    acetate at 0.19%. Both net-neutral ones disagree by 8-10%: a +1/-1 pair at
    0.900 and ALA-GLY at 0.922, stable across 0.5 / 0.35 / 0.25 / 0.2 A, so it
    is a convention difference rather than grid noise. DelPhi C++ sits on the
    other side of debye, giving a 22% spread with no closed form in reach.

    So M3 gates the monopole and records this, the same call M2 made for its
    non-monotonic rungs. Pinned here so that a change which *does* bring debye
    onto APBS's neutral-solute convention is noticed rather than absorbed.
    debye alone, so it needs no binary; the incumbents' numbers above are the
    control that made recording the honest option rather than the convenient
    one.
    """
    sphere = ionic_contribution("born-ion-vdw-salt")
    peptide = solve_energy(case_named("peptide-vdw")) - solve_energy(
        case_named("peptide-vdw-no-salt")
    )

    # APBS on the same two, measured 2026-08-14 and quoted rather than solved so
    # this runs on a machine with no binary.
    apbs_sphere, apbs_peptide = -0.6878, -0.2122
    assert abs(sphere / apbs_sphere - 1) < 0.02, "the monopole agreement is what M3 gates"
    assert abs(peptide / apbs_peptide - 1) > 0.05, (
        f"debye's neutral-solute ionic term is now {peptide:.4f} against APBS's "
        f"{apbs_peptide:.4f}, inside the band the charged cases meet. The "
        "conventions have converged — that is a real result, so update "
        "ROADMAP.md section 12 M3 rather than deleting this test."
    )


def test_sharing_the_distance_pass_changes_no_boundary_value():
    """Two states out of one pass must equal two states out of two passes.

    The distance from a boundary node to an atom does not depend on the
    solvent, and it is the expensive half — 1.5 billion pairs on serum albumin,
    which was being computed twice. Sharing it is worth 5.4 s of a 45 s solve
    and is required to change nothing, so this compares against the one-state
    function each state used to call.

    Both states are asked for, and they are not symmetric: the solvated one
    carries screening and an `exp`, the reference has `kappa = 0` and neither.
    A shared loop that leaked one state's screening into the other would pass a
    test that only looked at the solvated field.
    """
    from sashimi.debye.grid import size_grid  # noqa: PLC0415
    from sashimi.debye.sources import (  # noqa: PLC0415
        debye_huckel_boundaries,
        debye_huckel_boundary,
    )

    pqr = read_pqr("tests/data/ala-gly.pqr")
    solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR)
    reference_solvent = dataclasses.replace(
        solvent, solvent_dielectric=solvent.solute_dielectric, ionic_strength=0.0
    )
    grid = size_grid(pqr, GridSpec(resolution=0.5, padding=10.0))

    separately = [
        debye_huckel_boundary(grid, pqr, solvent, homogeneous=False),
        debye_huckel_boundary(grid, pqr, reference_solvent, homogeneous=True),
    ]
    together = debye_huckel_boundaries(grid, pqr, [(solvent, False), (reference_solvent, True)])
    assert np.any(separately[0]), "the solvated boundary is empty, so this proves nothing"
    for alone, shared in zip(separately, together, strict=True):
        assert np.array_equal(alone, shared)
    assert not np.array_equal(separately[0], separately[1]), (
        "the two states produced the same field, so the comparison cannot see them swap"
    )
