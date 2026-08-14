"""M2: debye with the charge off centre, graded against the Kirkwood series.

ROADMAP.md section 12 M2. Born is the d = 0 special case; the rest of the series
is where a solver's charge assignment and its boundary handling stop being
independent, because the charge approaches the interface.

**Why these cases had to be added before M2 could be attempted at all.** Every
Kirkwood rung in the corpus was on `smoothed-molecular` or `molecular`, and
debye's `SUPPORTED_SURFACES` is `van-der-waals` alone — so M2's exit criterion
named cases debye refuses by name. M0 had dropped the van der Waals Kirkwood
deliberately ("another sphere geometry re-measures what the existing rungs
already measure"), which is true for the two backends that build both surfaces
and false for the one M2 exists to grade. That is the same shape as the
`smoothed-molecular` gap M0 itself was created to close, one surface along, and
it is worth naming as a *class*: **a case added for coverage of the incumbents
is not automatically coverage of the candidate.**

**The bar is 1.5% at every gated rung, decided 2026-08-14 by Charlie**, and it is
deliberately not the shared per-case tolerance. debye reproduces APBS's
discretization, and APBS is what sets the shared number, so grading debye there
is a bar it meets by construction — section 7's check that cannot fail. 1.5% is
set independently of what debye does: it would have failed had debye read 1.6% at
d/a = 0.7, and it is *stricter than APBS manages there* (3.896%).
"""

from __future__ import annotations

import dataclasses
from itertools import pairwise

import pytest

from sashimi.analytic import kirkwood_solvation_energy
from sashimi.corpus import MANIFEST, Case
from sashimi.debye import DebyeSolver

# The gated rungs. d/a = 0.9 is recorded and not gated, for the reason its
# `description` gives: no shipped solver reproduces it.
GATED_RUNGS = ("kirkwood-vdw-03", "kirkwood-vdw-05", "kirkwood-vdw-07")

M2_BAR = 0.015


def case_named(name: str) -> Case:
    return next(case for case in MANIFEST if case.name == name)


@pytest.mark.parametrize("name", GATED_RUNGS)
def test_debye_reproduces_kirkwood_within_the_bar_m2_sets(name):
    """M2's exit criterion. Needs no binary — the reference is a closed form."""
    case = case_named(name)
    reference = case.analytic
    assert reference is not None
    tolerance = reference.rtol_for(DebyeSolver().label)

    result = DebyeSolver().solve(case.request())
    assert result.energy_kj_mol is not None
    error = abs(result.energy_kj_mol - reference.energy_kj_mol) / abs(reference.energy_kj_mol)

    assert error <= tolerance, (
        f"{name}: debye is {error:.4%} from the Kirkwood closed form, past the "
        f"{tolerance:.2%} ROADMAP.md section 12 M2 holds it to."
    )


@pytest.mark.parametrize("name", GATED_RUNGS)
def test_the_m2_bar_is_tighter_than_the_shared_tolerance(name):
    """The bar has to be debye's, not the one APBS's error sets.

    Same guard M1 carries, and it is the reason M2 is a claim: at d/a = 0.7 the
    shared tolerance is 8% because APBS reads 3.896% there. A solver three
    percent from the closed form would clear that while failing the milestone's
    stated criterion. Needs no binary.
    """
    reference = case_named(name).analytic
    assert reference is not None
    tolerance = reference.rtol_for(DebyeSolver().label)

    assert tolerance == M2_BAR
    assert tolerance < reference.rtol, (
        f"{name} grades debye at the shared tolerance {reference.rtol}, which is set by "
        "the least accurate backend that runs it. Give it a `debye_rtol`."
    )


def test_the_m2_bar_is_stricter_than_apbs_manages_at_the_hardest_rung():
    """What stops 1.5% being a number chosen to be met.

    `kirkwood-vdw-07`'s shared tolerance is 0.08 because APBS measures 3.896%
    there. debye is held to 1.5% on the same case, so the bar is not "whatever
    the incumbents do" — it is stricter than one of them, on the rung where they
    diverge most. If this ever inverts, the bar has stopped being a claim.
    """
    reference = case_named("kirkwood-vdw-07").analytic
    assert reference is not None
    assert reference.rtol_for("debye-0.1") < reference.rtol_for("apbs-3.4.1")
    assert reference.rtol_for("debye-0.1") < 0.03896, (
        "the bar must sit below APBS's measured error on this rung, or M2 asks "
        "less of debye than APBS already delivers"
    )


def test_the_ninth_rung_is_recorded_and_not_judged():
    """d/a = 0.9 stays ungated, and its tolerance stays unmeetable.

    Two halves, and the second is the one that rots: a case flagged "do not
    judge" whose tolerance is set to what the codes happen to achieve goes
    *vacuously green* the moment someone ungates it. Its molecular twin records
    that trap; this asserts the vdW copy did not reintroduce it.
    """
    reference = case_named("kirkwood-vdw-09").analytic
    assert reference is not None
    assert not reference.gated
    # Measured: APBS 9.854%, DelPhi 4.288%, debye 8.280%. The tolerance is below
    # all three, so ungating reddens rather than passing.
    assert reference.rtol < 0.04288


def test_the_near_boundary_charge_wobbles_for_every_backend_not_just_debye():
    """The M2 finding that is recorded rather than gated.

    M1 required the Born error to fall at every refinement step. That cannot be
    asked of Kirkwood at d/a >= 0.5 on a sharp boundary, because **no shipped
    solver manages it** — measured across the ladder 1.0 / 0.5 / 0.35 / 0.25 /
    0.2 A at d/a = 0.7:

        APBS       -38.311  -2.136  -2.713  -3.896  -0.381
        DelPhi C++  -4.718  -0.493  -1.093  -0.416  -0.371
        debye        0.726 -28.171  -6.211  -1.328  -1.893

    So requiring monotonicity here would be the mirror of a check that cannot
    fail — a check that cannot pass — and ROADMAP.md section 12 already made the
    same call at d/a = 0.9. Recorded instead, and pinned here so that a future
    solver change which *does* make debye monotonic is noticed rather than
    silently absorbed.

    debye alone, so this needs no binary; the incumbents' rows above are the
    control that made "record, do not gate" the right call rather than a
    convenient one.
    """
    case = case_named("kirkwood-vdw-07")
    base = case.request()
    exact = kirkwood_solvation_energy(3.0, 3.0 * 0.7, 1.0, 1.0, 78.54)

    errors = []
    for resolution in (0.5, 0.35, 0.25, 0.2):
        request = dataclasses.replace(
            base, grid=dataclasses.replace(base.grid, resolution=resolution)
        )
        energy = DebyeSolver().solve(request).energy_kj_mol
        assert energy is not None
        errors.append((energy - exact) / abs(exact))

    falls_every_step = all(abs(a) >= abs(b) for a, b in pairwise(errors))
    assert not falls_every_step, (
        "debye is now monotonic on the d/a = 0.7 rung, where it and both "
        f"incumbents were not: {[f'{e:.3%}' for e in errors]}. That is a real "
        "improvement — revisit M2's 'record, do not gate' call rather than "
        "deleting this test."
    )
