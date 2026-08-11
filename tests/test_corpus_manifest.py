"""The manifest and its recorded summaries, without running a solver.

`tests/test_corpus.py` needs APBS because it re-solves. Everything here reads
what is already checked in, which makes it the part of the corpus that runs on
every machine — including the assertions that the recorded numbers obey physics
rather than merely obeying themselves.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sashimi.corpus import (
    MANIFEST,
    TIER_ORDER,
    AnalyticReference,
    Case,
    CaseTier,
    build_case,
    cases_for_tier,
    load_summary,
    verify_case,
)
from sashimi.protocol import (
    FiniteDifferenceRequest,
    PotentialGrid,
    Provenance,
    SolveResult,
)


def recorded(name: str) -> dict[str, Any]:
    case = next(c for c in MANIFEST if c.name == name)
    return load_summary(case)


# --- tiers -------------------------------------------------------------------


def test_tiers_are_cumulative():
    counts = [len(cases_for_tier(tier)) for tier in TIER_ORDER]
    assert counts == sorted(counts)
    assert len(cases_for_tier(CaseTier.FULL)) == len(MANIFEST)


def test_every_case_declares_a_tier_that_exists():
    assert all(case.tier in TIER_ORDER for case in MANIFEST)


def test_the_fast_tier_is_not_empty():
    """Whatever else is true, some corpus runs on every push."""
    assert cases_for_tier(CaseTier.FAST)


# --- the recorded numbers, against physics rather than against themselves ----


@pytest.mark.parametrize(
    "case", [c for c in MANIFEST if c.analytic is not None], ids=lambda c: c.name
)
def test_recorded_energies_sit_within_their_analytic_tolerance(case: Case):
    """The check that would fail if a backend had been wrong since the first build.

    Every other corpus assertion compares a solver against its own recording, so
    a number that was wrong on day one stays wrong forever and passes. This one
    compares the recording against the closed form.
    """
    assert case.analytic is not None
    summary = load_summary(case)
    assert summary["analytic"]["relative_error"] <= case.analytic.rtol


def test_solvation_energy_does_not_depend_on_the_sign_of_the_charge():
    """dG goes as q^2, so -1e must reproduce +1e exactly.

    A sign mishandled anywhere between the PQR reader and the energy — in charge
    parsing, in the source term, in the integration — breaks this and nothing
    else in the corpus, because every other case is positively charged.
    """
    positive = recorded("born-ion-coarse")["energy_kj_mol"]
    negative = recorded("born-ion-negative")["energy_kj_mol"]

    assert negative == pytest.approx(positive, rel=1e-9)


def test_solvation_energy_scales_as_the_square_of_the_charge():
    """+2e on the same sphere is exactly 4x +1e, in the recording and in theory."""
    monovalent = recorded("born-ion-coarse")["energy_kj_mol"]
    divalent = recorded("born-ion-divalent")["energy_kj_mol"]

    assert divalent == pytest.approx(4.0 * monovalent, rel=1e-6)


def test_refining_the_grid_moves_the_answer_toward_the_closed_form():
    """The convergence pairs are the corpus's only claim about correctness."""
    for coarse, fine in (
        ("born-ion-coarse", "born-ion-fine"),
        ("born-ion-r1-coarse", "born-ion-r1-fine"),
    ):
        assert (
            recorded(fine)["analytic"]["relative_error"]
            < (recorded(coarse)["analytic"]["relative_error"])
        ), f"{fine} should be closer to exact than {coarse}"


def test_analytic_references_are_computed_rather_than_quoted():
    """The +2e reference is 4x the +1e one to the last bit, because both came
    from the same expression rather than from a table someone typed."""
    monovalent = next(c for c in MANIFEST if c.name == "born-ion-coarse").analytic
    divalent = next(c for c in MANIFEST if c.name == "born-ion-divalent").analytic
    assert monovalent is not None
    assert divalent is not None

    assert divalent.energy_kj_mol == pytest.approx(4.0 * monovalent.energy_kj_mol, rel=1e-15)


def test_the_salted_case_declines_an_analytic_reference():
    """Deliberate: the two backends disagree 39% on the mobile-ion term, so
    pinning either convention as "the" closed form would encode a choice as physics."""
    assert next(c for c in MANIFEST if c.name == "born-ion-salt").analytic is None


# --- the analytic check catches a wrong answer -------------------------------


class WrongSolver:
    """Returns a plausible grid and an energy off by a factor the tolerance rejects."""

    def __init__(self, scale: float = 1.5):
        self.scale = scale

    def solve(self, request: FiniteDifferenceRequest) -> SolveResult:  # noqa: ARG002 — the protocol's signature
        potential = PotentialGrid(
            values=np.zeros((9, 9, 9)),
            origin=np.zeros(3),
            spacing=np.full(3, 0.5),
        )
        exact = -228.61080098772135
        return SolveResult(
            provenance=Provenance(backend="wrong"),
            energy_kj_mol=exact * self.scale,
            potential=potential,
        )


def test_a_backend_that_is_wrong_from_the_first_build_is_caught():
    """The failure mode a self-recorded corpus cannot see, now visible.

    The reference here is the wrong solver's *own* summary, so every
    recording-based check agrees with itself. Only the closed form disagrees.
    """
    case = next(c for c in MANIFEST if c.name == "born-ion-coarse")
    solver = WrongSolver()
    self_consistent = build_case(solver, case)

    found = verify_case(solver, case, self_consistent)

    assert [d.field for d in found] == ["analytic.energy_kj_mol"]
    assert "from the closed form" in found[0].detail


def test_a_case_without_a_closed_form_is_not_checked_against_one():
    case = next(c for c in MANIFEST if c.name == "born-ion-salt")
    solver = WrongSolver()

    assert verify_case(solver, case, build_case(solver, case)) == []


def test_an_analytic_reference_records_how_it_was_derived():
    """A bare number in a summary is unauditable; this says where it came from."""
    reference = next(c for c in MANIFEST if c.analytic is not None).analytic
    assert isinstance(reference, AnalyticReference)
    assert reference.source
