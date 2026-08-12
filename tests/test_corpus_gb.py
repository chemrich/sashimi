"""The golden corpus for a backend that needs no binary — including ours.

Every other corpus test needs APBS, so the regression net has always been on
somebody else's compiled code. Generalized Born is *our* code: a few hundred
lines of numpy that we will keep changing, and until now it had unit tests
against closed forms but no recorded answers on real structures.

This runs everywhere, because there is nothing to install. It is the part of the
corpus that cannot skip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sashimi.corpus import MANIFEST, Case, load_summary, verify_case
from sashimi.gb import GbSolver
from sashimi.protocol import SolverFamily, SurfaceModel

GB_DIRECTORY = Path("tests/corpus/gb")

# Generalized Born answers on the molecular surface and nowhere else, so these
# are the corpus cases it can be asked at all. `sashimi.gb.options` records why
# the intuitive `van-der-waals` reading is wrong.
GB_CASES = tuple(c for c in MANIFEST if c.solvent.surface_model is SurfaceModel.MOLECULAR)


@pytest.fixture(scope="module")
def solver():
    return GbSolver()


def test_the_gb_corpus_is_not_empty():
    """A guard on the selection above: a typo here would silently test nothing."""
    assert len(GB_CASES) >= 19


@pytest.mark.parametrize("case", GB_CASES, ids=lambda c: c.name)
def test_gb_reproduces_its_recorded_answer(case: Case, solver):
    """Our own solver against its own recorded numbers, on real structures."""
    recorded = load_summary(case, GB_DIRECTORY)
    assert verify_case(solver, case, recorded, family=SolverFamily.ANALYTIC) == []


@pytest.mark.parametrize("case", GB_CASES, ids=lambda c: c.name)
def test_an_analytic_summary_records_no_field_it_did_not_compute(case: Case):
    """Shape follows the answer, not a schema.

    Generalized Born returns an energy and no field, so its summaries carry no
    geometry, no probes and no surface statistics. A corpus that demanded them
    would have to invent them.
    """
    summary = load_summary(case, GB_DIRECTORY)

    assert summary["family"] == "analytic"
    assert summary["energy_kj_mol"] is not None
    assert "geometry" not in summary
    assert "probes" not in summary
    assert "surface" not in summary


def test_a_changed_gb_answer_is_caught(solver):
    """The regression net doing its job on the code most likely to change."""
    case = next(c for c in GB_CASES if c.name == "peptide-molecular")
    recorded = dict(load_summary(case, GB_DIRECTORY))
    recorded["energy_kj_mol"] = recorded["energy_kj_mol"] * 1.01

    found = verify_case(solver, case, recorded, family=SolverFamily.ANALYTIC)

    assert [d.field for d in found] == ["energy_kj_mol"]


# --- the approximate tier against the reference tier -------------------------
#
# Everything above compares Generalized Born against its own past, which a
# backend that was wrong on day one passes forever — the failure mode
# `AnalyticReference` exists to close for the cases that have a closed form.
# Most of these do not. What they have instead is an APBS recording of the same
# question, sitting in `tests/corpus/`, and the relationship between the two
# files is a fact about the physics rather than about either recording.
#
# It is also the one check that survives `corpus build --force`: rebuilding the
# GB summaries makes every test above agree with the new numbers, and makes
# these fail by however far the method moved.


def deviation(name: str) -> float:
    """How far the approximate tier sits from the reference tier, from the two recordings."""
    case = next(c for c in GB_CASES if c.name == name)
    reference = float(load_summary(case)["energy_kj_mol"])
    approximate = float(load_summary(case, GB_DIRECTORY)["energy_kj_mol"])
    return abs(approximate - reference) / abs(reference)


# Measured on APBS 3.4.1 and `sashimi.gb` at OBC2/mbondi, 2026-08-12. Not a
# tolerance the method is required to meet — a record of what it does, so that a
# change in either tier is visible as a named case rather than as a diff. The
# spread is the point: 0.7% to 28%, and which end a case lands on is not
# predicted by its size.
GB_DEVIATION: dict[str, float] = {
    # Closed-form geometry: the 3.09% is OBC2's offset on a lone sphere, and it
    # is the same number at both solute dielectrics, which is what says the
    # dielectric factor itself is not where the error is.
    "born-ion-molecular": 0.0072,
    "born-ion-molecular-eps2": 0.0081,
    # All-atom AMBER structures, where mbondi is the right radius set.
    "fkbp-apo-molecular": 0.0255,
    "acetate-molecular": 0.0256,
    "fas2-molecular": 0.0258,
    "fkbp-dmso-molecular": 0.0323,
    "aspartate-residue-molecular": 0.0356,
    "ion-protein-complex-molecular": 0.0359,
    "protein-rna-molecular": 0.0389,
    "barstar-molecular": 0.0424,
    "acetic-acid-molecular": 0.0451,
    "peptide-molecular": 0.0590,
    "peptide-molecular-cold": 0.0591,
    "peptide-molecular-no-salt": 0.0595,
    "peptide-molecular-high-salt": 0.0608,
    # Small absolute energies, where a few kJ/mol is a large fraction.
    "methoxide-molecular": 0.1313,
    "methanol-molecular": 0.2821,
    # Polar-hydrogen structures, where substituting an all-atom radius set hands
    # the two solvers measurably different solutes. ROADMAP.md section 7 records
    # the measurement; `test_which_radii_are_right_depends_on_the_structure`
    # below is the controlled version of it.
    "lysozyme-molecular": 0.1345,
    "hca-molecular": 0.2113,
}

# Both sides are checked-in recordings, so each deviation is exact and this band
# only has to absorb float formatting. Anything larger is a real move.
DEVIATION_BAND = 0.0005


def test_every_shared_case_has_a_recorded_deviation():
    """A case added to one tier and not the other is the gap this is meant to close."""
    assert sorted(GB_DEVIATION) == sorted(c.name for c in GB_CASES)


@pytest.mark.parametrize("name", sorted(GB_DEVIATION), ids=lambda n: n)
def test_the_approximation_sits_where_it_was_measured(name: str):
    """The approximate tier against the reference tier, case by case."""
    assert deviation(name) == pytest.approx(GB_DEVIATION[name], abs=DEVIATION_BAND)


def test_both_tiers_agree_which_way_salt_moves_a_solute():
    """Screening, as a grid solver and an analytic one each see it.

    The salt arm was `smoothed-molecular` until now, so it existed only for
    APBS. Neither tier's absolute number matters here: what has to agree is the
    direction, and it is the first physical claim the corpus can make across two
    solver families rather than about one.
    """
    ladder = ("peptide-molecular-no-salt", "peptide-molecular", "peptide-molecular-high-salt")
    for directory in (None, GB_DIRECTORY):
        energies = [
            load_summary(next(c for c in GB_CASES if c.name == name), directory)["energy_kj_mol"]
            for name in ladder
        ]
        assert energies[0] > energies[1] > energies[2], (directory, energies)


def test_both_tiers_agree_that_the_anion_is_the_solvated_half():
    """Methanol against methoxide: an ionization pair both tiers can be handed.

    An order of magnitude apart in both, because one carries a charge and the
    other does not. A solvation model that got this wrong would be useless for
    the pKa-shaped questions these structures come from.
    """
    for directory in (None, GB_DIRECTORY):
        neutral, anion = (
            load_summary(next(c for c in GB_CASES if c.name == name), directory)["energy_kj_mol"]
            for name in ("methanol-molecular", "methoxide-molecular")
        )
        assert anion < 5 * neutral < 0, (directory, neutral, anion)


def test_a_binding_difference_is_not_a_difference_of_approximations():
    """The finding this pair was added to look for, recorded rather than hidden.

    FKBP with and without DMSO is 2.6% and 3.2% from the reference tier — well
    inside anything anyone would call agreement — and the *difference* of those
    two numbers has the wrong sign: APBS pays 6.3 kJ/mol to bury the ligand,
    Generalized Born is handed 8.3. A binding energy is that difference, so an
    absolute error small enough to look harmless is not evidence the tier can be
    used for the quantity most callers actually want.

    Pinned deliberately. If the approximation improves enough to get the sign
    right, this test fails and should be rewritten to say so.
    """

    def difference(directory: Path | None) -> float:
        apo, holo = (
            float(
                load_summary(next(c for c in GB_CASES if c.name == name), directory)[
                    "energy_kj_mol"
                ]
            )
            for name in ("fkbp-apo-molecular", "fkbp-dmso-molecular")
        )
        return holo - apo

    reference, approximate = difference(None), difference(GB_DIRECTORY)

    assert reference == pytest.approx(6.25, abs=0.05)
    assert approximate == pytest.approx(-8.32, abs=0.05)
    assert reference * approximate < 0, "the sign agreeing is the interesting failure"


def test_which_radii_are_right_depends_on_the_structure(solver):
    """Two radius dialects, two structures, and the ordering flips between them.

    ROADMAP.md section 7 records this from ten structures at once; the corpus now
    holds one of each kind, so it can be asserted in a controlled pair instead.
    `hca` is a polar-hydrogen structure whose heavy-atom radii carry the volume
    of hydrogens absent from the file; `fas2` is all-atom. Substituting mbondi is
    right for the second and wrong for the first, and nothing in a PQR says which
    kind it is holding — which is why the default is chosen for what
    `sashimi_prepare_structure` emits rather than per structure.
    """
    from sashimi.gb.options import GbOptions, GbRadii  # noqa: PLC0415 — local to this test

    as_given = GbSolver(GbOptions(radii=GbRadii.AS_GIVEN))
    for name, mbondi_is_better in (("fas2-molecular", True), ("hca-molecular", False)):
        case = next(c for c in GB_CASES if c.name == name)
        request = case.system(want_potential=False).request_for(SolverFamily.ANALYTIC)
        reference = load_summary(case)["energy_kj_mol"]

        errors = {
            radii: abs(s.solve(request).energy_kj_mol - reference) / abs(reference)
            for radii, s in (("mbondi", solver), ("as-given", as_given))
        }

        assert (errors["mbondi"] < errors["as-given"]) is mbondi_is_better, (name, errors)
