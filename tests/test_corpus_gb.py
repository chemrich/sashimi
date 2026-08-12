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
    assert len(GB_CASES) >= 5


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
