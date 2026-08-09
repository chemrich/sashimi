"""Golden-corpus regression net.

With APBS coming from the system package manager rather than a lockfile, its
version is no longer pinned by the build. This is what replaces that pin: if a
`brew upgrade` or a new Ubuntu image moves APBS underneath us, the Born numbers
shift and these tests say so, in kJ/mol, instead of the change landing silently.

Regenerate deliberately with `uv run python scripts/build_corpus.py` — never to
make a red test go green.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sashimi.apbs import ApbsSolver, discover_apbs
from sashimi.protocol import GridSpec, SolventModel
from scripts.build_corpus import SOLVENT, born_ion

pytestmark = pytest.mark.apbs

CORPUS_PATH = Path(__file__).parent / "corpus" / "born-sashimi.json"

# CLAUDE.md freezes the backend at 3.4.1. brew and apt both ship exactly this.
EXPECTED_APBS_VERSION = "3.4.1"

# Energies are a converged scalar and reproduce tightly across platforms.
ENERGY_RTOL = 1e-4
# Pointwise potentials are interpolated off the grid, so they carry a little
# more platform-dependent float noise than the integrated energy does.
POTENTIAL_RTOL = 1e-3


@pytest.fixture(scope="module")
def corpus() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(CORPUS_PATH.read_text())
    return loaded


def test_backend_is_the_pinned_version():
    """A drifted system APBS must fail here, not silently change results."""
    binary = discover_apbs()
    assert binary.version == EXPECTED_APBS_VERSION, (
        f"expected APBS {EXPECTED_APBS_VERSION}, found {binary.version} at {binary.path}. "
        "Either pin the system package back, or regenerate the corpus deliberately "
        "and update EXPECTED_APBS_VERSION."
    )


def test_corpus_records_the_expected_backend(corpus):
    assert corpus["reference_backend"] == f"apbs-{EXPECTED_APBS_VERSION}"


def test_solvent_model_matches_the_recorded_one(corpus):
    """A changed default would invalidate every number in the file."""
    recorded = corpus["solvent_model"]
    for field, value in recorded.items():
        assert getattr(SOLVENT, field) == value, f"{field} drifted from the corpus"


@pytest.mark.parametrize("case_name", ["res=0.5", "res=0.25"])
def test_energy_reproduces_the_corpus(corpus, case_name):
    case = corpus["cases"][case_name]
    result = ApbsSolver().solve_lpbe(
        born_ion(),
        GridSpec(**case["grid_spec"]),
        SolventModel(**corpus["solvent_model"]),
        compute_energy=True,
    )
    assert result.diagnostics["dime"] == case["dime"]
    assert result.energy_kj_mol == pytest.approx(case["energy_kj_mol"], rel=ENERGY_RTOL), (
        f"{case_name}: {result.energy_kj_mol:.6f} kJ/mol vs corpus "
        f"{case['energy_kj_mol']:.6f} — the solver moved."
    )


@pytest.mark.parametrize("case_name", ["res=0.5", "res=0.25"])
def test_potential_probes_reproduce_the_corpus(corpus, case_name):
    case = corpus["cases"][case_name]
    result = ApbsSolver().solve_lpbe(
        born_ion(),
        GridSpec(**case["grid_spec"]),
        SolventModel(**corpus["solvent_model"]),
    )
    xs = [float(x) for x in case["potential_at_x"]]
    expected = np.array(list(case["potential_at_x"].values()))
    points = np.zeros((len(xs), 3))
    points[:, 0] = xs

    np.testing.assert_allclose(result.potential.value_at(points), expected, rtol=POTENTIAL_RTOL)
