"""APBS against DelPhi, on the surface models they actually share.

This is the phase 7 deliverable ROADMAP.md section 8 describes: run one system
through two backends and report the spread. It needs both binaries, so it is
marked for both and skips unless both are installed — in practice, the Linux CI
leg, where the C++ DelPhi is built from source.

**It refuses to compare across mismatched surface models**, which is the whole
discipline of section 8 rather than a detail. Surface definition moves a
dipeptide's solvation energy across 25.7% on this code, so a spread computed
between an APBS `smoothed-molecular` and a DelPhi `molecular` would be a
modelling difference misreported as a solver disagreement — worse than no
number. `comparable_surface_models()` is the precondition, and it is frequently
empty: with pyDelPhi installed instead of the C++ build it is empty by
definition, and this file skips entirely rather than inventing a comparison.

The gate is deliberately loose. Two independent finite-difference codes on
different grids will never agree to corpus tolerance, and pinning the measured
2-5% would make this a change-detector for physics that is allowed to change.
What it catches is the class of failure that actually threatens a second
backend: a unit error, a factor of two, a definitional swap. Those move the
answer by tens of percent or more, and this fails loudly on them.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from sashimi.apbs import ApbsSolver
from sashimi.capabilities import comparable_surface_models
from sashimi.delphi import DelphiSolver
from sashimi.pqr import parse_pqr, read_pqr
from sashimi.protocol import (
    FiniteDifferenceRequest,
    GridSpec,
    SolventModel,
    SurfaceModel,
)

pytestmark = [pytest.mark.apbs, pytest.mark.delphi]

# Two independent FD codes on different grids. Measured across the shared
# models: 2.30% (Born ion), 2.31% (ALA-GLY molecular), 4.02% (ALA-GLY
# van-der-waals), 2.44% (hen lysozyme). See the module docstring on why this is
# not tightened to those numbers.
MAX_SPREAD = 0.10

BORN_ION_PQR = "ATOM      1  I   ION     1       0.000   0.000  0.000  1.00  3.00\n"

# Two distinct reasons produce the same empty list, and the message should not
# guess which: fewer than two backends installed, or two that overlap nowhere.
NO_SHARED_MODEL = (
    "no surface model is comparable across the installed backends — either fewer "
    "than two are installed, or they overlap nowhere, which is the case for APBS "
    "and pyDelPhi. Cross-validation needs the C++ DelPhi build."
)


def shared_models() -> list[SurfaceModel]:
    return [SurfaceModel(name) for name in comparable_surface_models()]


@pytest.fixture(scope="module")
def structures():
    peptide = read_pqr(Path(__file__).resolve().parent / "data" / "ala-gly.pqr")
    return {"born-ion": parse_pqr(BORN_ION_PQR), "ala-gly": peptide}


def _solve_both(structure, solvent) -> tuple[float, float]:
    request = FiniteDifferenceRequest(
        structure=structure,
        solvent=solvent,
        grid=GridSpec(resolution=0.5, padding=10.0),
        want_potential=False,  # energy is the comparable quantity; grids differ
    )
    apbs = ApbsSolver().solve(request)
    delphi = DelphiSolver().solve(request)
    assert apbs.energy_kj_mol is not None
    assert delphi.energy_kj_mol is not None
    return apbs.energy_kj_mol, delphi.energy_kj_mol


@pytest.mark.parametrize("case", ["born-ion", "ala-gly"])
def test_backends_agree_on_shared_surface_models(structures, case):
    models = shared_models()
    if not models:
        pytest.skip(NO_SHARED_MODEL)

    for model in models:
        solvent = SolventModel(surface_model=model)
        if case == "born-ion":
            solvent = dataclasses.replace(solvent, solute_dielectric=1.0, ionic_strength=0.0)

        apbs, delphi = _solve_both(structures[case], solvent)
        spread = abs(apbs - delphi) / abs(apbs)

        assert spread < MAX_SPREAD, (
            f"{case} / {model.value}: APBS {apbs:.3f} vs DelPhi {delphi:.3f} kJ/mol "
            f"= {spread:.2%} apart. Two FD codes should agree to a few percent on a "
            "shared surface model; a gap this size is a unit, sign or definitional "
            "error rather than discretization."
        )


def test_both_backends_agree_on_the_born_ion_closed_form(structures):
    """Where an analytic answer exists, agreement with *it* is the real check.

    Two backends agreeing with each other could still both be wrong. The Born
    ion is the one case in the suite where a third, independent answer exists.
    """
    if not shared_models():
        pytest.skip(NO_SHARED_MODEL)

    from tests.born_reference import born_solvation_energy  # noqa: PLC0415

    solvent = SolventModel(
        surface_model=shared_models()[0], solute_dielectric=1.0, ionic_strength=0.0
    )
    apbs, delphi = _solve_both(structures["born-ion"], solvent)
    expected = born_solvation_energy(3.0, solute_dielectric=1.0)

    # Both are held to the same 3% here rather than the 1% the analytic test
    # applies to a converged grid: this runs at a fixed 0.5 A for speed, and
    # APBS is measurably further from the closed form there than DelPhi is.
    assert apbs == pytest.approx(expected, rel=0.03)
    assert delphi == pytest.approx(expected, rel=0.03)
