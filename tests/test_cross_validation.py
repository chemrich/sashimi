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
from sashimi.validate import Comparison, Incomparable, validate

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


def _compare(structure, solvent, *, want_potential: bool = False) -> Comparison:
    """Run both backends through the real `sashimi.validate` engine.

    Deliberately not a private comparison: this file used to subtract two
    energies itself, which meant the shipped engine — including every refusal
    that makes a spread trustworthy — was exercised only by stubs. Going through
    `validate` means the CLI, the library and this test all agree by
    construction.
    """
    request = FiniteDifferenceRequest(
        structure=structure,
        solvent=solvent,
        grid=GridSpec(resolution=0.5, padding=10.0),
        want_potential=want_potential,
    )
    return validate(
        {"apbs": ApbsSolver(), "delphi": DelphiSolver()},
        request,
        tolerance=MAX_SPREAD,
    )


@pytest.mark.parametrize("case", ["born-ion", "ala-gly"])
def test_backends_agree_on_shared_surface_models(structures, case):
    models = shared_models()
    if not models:
        pytest.skip(NO_SHARED_MODEL)

    for model in models:
        # Zero salt, deliberately: APBS reports a polar solvation energy and
        # DelPhi a reaction-field energy, and those coincide only where there is
        # no mobile-ion contribution. `validate` refuses the salted comparison,
        # which `test_salted_comparison_is_refused` covers.
        solvent = SolventModel(
            surface_model=model, solute_dielectric=1.0 if case == "born-ion" else 2.0
        )
        solvent = dataclasses.replace(solvent, ionic_strength=0.0)

        comparison = _compare(structures[case], solvent)

        assert comparison.agrees, f"{case} / {model.value}: {comparison.summary()}"


def test_salted_comparison_is_refused(structures):
    """The engine's central refusal, against the real backends.

    APBS's difference-of-blocks carries the mobile-ion term and DelPhi's
    reaction field does not, so at 0.15 M these are different quantities. The
    spread would look like a modest disagreement rather than the definitional
    gap it is, which is precisely the failure mode worth refusing.
    """
    models = shared_models()
    if not models:
        pytest.skip(NO_SHARED_MODEL)

    solvent = SolventModel(surface_model=models[0], ionic_strength=0.15)
    with pytest.raises(Incomparable, match="mobile-ion contribution"):
        _compare(structures["ala-gly"], solvent)


def test_potentials_are_comparable_across_incompatible_grids(structures):
    """The blocker that kept this out of `corpus.verify_case`, resolved.

    APBS's dime must be 32c+1 and DelPhi's gsize is any odd integer, so the two
    never produce the same grid. Sampling both at shared physical coordinates
    sidesteps that entirely.
    """
    models = shared_models()
    if not models:
        pytest.skip(NO_SHARED_MODEL)

    solvent = SolventModel(surface_model=models[0], solute_dielectric=1.0, ionic_strength=0.0)
    comparison = _compare(structures["born-ion"], solvent, want_potential=True)

    grids = [r.potential for r in comparison.runs if r.potential is not None]
    assert len(grids) == 2
    assert grids[0].shape != grids[1].shape, "expected incompatible grids to compare across"
    assert comparison.n_probes > 0
    assert comparison.potential_rmsd_kt_e is not None


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
    comparison = _compare(structures["born-ion"], solvent)
    expected = born_solvation_energy(3.0, solute_dielectric=1.0)

    # Both are held to the same 3% here rather than the 1% the analytic test
    # applies to a converged grid: this runs at a fixed 0.5 A for speed, and
    # APBS is measurably further from the closed form there than DelPhi is.
    for backend_run in comparison.runs:
        assert backend_run.energy_kj_mol == pytest.approx(expected, rel=0.03), (
            f"{backend_run.name} is {backend_run.energy_kj_mol} against a closed form of {expected}"
        )
