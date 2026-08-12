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
number. `comparable_surface_models()` is the precondition for *which model* to compare
on, and this file skips entirely rather than inventing a comparison.

It is not the precondition for whether the comparison can run at all, and
reading it as one was a bug that survived from phase 7 to 2026-08-12. That
function counts backends that share a model, and Generalized Born is always
available and shares `molecular` with APBS — so on a machine with APBS and no
DelPhi it returns `["molecular"]`, the skip does not fire, and `DelphiSolver()`
raises `DelphiNotFound` five tests running. That is the README's own recommended
install: APBS from conda-forge and nothing else. Neither CI leg saw it because
both always carry a second *real* backend, and it was found by asking what a
runner with only APBS would do.

So the guard is now what the file's name says it is: both of the backends *this
file compares* must be installed.

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
from sashimi.delphi import DelphiSolver, discover_delphi
from sashimi.delphi.discover import DelphiFlavour
from sashimi.pqr import parse_pqr, read_pqr
from sashimi.protocol import (
    EnergyTerm,
    FiniteDifferenceRequest,
    GridSpec,
    SolventModel,
    SurfaceModel,
)
from sashimi.validate import Comparison, Incomparable, validate
from tests.helpers import installed_or_skip

pytestmark = [pytest.mark.apbs, pytest.mark.delphi]


@pytest.fixture(autouse=True)
def _delphi_installed():
    """Both backends, or a skip. The marker selects; this is what skips.

    Autouse so it cannot be forgotten on a test added later — which is how the
    equivalent guard came to be missing from the boundary-element MCP test.
    """
    installed_or_skip(discover_delphi, "SASHIMI_DELPHI_PATH")


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
    "than two are installed, or the installed pair overlaps nowhere. Both shipped "
    "DelPhi flavours share `molecular` with APBS, so in practice this means a "
    "backend is missing."
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
        # Zero salt, so this case is comparable whichever DelPhi flavour is
        # installed: pyDelPhi reports the reaction field only, which coincides
        # with APBS's polar solvation energy exactly when there are no mobile
        # ions. The salted case is
        # `test_salted_comparison_depends_on_the_flavours_energy_term`.
        solvent = SolventModel(
            surface_model=model, solute_dielectric=1.0 if case == "born-ion" else 2.0
        )
        solvent = dataclasses.replace(solvent, ionic_strength=0.0)

        comparison = _compare(structures[case], solvent)

        assert comparison.agrees, f"{case} / {model.value}: {comparison.summary()}"


def test_salted_comparison_depends_on_the_flavours_energy_term(structures):
    """Salt is where the two backends' definitions used to diverge.

    The C++ build is asked for the ion-inclusive quantity, so it reports APBS's
    term and a salted comparison is legitimate — which is the whole reason for
    reconstructing that term rather than reading DelPhi's headline line.
    pyDelPhi cannot report it, so the same request is refused there.

    Both branches are the engine working. Which one runs is a property of the
    installed flavour, so the test asks rather than assumes.
    """
    models = shared_models()
    if not models:
        pytest.skip(NO_SHARED_MODEL)

    solvent = SolventModel(surface_model=models[0], ionic_strength=0.15)

    if discover_delphi().flavour is DelphiFlavour.CPP:
        comparison = _compare(structures["ala-gly"], solvent)
        assert comparison.agrees, comparison.summary()
        assert all(r.energy_term is EnergyTerm.POLAR_SOLVATION for r in comparison.runs), (
            "a salted comparison is only meaningful when both report the same term"
        )
    else:
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
