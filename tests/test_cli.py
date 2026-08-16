"""Backend selection in the CLI, without running a solver.

`sashimi validate` picks which backends to compare. Both bugs here were the
same shape as the one that made `tests/test_cross_validation.py` fail on the
README's own install: code that reasons about *registered* backends where it
means *installed* ones.
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import Any

import pytest

from sashimi import backends
from sashimi.backends import BackendReport
from sashimi.cli import _backends_supporting, _refuses, _select, _validate, build_parser
from sashimi.corpus import (
    CORPUS_DIR,
    MANIFEST,
    ROOT_BACKEND,
    CaseTier,
    cases_for_tier,
    corpus_dir_for,
)
from sashimi.protocol import SurfaceModel


def reports(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"backends": list(entries)}


def backend(
    name: str, *, available: bool, models: tuple[str, ...], detail: str = ""
) -> dict[str, Any]:
    return {
        "name": name,
        "available": available,
        "surface_models": list(models),
        "detail": detail,
    }


def test_an_absent_backend_is_excluded_for_being_absent(monkeypatch):
    """Not installed and does not support are different reasons.

    An unavailable backend reports an empty `surface_models`, which reads
    exactly like genuine non-support. So a machine without DelPhi was told
    "excluding delphi: does not support the molecular surface" — false, about
    the project's primary cross-validation partner, and the sort of false that
    stops someone installing it.
    """
    monkeypatch.setattr(
        "sashimi.cli.describe_capabilities",
        lambda: reports(
            backend("apbs", available=True, models=("molecular",)),
            backend(
                "delphi", available=False, models=(), detail="No usable DelPhi found.\nBuild it"
            ),
            backend("gb", available=True, models=("molecular",)),
        ),
    )

    kept, excluded = _backends_supporting(
        ["apbs", "delphi", "gb"], SurfaceModel.MOLECULAR, explicit=False
    )

    assert kept == ["apbs", "gb"]
    assert "not installed" in excluded["delphi"]
    assert "does not support" not in excluded["delphi"]


def test_a_present_backend_that_cannot_take_the_surface_still_says_so(monkeypatch):
    """The other reason must survive the fix that separated them."""
    monkeypatch.setattr(
        "sashimi.cli.describe_capabilities",
        lambda: reports(
            backend("apbs", available=True, models=("molecular", "smoothed-molecular")),
            backend("gb", available=True, models=("molecular",)),
        ),
    )

    kept, excluded = _backends_supporting(
        ["apbs", "gb"], SurfaceModel.SMOOTHED_MOLECULAR, explicit=False
    )

    assert kept == ["apbs"]
    assert excluded["gb"] == "does not support the smoothed-molecular surface"


def test_validate_defaults_to_installed_backends_not_registered_ones(monkeypatch):
    """The README's install is APBS and nothing else.

    Defaulting to every registered backend put TABI-PB in the comparison on a
    machine that has never built it — unavailable, but still advertising
    `molecular`, so the surface filter kept it and every case died on
    `TabipbNotFound`. Reproduced here by leaving exactly one backend installed:
    the run must refuse for the honest reason rather than selecting solvers that
    are not there.
    """
    monkeypatch.setattr("sashimi.backends.available_names", lambda: ["apbs"])

    args = argparse.Namespace(
        backend=None,
        surface=None,
        case=None,
        mesh_density=2.0,
        tolerance=0.1,
        approximation_tolerance=0.15,
        allow_mismatched=False,
    )

    with pytest.raises(SystemExit) as caught:
        _validate(args)

    message = str(caught.value)
    assert "needs at least two backends" in message
    assert "Installed here: apbs" in message


def test_a_case_outside_the_tier_is_not_reported_as_unknown():
    """Two different mistakes with two different fixes.

    A CI step named ten real cases without `--tier full`, was told they did not
    exist, swallowed the exit and reported nothing while looking green. "Raise
    the tier" and "check the spelling" should not print the same sentence.
    """
    with pytest.raises(SystemExit) as caught:
        _select(cases_for_tier(CaseTier.FAST), ["lysozyme-molecular"])

    message = str(caught.value)
    assert "not in the selected tier" in message
    assert "--tier full" in message
    assert "unknown case" not in message


def test_validate_defaults_to_the_cheapest_tier():
    """A default nobody can wait out is not a default.

    `validate` gained `--tier` at M5 but kept defaulting to the whole manifest,
    which was measured at over 40 minutes without finishing on a fully-installed
    machine — and that predates debye, so it is not one backend's cost. Unlike
    `corpus verify`, `validate` re-solves every case in every backend with no
    recordings to fall back on, so its default has to be the cheapest tier the
    corpus offers rather than merely a small one. Widening it again should mean
    deleting this test on purpose.
    """
    parser = build_parser()
    default = CaseTier(parser.parse_args(["validate"]).tier)

    assert default is min(CaseTier, key=lambda tier: len(cases_for_tier(tier)))
    assert len(cases_for_tier(default)) < len(MANIFEST)
    assert CaseTier(parser.parse_args(["validate", "--tier", "full"]).tier) is CaseTier.FULL


def test_a_misspelt_case_still_says_unknown():
    with pytest.raises(SystemExit) as caught:
        _select(cases_for_tier(CaseTier.FULL), ["lysozime"])

    assert "unknown case" in str(caught.value)


# --- where a backend's recordings go, and what a refusal means (M5) ----------


def test_a_backends_recordings_go_to_its_own_directory():
    """The footgun M5 removed, pinned so it cannot come back.

    `summary_path` used to ignore `--backend` and default to the corpus root,
    which is APBS's. So `corpus build --backend delphi` wrote DelPhi's answers
    into APBS's files unless the caller remembered `--directory`, and it failed
    safe only where APBS had already recorded — printing `skip (exists)` instead
    of overwriting. On any case APBS had not recorded, it filed a wrong answer
    silently. Every backend the registry knows is checked, so a sixth one cannot
    arrive without a home.
    """
    assert corpus_dir_for(ROOT_BACKEND) == CORPUS_DIR
    for name in backends.names():
        expected = CORPUS_DIR if name == ROOT_BACKEND else CORPUS_DIR / name
        assert corpus_dir_for(name) == expected
    assert len({corpus_dir_for(n) for n in backends.names()}) == len(backends.names())


def test_a_case_a_backend_refuses_is_not_a_missing_recording():
    """`n/a` and MISS are different facts and had the same symptom.

    debye declines `smoothed-molecular` and `gaussian` on purpose — they are
    APBS's and DelPhi's discretizations — so a third of the corpus has no debye
    recording and never will. Counting those as discrepancies made a
    partial-coverage backend permanently red, which is what put M5's stated exit
    criterion out of reach.

    Read from the backend's published report rather than a table here, so this
    cannot drift from what `sashimi_capabilities` tells a caller.
    """
    smoothed = next(
        c for c in MANIFEST if c.solvent.surface_model is SurfaceModel.SMOOTHED_MOLECULAR
    )
    sharp = next(c for c in MANIFEST if c.solvent.surface_model is SurfaceModel.VAN_DER_WAALS)

    assert _refuses("debye", smoothed)
    assert not _refuses("debye", sharp)


def test_an_unavailable_backend_is_not_mistaken_for_a_refusing_one():
    """Conservative where it cannot tell, which is the DelPhi-shaped trap.

    An undiscoverable DelPhi reports *no* surface models, because which ones it
    has depends on the flavour found. Reading that as "refuses everything" would
    turn a missing binary into a clean bill of health for a backend with no
    recordings at all, so an unavailable backend falls through to the ordinary
    missing-recording path.
    """
    entry = backends.REGISTRY["delphi"]
    hidden = BackendReport("delphi", available=False, family="finite-difference")
    case = next(c for c in MANIFEST if c.solvent.surface_model is SurfaceModel.MOLECULAR)

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(
            backends.REGISTRY,
            "delphi",
            dataclasses.replace(entry, describe=lambda: hidden),
        )
        assert not _refuses("delphi", case)
