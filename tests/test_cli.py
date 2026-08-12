"""Backend selection in the CLI, without running a solver.

`sashimi validate` picks which backends to compare. Both bugs here were the
same shape as the one that made `tests/test_cross_validation.py` fail on the
README's own install: code that reasons about *registered* backends where it
means *installed* ones.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from sashimi.cli import _backends_supporting, _select, _validate
from sashimi.corpus import CaseTier, cases_for_tier
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


def test_a_misspelt_case_still_says_unknown():
    with pytest.raises(SystemExit) as caught:
        _select(cases_for_tier(CaseTier.FULL), ["lysozime"])

    assert "unknown case" in str(caught.value)
