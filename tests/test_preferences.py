"""Choosing a backend by what you want from the answer, not by its name.

`Preference` exists because a caller usually knows they want the fast one or
the steady one, and should not have to know that APBS focuses, that pyDelPhi
cannot build a van der Waals boundary, or that debye is the only thing that
runs on linux-aarch64. ROADMAP.md section 12 records the measurements the
ordering rests on.

**The surface model decides as much as availability does.** pyDelPhi answers 35
of the corpus's 100 cases — no `van-der-waals`, no `smoothed-molecular` — so a
resolver that checked only "is it installed" would hand two thirds of `stable`
requests to a backend that refuses them. Most of this file is that fall-through.
"""

from __future__ import annotations

import pytest

from sashimi import backends
from sashimi.errors import InputError
from sashimi.protocol import Preference, SurfaceModel

SURFACES = sorted(m.value for m in SurfaceModel)


def test_every_preference_has_an_order_and_every_name_is_registered():
    """A typo in the table would silently make a preference unreachable."""
    for preference in Preference:
        order = backends._PREFERENCE_ORDER[preference.value]
        assert order, f"{preference} has no candidates"
        for name in order:
            assert name in backends.names(), f"{preference} names unregistered {name!r}"


def test_the_orders_differ_or_the_preferences_mean_nothing():
    """Three names for one ordering would be three names for nothing."""
    orders = {p.value: backends._PREFERENCE_ORDER[p.value] for p in Preference}
    assert orders["fast"][0] != orders["stable"][0]
    assert orders["portable"][0] not in (orders["fast"][0], orders["stable"][0])


def test_fast_leads_with_apbs_and_stable_with_pydelphi():
    """The ordering is a claim about measurements, so it is asserted here.

    APBS is 12.4 s where pyDelPhi is 27.3 s at 1,156 residues, and answers every
    surface model in the corpus. pyDelPhi is the more pose-stable of the two at
    every resolution measured — 0.15-0.52% against 0.42-1.07% across two
    structures and three resolutions, and on a *coarser* effective grid in all
    six, which is the harder condition.
    """
    assert backends._PREFERENCE_ORDER["fast"][0] == "apbs"
    assert backends._PREFERENCE_ORDER["stable"][0] == "pydelphi"
    assert backends._PREFERENCE_ORDER["portable"][0] == "debye"


def test_portable_never_reaches_for_a_binary():
    """Its whole point is the machine with nothing installed — linux-aarch64,
    where conda-forge has no APBS at all."""
    for name in backends._PREFERENCE_ORDER["portable"]:
        report = next(r for r in backends.reports() if r.name == name)
        assert report.available, f"{name} should need no install, but reports unavailable"


@pytest.mark.parametrize("preference", [p.value for p in Preference])
@pytest.mark.parametrize("surface", SURFACES)
def test_resolution_either_answers_or_explains(preference: str, surface: str):
    """Whatever is installed, the resolver returns a usable backend or says why not.

    Both outcomes are correct — `portable` genuinely cannot do
    `smoothed-molecular`, since that is APBS's own averaging — and what must not
    happen is a backend that cannot answer being handed the request.
    """
    try:
        name, because = backends.resolve(preference, surface)
    except InputError as exc:
        assert surface in str(exc)
        assert preference in str(exc)
        return
    report = next(r for r in backends.reports() if r.name == name)
    assert report.available, f"{preference}/{surface} chose an unavailable {name}"
    assert surface in report.surface_models, (
        f"{preference}/{surface} chose {name}, which supports {report.surface_models}"
    )
    assert because, "a resolved preference must say why"


def test_the_reason_names_what_was_skipped():
    """A caller who asked for `stable` and got something else needs to see why.

    The interesting case is a fall-through for *capability* rather than for
    absence: pyDelPhi is installed and still cannot take a van der Waals
    boundary, and "not installed" would be the wrong explanation.
    """
    available = {r.name for r in backends.reports() if r.available}
    if "pydelphi" not in available:
        pytest.skip("pyDelPhi is not installed here, so there is no fall-through to see")
    name, because = backends.resolve("stable", "van-der-waals")
    assert name != "pydelphi"
    assert "pydelphi" in because
    assert "van-der-waals" in because


def test_an_unknown_preference_lists_the_known_ones():
    with pytest.raises(InputError, match="unknown preference"):
        backends.resolve("accurate", "molecular")


def test_stable_is_not_called_accurate():
    """Naming, asserted, because the wrong name here is a claim we cannot support.

    Above a two-atom solute the corpus has no ground truth at all — 37
    closed-form energies, every one a Born ion or a Kirkwood sphere. What is
    measured is how little an answer moves under rigid motion, which is
    discretization noise, not distance from a right answer.
    """
    assert {p.value for p in Preference} == {"fast", "stable", "portable"}
    assert "accurate" not in {p.value for p in Preference}
