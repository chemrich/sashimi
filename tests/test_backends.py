"""The backend registry: one list of solvers, and the drift it exists to stop.

Three callers used to know the backends separately — the CLI held factories,
`capabilities` held report functions, and a library consumer held nothing. The
failure that motivates this file is not a crash: it is `sashimi_capabilities`
reporting a backend that `--backend` cannot name, which is a report about a
solver the caller then cannot run.

Nothing here needs a binary. Constructing a solver is the only part that
discovers one, and that is exactly what the registry does *not* do until asked.
"""

from __future__ import annotations

import pytest

from sashimi import backends
from sashimi.apbs.options import SURFACE_KEYWORD
from sashimi.cli import BACKEND_NAMES
from sashimi.delphi.options import SUPPORTED_SURFACES as DELPHI_SURFACES
from sashimi.errors import InputError
from sashimi.gb.options import SUPPORTED_SURFACES as GB_SURFACES
from sashimi.protocol import AccuracyTier, SolventModel, SolverFamily
from sashimi.tabipb.options import SUPPORTED_SURFACES as TABIPB_SURFACES

EXPECTED = ("apbs", "delphi", "tabipb", "gb")


def test_the_registry_holds_every_shipped_backend():
    """A guard on the list itself; debye adds one entry here and nowhere else."""
    assert backends.names() == EXPECTED


@pytest.mark.parametrize("name", EXPECTED)
def test_a_backend_agrees_with_itself_about_its_family(name: str):
    """The registry says `SolverFamily`, the report says a string, and they must match.

    Two spellings of the same fact, one typed and one for display. They are what
    would drift apart silently: a caller dispatching on the enum and an agent
    reading the report would then be told different things about the same
    solver.
    """
    entry = backends.REGISTRY[name]

    assert entry.report().family == entry.family.value


@pytest.mark.parametrize("name", EXPECTED)
def test_a_backend_reports_the_name_it_is_registered_under(name: str):
    """`--backend <name>` and the name in the report have to be the same string."""
    assert backends.REGISTRY[name].report().name == name


def test_the_cli_offers_exactly_what_the_registry_holds():
    """`--backend` choices come from the registry rather than a parallel list."""
    assert tuple(BACKEND_NAMES) == backends.names()


def test_an_unknown_backend_says_what_the_alternatives_are():
    """A caller that guessed the name wrong has no other way to learn it."""
    with pytest.raises(InputError) as caught:
        backends.get("apbs3")

    message = str(caught.value)
    assert "apbs3" in message
    for name in EXPECTED:
        assert name in message


def test_the_in_process_backend_is_always_available():
    """Generalized Born has nothing to discover, so it cannot be missing.

    Which makes it the one backend a consumer can rely on being there — and the
    reason `available_names()` is never empty, however bare the machine.
    """
    assert "gb" in backends.available_names()


def test_every_backend_can_answer_the_default_surface_model():
    """What the default is *for*, and the property it was chosen to have.

    `smoothed-molecular` was the default until 2026-08-13 and is APBS's harmonic
    averaging alone, so a request that named no surface refused on three
    backends out of four — including `gb`, the tier that needs no binary and is
    the only one guaranteed to be there. `molecular` is the one boundary every
    shipped backend defines, and this is where a fifth backend that cannot
    answer on it has to argue the case rather than quietly narrow the default.

    Read from the modules that own the mapping rather than from
    `reports()`, which is not the same claim: an undiscoverable DelPhi reports
    *no* surface models, because which ones it has depends on the flavour found,
    so a report-based version of this passed with a binary installed and failed
    on the bare leg. Both DelPhi flavours are checked for the same reason — the
    default has to be answerable by whichever one a machine happens to have.
    """
    declared = {
        "apbs": frozenset(SURFACE_KEYWORD),
        "tabipb": TABIPB_SURFACES,
        "gb": GB_SURFACES,
        **{f"delphi/{flavour.value}": models for flavour, models in DELPHI_SURFACES.items()},
    }
    # Every registered backend appears above, so a fifth one cannot arrive
    # without either supporting the default or arguing here for narrowing it.
    assert {name.split("/")[0] for name in declared} == set(backends.names())

    default = SolventModel().surface_model
    unsupported = sorted(name for name, models in declared.items() if default not in models)
    assert unsupported == [], f"{unsupported} cannot answer the default {default.value!r}"


def test_only_the_approximate_tier_says_it_approximates():
    """`AccuracyTier` is what stops a triage number being read as an answer."""
    tiers = {report.name: report.accuracy_tier for report in backends.reports()}

    assert tiers["gb"] == AccuracyTier.APPROXIMATE.value
    assert {tiers[n] for n in ("apbs", "delphi", "tabipb")} == {AccuracyTier.REFERENCE.value}


def test_the_registry_does_not_construct_a_solver_to_describe_one():
    """Reporting must not depend on the binary being there.

    `sashimi_capabilities` is the one tool that has to work when a backend is
    missing — if describing it went through construction, the tool that explains
    an absent APBS would be the tool that cannot run.
    """
    reports = backends.reports()

    assert [r.name for r in reports] == list(EXPECTED)
    assert all(isinstance(r.available, bool) for r in reports)


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("apbs", SolverFamily.FINITE_DIFFERENCE),
        ("delphi", SolverFamily.FINITE_DIFFERENCE),
        ("tabipb", SolverFamily.BOUNDARY_ELEMENT),
        ("gb", SolverFamily.ANALYTIC),
    ],
)
def test_each_backend_declares_the_dialect_it_speaks(name: str, family: SolverFamily):
    """Dispatch through the registry happens at runtime, where the static
    guarantee that a BEM backend never sees an FD request does not reach."""
    assert backends.get(name).family is family
