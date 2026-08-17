"""Backend selection in the CLI, without running a solver.

`sashimi validate` picks which backends to compare. Both bugs here were the
same shape as the one that made `tests/test_cross_validation.py` fail on the
README's own install: code that reasons about *registered* backends where it
means *installed* ones.
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import Any, NamedTuple, cast

import numpy as np
import pytest

from sashimi import backends
from sashimi.backends import BackendReport
from sashimi.cli import (
    _backends_answering,
    _backends_supporting,
    _print_case,
    _refuses,
    _select,
    _system_fingerprint,
    _validate,
    build_parser,
)
from sashimi.corpus import (
    CORPUS_DIR,
    MANIFEST,
    ROOT_BACKEND,
    CaseTier,
    cases_for_tier,
    corpus_dir_for,
)
from sashimi.errors import SashimiError
from sashimi.protocol import (
    EnergyTerm,
    Equation,
    PQRData,
    Solver,
    SolverFamily,
    SurfaceModel,
    System,
)
from sashimi.validate import Backend, BackendRun, Comparison


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
        # `_validate` reads this before it selects cases; leaving it out passed
        # only because the backend-count guard raises first, so reordering the
        # two would have turned this into an AttributeError.
        tier=CaseTier.FAST.value,
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
    assert "--tier full" in message  # and `lysozyme-molecular` is genuinely a full case
    assert "unknown case" not in message


def test_the_tier_hint_names_the_tier_the_case_is_in():
    """`--tier full` is the wrong advice for a `standard` case.

    It bills the whole 98-case manifest to reach one case, and `validate`'s
    default dropping to `fast` is what made this message common enough for the
    difference to cost anyone time.
    """
    standard = next(case for case in MANIFEST if case.tier is CaseTier.STANDARD)

    with pytest.raises(SystemExit) as caught:
        _select(cases_for_tier(CaseTier.FAST), [standard.name])

    message = str(caught.value)
    assert "--tier standard" in message
    assert "--tier full" not in message


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


class Asked(NamedTuple):
    """What a stubbed `_validate` run put to a solver: the systems, and who."""

    systems: list[Any]
    backends: list[list[str]]


def _validate_asking(monkeypatch: Any, **overrides: Any) -> Asked:
    """Run `_validate` with the solving stubbed out, and report what it asked.

    The tier is a *cost* control, so the property worth pinning is which cases
    reach a solver — not what argparse stored. Backends are named explicitly and
    the surface is given, which is what lets this run with nothing installed.
    """
    asked = Asked([], [])

    def stub_solver(_name: str) -> tuple[object, SolverFamily]:
        return object(), SolverFamily.FINITE_DIFFERENCE

    def record(system: Any, running: Any = (), *_args: Any, **_kwargs: Any) -> None:
        # Recorded, then refused: reaching a solver is the property under test,
        # and refusing keeps this runnable with nothing installed. The *backend
        # list* is recorded beside the system, because "which backends were
        # asked" is a question this file twice answered against a helper instead
        # of against the code path that calls it — and a mutation that reverted
        # the call site stayed green both times.
        asked.systems.append(system)
        asked.backends.append([b.name for b in running])
        raise SashimiError("stubbed out")

    monkeypatch.setattr("sashimi.backends.solver_for", stub_solver)
    monkeypatch.setattr("sashimi.cli.validate_system", record)

    fields: dict[str, Any] = {
        "backend": ["apbs", "delphi"],
        "surface": "molecular",
        "case": None,
        "tier": CaseTier.FAST.value,
        "mesh_density": 2.0,
        "tolerance": 0.1,
        "approximation_tolerance": 0.15,
        "allow_mismatched": False,
    }
    _validate(argparse.Namespace(**{**fields, **overrides}))
    return asked


def _fingerprints_of(cases) -> set[str]:
    """What these cases resolve to once `validate`'s overrides are applied."""
    return {
        _system_fingerprint(
            dataclasses.replace(
                case.system(),
                solvent=dataclasses.replace(case.solvent, surface_model=SurfaceModel.MOLECULAR),
                mesh_density=2.0,
                want_potential=False,
            )
        )
        for case in cases
    }


def test_the_tier_actually_bounds_what_validate_solves(monkeypatch, capsys):
    """The flag has to reach the loop, not just the Namespace.

    Pinning `parse_args(["validate"]).tier` alone would leave the pre-M5 bug —
    `_select(MANIFEST, args.case)`, ignoring the tier entirely — green, which is
    the exact regression this change exists to prevent.

    Stated as set membership rather than a count, so that deduplicating
    identical systems — which legitimately reduces the count — does not read as
    the tier leaking.
    """
    asked = _validate_asking(monkeypatch).systems
    capsys.readouterr()

    in_tier = _fingerprints_of(cases_for_tier(CaseTier.FAST))
    beyond = _fingerprints_of(MANIFEST) - in_tier

    assert {_system_fingerprint(s) for s in asked} == in_tier
    assert not {_system_fingerprint(s) for s in asked} & beyond


def test_one_backends_refusal_does_not_discard_the_others(monkeypatch, capsys):
    """A comparison needs two backends, not every backend.

    TABI-PB cannot mesh a solute with fewer than four atoms. Discarding the
    whole case for that threw away APBS's and DelPhi's answers to the same
    question — 27 of the fast tier's 40 cases, *after* those two had solved.
    """
    born = next(case for case in MANIFEST if case.name == "born-ion-coarse")
    system = dataclasses.replace(
        born.system(),
        solvent=dataclasses.replace(born.solvent, surface_model=SurfaceModel.MOLECULAR),
        want_potential=False,
    )
    # The solver is never called — `_backends_answering` reads preconditions,
    # which is what lets this run with nothing installed.
    solver = cast("Solver[Any]", object())
    selected = [
        Backend("apbs", solver, SolverFamily.FINITE_DIFFERENCE),
        Backend("tabipb", solver, SolverFamily.BOUNDARY_ELEMENT),
        Backend("debye", solver, SolverFamily.FINITE_DIFFERENCE),
    ]

    running, refused = _backends_answering(selected, system)

    assert [b.name for b in running] == ["apbs", "debye"]
    assert "at least 4 atoms" in refused["tabipb"]
    # One sentence: this is a footnote under every case in the run, and length
    # is what stops it being read.
    assert refused["tabipb"].count(".") == 0


def test_validate_asks_the_answering_backends_and_keeps_the_case(monkeypatch, capsys):
    """The wiring, not the helper — `_backends_answering` has to be *used*.

    `test_one_backends_refusal_does_not_discard_the_others` calls the helper
    directly, so replacing `_validate`'s call to it with `selected, {}` — which
    reverts this whole change — left the suite green. That is the third time in
    two PRs that a guard covered a helper and not the code path, so this asserts
    on what `_validate` actually hands `validate_system`.
    """
    asked = _validate_asking(
        monkeypatch,
        backend=["apbs", "tabipb", "debye"],
        case=["born-ion-coarse"],
        tier=CaseTier.FULL.value,
    )
    capsys.readouterr()

    # The case survives — one solve was attempted, not a SKIP...
    assert len(asked.systems) == 1
    # ...and TABI-PB, which cannot mesh one atom, was not in it.
    assert asked.backends == [["apbs", "debye"]]


def test_the_fingerprint_reads_bytes_rather_than_repr():
    """`repr` elides a large numpy array, and two proteins would collide.

    Grouping by `repr(system)` works on the corpus's small synthetic cases and
    silently merges big ones: numpy prints `[ 1. 2. ... 9. 10.]` past its
    threshold, so two structures agreeing at the ends and differing in the
    middle share a repr. That would report one protein's energy under another's
    name — the failure mode the grouping exists to prevent, inverted.
    """
    size = 2000
    coords = np.zeros((size, 3))
    charges = np.ones(size)
    radii = np.full(size, 1.5)
    other = charges.copy()
    other[size // 2] = -1.0  # differs only where repr elides

    def system_for(q):
        return System(structure=PQRData(coords=coords, charges=q, radii=radii))

    assert repr(system_for(charges)) == repr(system_for(other)), "the trap is still real"
    assert _system_fingerprint(system_for(charges)) != _system_fingerprint(system_for(other))


def test_identical_systems_are_solved_once(monkeypatch, capsys):
    """The surface override collapses cases that differ only in surface model.

    `born-ion-coarse`, `born-ion-molecular` and `born-ion-vdw` become one
    question once `molecular` is imposed on all three, so asking it three times
    is three times the cost for one answer. Measured on the whole fast tier: 40
    cases, 18 distinct systems.
    """
    trio = ["born-ion-coarse", "born-ion-molecular", "born-ion-vdw"]

    asked = _validate_asking(monkeypatch, case=trio, tier=CaseTier.FULL.value).systems
    capsys.readouterr()

    assert len(asked) == 1


def test_the_collapse_is_reported_rather_than_silent(capsys):
    """Solving once is the cheap half; saying so is the half that matters.

    Three identical rows read as a measurement — a reader compares the `-vdw`
    row against the `-molecular` row, sees no difference, and concludes the
    probe is worth nothing. M4 measured it worth +5.72% on ALA-GLY. The rows
    were identical because both were solved on `molecular`, not because the
    surfaces agree.
    """
    case = next(c for c in MANIFEST if c.name == "born-ion-coarse")
    comparison = Comparison(
        runs=[
            BackendRun(
                name="apbs",
                energy_kj_mol=-234.0,
                energy_term=EnergyTerm.POLAR_SOLVATION,
                surface_model=SurfaceModel.MOLECULAR,
                equation=Equation.LINEAR,
                potential=None,
            )
        ],
        agrees=True,
    )

    _print_case(
        case,
        comparison,
        refused={"tabipb": "tabipb needs at least 4 atoms"},
        aliases=("born-ion-molecular", "born-ion-vdw"),
        model=SurfaceModel.MOLECULAR,
    )
    printed = capsys.readouterr().out

    assert "same system as born-ion-molecular, born-ion-vdw" in printed
    assert "molecular surface is applied" in printed
    # A backend that sat the case out must not read as one that agreed.
    assert "not asked: tabipb" in printed


def test_a_named_case_is_reachable_from_outside_the_default_tier(monkeypatch, capsys):
    """Lowering a cost default must not lower reach.

    The first cut of the `fast` default narrowed `--case` along with it, so
    `sashimi validate --case lysozyme-molecular` — which worked before — started
    exiting 1 for the 58 cases outside `fast`.
    """
    outside = next(case for case in MANIFEST if case.tier is not CaseTier.FAST)

    asked = _validate_asking(monkeypatch, case=[outside.name]).systems
    capsys.readouterr()

    assert len(asked) == 1


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
