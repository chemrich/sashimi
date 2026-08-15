"""The debye tier of the corpus, on a machine with no binaries at all.

Every other recorded tier needs something installed to re-solve: `tests/corpus/`
needs APBS, `tests/corpus/delphi/` a compiler, `tests/corpus/tabipb/` a mesher
as well. debye needs nothing, which makes it **the one tier that can be
re-solved on every leg of CI** — and until this file it was the one tier nothing
re-solved at all. `test_every_cross_backend_recording_answers_a_manifest_case`
checks filenames; the CI corpus step runs the default backend, which is APBS.
So all 56 recordings could have gone stale silently, which is the same
"a skipped tier and a passing tier look identical" trap `.github/workflows/ci.yml`
already records being burned by.

Nothing here is marked and nothing here can skip, for the reason
`tests/test_debye_solver.py` gives: a suite that quietly skipped debye on a
binary-free machine would be testing the opposite of debye's central claim.

Scoped to the fast tier so the edit-test loop stays a loop. The standard and
full tiers are `sashimi corpus verify --backend debye --tier standard|full`,
which is what M5 was graded on and what a release should run.
"""

from __future__ import annotations

import json

import pytest

from sashimi.corpus import (
    MANIFEST,
    Case,
    CaseTier,
    Tolerances,
    cases_for_tier,
    corpus_dir_for,
    load_summary,
    verify_case,
)
from sashimi.debye import DebyeSolver
from sashimi.protocol import SolverFamily

DEBYE_DIR = corpus_dir_for("debye")

# The recorded fast-tier cases. Derived by asking which recordings exist rather
# than by listing names, so a case recorded later joins without an edit here —
# and `test_every_case_debye_can_answer_is_recorded` is the other half, which
# fails if one it *could* answer is missing.
FAST_RECORDED = [
    case for case in cases_for_tier(CaseTier.FAST) if (DEBYE_DIR / f"{case.name}.json").is_file()
]


def test_the_debye_tier_is_not_empty():
    """Whatever else is true, something here runs."""
    assert FAST_RECORDED, f"no debye recordings in {DEBYE_DIR}"


@pytest.mark.parametrize("case", FAST_RECORDED, ids=lambda c: c.name)
def test_debye_reproduces_its_recorded_answer(case: Case):
    """Re-solve and compare, with no binary installed.

    This is `sashimi corpus verify --backend debye --tier fast` as a test, which
    is M5's exit criterion. The CLI is what a release runs; this is what every
    push runs, on every platform, including the leg with nothing installed.
    """
    found = verify_case(
        DebyeSolver(),
        case,
        load_summary(case, backend="debye"),
        Tolerances(),
        SolverFamily.FINITE_DIFFERENCE,
    )
    assert found == [], f"{case.name}: {'; '.join(str(item) for item in found)}"


def test_every_case_debye_can_answer_is_recorded():
    """A gap in coverage, told apart from a principled refusal.

    debye declines `smoothed-molecular` and `gaussian` on purpose, so most of
    the corpus has no debye recording and never will. What *would* be a gap is a
    sharp-boundary case with no recording, and the two look identical from a
    file listing. Asked of the solver's own supported set, so this cannot drift
    from what it actually builds.
    """
    from sashimi.debye.options import SUPPORTED_SURFACES  # noqa: PLC0415

    missing = [
        case.name
        for case in MANIFEST
        if case.solvent.surface_model in SUPPORTED_SURFACES
        and not (DEBYE_DIR / f"{case.name}.json").is_file()
    ]
    assert not missing, (
        f"debye builds these surfaces but has no recording for: {', '.join(missing)}. "
        "Record them with `sashimi corpus build --backend debye --tier full`."
    )


def test_no_recording_carries_a_tolerance_the_manifest_has_since_changed():
    """The failure mode that shipped in this milestone and nothing caught.

    `build_case` writes the tolerance *that applies to this backend* into each
    summary, calling it "the one a reader of the file needs". Adding a
    `debye_rtol` to a case already recorded leaves that field behind, and
    `corpus build` skips existing files without `--force`, so the recording
    keeps the old number. `verify` grades the freshly-solved answer rather than
    the recording, so nothing goes red — the artifact just quietly disagrees
    with the manifest, and three of them shipped reading
    `relative_error 0.01504` beside `rtol 0.01`.

    Covers every backend's recordings rather than debye's, since the mechanism
    is not debye's.
    """
    from sashimi import backends  # noqa: PLC0415

    stale = []
    for name in backends.names():
        directory = corpus_dir_for(name)
        for case in MANIFEST:
            path = directory / f"{case.name}.json"
            if not path.is_file():
                continue
            summary = json.loads(path.read_text())
            label = summary.get("backend", name)
            for key, reference in (
                ("analytic", case.analytic),
                ("analytic_field", case.analytic_field),
            ):
                block = summary.get(key)
                if reference is None or not block or "rtol" not in block:
                    continue
                expected = reference.rtol_for(label)
                if block["rtol"] != expected:
                    stale.append(f"{name}/{case.name} {key}: {block['rtol']} != {expected}")

    assert not stale, (
        "recording(s) carry a tolerance the manifest has since changed — re-record "
        "with `--force`:\n  " + "\n  ".join(stale)
    )
