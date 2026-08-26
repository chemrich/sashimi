"""The one thing that stops `studies/` rotting, and the reason it is only one.

ROADMAP.md section 12 cites tables that no corpus recording holds, and until
2026-08-26 their generators lived in a scratch directory. `studies/` checks them
in; this checks that at least one of them still produces what it produced.

**Why a single study and not the campaign.** The field-axis campaign is tens of
minutes of solves. Section 7 records the DelPhi tier silently skipping every test
for a year while CI stayed green, and section 12 records "too slow to check here"
being one step from "quietly absent" — so the choice is between one cheap study
that actually runs on every push and a comprehensive suite that gets marked slow
and stops. This is the cheap one.

**What a failure here means.** Not "fix the test". It means the solver has moved
under a table section 12 still quotes, and the question is whether the document
needs correcting. `studies/README.md` says so too, because that is the sort of
instruction a reader needs at the moment the test goes red rather than later.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

STUDIES = Path("studies")


def test_every_study_names_a_result_that_exists():
    """The map in `studies/README.md` is only worth having if it resolves.

    Cheap, and it catches the shape of rot that arrives first: a script renamed
    or deleted while the README and section 12 go on pointing at it.
    """
    scripts = sorted(STUDIES.glob("*/*.py"))
    assert len(scripts) > 15, f"only {len(scripts)} study scripts; the directory has been thinned"

    readme = (STUDIES / "README.md").read_text()
    missing = [s.name for s in scripts if s.name not in readme]
    assert not missing, f"study scripts absent from studies/README.md's map: {missing}"

    for directory in sorted({s.parent for s in scripts}):
        results = directory / "results"
        assert results.is_dir(), f"{directory} has no results/ beside it"
        assert any(results.iterdir()), f"{results} is empty"


def test_the_cheapest_study_still_reproduces_its_recorded_table():
    """`sphere_shell.py` end to end, against the output checked in beside it.

    About ten seconds: four small Born solves per width at two resolutions, read
    on a shell against the exact closed form. It exercises the ramp at six
    widths, the solvent-excluded surface, `_union_gap` on the accessible
    boundary, and `PotentialGrid.value_at` — so most of what the field-axis
    tables were taken through moves this number if it moves at all.

    Compared as text rather than parsed, deliberately. The recorded file is what
    a reader of section 12 would diff against, and a comparison that reformats
    both sides first would pass through a change to the formatting that makes the
    two no longer comparable by eye.
    """
    study = STUDIES / "field_axis" / "sphere_shell.py"
    recorded = (STUDIES / "field_axis" / "results" / "sphere_shell.txt").read_text().strip()

    finished = subprocess.run(
        [sys.executable, str(study)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert finished.returncode == 0, f"{study} failed:\n{finished.stderr[-2000:]}"

    produced = finished.stdout.strip()
    assert produced, f"{study} printed nothing"
    if produced == recorded:
        return

    got = produced.splitlines()
    want = recorded.splitlines()
    moved = [f"  recorded: {w}\n  produced: {g}" for g, w in zip(got, want, strict=False) if g != w]
    raise AssertionError(
        "the cheapest study no longer reproduces its recorded table, so a number "
        "ROADMAP.md section 12 quotes has moved:\n" + "\n".join(moved[:4])
    )
