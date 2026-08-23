"""The instrument's own tests, which are mostly about what it refuses to say.

A benchmark harness is the one kind of code whose bugs flatter it: a comparison
that quietly measured two different grids, or that called two different answers
the same, reports a speed-up either way and looks right doing it. So the tests
here spend most of their effort on the three claims that could be false without
anything looking wrong — that a ratio is oriented the way it reads, that
"identical" means bit-identical rather than close, and that both sides of a
comparison are handed the same case.

Timing itself is deliberately barely tested: an assertion about how long
something takes is the flakiest thing a suite can contain, and this module
exists precisely because the machine's clock is untrustworthy under load.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import pytest

from sashimi.bench import (
    CaseSpec,
    Comparison,
    Measurement,
    _baseline_environment,
    children_counted,
    cpu_seconds,
    remote_snippet,
    render_comparison,
    solve_case,
)
from sashimi.cli import main
from sashimi.protocol import SurfaceModel

ALA_GLY = Path("tests/data/ala-gly.pqr")

# The recorded answer for this structure at the default grid, so a bench run
# that silently solved something else has something to fail against.
#
# Read from the corpus rather than restated. A literal here is a second copy of
# a number the corpus already owns, and it goes stale in exactly the case that
# matters — a deliberate re-record, where every corpus test moves to the new
# answer and this one is left behind pointing at the old. It was also an
# absolute digit string, and ROADMAP.md section 12 records bit-identity in this
# solver being a per-platform property: the same expression returns
# ...4118 on macOS and ...4138 on linux/amd64.
ALA_GLY_ENERGY = float(
    json.loads(Path("tests/corpus/debye/peptide-molecular.json").read_text())["energy_kj_mol"]
)


def _spec(
    path: Path = ALA_GLY,
    resolution: float = 0.5,
    padding: float = 10.0,
    surface: SurfaceModel = SurfaceModel.MOLECULAR,
    backend: str = "debye",
) -> CaseSpec:
    return CaseSpec(
        path=path, resolution=resolution, padding=padding, surface=surface, backend=backend
    )


def _comparison(baseline: float, candidate: float, *, energies: tuple[float, float]) -> Comparison:
    return Comparison(
        baseline=Measurement(label="baseline", samples=(baseline,)),
        candidate=Measurement(label="candidate", samples=(candidate,)),
        baseline_energy=energies[0],
        candidate_energy=energies[1],
    )


def test_the_report_is_the_minimum_rather_than_the_mean():
    """The least-contaminated sample, which is the whole reason for minimum-of-N."""
    measurement = Measurement(label="x", samples=(9.0, 4.0, 7.0))
    assert measurement.best == 4.0
    assert measurement.best != sum(measurement.samples) / len(measurement.samples)


def test_the_spread_reports_how_contaminated_the_run_was():
    """Worst over best: a spread near one is what makes a small ratio believable."""
    assert Measurement(label="x", samples=(2.0, 4.0)).spread == 2.0
    assert Measurement(label="x", samples=(3.0,)).spread == 1.0


def test_a_ratio_above_one_means_the_candidate_is_faster():
    """The orientation, which is a coin-flip to get wrong and silent when wrong."""
    faster = _comparison(10.0, 5.0, energies=(1.0, 1.0))
    assert faster.ratio == pytest.approx(2.0)
    slower = _comparison(5.0, 10.0, energies=(1.0, 1.0))
    assert slower.ratio == pytest.approx(0.5)


def test_identical_means_bit_identical_and_not_merely_close():
    """The guard this module exists to make unfalsifiable.

    M7's change must leave the energy alone to the last digit, so a tolerance
    here would pass exactly the bug the check is for. One unit in the last place
    is the smallest difference a float can carry, and it has to count.
    """
    energy = ALA_GLY_ENERGY
    nudged = math.nextafter(energy, math.inf)
    assert nudged != energy

    assert _comparison(2.0, 1.0, energies=(energy, energy)).identical
    assert not _comparison(2.0, 1.0, energies=(energy, nudged)).identical


def test_a_ratio_between_two_different_answers_is_reported_as_not_a_speed_up():
    differing = _comparison(2.0, 1.0, energies=(1.0, 2.0))
    rendered = render_comparison(differing)
    assert "ENERGIES DIFFER" in rendered
    assert "not a speed-up" in rendered


def test_the_baseline_snippet_carries_every_field_of_the_case():
    """Both sides solve one case, or the comparison is between two programs.

    Each field is checked by a value that is *not* the default, because a
    snippet that dropped the field entirely would still produce a working
    baseline — one solving the default grid, faster or slower for reasons that
    have nothing to do with the revision.
    """
    snippet = remote_snippet(
        _spec(resolution=0.37, padding=7.5, surface=SurfaceModel.VAN_DER_WAALS)
    )
    assert "0.37" in snippet
    assert "7.5" in snippet
    assert repr(SurfaceModel.VAN_DER_WAALS.value) in snippet
    assert str(ALA_GLY.resolve()) in snippet
    # An absolute path, so the child's own working directory cannot change which
    # file is read.
    assert str(ALA_GLY.resolve()) != str(ALA_GLY)


def test_the_baseline_snippet_is_a_program_rather_than_a_string():
    compile(remote_snippet(_spec()), "<remote>", "exec")


def test_a_missing_structure_fails_before_anything_is_timed():
    """`solve_case` reads the file, so the failure lands at setup rather than in a sample."""
    with pytest.raises(FileNotFoundError):
        solve_case(_spec(path=Path("tests/data/does-not-exist.pqr")))


def test_the_command_names_the_structure_it_could_not_find():
    """The CLI guards the path itself, because a traceback is not a message."""
    with pytest.raises(SystemExit, match="no such structure"):
        main(["bench", "--structure", "tests/data/does-not-exist.pqr"])


def test_reading_the_structure_is_outside_the_timed_closure(tmp_path):
    """Parsing a PQR is not what is being measured, and it contends like everything else.

    Demonstrated by deleting the file the closure was built from, which only a
    closure that had already read it can survive. The copy lives in `tmp_path`
    rather than the fixture being moved aside and restored: a `finally` does not
    run through a `SIGKILL` or a CI timeout, and the first draft of this test
    could leave the checkout missing a tracked fixture — with a symptom in the
    corpus tests pointing nowhere near the cause.
    """
    copied = tmp_path / ALA_GLY.name
    copied.write_bytes(ALA_GLY.read_bytes())
    work = solve_case(_spec(path=copied))
    copied.unlink()
    assert work() == pytest.approx(ALA_GLY_ENERGY)


def test_a_repeat_count_below_one_is_refused(capsys):
    """`measure` guards it and so must the interleaved path, which has its own loop."""
    with pytest.raises(SystemExit, match="at least 1"):
        main(["bench", "--structure", str(ALA_GLY), "--repeats", "0"])


@pytest.mark.skipif(
    not children_counted(), reason="no POSIX `resource`, so cpu_seconds() cannot see children"
)
def test_cpu_seconds_counts_a_reaped_child():
    """The half `time.process_time()` cannot see, and the reason `--backend` works.

    Three of the five backends are subprocesses, so a benchmark without
    `RUSAGE_CHILDREN` reads their cost as this process's bookkeeping and nothing
    else — a wall-clock benchmark wearing a CPU-time label. Checked against a
    child that deliberately burns CPU rather than sleeping, since sleeping would
    pass with or without the fix.

    **Skipped where there is no `resource` module**, which is the platform
    `cpu_seconds` has a documented fallback for. Without the guard this fails on
    Windows — `spent_total` and `spent_here` become the same measurement, the
    difference is 0.0 — and blames `RUSAGE_CHILDREN` on a system that has none.
    `children_counted()` is exported for exactly this question and `cli.py`
    already branches on it.

    **The assertion is on the child's contribution, not on a ratio, and that is
    a correction.** This first read `spent_total > spent_here * 2`, which grades
    the child's work against however much CPU *this* process happened to spend
    inside `subprocess.run` — a fork and an exec, whose cost is the runner's
    business and not this project's. In CI run 32545677587 that reached 0.298 s
    on GitHub's Linux runners while the whole window totalled 0.519 s, and all
    three legs went red on a documentation-only change.

    **The numbers in that failure proved the feature was working**: 0.519 less 0.298
    means the child contributed 0.221 s, which is what the test exists to detect.
    `cpu_seconds()` is `RUSAGE_SELF + RUSAGE_CHILDREN` and `process_time()` is
    the self half alone, so the difference is the child's time and nothing else.

    **Narrower, not stricter** — the old form is harsher whenever the parent
    burns CPU, and deliberately so: at `spent_here = 0.30, spent_total = 0.40`
    the ratio fails where this passes. What it buys is measuring the right
    quantity rather than a proxy for it, so it cannot be reddened by a slow
    fork. It is stricter only than the other assertion it replaces,
    `spent_total > 0.05`, which it implies.

    *One thing this does rest on:* the two clocks are different sources —
    `CLOCK_PROCESS_CPUTIME_ID` against `getrusage`'s tick-derived fields — so
    their disagreement lands in the residual. Measured over five 0.11 s burns
    they differ by at most **4 µs**, four orders of magnitude under the 50 ms
    floor, so the residual is the child's time to well within the margin.
    """
    import subprocess  # noqa: PLC0415 — local to the one test that shells out

    # Read in mirrored order so the two windows nest rather than stagger: the
    # self CPU spent between the reads then falls inside both and cancels.
    before_total, before_process = cpu_seconds(), time.process_time()
    subprocess.run([sys.executable, "-c", "sum(i*i for i in range(4_000_000))"], check=True)
    spent_here = time.process_time() - before_process
    spent_total = cpu_seconds() - before_total
    from_the_child = spent_total - spent_here

    assert from_the_child > 0.05, (
        f"the child contributed {from_the_child:.3f}s of CPU "
        f"(window total {spent_total:.3f}s, this process {spent_here:.3f}s), "
        "so RUSAGE_CHILDREN is not reaching cpu_seconds()"
    )


@pytest.mark.parametrize("backend", ["debye", "gb", "tabipb", "apbs", "delphi"])
def test_bench_times_any_registered_backend(backend: str, capsys):
    """Not only debye, and not only the finite-difference family.

    `tabipb` is here because leaving it out is what let the first draft ship
    broken: `solve_case` threw away the family the registry hands back and built
    a `FiniteDifferenceRequest` for everything, so `--backend tabipb` died on a
    missing `mesh_density` and `--backend gb` silently ignored `--resolution`.
    Both in-process backends happen to accept the wrong request type, so a list
    of only those two could not fail.

    `apbs` and `delphi` are here because they are the whole point: they are the
    subprocess backends `time.process_time()` could not see, and every row of the
    roadmap's ladder is a `--backend apbs` or `--backend delphi` run. A list
    without them would ship a broken finite-difference dispatch green.
    """
    argv = ["bench", "--structure", str(ALA_GLY), "--backend", backend, "--repeats", "1", "--json"]
    # `main` catches SashimiError itself and returns 2 — it does not raise — so a
    # `try/except SashimiError` here would never fire and a machine without the
    # binary would fail rather than skip. The first draft did exactly that.
    code = main(argv)
    if code == 2:
        pytest.skip(f"{backend} is not available here: {capsys.readouterr().err.strip()}")
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["energy"] < 0
    assert report["best"] > 0


def test_the_grid_reaches_a_backend_that_uses_one():
    """`--resolution` must change the answer, or the flag is decoration.

    The dropped-family bug made `gb` ignore the grid entirely — two very
    different resolutions returned an identical energy, so a resolution sweep
    would report flat timings with no sign anything was wrong.
    """
    assert solve_case(_spec(resolution=1.0))() != solve_case(_spec(resolution=0.4))()


def test_a_gridless_backend_is_not_handed_a_grid():
    """The other half: `gb` discretizes nothing, so its answer must *not* move."""
    coarse = solve_case(_spec(backend="gb", resolution=1.0))()
    assert coarse == solve_case(_spec(backend="gb", resolution=0.4))()


def test_the_backend_reaches_the_baseline_snippet_too():
    """A comparison that solved with two different backends would be nonsense."""
    snippet = remote_snippet(_spec(backend="gb"))
    assert "'gb'" in snippet


def test_the_debye_baseline_still_reaches_revisions_before_the_registry():
    """`--against` exists to measure older trees, so it must not need new imports.

    `sashimi.backends` arrived in PR #28 and `System.request_for` in PR #26.
    A snippet that reaches for either dies with an ImportError against exactly
    the revisions it was written to compare against — and debye is what almost
    every comparison is about, so its path stays on `DebyeSolver` and
    `FiniteDifferenceRequest`, which have been there since M1.
    """
    snippet = remote_snippet(_spec(backend="debye"))
    assert "DebyeSolver" in snippet
    assert "FiniteDifferenceRequest" in snippet


def test_the_baseline_snippet_counts_its_own_children():
    """The comparison path had the same blind spot the module was fixing.

    `measure` was converted to `cpu_seconds` and `compare_against` was not, so
    `--against` with a subprocess backend timed ~0.3% of the work: APBS read
    0.0085 s against 2.95 s standalone, and two identical trees compared at
    0.948 instead of ~1.000. Both sides need children counted — this one is the
    far side, which does its own timing so that `uv run`'s resolution and a
    second interpreter's start-up stay out of the sample.
    """
    snippet = remote_snippet(_spec(backend="apbs"))
    assert "RUSAGE_CHILDREN" in snippet
    assert "request_for(family)" in snippet
    # And it resolves the binary before the clock starts, for the same reason
    # `solve_case` does — discovery shells out and hashes the executable — but
    # by touching the cached property rather than running a whole extra solve.
    assert "solver.binary" in snippet
    assert snippet.count("solver.solve(request)") == 1


def test_the_bench_command_solves_the_structure_it_was_given(capsys):
    """End to end, against the energy the corpus records for this structure."""
    assert main(["bench", "--structure", str(ALA_GLY), "--repeats", "1", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["energy"] == pytest.approx(ALA_GLY_ENERGY)
    assert len(report["samples"]) == 1
    assert report["best"] == report["samples"][0]


def test_repeats_is_how_many_samples_are_taken(capsys):
    assert main(["bench", "--structure", str(ALA_GLY), "--repeats", "3", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["samples"]) == 3
    assert report["best"] == min(report["samples"])


@pytest.mark.parametrize("variable", ["PYTHONPATH", "VIRTUAL_ENV"])
def test_the_baseline_cannot_inherit_a_path_to_this_tree(variable: str, monkeypatch):
    """The silent failure this instrument would be worst at reporting.

    `PYTHONPATH` outranks a project's own site-packages, and `uv run` passes it
    to the child — verified against a real subprocess, not assumed. So a
    developer with `PYTHONPATH=$PWD/src` exported, which is ordinary for a
    src-layout checkout, would have the baseline import the *working tree* and
    measure it against itself. It reads as ~1.000x with bit-identical energies,
    which is precisely what a correct no-op change looks like — from the one
    tool whose stated job is refusing to compare two different programs.
    """
    monkeypatch.setenv(variable, "/tmp/definitely-not-the-baseline")
    assert variable not in _baseline_environment()
    # Everything else is still handed through; this is a filter, not a scrub.
    monkeypatch.setenv("SASHIMI_BENCH_CANARY", "kept")
    assert _baseline_environment()["SASHIMI_BENCH_CANARY"] == "kept"


def test_a_baseline_that_is_not_a_checkout_says_so(tmp_path, capsys):
    """A `SashimiError` the CLI turns into a message and exit 2, not a traceback."""
    code = main(
        ["bench", "--structure", str(ALA_GLY), "--repeats", "1", "--against", str(tmp_path)]
    )
    assert code == 2
    assert "does not look like a sashimi checkout" in capsys.readouterr().err
