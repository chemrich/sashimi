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
from pathlib import Path

import pytest

from sashimi.bench import (
    CaseSpec,
    Comparison,
    Measurement,
    remote_snippet,
    render_comparison,
    solve_case,
)
from sashimi.cli import main
from sashimi.protocol import SurfaceModel

ALA_GLY = Path("tests/data/ala-gly.pqr")

# The recorded answer for this structure at the default grid, which is what the
# corpus holds. Repeated here so a bench run that silently solved something else
# has something to fail against.
ALA_GLY_ENERGY = -218.62772042354118


def _spec(
    path: Path = ALA_GLY,
    resolution: float = 0.5,
    padding: float = 10.0,
    surface: SurfaceModel = SurfaceModel.MOLECULAR,
) -> CaseSpec:
    return CaseSpec(path=path, resolution=resolution, padding=padding, surface=surface)


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


def test_reading_the_structure_is_outside_the_timed_closure():
    """Parsing a PQR is not what is being measured, and it contends like everything else."""
    work = solve_case(_spec())
    ALA_GLY.rename(moved := ALA_GLY.with_suffix(".pqr.moved"))
    try:
        assert work() == pytest.approx(ALA_GLY_ENERGY)
    finally:
        moved.rename(ALA_GLY)


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


def test_a_baseline_that_is_not_a_checkout_says_so(tmp_path, capsys):
    """A `SashimiError` the CLI turns into a message and exit 2, not a traceback."""
    code = main(
        ["bench", "--structure", str(ALA_GLY), "--repeats", "1", "--against", str(tmp_path)]
    )
    assert code == 2
    assert "does not look like a sashimi checkout" in capsys.readouterr().err
