"""The instrument M7's claim is measured on: CPU time, minimum of N, interleaved.

**Why this is a module rather than a stopwatch.** ROADMAP.md section 11 deferred
a benchmark VM until debye made a performance claim against APBS, and M7's
groundwork found that the VM was not what was missing — the *instrument* was.
Interleaved on one machine, alternating between two revisions of the same case,
identical code read 61.8 / 79.1 / 42.5 s against 52.2 / 44.6 / 60.5 s: a 1.9x
spread with the ranges overlapping completely, which is larger than anything M7
would claim. That is contention rather than architecture, and a VM on the same
host contends the same way.

Three choices follow from it, and each is the whole reason a line below exists:

- **CPU time, not wall clock.** Other processes take wall-clock time away from a
  run; they mostly do not change how many CPU-seconds the work costs — memory
  and cache contention do inflate it, so this is a large improvement rather than
  an immunity. On the same pair that read as a 40% regression on wall clock, CPU
  time read 44.96 s against 42.04 s — the 6.5% that was really there. Measured
  across two runs at load averages 4.7 and 5.9, CPU samples of the same solve
  agree to under 2.5% where wall clock spread 1.4-2.7x.
- **Children counted, because three of the five backends are subprocesses (apbs, delphi, tabipb).**
  `time.process_time()` excludes them by definition, so timing APBS or DelPhi
  with it alone silently measures nothing but this process's bookkeeping and
  falls back to wall clock in practice. `resource.getrusage(RUSAGE_CHILDREN)`
  supplies the rest — see `cpu_seconds`.
- **Minimum of N, not a mean.** The minimum is the least-contaminated sample. A
  mean averages the contamination in, and a machine under variable load has no
  central tendency worth reporting.
- **Interleaved, not batched.** `ABABAB` rather than `AAABBB`, so a drift in
  machine state over the run lands on both sides instead of on whichever went
  second. This is what "measured back to back" has to mean to be worth anything.

**The energy is part of the measurement, not a separate check.** A solver can be
made arbitrarily fast by computing something else, so every comparison here
reports both revisions' energies to full precision and refuses to call a ratio a
speed-up when they differ. M7's own change — batching the rim loop — is one
whose answer must come out *bit*-identical, so `--against` compares the repr
rather than a tolerance.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from sashimi import backends
from sashimi.errors import SashimiError
from sashimi.pqr import read_pqr
from sashimi.protocol import GridSpec, SolventModel, SurfaceModel, System

__all__ = [
    "CaseSpec",
    "Comparison",
    "Measurement",
    "children_counted",
    "compare_against",
    "cpu_seconds",
    "measure",
    "remote_snippet",
    "render",
    "render_comparison",
    "solve_case",
]

# Enough samples for a minimum to mean something, few enough that a
# protein-scale solve is still an edit-test loop. Three is what the groundwork
# used; the flag exists because the right number depends on the load.
DEFAULT_REPEATS = 3

# A baseline solve that has not finished in this long has hung rather than been
# slow: the whole corpus at protein scale is minutes, and an interleaved run
# blocked forever is indistinguishable from the instrument being broken.
BASELINE_TIMEOUT = 3600


@cache
def _rusage() -> Any:
    """The `resource` module, or None where there is not one.

    POSIX-only, and `cli.py` imports this module unconditionally — so a
    module-level `import resource` makes the whole `sashimi` command
    unimportable on Windows, for a package whose entire proposition is that it
    installs anywhere. Imported here instead, once, with `cpu_seconds` falling
    back to `process_time()` when it is absent. That fallback is *wrong* for
    subprocess backends, which is exactly why it says so out loud.
    """
    try:
        import resource  # noqa: PLC0415 — POSIX-only, so it cannot be top-level
    except ImportError:  # pragma: no cover - exercised only off POSIX
        return None
    return resource


def children_counted() -> bool:
    """Whether `cpu_seconds` can see subprocess backends on this platform."""
    return _rusage() is not None


def cpu_seconds() -> float:
    """This process's CPU time plus every child it has reaped.

    The half `time.process_time()` cannot see. APBS, DelPhi C++ and pyDelPhi are
    all subprocesses, so a benchmark that omits `RUSAGE_CHILDREN` is a
    wall-clock benchmark wearing a CPU-time label — which is what a first pass
    at the cross-backend baseline turned out to be, reading the same APBS case
    at 13.8 s and 32.9 s in two runs.

    **Two preconditions, because the counter is process-wide.** A child only
    contributes once it has been *reaped*, so a delta taken before the wait
    returns reads zero; and any other child reaped inside the window is counted
    too, so this is not safe across concurrent subprocess work. Both hold here:
    the solvers run one binary at a time and `subprocess.run` waits.
    """
    module = _rusage()
    if module is None:  # pragma: no cover - exercised only off POSIX
        return time.process_time()
    here = module.getrusage(module.RUSAGE_SELF)
    reaped = module.getrusage(module.RUSAGE_CHILDREN)
    return float(here.ru_utime + here.ru_stime + reaped.ru_utime + reaped.ru_stime)


@dataclass(frozen=True)
class CaseSpec:
    """What to solve, in the terms both sides of a comparison have to agree on.

    One object rather than four arguments threaded twice, because the baseline
    runs in another process and every field has to reach it verbatim: a
    comparison where the two sides solved different grids is worse than no
    comparison, and it would not look wrong in the output.
    """

    path: Path
    resolution: float
    padding: float
    surface: SurfaceModel
    backend: str = "debye"


@dataclass(frozen=True)
class Measurement:
    """CPU seconds for one variant, kept as every sample rather than a summary.

    The samples are carried rather than reduced because the *spread* is the
    evidence that the minimum is worth trusting: two runs whose minima differ by
    5% and whose spreads are 90% have not measured anything, and only the raw
    samples can say so.
    """

    label: str
    samples: tuple[float, ...]

    @property
    def best(self) -> float:
        return min(self.samples)

    @property
    def spread(self) -> float:
        """Worst sample over best, as a ratio: how contaminated the run was."""
        return max(self.samples) / min(self.samples)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "samples": list(self.samples),
            "best": self.best,
            "spread": self.spread,
        }


@dataclass(frozen=True)
class Comparison:
    """Two revisions on one machine, and whether they computed the same thing."""

    baseline: Measurement
    candidate: Measurement
    baseline_energy: float
    candidate_energy: float

    @property
    def ratio(self) -> float:
        """Baseline over candidate: above one is the candidate being faster."""
        return self.baseline.best / self.candidate.best

    @property
    def identical(self) -> bool:
        """Bit-identical energies, which is the bar an answer-preserving change meets."""
        return repr(self.baseline_energy) == repr(self.candidate_energy)

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.as_dict(),
            "candidate": self.candidate.as_dict(),
            "baseline_energy": repr(self.baseline_energy),
            "candidate_energy": repr(self.candidate_energy),
            "ratio": self.ratio,
            "identical": self.identical,
        }


def measure[T](work: Callable[[], T], *, repeats: int, label: str) -> tuple[Measurement, T]:
    """Run `work` `repeats` times, keeping every CPU sample and the last result.

    `cpu_seconds` counts this process and its reaped children, so a subprocess
    backend is measured rather than silently timed at zero. It excludes time the
    scheduler gave to something else, which is the entire point.

    **`--against` still has the far side report its own number, and not because
    `RUSAGE_CHILDREN` cannot see it.** It can: the other checkout runs under
    `subprocess.run`, which waits, so it is a reaped descendant like any other.
    The reason is that measuring it from here would charge `uv run`'s dependency
    resolution and a second interpreter's start-up into the sample. An earlier
    draft of this docstring gave the wrong reason, and that is how the
    comparison path kept `process_time()` through the very change that added
    `cpu_seconds` — a wrong *why* is what let a wrong *what* survive review.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    samples = []
    result: Any = None
    for _ in range(repeats):
        started = cpu_seconds()
        result = work()
        samples.append(cpu_seconds() - started)
    return Measurement(label=label, samples=tuple(samples)), result


def solve_case(spec: CaseSpec) -> Callable[[], float]:
    """A no-argument solve of one structure by one backend, returning its energy.

    The structure is read once, outside the returned closure: parsing a PQR is
    not what is being measured, and leaving it inside would put a few
    milliseconds of file I/O — which contends with everything else on the
    machine — inside every sample.

    **Any registered backend, not only debye.** The instrument was written for
    "did this change to debye help", and the cross-backend baseline in
    ROADMAP.md section 12 then had to be produced by a script that lived nowhere
    — twenty CPU numbers with no way to re-run them, which is the failure this
    module's own docstring says it exists to prevent.
    """
    solver, family = backends.solver_for(spec.backend)
    # `request_for`, not a hardcoded `FiniteDifferenceRequest`. The registry
    # returns the family for exactly this reason and the first draft threw it
    # away, which made `--backend tabipb` die on a missing `mesh_density` and
    # made `--backend gb` silently ignore `--resolution` — a resolution sweep
    # that reported flat timings and identical energies with no warning.
    request = System(
        structure=read_pqr(spec.path),
        solvent=SolventModel(surface_model=spec.surface),
        grid=GridSpec(resolution=spec.resolution, padding=spec.padding),
        want_energy=True,
        want_potential=False,
    ).request_for(family)

    # Binary discovery is lazy — `ApbsSolver.binary` shells out for a version
    # and hashes the executable on first use — and with children now counted
    # that lands in sample one. Pay it here, where the PQR parsing already is.
    solver.solve(request)

    def work() -> float:
        energy = solver.solve(request).energy_kj_mol
        if energy is None:  # pragma: no cover - `want_energy` defaults to True
            raise SashimiError("the solve returned no energy, so there is nothing to check")
        return float(energy)

    return work


def compare_against(
    checkout: Path,
    spec: CaseSpec,
    local: Callable[[], float],
    *,
    repeats: int,
) -> Comparison:
    """Interleave this tree against another checkout of the repo, one sample each.

    The other side runs as a subprocess because a revision cannot be imported
    twice into one interpreter — which is also why its CPU time cannot be read
    from here. `time.process_time()` in the child excludes the parent's work and
    the interpreter's own start-up is charged before the timer starts, so the
    child reporting its own sample is both simpler and more accurate than
    `getrusage(RUSAGE_CHILDREN)` around the call.

    The alternation is `candidate` then `baseline` within each repeat, so the
    two sides sit adjacent in time. Which goes first inside a repeat does not
    matter; that they are never blocked apart does.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    here: list[float] = []
    there: list[float] = []
    energy_here = 0.0
    energy_there = 0.0
    for _ in range(repeats):
        started = cpu_seconds()
        energy_here = local()
        here.append(cpu_seconds() - started)
        sample, energy_there = _remote_sample(checkout, spec)
        there.append(sample)
    return Comparison(
        baseline=Measurement(label=str(checkout), samples=tuple(there)),
        candidate=Measurement(label="working tree", samples=tuple(here)),
        baseline_energy=energy_there,
        candidate_energy=energy_here,
    )


# The baseline side, as a snippet run inside the other checkout rather than as a
# call to its own `sashimi bench`. This matters: the instrument landed in M7, so
# requiring the baseline to contain it would make every revision before M7
# unmeasurable — including the `main` this milestone has to be graded against.
# What the snippet needs from the other revision is `DebyeSolver` and the
# protocol types, which have been there since M1.
_REMOTE = """
import json, time
try:
    import resource
except ImportError:
    resource = None
from sashimi.backends import solver_for
from sashimi.pqr import read_pqr
from sashimi.protocol import GridSpec, SolventModel, SurfaceModel, System


def cpu():
    if resource is None:
        return time.process_time()
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


solver, family = solver_for({backend!r})
request = System(
    structure=read_pqr({structure!r}),
    solvent=SolventModel(surface_model=SurfaceModel({surface!r})),
    grid=GridSpec(resolution={resolution!r}, padding={padding!r}),
    want_energy=True,
    want_potential=False,
).request_for(family)
solver.solve(request)
started = cpu()
energy = solver.solve(request).energy_kj_mol
print(json.dumps({{"best": cpu() - started, "energy": energy}}))
"""


def remote_snippet(spec: CaseSpec) -> str:
    """The baseline-side program, kept public so a test can read what will run."""
    return _REMOTE.format(
        structure=str(spec.path.resolve()),
        surface=spec.surface.value,
        resolution=spec.resolution,
        padding=spec.padding,
        backend=spec.backend,
    )


def _remote_sample(checkout: Path, spec: CaseSpec) -> tuple[float, float]:
    """One CPU sample from another checkout, measured inside its own interpreter.

    `uv run --project` rather than this interpreter: the other checkout has its
    own lockfile, and the point of measuring it at all is that it is a different
    revision. Its CPU time cannot be read from here — `time.process_time()`
    excludes children — so the child times itself, which also keeps interpreter
    start-up out of the sample.
    """
    if not (checkout / "pyproject.toml").is_file():
        raise SashimiError(f"{checkout} does not look like a sashimi checkout (no pyproject.toml)")
    command = ["uv", "run", "--project", str(checkout), "python", "-c", remote_snippet(spec)]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=checkout,
        env=_baseline_environment(),
        timeout=BASELINE_TIMEOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise SashimiError(
            f"the baseline checkout at {checkout} failed to solve:\n"
            f"{completed.stderr.strip()[-2000:]}"
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SashimiError(
            f"the baseline checkout printed something other than its measurement.\n"
            f"stdout: {completed.stdout.strip()[:400]}"
        ) from exc
    if report["energy"] is None:  # pragma: no cover - the request asks for energy
        raise SashimiError("the baseline checkout returned no energy")
    return float(report["best"]), float(report["energy"])


def _baseline_environment() -> dict[str, str]:
    """The parent environment with anything that could import *this* tree removed.

    `PYTHONPATH` takes precedence over a project's own site-packages, so a
    developer with `PYTHONPATH=$PWD/src` exported — ordinary for a src-layout
    checkout — would have the baseline subprocess import the working tree and
    measure it against itself. That failure is silent and it is the worst one
    available here: it reads as ~1.000x with bit-identical energies, which is
    exactly what a correct no-op change looks like, reported by the one tool
    whose whole job is refusing to compare two different programs.

    `VIRTUAL_ENV` goes for the same reason: `uv run` will warn and prefer the
    project's environment, but leaving the parent's active is asking the child
    to resolve a conflict it should never see.
    """
    return {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}


def render(measurement: Measurement, energy: float) -> str:
    """One revision's result, for a human."""
    samples = " / ".join(f"{s:.2f}" for s in measurement.samples)
    return (
        f"{measurement.best:.2f} s CPU (best of {len(measurement.samples)}: {samples}; "
        f"spread {measurement.spread:.2f}x)\nenergy {energy!r} kJ/mol"
    )


def render_comparison(comparison: Comparison) -> str:
    """Both revisions, the ratio, and whether the ratio means anything."""
    lines = [
        (
            f"baseline  {comparison.baseline.best:.2f} s CPU  "
            f"(spread {comparison.baseline.spread:.2f}x)  {comparison.baseline.label}"
        ),
        (
            f"candidate {comparison.candidate.best:.2f} s CPU  "
            f"(spread {comparison.candidate.spread:.2f}x)  {comparison.candidate.label}"
        ),
        "",
        (
            f"ratio {comparison.ratio:.3f}x on CPU time, minimum of "
            f"{len(comparison.candidate.samples)}, interleaved"
        ),
    ]
    if comparison.identical:
        lines.append(f"energies bit-identical: {comparison.candidate_energy!r} kJ/mol")
    else:
        # Loud, and a non-zero exit, because a ratio between two different
        # answers is not a speed-up — it is two programs.
        lines.append(
            "ENERGIES DIFFER, so this ratio is not a speed-up:\n"
            f"  baseline  {comparison.baseline_energy!r}\n"
            f"  candidate {comparison.candidate_energy!r}"
        )
    return "\n".join(lines)
