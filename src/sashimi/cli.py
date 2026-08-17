"""`sashimi` command line: corpus build and verify.

argparse rather than a CLI framework — two subcommands do not justify a
dependency, and the runtime tree stays at numpy/pydantic/fastmcp.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sashimi import backends
from sashimi.capabilities import comparable_surface_models, describe_capabilities
from sashimi.corpus import (
    MANIFEST,
    Case,
    CaseTier,
    Tolerances,
    build_case,
    cases_for_tier,
    load_summary,
    summary_path,
    verify_case,
    write_summary,
)
from sashimi.errors import SashimiError, UnsupportedRequest
from sashimi.protocol import Solver, SurfaceModel
from sashimi.validate import (
    DEFAULT_APPROXIMATION_TOLERANCE,
    DEFAULT_ENERGY_TOLERANCE,
    Backend,
    SolverFamily,
    validate_system,
)

__all__ = ["main"]

MIN_BACKENDS = 2  # a spread needs two things to spread between

# Backends come from `sashimi.backends`, which is also what `sashimi_solve` and
# `sashimi_capabilities` read. debye registers itself there when it exists; the
# corpus is designed to be its acceptance gate, and this file needs no edit.
BACKEND_NAMES = backends.names()


def _corpus_solver(name: str) -> tuple[Solver[Any], SolverFamily]:
    """A corpus backend and the dialect to ask it in.

    Any family, since a `Case` is a physical question rather than a grid: the
    summary records whatever the backend returned. What each backend can
    *answer* still varies — Generalized Born takes only the molecular-surface
    cases, TABI-PB only those with four atoms or more — and refusing a case it
    cannot take is the backend's own job, in its own words.
    """
    return backends.solver_for(name)


def _select(
    cases: tuple[Case, ...], names: Sequence[str] | None, *, tier_bounds_names: bool = True
) -> tuple[Case, ...]:
    """The named cases, or every case in the tier.

    A case that exists but sits outside the selected tier is reported as such
    rather than as unknown. Those are different mistakes with different fixes —
    raise `--tier`, versus check the spelling — and calling both "unknown" cost
    a CI round trip: a measurement step named ten real cases, was told they did
    not exist, and reported nothing while looking green. The hint names the tier
    the case is *in*, because sending someone to `--tier full` to reach a
    `standard` case bills them the whole manifest to run one case.

    `tier_bounds_names=False` lets a named case come from outside the tier
    entirely. `validate` needs that: its tier is purely a cost default, and
    narrowing that default must not narrow what a caller is allowed to name.
    """
    if not names:
        return cases
    pool = MANIFEST if not tier_bounds_names else cases
    known = {case.name: case for case in pool}
    tier_of = {case.name: case.tier.value for case in MANIFEST}
    unknown = [n for n in names if n not in known]
    if unknown:
        out_of_tier = [n for n in unknown if n in tier_of]
        misspelt = [n for n in unknown if n not in tier_of]
        problems = []
        if out_of_tier:
            wanted = sorted({tier_of[n] for n in out_of_tier})
            problems.append(
                f"not in the selected tier: {', '.join(out_of_tier)} — "
                f"pass --tier {' or --tier '.join(wanted)}"
            )
        if misspelt:
            problems.append(
                f"unknown case(s): {', '.join(misspelt)}\navailable: {', '.join(known)}"
            )
        raise SystemExit("\n".join(problems))
    return tuple(known[n] for n in names)


def _refuses(backend: str, case: Case) -> bool:
    """Whether this backend declines the case by design rather than lacking a recording.

    Read from the backend's own `BackendReport` — the same `surface_models` and
    preconditions `sashimi_capabilities` publishes to callers — rather than from
    a table here. That matters twice over: a second copy of "which backend
    supports which surface" is the guards file's recurring failure mode, and
    because this is the *published* field, a backend that misreports it
    misreports it to every caller and not only to `verify`.

    **Conservative when it cannot tell.** An undiscoverable DelPhi reports no
    surface models at all, since which ones it has depends on the flavour found,
    so an unavailable backend falls through to the ordinary missing-recording
    path. Calling those refusals would turn "DelPhi is not installed" into
    "DelPhi does not support anything", which is the failure
    `test_every_backend_can_answer_the_default_surface_model` was written after.
    """
    entry = backends.get(backend)
    report = entry.report()
    if not report.available or not report.surface_models:
        return False
    if case.solvent.surface_model.value not in report.surface_models:
        return True
    return bool(entry.check(case.system(want_potential=False)))


def _build(args: argparse.Namespace) -> int:
    """Record a backend's answers, skipping the questions it refuses to be asked.

    **A refusal is a result, not a crash.** No backend answers the whole corpus:
    Generalized Born takes only the molecular surface, TABI-PB needs four atoms,
    and debye builds the two sharp boundaries and declines APBS's harmonic
    averaging and DelPhi's Gaussian by name. Before M5 this loop had no
    `except`, so recording a partial-coverage backend meant naming its cases by
    hand and the first refusal killed the run — which is also why the earlier
    tiers were recorded case by case.
    """
    solver, family = _corpus_solver(args.backend)
    directory = Path(args.directory) if args.directory else None
    cases = _select(cases_for_tier(CaseTier(args.tier)), args.case)

    written = refused = skipped = 0
    for case in cases:
        path = summary_path(case, directory, args.backend)
        if path.exists() and not args.force:
            print(f"  skip  {case.name} (exists; pass --force to overwrite)")
            skipped += 1
            continue
        try:
            summary = build_case(solver, case, family)
        except UnsupportedRequest as exc:
            print(f"  n/a   {case.name}: {exc}")
            refused += 1
            continue
        write_summary(summary, path)
        energy = summary["energy_kj_mol"]
        shown = f"{energy:.6f} kJ/mol" if energy is not None else "no energy"
        print(f"  wrote {case.name:<24} {shown}")
        written += 1

    # Every case is accounted for. Reporting only `written` and `refused` made a
    # fully-recorded rerun print "0 written" with 23 cases unmentioned, which
    # reads as nothing having worked.
    parts = [f"{written} written"]
    if skipped:
        parts.append(f"{skipped} already recorded")
    if refused:
        parts.append(f"{refused} refused by the backend")
    print(f"\n{len(cases)} case(s) against {args.backend}: {', '.join(parts)}.")
    return 0


def _verify(args: argparse.Namespace) -> int:
    solver, family = _corpus_solver(args.backend)
    directory = Path(args.directory) if args.directory else None
    cases = _select(cases_for_tier(CaseTier(args.tier)), args.case)
    tolerances = Tolerances(
        energy_rtol=args.energy_rtol,
        potential_rtol=args.potential_rtol,
    )

    failures: list[str] = []
    checked = refused = 0
    for case in cases:
        try:
            recorded = load_summary(case, directory, args.backend)
        except FileNotFoundError as exc:
            # A backend that *refuses* the case has nothing to reproduce, and
            # calling that a discrepancy makes a partial-coverage backend
            # permanently red — which is what made M5's stated exit criterion
            # unreachable for a solver that declines two surface models on
            # purpose. Asked rather than looked up, so this cannot drift from
            # what the backend actually does.
            if _refuses(args.backend, case):
                print(f"  n/a   {case.name} (this backend does not build that surface)")
                refused += 1
                continue
            print(f"  MISS  {case.name}: {exc}")
            failures.append(f"{case.name}: no recorded summary")
            continue

        try:
            found = verify_case(solver, case, recorded, tolerances, family)
        except UnsupportedRequest as exc:
            # A recording exists for a case the backend now refuses. `_refuses`
            # never sees this, because it is only consulted when the recording
            # is *missing* — so without this the first such case escapes the
            # loop and kills the whole run, where `build` would have carried on.
            # Reachable today through `--directory`, and by any backend that
            # narrows `SUPPORTED_SURFACES` with recordings already on disk.
            print(f"  n/a   {case.name}: {exc}")
            refused += 1
            continue

        if found:
            print(f"  FAIL  {case.name}")
            for item in found:
                print(f"          {item}")
            failures.extend(str(item) for item in found)
        else:
            print(f"  ok    {case.name}")
            checked += 1

    if failures:
        print(
            f"\n{len(failures)} discrepancy(ies) against {args.backend}. "
            "A corpus diff is a real result change — investigate before rebuilding."
        )
        return 1

    if cases and not checked:
        # Verifying nothing is not passing. Every selected case was refused, so
        # this would otherwise print "All 0 case(s) reproduce" and exit 0 — and
        # since refusal is read from the backend's own published
        # `surface_models`, a backend that regressed to reporting an empty or
        # wrong list would turn its entire acceptance gate green by the same
        # mechanism that makes a principled refusal work. That is the exact
        # shape of guard this project keeps finding: one that cannot fail.
        print(
            f"\nNothing verified: all {refused} selected case(s) were refused by "
            f"{args.backend}. Either the selection contains only surfaces it does "
            "not build, or the backend is misreporting what it supports."
        )
        return 1

    tail = f" ({refused} refused by the backend)" if refused else ""
    print(f"\nAll {checked} case(s) reproduce against {args.backend}{tail}.")
    return 0


def _pick_surface_model(requested: str | None) -> SurfaceModel:
    """A surface model the installed backends can all actually run.

    Cross-validation has to pick one rather than take the protocol default,
    because what every *installed* backend shares is a property of the machine:
    `molecular` is the default precisely because all four shipped backends
    answer on it, but the comparison has to hold when the set is something else.
    Choosing is not the same as substituting silently: the choice is printed,
    and an explicit `--surface` that some backend cannot honour fails at solve
    time with that backend's own message rather than being quietly replaced.

    Scoped to *installed* backends rather than the selected subset, which is
    where it would be narrowed if that distinction ever mattered.
    """
    if requested is not None:
        return SurfaceModel(requested)

    shared = comparable_surface_models()
    if not shared:
        raise SystemExit(
            "The installed backends share no surface model, so no comparison is "
            "possible. `sashimi capabilities` reports what each one supports; "
            "`--surface` overrides this choice."
        )
    # Prefer the solvent-excluded surface: it is the one every shipped backend
    # supports and the one most published numbers use.
    for preferred in (SurfaceModel.MOLECULAR, SurfaceModel.VAN_DER_WAALS):
        if preferred.value in shared:
            return preferred
    return SurfaceModel(shared[0])


def _backends_supporting(
    names: list[str], model: SurfaceModel, *, explicit: bool
) -> tuple[list[str], dict[str, str]]:
    """Drop backends that cannot answer on this surface, and say which and why.

    Backends no longer support the same surfaces — Generalized Born answers only
    on `molecular`, TABI-PB cannot mesh `van-der-waals`, `smoothed-molecular` is
    APBS's alone. Handing a backend a surface it refuses makes every case report
    as incomparable, which names neither the backend nor the reason.

    **Not installed and does not support are different reasons**, and they used
    to print the same sentence: an absent backend reports an empty
    `surface_models`, which is indistinguishable from genuine non-support unless
    `available` is read too. So a machine without DelPhi was told "excluding
    delphi: does not support the molecular surface" — false, and the sort of
    false that stops someone installing the project's primary cross-validation
    partner.

    A backend the caller named explicitly is kept, so `--backend gb --surface
    van-der-waals` fails with GB's own message rather than being silently
    dropped: asking for something impossible should say so.
    """
    if explicit:
        return names, {}

    reports = {report["name"]: report for report in describe_capabilities()["backends"]}
    kept, excluded = [], {}
    for name in names:
        report = reports.get(name)
        if report is None or model.value in report["surface_models"]:
            kept.append(name)
        elif not report["available"]:
            excluded[name] = f"not installed — {report['detail'].splitlines()[0]}"
        else:
            excluded[name] = f"does not support the {model.value} surface"
    return kept, excluded


def _validate(args: argparse.Namespace) -> int:
    # Installed, not registered. Defaulting to every *registered* backend put
    # TABI-PB in the comparison on a machine that has never built it: it is
    # unavailable but still advertises `molecular`, so the surface filter kept
    # it, `solver_for` constructed it lazily without complaint, and every case
    # then died on `TabipbNotFound`. On the README's own install — conda-forge
    # APBS and nothing else — `sashimi validate` skipped all 64 cases and exited
    # 1, while apbs against gb was a perfectly good comparison sitting right
    # there. Same "what would a machine with only APBS do" bug this branch fixed
    # in the cross-validation tests.
    names = args.backend or backends.available_names()
    unknown = [n for n in names if n not in BACKEND_NAMES]
    if unknown:
        raise SystemExit(f"unknown backend(s): {', '.join(unknown)}")
    if len(names) < MIN_BACKENDS:
        installed = ", ".join(backends.available_names()) or "none"
        raise SystemExit(
            f"cross-validation needs at least two backends, got {len(names)}. "
            f"One backend trivially agrees with itself. Installed here: {installed}. "
            "`sashimi capabilities` reports what each one needs."
        )

    model = _pick_surface_model(args.surface)
    # Tier-aware since M5. It was always the whole manifest, which was tolerable
    # while every installed backend was either a fast binary or `gb`; debye is a
    # reference-tier solver running in this process, so the giants cost minutes
    # apiece and a caller needs a way to say "not those" short of naming cases.
    #
    # The default is `fast` since 2026-08-16. `full` was measured at over 40
    # minutes without finishing on a fully-installed machine — and that was true
    # before debye registered, so it is not one backend's cost. Three things
    # compound: the tier runs every case rather than the ones each backend
    # recorded, the surface override below asks every backend all of them, and
    # the per-case cost is the slowest backend's. A default nobody can wait out
    # is a trap rather than a default; exhaustive protein-scale verification is
    # the corpus's job, where the answers are recorded. `--tier full` is one flag.
    #
    # `--case` is deliberately not bounded by that tier. The first cut of this
    # change narrowed both at once, which silently broke `sashimi validate
    # --case lysozyme-molecular` — a command that worked before it — for the 58
    # cases outside `fast`. Lowering a cost default must not remove reach.
    cases = _select(cases_for_tier(CaseTier(args.tier)), args.case, tier_bounds_names=False)
    names, excluded = _backends_supporting(names, model, explicit=bool(args.backend))
    if len(names) < MIN_BACKENDS:
        raise SystemExit(
            f"only {len(names)} of the selected backends support the "
            f"{model.value} surface. `sashimi capabilities` reports what each "
            "one supports; `--surface` chooses a different one."
        )
    selected = [Backend(name, *backends.solver_for(name)) for name in names]

    families = {b.family.value for b in selected}
    across = f" across {len(families)} solver families" if len(families) > 1 else ""
    print(f"Comparing {', '.join(names)} on the {model.value} surface{across}.")
    for name, reason in excluded.items():
        # An excluded *backend* is not a skipped *case*: without this the run
        # reports every case as incomparable and never says which backend did it.
        print(f"  excluding {name}: {reason}")
    print()

    disagreed: list[str] = []
    incomparable: list[str] = []

    for case in cases:
        # `Case.system()` is the seam; this only overrides the two things a
        # cross-solver run has to choose for itself — the shared surface model,
        # and a mesh density the corpus has no opinion about. Potentials are off
        # because a volume and a triangulated surface have nothing to compare.
        system = dataclasses.replace(
            case.system(),
            solvent=dataclasses.replace(case.solvent, surface_model=model),
            mesh_density=args.mesh_density,
            want_potential=False,
        )
        try:
            comparison = validate_system(
                system,
                selected,
                tolerance=args.tolerance,
                approximation_tolerance=args.approximation_tolerance,
                allow_mismatch=args.allow_mismatched,
            )
        except SashimiError as exc:
            print(f"  SKIP  {case.name}: {exc}\n")
            incomparable.append(case.name)
            continue

        marker = "ok   " if comparison.agrees else "DIFF "
        print(f"  {marker} {case.name:<24} {comparison.summary()}")
        for run in comparison.runs:
            energy = (
                f"{run.energy_kj_mol:12.3f}" if run.energy_kj_mol is not None else "          -"
            )
            deviation = comparison.approximation_deviation.get(run.name)
            # Named on the row it belongs to: an approximation's distance from the
            # reference is not part of the spread and must not read as if it were.
            tier = f"  [{run.accuracy_tier}, {deviation:.2%} from reference]" if deviation else ""
            print(f"          {run.name:<10} {energy} kJ/mol  ({run.energy_term}){tier}")
        for note in comparison.notes:
            print(f"          note: {note}")
        if not comparison.agrees:
            disagreed.append(case.name)

    compared = len(cases) - len(incomparable)
    print()
    if incomparable:
        # Not a failure on its own: refusing to compare incomparable things is
        # this tool working, not this tool breaking.
        print(f"{len(incomparable)} case(s) could not be compared: {', '.join(incomparable)}")
    if disagreed:
        print(
            f"{len(disagreed)} of {compared} compared case(s) disagreed: "
            f"{', '.join(disagreed)}. A spread beyond discretization is an "
            "input-generation bug or real parameter sensitivity — both worth knowing."
        )
        return 1
    if not compared:
        print("Nothing was comparable, so nothing was validated.")
        return 1
    print(f"All {compared} compared case(s) agree across {len(names)} backends.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sashimi", description=__doc__)
    subcommands = parser.add_subparsers(dest="group", required=True)

    corpus = subcommands.add_parser("corpus", help="golden-corpus operations")
    actions = corpus.add_subparsers(dest="action", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--backend",
        choices=sorted(BACKEND_NAMES),
        default="apbs",
        help="solver to run (default: apbs)",
    )
    common.add_argument("--case", action="append", help="limit to a named case; repeatable")
    common.add_argument("--directory", help="where summaries live (default: tests/corpus)")
    common.add_argument(
        "--tier",
        choices=[t.value for t in CaseTier],
        default=CaseTier.STANDARD.value,
        help=(
            "how much of the corpus to run; cumulative "
            f"(default: {CaseTier.STANDARD.value}, which is what CI runs per push)"
        ),
    )

    builder = actions.add_parser("build", parents=[common], help="record summaries")
    builder.add_argument("--force", action="store_true", help="overwrite existing summaries")
    builder.set_defaults(func=_build)

    verifier = actions.add_parser("verify", parents=[common], help="re-run and diff")
    verifier.add_argument("--energy-rtol", type=float, default=Tolerances().energy_rtol)
    verifier.add_argument("--potential-rtol", type=float, default=Tolerances().potential_rtol)
    verifier.set_defaults(func=_verify)

    validator = subcommands.add_parser(
        "validate",
        help="compare backends against each other on the same system",
        description=(
            "Run one system through several backends and report the spread. "
            "Refuses to compare across differing surface models, energy terms or "
            "equations, because such a spread is a modelling difference reported "
            "as a solver disagreement."
        ),
    )
    validator.add_argument(
        "--backend",
        action="append",
        choices=sorted(BACKEND_NAMES),
        help="backend to include; repeatable (default: all installed)",
    )
    validator.add_argument("--case", action="append", help="limit to a named case; repeatable")
    validator.add_argument(
        "--tier",
        choices=[t.value for t in CaseTier],
        default=CaseTier.FAST.value,
        help=(
            "how much of the corpus to compare; cumulative (default: fast). "
            "Cost scales with the slowest backend selected, so an in-process "
            "solver on the 8,279-atom cases is minutes each — `--tier full` is "
            "every case, and on a fully-installed machine that is not minutes"
        ),
    )
    validator.add_argument(
        "--surface",
        choices=sorted(m.value for m in SurfaceModel),
        help="surface model to compare on (default: one the backends share)",
    )
    validator.add_argument(
        "--mesh-density",
        type=float,
        default=2.0,
        help=(
            "vertices per square angstrom for boundary-element backends "
            "(default: 2.0; below 1.5 TABI-PB will not solve)"
        ),
    )
    validator.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_ENERGY_TOLERANCE,
        help=f"relative energy spread treated as agreement (default: {DEFAULT_ENERGY_TOLERANCE})",
    )
    validator.add_argument(
        "--approximation-tolerance",
        type=float,
        default=DEFAULT_APPROXIMATION_TOLERANCE,
        help=(
            "how far an approximate backend may sit from the reference consensus "
            f"before it is treated as broken (default: {DEFAULT_APPROXIMATION_TOLERANCE})"
        ),
    )
    validator.add_argument(
        "--allow-mismatched",
        action="store_true",
        help="report a spread even across differing surface models or energy terms",
    )
    validator.set_defaults(func=_validate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result: int = args.func(args)
    except SashimiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return result


if __name__ == "__main__":
    raise SystemExit(main())
