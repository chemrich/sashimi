"""`sashimi` command line: corpus build and verify.

argparse rather than a CLI framework — two subcommands do not justify a
dependency, and the runtime tree stays at numpy/pydantic/fastmcp.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from sashimi.capabilities import comparable_surface_models
from sashimi.corpus import (
    MANIFEST,
    Case,
    Tolerances,
    build_case,
    load_summary,
    summary_path,
    verify_case,
    write_summary,
)
from sashimi.errors import SashimiError
from sashimi.protocol import FiniteDifferenceRequest, Solver, SurfaceModel
from sashimi.validate import (
    DEFAULT_APPROXIMATION_TOLERANCE,
    DEFAULT_ENERGY_TOLERANCE,
    Backend,
    SolverFamily,
    System,
    validate_system,
)

__all__ = ["main"]

MIN_BACKENDS = 2  # a spread needs two things to spread between

# Backends `--backend` can name. debye registers itself here when it exists;
# the corpus is designed to be its acceptance gate.
BACKENDS: dict[str, Callable[[], Solver[FiniteDifferenceRequest]]] = {}


def _apbs_solver() -> Solver[FiniteDifferenceRequest]:
    from sashimi.apbs import ApbsSolver  # noqa: PLC0415 — keeps `--help` binary-free

    return ApbsSolver()


def _delphi_solver() -> Solver[FiniteDifferenceRequest]:
    from sashimi.delphi import DelphiSolver  # noqa: PLC0415 — keeps `--help` binary-free

    return DelphiSolver()


def _tabipb_solver() -> Solver[Any]:
    from sashimi.tabipb import TabipbSolver  # noqa: PLC0415 — keeps `--help` binary-free

    return TabipbSolver()


BACKENDS["apbs"] = _apbs_solver
BACKENDS["delphi"] = _delphi_solver
BACKENDS["tabipb"] = _tabipb_solver

# Which request family each backend speaks. `Solver` is generic in its request
# type, so a checker already refuses to hand a boundary-element request to an FD
# backend — but that guarantee is static, and dispatch here happens at runtime.
FAMILIES: dict[str, SolverFamily] = {
    "apbs": SolverFamily.FINITE_DIFFERENCE,
    "delphi": SolverFamily.FINITE_DIFFERENCE,
    "tabipb": SolverFamily.BOUNDARY_ELEMENT,
}


def _select(cases: tuple[Case, ...], names: Sequence[str] | None) -> tuple[Case, ...]:
    if not names:
        return cases
    known = {case.name: case for case in cases}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(f"unknown case(s): {', '.join(unknown)}\navailable: {', '.join(known)}")
    return tuple(known[n] for n in names)


def _fd_solver(name: str) -> Solver[FiniteDifferenceRequest]:
    """A corpus backend. The corpus is finite-difference by construction — every
    case records a grid — so a BEM backend cannot build or verify one."""
    if FAMILIES[name] is not SolverFamily.FINITE_DIFFERENCE:
        raise SystemExit(
            f"{name} is a {FAMILIES[name].value} solver, and the corpus records grid "
            "geometry for every case. Use `sashimi validate` to compare it against "
            "the finite-difference backends."
        )
    return BACKENDS[name]()


def _build(args: argparse.Namespace) -> int:
    solver = _fd_solver(args.backend)
    directory = Path(args.directory) if args.directory else None
    cases = _select(MANIFEST, args.case)

    for case in cases:
        path = summary_path(case, directory)
        if path.exists() and not args.force:
            print(f"  skip  {case.name} (exists; pass --force to overwrite)")
            continue
        summary = build_case(solver, case)
        write_summary(summary, path)
        energy = summary["energy_kj_mol"]
        shown = f"{energy:.6f} kJ/mol" if energy is not None else "no energy"
        print(f"  wrote {case.name:<24} {shown}")

    print(f"\n{len(cases)} case(s) against {args.backend}.")
    return 0


def _verify(args: argparse.Namespace) -> int:
    solver = _fd_solver(args.backend)
    directory = Path(args.directory) if args.directory else None
    cases = _select(MANIFEST, args.case)
    tolerances = Tolerances(
        energy_rtol=args.energy_rtol,
        potential_rtol=args.potential_rtol,
    )

    failures: list[str] = []
    for case in cases:
        try:
            recorded = load_summary(case, directory)
        except FileNotFoundError as exc:
            print(f"  MISS  {case.name}: {exc}")
            failures.append(f"{case.name}: no recorded summary")
            continue

        found = verify_case(solver, case, recorded, tolerances)
        if found:
            print(f"  FAIL  {case.name}")
            for item in found:
                print(f"          {item}")
            failures.extend(str(item) for item in found)
        else:
            print(f"  ok    {case.name}")

    if failures:
        print(
            f"\n{len(failures)} discrepancy(ies) against {args.backend}. "
            "A corpus diff is a real result change — investigate before rebuilding."
        )
        return 1

    print(f"\nAll {len(cases)} case(s) reproduce against {args.backend}.")
    return 0


def _pick_surface_model(requested: str | None) -> SurfaceModel:
    """A surface model the installed backends can all actually run.

    Cross-validation has to pick one, because sashimi's default
    (`smoothed-molecular`) is APBS-only and every DelPhi solve at it refuses.
    Choosing is not the same as substituting silently: the choice is printed,
    and an explicit `--surface` that some backend cannot honour fails at solve
    time with that backend's own message rather than being quietly replaced.

    Scoped to *installed* backends rather than the selected subset, which is the
    same set while `BACKENDS` holds two entries. A third backend would make the
    distinction real, and this is where it would be narrowed.
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


def _validate(args: argparse.Namespace) -> int:
    names = args.backend or sorted(BACKENDS)
    unknown = [n for n in names if n not in BACKENDS]
    if unknown:
        raise SystemExit(f"unknown backend(s): {', '.join(unknown)}")
    if len(names) < MIN_BACKENDS:
        raise SystemExit(
            f"cross-validation needs at least two backends, got {len(names)}. "
            "One backend trivially agrees with itself."
        )

    model = _pick_surface_model(args.surface)
    cases = _select(MANIFEST, args.case)
    backends = [Backend(name, BACKENDS[name](), FAMILIES[name]) for name in names]

    families = {b.family.value for b in backends}
    across = f" across {len(families)} solver families" if len(families) > 1 else ""
    print(f"Comparing {', '.join(names)} on the {model.value} surface{across}.\n")

    disagreed: list[str] = []
    incomparable: list[str] = []

    for case in cases:
        system = System(
            structure=case.structure(),
            solvent=dataclasses.replace(case.solvent, surface_model=model),
            grid=case.grid,
            mesh_density=args.mesh_density,
            want_energy=case.compute_energy,
        )
        try:
            comparison = validate_system(
                system,
                backends,
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
        "--backend", choices=sorted(BACKENDS), default="apbs", help="solver to run (default: apbs)"
    )
    common.add_argument("--case", action="append", help="limit to a named case; repeatable")
    common.add_argument("--directory", help="where summaries live (default: tests/corpus)")

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
        choices=sorted(BACKENDS),
        help="backend to include; repeatable (default: all installed)",
    )
    validator.add_argument("--case", action="append", help="limit to a named case; repeatable")
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
