"""`sashimi` command line: corpus build and verify.

argparse rather than a CLI framework — two subcommands do not justify a
dependency, and the runtime tree stays at numpy/pydantic/fastmcp.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

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
from sashimi.protocol import FiniteDifferenceRequest, Solver

__all__ = ["main"]

# Backends `--backend` can name. debye registers itself here when it exists;
# the corpus is designed to be its acceptance gate.
BACKENDS: dict[str, Callable[[], Solver[FiniteDifferenceRequest]]] = {}


def _apbs_solver() -> Solver[FiniteDifferenceRequest]:
    from sashimi.apbs import ApbsSolver  # noqa: PLC0415 — keeps `--help` binary-free

    return ApbsSolver()


BACKENDS["apbs"] = _apbs_solver


def _select(cases: tuple[Case, ...], names: Sequence[str] | None) -> tuple[Case, ...]:
    if not names:
        return cases
    known = {case.name: case for case in cases}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(f"unknown case(s): {', '.join(unknown)}\navailable: {', '.join(known)}")
    return tuple(known[n] for n in names)


def _build(args: argparse.Namespace) -> int:
    solver = BACKENDS[args.backend]()
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
    solver = BACKENDS[args.backend]()
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
