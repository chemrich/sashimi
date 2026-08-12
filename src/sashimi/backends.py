"""Which solvers this installation has: identity, construction, self-description.

Three callers need to know what backends exist, and until this module they each
knew separately. `sashimi.cli` held a dict of factories and a parallel dict of
request families; `sashimi.capabilities` held four report functions and listed
them twice; and a library consumer — the case ROADMAP.md §9 exists for — had
neither, so it had to write its own dispatch to get at anything but APBS.

Three copies of "the backends are apbs, delphi, tabipb, gb" is three places to
edit when debye arrives, and they can disagree in a way nothing notices: a
backend `capabilities` reports as available but `--backend` cannot name is a
report about a solver the caller then cannot run.

So the registry holds the name, the request family it speaks, how to build it,
how it describes itself, how it would size a grid and what it requires — and
everything else asks here. It deliberately does *not* hold an instance:
`sashimi_capabilities` must be able to report a missing binary rather than fail
importing, so nothing here may depend on a solver being constructible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from sashimi.errors import BackendUnavailable, InputError
from sashimi.protocol import (
    AccuracyTier,
    Equation,
    GridSpec,
    PQRData,
    Solver,
    SolverFamily,
    System,
)


class SizedGrid(Protocol):
    """What a backend's grid sizer returns, in the part callers above it share.

    `ApbsGrid` and `DelphiGrid` describe different lattices — a multigrid `dime`
    against an odd cubic `gsize` — and neither shape belongs above its backend.
    What they agree on is what a cost estimate needs.
    """

    @property
    def spacing(self) -> tuple[float, float, float]: ...

    @property
    def n_points(self) -> int: ...

    def as_diagnostics(self) -> dict[str, Any]: ...


__all__ = [
    "REGISTRY",
    "BackendEntry",
    "BackendReport",
    "available_names",
    "get",
    "names",
    "reports",
    "solver_for",
]

# Equations sashimi will actually solve, as opposed to represent. ROADMAP.md
# §14 Q1: nonlinear is expressible so BEM backends cannot be handed one, but no
# tested code path produces it, and refusing beats returning untested numbers.
IMPLEMENTED_EQUATIONS = (Equation.LINEAR,)


@dataclass(frozen=True)
class BackendReport:
    """One backend's state, including the reason it is unusable."""

    name: str
    available: bool
    family: str
    version: str | None = None
    detail: str = ""
    surface_models: tuple[str, ...] = ()
    equations: tuple[str, ...] = ()
    # Whether this backend discretizes the equation or approximates it. An agent
    # choosing a backend for triage rather than for an answer needs to see the
    # difference, and `sashimi validate` reports the two tiers separately.
    accuracy_tier: str = AccuracyTier.REFERENCE.value
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "family": self.family,
            "version": self.version,
            "detail": self.detail,
            "surface_models": list(self.surface_models),
            "equations": list(self.equations),
            "accuracy_tier": self.accuracy_tier,
            **self.extras,
        }


# --- how each backend describes itself ---------------------------------------
#
# Imports are local to each function throughout: `sashimi_capabilities` is the
# one tool that has to work when a backend is missing, and paying four backends'
# import cost to answer "what is installed" would also make `--help` slow.


def _apbs_report() -> BackendReport:
    from sashimi.apbs import discover_apbs  # noqa: PLC0415 — keeps import cost off other paths
    from sashimi.apbs.options import SURFACE_KEYWORD  # noqa: PLC0415

    supported = tuple(sorted(m.value for m in SURFACE_KEYWORD))
    equations = tuple(e.value for e in IMPLEMENTED_EQUATIONS)
    try:
        binary = discover_apbs()
    except BackendUnavailable as exc:
        return BackendReport(
            name="apbs",
            available=False,
            family="finite-difference",
            detail=str(exc),
            surface_models=supported,
            equations=equations,
        )
    return BackendReport(
        name="apbs",
        available=True,
        family="finite-difference",
        version=binary.version,
        detail=f"resolved to {binary.path}",
        surface_models=supported,
        equations=equations,
        extras={
            "binary_sha256": binary.sha256,
            "representable_equations": [e.value for e in Equation],
            "native_surface_escape_hatch": "ApbsOptions.srfm_override (spl2, spl4)",
        },
    )


def _delphi_report() -> BackendReport:
    """DelPhi's state, and which of its two executables was found.

    Reported as one backend rather than two, because a caller asks "can I run
    DelPhi", but the flavour is carried in `extras` and in the supported
    surface models — which genuinely differ between them, so a single answer to
    "which surfaces does DelPhi support" would be wrong.
    """
    from sashimi.delphi.discover import discover_delphi  # noqa: PLC0415
    from sashimi.delphi.options import SUPPORTED_SURFACES, UNVALIDATED_SURFACES  # noqa: PLC0415

    equations = tuple(e.value for e in IMPLEMENTED_EQUATIONS)
    try:
        binary = discover_delphi()
    except BackendUnavailable as exc:
        return BackendReport(
            name="delphi",
            available=False,
            family="finite-difference",
            detail=str(exc),
            equations=equations,
        )

    supported = SUPPORTED_SURFACES[binary.flavour]
    return BackendReport(
        name="delphi",
        available=True,
        family="finite-difference",
        version=binary.version,
        detail=f"resolved to {binary.path} ({binary.flavour.value})",
        surface_models=tuple(sorted(m.value for m in supported)),
        equations=equations,
        extras={
            "flavour": binary.flavour.value,
            "binary_sha256": binary.sha256,
            "unvalidated_surface_models": sorted(
                m.value for m in (supported & UNVALIDATED_SURFACES)
            ),
            "energy_term": "corrected reaction field; excludes the mobile-ion osmotic term",
        },
    )


def _tabipb_report() -> BackendReport:
    """TABI-PB's state. The first backend that is not a grid.

    Reported with `family: boundary-element`, which is the field that tells a
    caller why it answers different questions: it returns potentials on the
    dielectric interface, so there is no volume to interpolate and no map to
    write for a viewer.
    """
    from sashimi.tabipb.discover import discover_tabipb  # noqa: PLC0415
    from sashimi.tabipb.options import SUPPORTED_SURFACES  # noqa: PLC0415
    from sashimi.tabipb.run import LOWEST_RELIABLE_MESH_DENSITY, MIN_ATOMS  # noqa: PLC0415

    supported = tuple(sorted(m.value for m in SUPPORTED_SURFACES))
    equations = tuple(e.value for e in IMPLEMENTED_EQUATIONS)
    try:
        binary = discover_tabipb()
    except BackendUnavailable as exc:
        return BackendReport(
            name="tabipb",
            available=False,
            family="boundary-element",
            detail=str(exc),
            surface_models=supported,
            equations=equations,
        )

    return BackendReport(
        name="tabipb",
        available=True,
        family="boundary-element",
        version=binary.mesher_version,
        detail=f"resolved to {binary.path}, meshing with {binary.mesher_path.name}",
        surface_models=supported,
        equations=equations,
        extras={
            "binary_sha256": binary.sha256,
            "mesher": binary.mesher_path.name,
            "returns": "potential on the dielectric interface, not a volumetric map",
            "min_atoms": MIN_ATOMS,
            "lowest_reliable_mesh_density": LOWEST_RELIABLE_MESH_DENSITY,
        },
    )


def _gb_report() -> BackendReport:
    """Generalized Born's state, which is always "available".

    The only backend with nothing to discover: no binary, no environment
    variable, no install step. `available` is therefore unconditionally true,
    and the honest caveat is not availability but `accuracy_tier` — it
    approximates the equation the others solve, and a caller choosing it for an
    answer rather than for triage needs to see that before it picks.
    """
    from sashimi.gb import BACKEND_VERSION, GbOptions  # noqa: PLC0415
    from sashimi.gb.options import SUPPORTED_SURFACES  # noqa: PLC0415

    options = GbOptions()
    return BackendReport(
        name="gb",
        available=True,
        family="analytic",
        version=BACKEND_VERSION,
        detail="in process; no binary, nothing to install",
        surface_models=tuple(sorted(m.value for m in SUPPORTED_SURFACES)),
        equations=(Equation.LINEAR.value,),
        accuracy_tier=AccuracyTier.APPROXIMATE.value,
        extras={
            "model": options.model.value,
            "returns": "solvation energy only; no potential field exists to sample",
            "expected_deviation": (
                "tens of percent from a Poisson-Boltzmann solver by construction; "
                "`sashimi validate` reports it as a deviation, not a disagreement"
            ),
        },
    )


# --- how each backend is built -----------------------------------------------


def _build_apbs() -> Solver[Any]:
    from sashimi.apbs import ApbsSolver  # noqa: PLC0415

    return ApbsSolver()


def _build_delphi() -> Solver[Any]:
    from sashimi.delphi import DelphiSolver  # noqa: PLC0415

    return DelphiSolver()


def _build_tabipb() -> Solver[Any]:
    from sashimi.tabipb import TabipbSolver  # noqa: PLC0415

    return TabipbSolver()


def _build_gb() -> Solver[Any]:
    from sashimi.gb import GbSolver  # noqa: PLC0415

    return GbSolver()


# --- what each backend would cost, and what it requires ----------------------


def _apbs_size_grid(structure: PQRData, spec: GridSpec) -> SizedGrid:
    from sashimi.apbs.grid import size_grid  # noqa: PLC0415

    return size_grid(structure, spec)


def _delphi_size_grid(structure: PQRData, spec: GridSpec) -> SizedGrid:
    from sashimi.delphi.grid import size_grid  # noqa: PLC0415

    return size_grid(structure, spec)


def _tabipb_preconditions(system: System) -> list[str]:
    """What TABI-PB refuses, checked before a structure has been prepared.

    Both numbers are already in its report; neither was checked, so a one-atom
    Born ion — the corpus's own anchor case — validated as fine and then failed
    inside the mesher. Measured for both, in ROADMAP.md §7.
    """
    from sashimi.tabipb.run import LOWEST_RELIABLE_MESH_DENSITY, MIN_ATOMS  # noqa: PLC0415

    problems = []
    if system.structure.n_atoms < MIN_ATOMS:
        problems.append(
            f"tabipb needs at least {MIN_ATOMS} atoms to triangulate a surface and this "
            f"structure has {system.structure.n_atoms}. There is no mesh to solve on, "
            "so the Born ion and other tiny solutes have no boundary-element answer."
        )
    if system.mesh_density < LOWEST_RELIABLE_MESH_DENSITY:
        problems.append(
            f"tabipb aborts below mesh_density {LOWEST_RELIABLE_MESH_DENSITY} and this "
            f"request asks for {system.mesh_density}. The abort is an uncaught C++ "
            "exception carrying no cause, which is why it is refused here instead."
        )
    return problems


@dataclass(frozen=True)
class BackendEntry:
    """One backend, as everything above the solver layer needs to know it."""

    name: str
    # `Solver` is generic in its request type, so a checker already refuses to
    # hand a boundary-element request to an FD backend — but that guarantee is
    # static, and dispatch through this registry happens at runtime.
    family: SolverFamily
    build: Callable[[], Solver[Any]]
    describe: Callable[[], BackendReport]
    # How this backend would size a grid, where it builds one. Two
    # finite-difference backends do not agree on what a grid is: APBS solves on
    # a `dime = c*2^(l+1)+1` multigrid lattice with per-axis spacing, DelPhi on
    # an odd cubic one. Costing one with the other's arithmetic produces a
    # confident wrong number — a point count, a map size and a "resolution was
    # relaxed" warning belonging to a solver the caller is not running.
    size_grid: Callable[[PQRData, GridSpec], SizedGrid] | None = None
    # What this backend requires beyond a surface model, in its own terms.
    # TABI-PB's mesher cannot triangulate fewer than four atoms and its solver
    # aborts below a mesh density of 1.5; both are already published in its
    # report, and both are free to check before a structure has been prepared.
    preconditions: Callable[[System], list[str]] | None = None

    def solver(self) -> Solver[Any]:
        """Construct it.

        Does **not** raise when the binary is missing: every shipped backend
        discovers lazily, so the failure arrives from `solve()`. That is
        deliberate — `describe_capabilities` has to be able to report an absent
        APBS, and it could not if naming a backend were enough to fail.
        """
        return self.build()

    def report(self) -> BackendReport:
        return self.describe()

    def check(self, system: System) -> list[str]:
        """Backend-specific reasons this request would be refused, if any."""
        return self.preconditions(system) if self.preconditions else []


# Registry order is report order, which is roughly the order the backends
# arrived and the order §8's table lists them. debye registers here when it
# exists, and that is the whole edit — `--backend`, `sashimi_solve` and
# `sashimi_capabilities` all read from this.
REGISTRY: dict[str, BackendEntry] = {
    "apbs": BackendEntry(
        "apbs",
        SolverFamily.FINITE_DIFFERENCE,
        _build_apbs,
        _apbs_report,
        size_grid=_apbs_size_grid,
    ),
    "delphi": BackendEntry(
        "delphi",
        SolverFamily.FINITE_DIFFERENCE,
        _build_delphi,
        _delphi_report,
        size_grid=_delphi_size_grid,
    ),
    "tabipb": BackendEntry(
        "tabipb",
        SolverFamily.BOUNDARY_ELEMENT,
        _build_tabipb,
        _tabipb_report,
        preconditions=_tabipb_preconditions,
    ),
    "gb": BackendEntry("gb", SolverFamily.ANALYTIC, _build_gb, _gb_report),
}


def names() -> tuple[str, ...]:
    """Every registered backend, whether or not it is installed."""
    return tuple(REGISTRY)


def get(name: str) -> BackendEntry:
    """One backend by name, or an `InputError` naming the alternatives.

    An unknown backend is the caller's mistake and is recoverable by picking a
    different one, so it is an `InputError` rather than a `KeyError` — the
    message has to carry the list, because a caller that guessed the name wrong
    has no other way to learn it.
    """
    try:
        return REGISTRY[name]
    except KeyError:
        raise InputError(
            f"unknown backend {name!r}. This installation registers: "
            f"{', '.join(names())}. `sashimi_capabilities` reports which of "
            "them are actually installed."
        ) from None


def solver_for(name: str) -> tuple[Solver[Any], SolverFamily]:
    """A constructed solver and the request dialect to ask it in."""
    entry = get(name)
    return entry.solver(), entry.family


def reports() -> list[BackendReport]:
    """Every backend's self-description, in registry order."""
    return [REGISTRY[name].report() for name in names()]


def available_names() -> list[str]:
    """The backends that could actually run a solve right now."""
    return [report.name for report in reports() if report.available]
