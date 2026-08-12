"""What this installation can actually do, and whether a request would work.

Two questions an agent should be able to ask before committing to a solve that
takes a minute and writes 50 MB:

- *What can you do?* Which backends are installed, which surface models they
  support, which equations they actually solve. Discovery beats guessing, and
  guessing is what an agent does otherwise.
- *Would this work?* Grid sizing is pure arithmetic — `size_grid` needs no
  subprocess — so the shape of the answer, the point count, whether the
  resolution would be relaxed and how large the map would be are all knowable
  for free, before anything runs.

Reporting a missing backend is not an error here. "APBS is not installed" is a
capability report, and raising instead would make the one tool that could
explain the problem the one tool that cannot run.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from sashimi.apbs.grid import size_grid
from sashimi.apbs.options import ApbsOptions, resolve_surface
from sashimi.artifacts import describe_cleanup, estimated_dx_bytes
from sashimi.backends import IMPLEMENTED_EQUATIONS, BackendReport, get, reports
from sashimi.errors import InputError, SashimiError
from sashimi.protocol import (
    Equation,
    GridSpec,
    PQRData,
    SolventModel,
    SolverFamily,
    SurfaceModel,
)

__all__ = [
    "UNITS",
    "BackendReport",
    "describe_capabilities",
    "validate_request",
]

UNITS = {
    "length": "angstrom",
    "potential": "kT/e",
    "energy": "kJ/mol",
    "ionic_strength": "molar (1:1 salt)",
    "temperature": "kelvin",
}

# A spread needs two things to spread between.
MIN_BACKENDS_TO_COMPARE = 2


def comparable_surface_models() -> list[str]:
    """Surface models on which two or more installed backends could be compared.

    The precondition for a cross-solver spread being a solver disagreement
    rather than a modelling one (ROADMAP.md section 8). It is frequently empty,
    for two quite different reasons: APBS and pyDelPhi have no surface model in
    common at all, and a single backend has nothing to be compared *with*.

    The second case is why this counts backends rather than reporting whatever
    a lone one supports. A single APBS trivially "shares" all three of its
    models with itself, and reporting those as comparable would tell a caller
    that cross-validation is available when nothing is installed to validate
    against.

    **Two or more, not all**, which is what the first line says and what the
    intersection this used to compute did not do. The bug was invisible while
    three backends all supported `molecular`, and it fails in the direction that
    hurts: one backend supporting a disjoint set empties the result for
    everybody, and an empty result here stops `sashimi validate` and skips the
    whole cross-validation tier. Adding a backend must not be able to switch off
    the comparisons between the others.

    Fixing it turned up a comparison that was legitimate all along and had never
    run: APBS against DelPhi on `van-der-waals`, which TABI-PB cannot mesh and
    which the intersection therefore hid.
    """
    supported = [set(r.surface_models) for r in reports() if r.available]
    if len(supported) < MIN_BACKENDS_TO_COMPARE:
        return []
    counts = Counter(model for models in supported for model in models)
    return sorted(model for model, n in counts.items() if n >= MIN_BACKENDS_TO_COMPARE)


def describe_capabilities() -> dict[str, Any]:
    """Everything a caller needs to plan a request without trial and error."""
    backends = reports()
    usable = [b.name for b in backends if b.available]
    defaults = GridSpec()

    return {
        "units": UNITS,
        "backends": [b.as_dict() for b in backends],
        "available_backends": usable,
        "surface_models": {
            "portable": sorted(m.value for m in SurfaceModel),
            # Empty is a real and common answer, not a missing one: which models
            # two backends share is what decides whether they can be compared.
            "comparable_across_available_backends": comparable_surface_models(),
            "note": (
                "Surface definition is the largest modelling choice in the "
                "calculation, moving solvation energies by tens of percent. It is "
                "recorded with every result. Spline surfaces are deliberately not "
                "portable members: they exist for force calculations and are a "
                "misuse for energies."
            ),
        },
        "grid_defaults": {
            "resolution": defaults.resolution,
            "padding": defaults.padding,
            "max_points": defaults.max_points,
        },
        "artifacts": describe_cleanup(),
        "not_supported": [
            "nonlinear Poisson-Boltzmann (representable in the request; no solver path yet)",
            "raw solver input passthrough (deliberately absent)",
            "FEM, geoflow, PBAM, PBSAM solvers",
        ],
        "summary": (
            f"{len(usable)}/{len(backends)} backend(s) available"
            + (f" ({', '.join(usable)})" if usable else "")
            + f"; potentials in {UNITS['potential']}, energies in {UNITS['energy']}."
        ),
    }


def validate_request(
    structure: PQRData,
    grid: GridSpec | None = None,
    solvent: SolventModel | None = None,
    *,
    equation: Equation = Equation.LINEAR,
    options: ApbsOptions | None = None,
    backend: str = "apbs",
) -> dict[str, Any]:
    """Would this solve work, and what would it cost? No subprocess involved.

    Grid sizing and surface mapping are both pure, so everything expensive about
    a request is knowable before paying for it: the grid that would be used, its
    point count, the map size on disk, and whether the memory guardrail would
    quietly coarsen the resolution.

    `backend` is checked for two things a caller can only otherwise discover by
    running it: whether it is installed, and whether it supports the surface
    model asked for. That second one is the common failure now that a backend
    can be chosen — three of the four refuse `smoothed-molecular`, and the
    refusal arrives after the structure has been prepared.

    The cost estimate is a *finite-difference* one, because a grid is what it
    estimates. For a boundary-element or analytic backend the grid section is
    omitted rather than filled in with a number from the wrong model, and the
    report says what governs cost instead.
    """
    grid = grid or GridSpec()
    solvent = solvent or SolventModel()
    options = options or ApbsOptions()
    entry = get(backend)

    problems: list[str] = []
    report: dict[str, Any] = {
        "n_atoms": structure.n_atoms,
        "total_charge": round(structure.total_charge, 4),
        "extent_a": [round(float(v), 3) for v in structure.extent()],
    }

    if equation not in IMPLEMENTED_EQUATIONS:
        problems.append(
            f"the {equation.value} equation is representable but has no solver path yet; use linear"
        )

    backend_report = entry.report()
    surface = solvent.surface_model.value
    if backend_report.surface_models and surface not in backend_report.surface_models:
        problems.append(
            f"{backend} does not support the {surface} surface. It supports: "
            f"{', '.join(backend_report.surface_models)}. Surface model is the "
            "largest modelling choice in the calculation, so it is refused "
            "rather than substituted."
        )

    report["surface"] = {"requested": surface}
    if entry.name == "apbs":
        # `resolve_surface` maps onto APBS's keywords, so the resolved value is
        # only meaningful for APBS. Reporting another backend's request through
        # it would print a keyword that backend has never heard of.
        try:
            resolved_surface, resolved_probe = resolve_surface(
                solvent.surface_model, solvent.surface_radius, options
            )
            report["surface"]["resolved_keyword"] = resolved_surface
            report["surface"]["resolved_probe_radius_a"] = resolved_probe
        except InputError as exc:
            problems.append(str(exc))

    if entry.family is not SolverFamily.FINITE_DIFFERENCE:
        report["cost"] = {
            "grid": None,
            "note": _NON_GRID_COST[entry.family],
        }
    else:
        _add_grid_cost(structure, grid, report, problems)

    backend_dict = {
        "name": backend_report.name,
        "available": backend_report.available,
        "family": entry.family.value,
        "accuracy_tier": backend_report.accuracy_tier,
    }
    report["backend"] = backend_dict
    if not backend_report.available:
        problems.append(backend_report.detail)

    # A relaxed grid is a warning, not a refusal: the solve would still run and
    # the result reports the relaxation. Anything else here would prevent it.
    blocking = [p for p in problems if "caps the grid" not in p]
    report["ok"] = not blocking
    report["problems"] = problems
    report["summary"] = _summarise(report, blocking, problems)
    return report


# What a caller should look at instead of a point count, when a grid is not what
# the backend builds. Both are measured claims: see ROADMAP.md §7.
_NON_GRID_COST = {
    SolverFamily.BOUNDARY_ELEMENT: (
        "no grid: cost is the surface mesh, which does not track atom count — "
        "906 atoms mesh in 48 s where a 260-atom united-atom structure takes "
        "450 s at three times the vertices. Mesh density is the knob."
    ),
    SolverFamily.ANALYTIC: (
        "no grid and no field: cost is O(N^2) in atoms and runs in process — "
        "under a second for a 2,500-atom protein. Returns an energy only."
    ),
}


def _add_grid_cost(
    structure: PQRData,
    grid: GridSpec,
    report: dict[str, Any],
    problems: list[str],
) -> None:
    """The finite-difference cost estimate: what grid, how many points, how big."""
    try:
        sized = size_grid(structure, grid)
    except SashimiError as exc:
        problems.append(str(exc))
    else:
        relaxed = any(s > grid.resolution + 1e-9 for s in sized.spacing)
        report["grid"] = {
            "dime": list(sized.dime),
            "n_points": sized.n_points,
            "spacing_achieved_a": [round(s, 6) for s in sized.spacing],
            "resolution_requested_a": grid.resolution,
            "resolution_relaxed": relaxed,
            "estimated_map_mb": round(estimated_dx_bytes(sized.n_points) / 1e6, 1),
        }
        if relaxed:
            problems.append(
                f"max_points={grid.max_points:,} caps the grid, so the achieved spacing "
                f"would be {max(sized.spacing):.4f} A rather than the requested "
                f"{grid.resolution} A. Raise max_points or accept the coarser grid."
            )


def _summarise(report: dict[str, Any], blocking: list[str], problems: list[str]) -> str:
    grid = report.get("grid")
    shape = "x".join(str(d) for d in grid["dime"]) if grid else "unknown"
    size = f", ~{grid['estimated_map_mb']} MB map" if grid else ""
    if blocking:
        return f"Would not run: {blocking[0]}"
    warned = f" {len(problems)} warning(s)." if problems else ""
    return f"Would run on a {shape} grid{size}.{warned}"
