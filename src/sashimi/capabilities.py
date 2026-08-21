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

import dataclasses
from collections import Counter
from typing import Any

from sashimi.apbs.options import ApbsOptions, resolve_surface
from sashimi.artifacts import describe_cleanup, estimated_dx_bytes
from sashimi.backends import (
    IMPLEMENTED_EQUATIONS,
    BackendEntry,
    BackendReport,
    get,
    reports,
)
from sashimi.errors import InputError, SashimiError
from sashimi.protocol import (
    Equation,
    GridSpec,
    PQRData,
    SolventModel,
    SolverFamily,
    SurfaceModel,
    System,
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


def _describe(solvent: SolventModel) -> dict[str, Any]:
    """A solvent model as JSON, enums by value. Every field, so adding one shows."""
    return {
        field.name: value.value if isinstance(value, SurfaceModel) else value
        for field in dataclasses.fields(solvent)
        if (value := getattr(solvent, field.name)) is not None
    }


def describe_capabilities() -> dict[str, Any]:
    """Everything a caller needs to plan a request without trial and error."""
    backends = reports()
    usable = [b.name for b in backends if b.available]
    defaults = GridSpec()
    solvent_defaults = SolventModel()

    return {
        "units": UNITS,
        "backends": [b.as_dict() for b in backends],
        "available_backends": usable,
        "surface_models": {
            "portable": sorted(m.value for m in SurfaceModel),
            # What a request that names no surface will actually be solved on.
            # It moved once — `smoothed-molecular` to `molecular`, 2026-08-13 —
            # so a caller that wants a specific boundary should read it here
            # rather than assume this year's answer.
            "default": solvent_defaults.surface_model.value,
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
        # The physics half of "what happens if I name nothing", which had no
        # answer here while the grid half did — so an agent could discover the
        # spacing a defaulted solve would use but not the boundary, and the
        # boundary is the larger modelling choice of the two.
        "solvent_defaults": _describe(solvent_defaults),
        "artifacts": describe_cleanup(),
        # Reported because it is the difference between a 1,200-residue protein
        # taking about half a minute and taking two and a half, and because a
        # caller cannot discover it from a result — the answer is identical
        # either way, only the wait changes.
        "acceleration": _describe_acceleration(),
        # What each preference resolves to *here*, since it depends on what is
        # installed and on the surface asked for. A caller cannot work this out
        # from the backend list alone.
        "preferences": _describe_preferences(),
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


def _describe_preferences() -> dict[str, Any]:
    """Where `prefer=fast|stable|portable` lands on this machine, per surface.

    Resolved rather than described, because the answer is a property of the
    installation: `stable` means pyDelPhi where it is installed and the request
    is `molecular`, and something else everywhere else.
    """
    from sashimi.backends import resolve  # noqa: PLC0415 — avoids an import cycle
    from sashimi.errors import SashimiError  # noqa: PLC0415

    table: dict[str, Any] = {}
    for preference in ("fast", "stable", "portable"):
        per_surface: dict[str, str] = {}
        for surface in sorted(m.value for m in SurfaceModel):
            try:
                name, _why = resolve(preference, surface)
            except SashimiError:
                per_surface[surface] = "no installed backend can answer this"
            else:
                per_surface[surface] = name
        table[preference] = per_surface
    table["note"] = (
        "`prefer` is a convenience for a caller who knows what they want from "
        "the answer but not which solver gives it. Naming `backend` overrides "
        "it. 'stable' is named for what is measured — how little the answer "
        "moves when the solute is rotated — and not 'accurate', because above a "
        "two-atom solute nothing here has a reference answer to be accurate "
        "against."
    )
    return table


def _describe_acceleration() -> dict[str, Any]:
    """Whether debye's compiled surface kernel is installed, and what it is worth.

    `sashimi-electro[fast]` is an extra rather than a dependency because numba
    brings llvmlite and the pair are ~145 MB — several times the rest of the
    install — against a package whose proposition is that it needs nothing
    fetched by hand. So the trade is the caller's to make, which means the
    caller has to be told it exists.
    """
    from sashimi.debye import kernel  # noqa: PLC0415 — avoids a debye import cost here

    reason = kernel.why_unavailable()
    return {
        "compiled_surface_kernel": kernel.available(),
        "applies_to": "debye",
        "worth": (
            "9-28x on the three surface-classification loops, which is 1.7-1.9x "
            "on a whole solve: measured 1.81x on a 382-residue protein and 1.73x "
            "on a 1,156-residue one (149 s to 86 s of CPU), with energies "
            "bit-identical either way. The rest is the reduced-surface "
            "construction, which is not compiled"
        ),
        "install": "pip install 'sashimi-electro[fast]'",
        "cost": "~145 MB, since numba brings llvmlite",
        "why_unavailable": reason,
    }


def validate_request(
    structure: PQRData,
    grid: GridSpec | None = None,
    solvent: SolventModel | None = None,
    *,
    equation: Equation = Equation.LINEAR,
    options: ApbsOptions | None = None,
    backend: str = "apbs",
    mesh_density: float | None = None,
) -> dict[str, Any]:
    """Would this solve work, and what would it cost? No subprocess involved.

    Grid sizing and surface mapping are both pure, so everything expensive about
    a request is knowable before paying for it: the grid that would be used, its
    point count, the map size on disk, and whether the memory guardrail would
    quietly coarsen the resolution.

    `backend` is checked for two things a caller can only otherwise discover by
    running it: whether it is installed, and whether it supports the surface
    model asked for. That second one is where a chosen backend refuses — three
    of the four decline `smoothed-molecular`, and the refusal arrives after the
    structure has been prepared. The default no longer walks into it: it is
    `molecular`, which every backend supports.

    The cost estimate is the **chosen backend's own arithmetic**, not a
    finite-difference estimate applied to everyone. Two FD backends do not agree
    on what a grid is — APBS solves on a `dime = c*2^(l+1)+1` multigrid lattice
    with per-axis spacing, DelPhi on an odd cubic one — so costing DelPhi with
    APBS's sizer produced a point count, a map size and a relaxation warning
    belonging to a solver the caller was not running. Backends that build no
    grid report what governs their cost instead, and every answer arrives under
    the same `cost` key so a caller does not have to know the family to read it.

    `mesh_density` is accepted because it is the boundary-element cost knob, and
    checking it here is the difference between a refusal now and an uncaught C++
    exception after a structure has been prepared.
    """
    grid = grid or GridSpec()
    solvent = solvent or SolventModel()
    options = options or ApbsOptions()
    entry = get(backend)
    # `mesh_density=None` means "whatever the protocol defaults to", rather
    # than a number this module invents. `System` and `BoundaryElementRequest`
    # currently disagree about that default — ROADMAP.md §14 has it open — and
    # duplicating either here would make a third place to fix.
    system = System(
        structure=structure,
        solvent=solvent,
        grid=grid,
        want_energy=True,
        want_potential=entry.family is not SolverFamily.ANALYTIC,
        **({} if mesh_density is None else {"mesh_density": mesh_density}),
    )

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

    problems.extend(entry.check(system))
    report["cost"] = _cost(entry, structure, grid, problems)

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


def _cost(
    entry: BackendEntry,
    structure: PQRData,
    grid: GridSpec,
    problems: list[str],
) -> dict[str, Any]:
    """What this request would cost, in the terms the chosen backend works in.

    One key for every family, with `grid` null where there is no grid, so a
    caller reads the same field whichever backend it picked. The alternative —
    `grid` for finite difference and `cost` for everything else — meant knowing
    the family before knowing which key existed.
    """
    cost: dict[str, Any] = {"family": entry.family.value, "grid": None}
    if entry.size_grid is None:
        cost["note"] = _NON_GRID_COST[entry.family]
        return cost

    try:
        sized = entry.size_grid(structure, grid)
    except SashimiError as exc:
        problems.append(str(exc))
        cost["note"] = "the grid could not be sized; see problems"
        return cost

    relaxed = any(s > grid.resolution + 1e-9 for s in sized.spacing)
    cost["grid"] = {
        "n_points": sized.n_points,
        "spacing_achieved_a": [round(s, 6) for s in sized.spacing],
        "resolution_requested_a": grid.resolution,
        "resolution_relaxed": relaxed,
        "estimated_map_mb": round(estimated_dx_bytes(sized.n_points) / 1e6, 1),
        # The backend's own description of the lattice, in its own vocabulary:
        # `dime` for APBS, `gsize` and `scale` for DelPhi. Above this line those
        # are not interchangeable, which is the whole reason each backend sizes
        # its own.
        "native": sized.as_diagnostics(),
    }
    cost["note"] = f"{entry.name} sizes this grid itself; the shape above is its own"
    if relaxed:
        problems.append(
            f"max_points={grid.max_points:,} caps the grid, so the achieved spacing "
            f"would be {max(sized.spacing):.4f} A rather than the requested "
            f"{grid.resolution} A. Raise max_points or accept the coarser grid."
        )
    return cost


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


def _summarise(report: dict[str, Any], blocking: list[str], problems: list[str]) -> str:
    """One line an agent can act on, true for a backend that builds no grid.

    This used to read "Would run on a unknown grid" for Generalized Born and
    TABI-PB — contradicting the report's own `cost.grid: null` and inviting a
    caller to lower `max_points` for a solver that has none.
    """
    if blocking:
        return f"Would not run: {blocking[0]}"

    warned = f" {len(problems)} warning(s)." if problems else ""
    cost = report.get("cost", {})
    grid = cost.get("grid")
    if grid is None:
        return f"Would run: {cost.get('note', 'no grid to size')}.{warned}"
    return (
        f"Would run on {grid['n_points']:,} grid points at "
        f"{max(grid['spacing_achieved_a']):.3f} A, ~{grid['estimated_map_mb']} MB map.{warned}"
    )
