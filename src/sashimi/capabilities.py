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

from dataclasses import dataclass, field
from typing import Any

from sashimi.apbs.grid import size_grid
from sashimi.apbs.options import SURFACE_KEYWORD, ApbsOptions, resolve_surface
from sashimi.artifacts import describe_cleanup, estimated_dx_bytes
from sashimi.errors import BackendUnavailable, InputError, SashimiError
from sashimi.protocol import Equation, GridSpec, PQRData, SolventModel, SurfaceModel

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
            **self.extras,
        }


def _apbs_report() -> BackendReport:
    from sashimi.apbs import discover_apbs  # noqa: PLC0415 — keeps import cost off other paths

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


def describe_capabilities() -> dict[str, Any]:
    """Everything a caller needs to plan a request without trial and error."""
    backends = [_apbs_report()]
    usable = [b.name for b in backends if b.available]
    defaults = GridSpec()

    return {
        "units": UNITS,
        "backends": [b.as_dict() for b in backends],
        "available_backends": usable,
        "surface_models": {
            "portable": sorted(m.value for m in SurfaceModel),
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
            "boundary-element backends (protocol admits them; none shipped)",
            "raw APBS input passthrough (deliberately absent)",
            "FEM, geoflow, BEM, PBAM, PBSAM solvers",
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
) -> dict[str, Any]:
    """Would this solve work, and what would it cost? No subprocess involved.

    Grid sizing and surface mapping are both pure, so everything expensive about
    a request is knowable before paying for it: the grid that would be used, its
    point count, the map size on disk, and whether the memory guardrail would
    quietly coarsen the resolution.
    """
    grid = grid or GridSpec()
    solvent = solvent or SolventModel()
    options = options or ApbsOptions()

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

    try:
        resolved_surface, resolved_probe = resolve_surface(
            solvent.surface_model, solvent.surface_radius, options
        )
        report["surface"] = {
            "requested": solvent.surface_model.value,
            "resolved_keyword": resolved_surface,
            "resolved_probe_radius_a": resolved_probe,
        }
    except InputError as exc:
        problems.append(str(exc))

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

    backend = _apbs_report()
    report["backend"] = {"name": backend.name, "available": backend.available}
    if not backend.available:
        problems.append(backend.detail)

    # A relaxed grid is a warning, not a refusal: the solve would still run and
    # the result reports the relaxation. Anything else here would prevent it.
    blocking = [p for p in problems if "caps the grid" not in p]
    report["ok"] = not blocking
    report["problems"] = problems
    report["summary"] = _summarise(report, blocking, problems)
    return report


def _summarise(report: dict[str, Any], blocking: list[str], problems: list[str]) -> str:
    grid = report.get("grid")
    shape = "x".join(str(d) for d in grid["dime"]) if grid else "unknown"
    size = f", ~{grid['estimated_map_mb']} MB map" if grid else ""
    if blocking:
        return f"Would not run: {blocking[0]}"
    warned = f" {len(problems)} warning(s)." if problems else ""
    return f"Would run on a {shape} grid{size}.{warned}"
