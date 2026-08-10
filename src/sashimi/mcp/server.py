"""FastMCP tools.

Four tools, all prefixed `sashimi_`. Parameters are flat and physically named
rather than nested config objects — agents call those more reliably — and every
response pairs structured content with a short human-readable summary.

Deliberately absent: a PDB-fetching tool (mcpymol already owns structure
acquisition) and raw APBS-input passthrough (it would defeat the abstraction;
anyone who needs `fe-manual` needs APBS, not sashimi).

Typed sashimi exceptions become ToolError with the actionable message intact —
`GridTooLarge` already states the achievable resolution, `ApbsNotFound` already
carries the install one-liner. Anything else is re-raised unwrapped rather than
dressed up as a solver problem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from sashimi.apbs import ApbsSolver
from sashimi.dx import read_dx
from sashimi.errors import SashimiError
from sashimi.pqr import read_pqr
from sashimi.prep import ForceField, prepare_structure
from sashimi.protocol import (
    FiniteDifferenceRequest,
    GridSpec,
    PotentialGrid,
    SolventModel,
    SurfaceModel,
)

__all__ = ["mcp"]

mcp: FastMCP[Any] = FastMCP(
    name="sashimi",
    instructions=(
        "Biomolecular electrostatics via APBS. Typical flow: "
        "sashimi_prepare_structure on a PDB to get a PQR (read its warnings — "
        "rebuilt sidechains change the energies), then sashimi_solve for a "
        "potential map and optional solvation energy, then sashimi_potential_at "
        "to query the saved map without re-solving. Potentials are kT/e, "
        "energies kJ/mol, distances angstroms."
    ),
)


def _fail(exc: SashimiError) -> ToolError:
    """Surface a typed sashimi failure with its message intact."""
    return ToolError(str(exc))


@mcp.tool
def sashimi_prepare_structure(
    *,
    pdb_path: Annotated[str, Field(description="Path to a PDB file.")],
    forcefield: Annotated[ForceField, Field(description="Charge/radius parameter set.")] = "AMBER",
    ph: Annotated[
        float | None,
        Field(
            description=(
                "pH for propka titration-state prediction. Omit to use the "
                "forcefield's default protonation, which is faster and deterministic."
            ),
            ge=0.0,
            le=14.0,
        ),
    ] = None,
    output_pqr: Annotated[
        str | None, Field(description="Where to write the PQR. Defaults next to the PDB.")
    ] = None,
) -> dict[str, Any]:
    """Assign charges and radii to a PDB structure, producing a PQR for sashimi_solve.

    The warnings in the response are the important part: pdb2pqr rebuilds
    missing heavy atoms and debumps clashes, and those edits change the charges
    the solver sees. Check them before trusting downstream energies.
    """
    source = Path(pdb_path).expanduser()
    try:
        result = prepare_structure(source, forcefield=forcefield, ph=ph)
    except SashimiError as exc:
        raise _fail(exc) from exc

    destination = Path(output_pqr).expanduser() if output_pqr else source.with_suffix(".pqr")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result.pqr_text)

    summary = result.summary()
    modified = "; structure was modified" if result.structure_was_modified else ""
    return {
        "pqr_path": str(destination),
        **summary,
        "summary": (
            f"Prepared {source.name}: {summary['n_atoms']} atoms, "
            f"net charge {summary['total_charge']:+.3f} e, "
            f"{summary['n_warnings']} warning(s){modified}."
        ),
    }


@mcp.tool
def sashimi_solve(
    *,
    pqr_path: Annotated[str, Field(description="Path to a PQR file.")],
    resolution: Annotated[
        float, Field(description="Target fine-grid spacing, angstroms.", gt=0, le=5.0)
    ] = 0.5,
    padding: Annotated[
        float,
        Field(description="Minimum solute-surface to grid-edge distance, angstroms.", ge=0),
    ] = 10.0,
    ionic_strength: Annotated[
        float, Field(description="1:1 salt concentration, molar.", ge=0)
    ] = 0.15,
    solute_dielectric: Annotated[float, Field(description="Solute dielectric.", gt=0)] = 2.0,
    solvent_dielectric: Annotated[float, Field(description="Solvent dielectric.", gt=0)] = 78.54,
    compute_energy: Annotated[
        bool, Field(description="Also compute total polar solvation energy.")
    ] = False,
    surface_model: Annotated[
        str,
        Field(
            description=(
                "Dielectric boundary definition. This is the single largest modelling "
                "choice in the calculation \u2014 it moves solvation energies by tens of "
                "percent \u2014 so it is recorded with every result. One of: "
                "molecular, smoothed-molecular, van-der-waals."
            )
        ),
    ] = "smoothed-molecular",
    output_dx: Annotated[
        str | None, Field(description="Where to write the OpenDX map. Defaults next to the PQR.")
    ] = None,
) -> dict[str, Any]:
    """Solve the linearized Poisson-Boltzmann equation for a prepared structure.

    Returns grid statistics, the path to an OpenDX map that PyMOL and ChimeraX
    can load, the solvation energy when requested, and which backend produced
    it. Note `spacing_achieved`: the memory guardrail may relax the requested
    resolution, and the response says so rather than hiding it.
    """
    source = Path(pqr_path).expanduser()
    try:
        pqr = read_pqr(source)
    except (OSError, ValueError) as exc:
        raise ToolError(f"could not read PQR {source}: {exc}") from exc

    try:
        result = ApbsSolver().solve(
            FiniteDifferenceRequest(
                structure=pqr,
                solvent=SolventModel(
                    solvent_dielectric=solvent_dielectric,
                    solute_dielectric=solute_dielectric,
                    ionic_strength=ionic_strength,
                    surface_model=SurfaceModel(surface_model),
                ),
                grid=GridSpec(resolution=resolution, padding=padding),
                want_energy=compute_energy,
                want_potential=True,
            )
        )
    except SashimiError as exc:
        raise _fail(exc) from exc

    potential = result.potential
    if not isinstance(potential, PotentialGrid):  # pragma: no cover — APBS is volumetric
        raise ToolError(f"expected a volumetric map, got {type(potential).__name__}")

    destination = Path(output_dx).expanduser() if output_dx else source.with_suffix(".dx")
    destination.parent.mkdir(parents=True, exist_ok=True)
    potential.to_dx(destination)

    stats = potential.stats()
    energy = (
        f" Polar solvation energy {result.energy_kj_mol:.3f} kJ/mol."
        if result.energy_kj_mol is not None
        else ""
    )
    relaxed = result.diagnostics.get("resolution_relaxed", False)
    note = " Requested resolution was relaxed to fit max_points." if relaxed else ""

    return {
        "dx_path": str(destination),
        "energy_kj_mol": result.energy_kj_mol,
        "backend": result.provenance.summary(),
        "resolved_parameters": result.provenance.resolved_parameters,
        "grid": {
            "shape": stats["shape"],
            "origin": stats["origin"],
            "spacing_achieved": result.diagnostics["spacing_achieved"],
            "resolution_relaxed": relaxed,
        },
        "potential_kT_e": {
            "min": stats["min"],
            "max": stats["max"],
            "mean": stats["mean"],
            "std": stats["std"],
        },
        "diagnostics": result.diagnostics,
        "summary": (
            f"Solved {source.name} on a {'x'.join(str(n) for n in stats['shape'])} grid "
            f"at {result.diagnostics['spacing_achieved'][0]:.3f} A.{energy}{note} "
            f"Map written to {destination.name} ({result.provenance.backend})."
        ),
    }


@mcp.tool
def sashimi_potential_at(
    *,
    dx_path: Annotated[str, Field(description="Path to an OpenDX map.")],
    points: Annotated[
        list[list[float]],
        Field(description="Coordinates to sample, as [[x, y, z], ...] in angstroms."),
    ],
) -> dict[str, Any]:
    """Sample a saved potential map at arbitrary coordinates, without re-solving.

    Trilinear interpolation. Points outside the grid come back as null rather
    than clamped to an edge value, because a clamped number reads as a real
    measurement.
    """
    grid = _load_grid(dx_path)
    try:
        sampled = grid.value_at(np.asarray(points, dtype=float))
    except ValueError as exc:
        raise ToolError(f"invalid points: {exc}") from exc

    values: list[float | None] = [None if np.isnan(v) else float(v) for v in sampled]
    outside = sum(1 for v in values if v is None)
    return {
        "values_kT_e": values,
        "n_outside_grid": outside,
        "summary": (
            f"Sampled {len(values)} point(s) from {Path(dx_path).name}"
            + (f"; {outside} fell outside the grid." if outside else ".")
        ),
    }


@mcp.tool
def sashimi_compare_maps(
    *,
    dx_a: Annotated[str, Field(description="Path to the first OpenDX map.")],
    dx_b: Annotated[str, Field(description="Path to the second OpenDX map.")],
) -> dict[str, Any]:
    """Compare two potential maps: RMSD, maximum absolute difference, correlation.

    Useful for mutant-versus-wildtype questions now, and it doubles as the
    solver-versus-solver validation tool when a second backend exists. The two
    maps must share a grid; differing geometry is an error rather than a
    silently resampled comparison.
    """
    a, b = _load_grid(dx_a), _load_grid(dx_b)

    if a.shape != b.shape:
        raise ToolError(f"grid shapes differ: {a.shape} vs {b.shape}; maps are not comparable")
    if not np.allclose(a.spacing, b.spacing) or not np.allclose(a.origin, b.origin):
        raise ToolError(
            "grid geometry differs (origin or spacing); re-solve both on the same grid "
            "rather than comparing across geometries"
        )

    diff = a.values - b.values
    rmsd = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.abs(diff).max())
    flat_a, flat_b = a.values.reshape(-1), b.values.reshape(-1)
    # A constant map has zero variance, so correlation is undefined, not 0.0.
    correlation = (
        float(np.corrcoef(flat_a, flat_b)[0, 1]) if flat_a.std() > 0 and flat_b.std() > 0 else None
    )

    return {
        "rmsd_kT_e": rmsd,
        "max_abs_diff_kT_e": max_abs,
        "correlation": correlation,
        "mean_diff_kT_e": float(diff.mean()),
        "shape": list(a.shape),
        "summary": (
            f"RMSD {rmsd:.4g} kT/e, max |diff| {max_abs:.4g} kT/e"
            + (f", correlation {correlation:.6f}" if correlation is not None else "")
            + f" over {a.values.size:,} grid points."
        ),
    }


def _load_grid(path: str) -> PotentialGrid:
    resolved = Path(path).expanduser()
    try:
        return read_dx(resolved)
    except (OSError, ValueError) as exc:
        raise ToolError(f"could not read DX {resolved}: {exc}") from exc


def main() -> None:
    """stdio entry point, matching the rest of the FastMCP fleet."""
    mcp.run()


if __name__ == "__main__":
    main()
