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
from typing import Annotated, Any, Literal

import numpy as np
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from sashimi.analysis import potential_extrema, potential_in_sphere, residue_potentials
from sashimi.apbs import ApbsSolver
from sashimi.artifacts import content_address, describe_cleanup, map_path
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
        "energies kJ/mol, distances angstroms. " + describe_cleanup()
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
        str | None,
        Field(
            description=(
                "Where to write the OpenDX map. Defaults to a content-addressed "
                "name next to the PQR, so re-solving with different parameters "
                "never overwrites an earlier map."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Solve the linearized Poisson-Boltzmann equation for a prepared structure.

    Returns grid statistics, the path to an OpenDX map that PyMOL and ChimeraX
    can load, the solvation energy when requested, and which backend produced
    it. Note `spacing_achieved`: the memory guardrail may relax the requested
    resolution, and the response says so rather than hiding it.

    Maps are files, not inline data — a default-resolution grid is megabytes.
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

    # Content-addressed by default: two solves that differ in anything that
    # changes the answer get different files, so re-solving cannot silently
    # overwrite a previous map. An explicit output_dx is honored as given.
    address = content_address(pqr, result.provenance.resolved_parameters)
    destination = Path(output_dx).expanduser() if output_dx else map_path(source, address)
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
        "content_address": address,
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


@mcp.tool
def sashimi_potential_extrema(
    *,
    dx_path: Annotated[str, Field(description="Path to an OpenDX map.")],
    n: Annotated[int, Field(description="How many peaks to return.", ge=1, le=50)] = 5,
    sign: Annotated[
        Literal["positive", "negative", "both"],
        Field(description="Which extremes to report."),
    ] = "both",
    min_separation: Annotated[
        float,
        Field(
            description=(
                "Minimum angstroms between reported peaks. Without this you get "
                "n neighbours of the same peak. Roughly a sidechain's reach by default."
            ),
            gt=0,
        ),
    ] = 5.0,
    pqr_path: Annotated[
        str | None,
        Field(
            description=(
                "The structure the map came from. Strongly recommended: without it "
                "the strongest values are the point-charge singularities at atom "
                "centres, so the answer is 'at the atoms' rather than 'at the "
                "binding sites'. Supplying it masks the solute interior."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Find where a map's strongest positive and negative patches are.

    This is the "where should I look" question — an anion-binding pocket shows
    up as a positive patch, a cation site as a negative one. Values very close
    to an atom centre are dominated by that charge's own self-energy, so treat
    the largest magnitudes with suspicion unless they sit away from the solute.
    """
    grid = _load_grid(dx_path)
    structure = None
    if pqr_path is not None:
        try:
            structure = read_pqr(Path(pqr_path).expanduser())
        except (OSError, ValueError) as exc:
            raise ToolError(f"could not read PQR {pqr_path}: {exc}") from exc

    out: dict[str, Any] = {
        "min_separation_a": min_separation,
        "solute_masked": structure is not None,
    }
    described = []

    if sign in ("positive", "both"):
        peaks = potential_extrema(
            grid, n=n, most_positive=True, min_separation=min_separation, exclude_near=structure
        )
        out["most_positive"] = [p.as_dict() for p in peaks]
        if peaks:
            described.append(f"most positive {peaks[0].value:+.3g} kT/e")
    if sign in ("negative", "both"):
        troughs = potential_extrema(
            grid, n=n, most_positive=False, min_separation=min_separation, exclude_near=structure
        )
        out["most_negative"] = [p.as_dict() for p in troughs]
        if troughs:
            described.append(f"most negative {troughs[0].value:+.3g} kT/e")

    caveat = "" if structure is not None else " (solute not masked — pass pqr_path)"
    out["summary"] = (
        f"{Path(dx_path).name}: " + ("; ".join(described) or "no extrema found") + caveat
    )
    return out


@mcp.tool
def sashimi_potential_in_sphere(
    *,
    dx_path: Annotated[str, Field(description="Path to an OpenDX map.")],
    centre: Annotated[list[float], Field(description="Sphere centre [x, y, z] in angstroms.")],
    radius: Annotated[float, Field(description="Sphere radius in angstroms.", gt=0)],
) -> dict[str, Any]:
    """Summarise the potential inside a sphere — a ligand pocket, say.

    Check `n_points` before trusting the mean: a sphere smaller than the grid
    spacing may contain almost nothing.
    """
    grid = _load_grid(dx_path)
    try:
        stats = potential_in_sphere(grid, np.asarray(centre, dtype=float), radius)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc

    if stats["n_points"] == 0:
        stats["summary"] = (
            f"No grid points inside a {radius} A sphere at {centre} — "
            "is it outside the map, or smaller than the grid spacing?"
        )
    else:
        stats["summary"] = (
            f"{stats['n_points']:,} points: mean {stats['mean_kT_e']:+.3g} kT/e, "
            f"range {stats['min_kT_e']:+.3g} to {stats['max_kT_e']:+.3g}."
        )
    return stats


@mcp.tool
def sashimi_residue_potentials(
    *,
    dx_path: Annotated[str, Field(description="Path to an OpenDX map.")],
    pqr_path: Annotated[
        str, Field(description="The PQR the map was computed from; supplies residue labels.")
    ],
    top: Annotated[
        int | None,
        Field(description="Return only the N most negative residues. Omit for all.", ge=1),
    ] = None,
    probe_offset: Annotated[
        float,
        Field(
            description=(
                "Angstroms outside each atom's radius to sample. Sampling at atom "
                "centres would report the atom's own self-energy, not its environment."
            ),
            ge=0,
        ),
    ] = 2.0,
) -> dict[str, Any]:
    """Mean potential around each residue, most negative first.

    Answers "which residues sit in negative potential" — the question behind
    cation binding, electrostatic steering and charge-complementarity work,
    without moving a grid anywhere.
    """
    grid = _load_grid(dx_path)
    try:
        structure = read_pqr(Path(pqr_path).expanduser())
    except (OSError, ValueError) as exc:
        raise ToolError(f"could not read PQR {pqr_path}: {exc}") from exc

    try:
        residues = residue_potentials(grid, structure, probe_offset=probe_offset, top=top)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc

    under_sampled = [r.label for r in residues if r.n_sampled < r.n_atoms]
    note = f" {len(under_sampled)} residue(s) partly outside the grid." if under_sampled else ""
    lead = residues[0] if residues else None
    return {
        "residues": [r.as_dict() for r in residues],
        "under_sampled": under_sampled,
        "summary": (
            f"{len(residues)} residue(s)."
            + (f" Most negative: {lead.label} at {lead.value:+.3g} kT/e." if lead else "")
            + note
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
