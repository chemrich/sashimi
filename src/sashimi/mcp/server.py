"""FastMCP tools.

All prefixed `sashimi_`. Parameters are flat and physically named
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

import dataclasses
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from sashimi.analysis import potential_extrema, potential_in_sphere, residue_potentials
from sashimi.artifacts import content_address, describe_cleanup, map_path
from sashimi.backends import get as get_backend
from sashimi.backends import resolve as resolve_preference
from sashimi.capabilities import describe_capabilities, validate_request
from sashimi.dx import read_dx
from sashimi.errors import SashimiError
from sashimi.pqr import read_pqr
from sashimi.prep import ForceField, prepare_structure
from sashimi.protocol import (
    GridSpec,
    PotentialGrid,
    SolventModel,
    SolverFamily,
    SurfaceModel,
    SurfacePotential,
    System,
)

__all__ = ["mcp"]

mcp: FastMCP[Any] = FastMCP(
    name="sashimi",
    instructions=(
        "Biomolecular electrostatics via APBS. Typical flow: "
        "sashimi_prepare_structure on a PDB to get a PQR (read its warnings — "
        "rebuilt sidechains change the energies), then sashimi_solve for a "
        "potential map and optional solvation energy, then sashimi_potential_at "
        "to query the saved map without re-solving. Call sashimi_capabilities "
        "to see what this installation supports, and sashimi_validate_inputs to "
        "check a solve's cost before running it. Potentials are kT/e, "
        "energies kJ/mol, distances angstroms. " + describe_cleanup()
    ),
)


def _fail(exc: SashimiError) -> ToolError:
    """Surface a typed sashimi failure with its message intact."""
    return ToolError(str(exc))


def _surface_model(name: str) -> SurfaceModel:
    """Parse a surface model, naming the alternatives when it is not one.

    Both tools take this as a string, and only `sashimi_validate_inputs` used
    to check it: an unknown value reached `sashimi_solve` as a bare `ValueError`
    and an agent got a server-side traceback saying what was wrong but not what
    would work. Moving the default to `molecular` made that likelier rather than
    rarer — asking for the old boundary now means typing `smoothed-molecular`,
    and `smoothed_molecular` is a near-miss that lands here.
    """
    try:
        return SurfaceModel(name)
    except ValueError as exc:
        supported = ", ".join(sorted(m.value for m in SurfaceModel))
        raise ToolError(f"unknown surface_model {name!r}; one of: {supported}") from exc


def _grid_spec(resolution: float | None, padding: float | None) -> GridSpec:
    """A grid spec where unset means the protocol's default, not this layer's."""
    spec = GridSpec()
    if resolution is not None:
        spec = dataclasses.replace(spec, resolution=resolution)
    if padding is not None:
        spec = dataclasses.replace(spec, padding=padding)
    return spec


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
        float | None,
        Field(
            description=(
                "Target fine-grid spacing, angstroms. Grid backends only "
                "(apbs, delphi); setting it for a backend that builds no grid "
                "is refused rather than ignored."
            ),
            gt=0,
            le=5.0,
        ),
    ] = None,
    padding: Annotated[
        float | None,
        Field(
            description=(
                "Minimum solute-surface to grid-edge distance, angstroms. Grid backends only."
            ),
            ge=0,
        ),
    ] = None,
    mesh_density: Annotated[
        float | None,
        Field(
            description=(
                "Vertices per square angstrom of dielectric surface \u2014 the "
                "boundary-element cost knob, and the analogue of `resolution` "
                "for a solver that meshes instead of filling a volume. tabipb "
                "only, and it aborts below 1.5. Cost is the mesh rather than "
                "the atom count: 906 atoms mesh in 48 s where a 260-atom "
                "united-atom structure takes 450 s at three times the vertices."
            ),
            gt=0,
        ),
    ] = None,
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
                "molecular, smoothed-molecular, van-der-waals. The default, "
                "molecular, is the only one every backend can answer on; "
                "smoothed-molecular is APBS's alone."
            )
        ),
    ] = "molecular",
    backend: Annotated[
        str,
        Field(
            description=(
                "Which solver to run. `sashimi_capabilities` lists what this "
                "installation has and what each supports. apbs and delphi fill a "
                "volume and return a map; tabipb solves on the dielectric "
                "surface and returns statistics over it, with no map to write; "
                "gb approximates the energy in process, needs no binary at all, "
                "and returns no field of any kind."
            )
        ),
    ] = "apbs",
    prefer: Annotated[
        str | None,
        Field(
            description=(
                "Pick the backend by what you want optimised instead of by "
                "name: 'fast' (lowest CPU), 'stable' (least sensitive to where "
                "the lattice falls), or 'portable' (no install step). Resolves "
                "against what is installed *and* what surface_model needs, so "
                "'stable' falls through to another solver for a surface "
                "pyDelPhi cannot build. The result reports which backend ran "
                "and why. Naming `backend` explicitly overrides this."
            )
        ),
    ] = None,
    output_dx: Annotated[
        str | None,
        Field(
            description=(
                "Where to write the OpenDX map. Defaults to a content-addressed "
                "name next to the PQR, so re-solving with different parameters "
                "never overwrites an earlier map. Ignored by backends that "
                "produce no volumetric map."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Solve the linearized Poisson-Boltzmann equation for a prepared structure.

    Returns the solvation energy when requested, which backend produced it, and
    — for the backends that fill a volume — grid statistics and the path to an
    OpenDX map that PyMOL and ChimeraX can load. Note `spacing_achieved`: the
    memory guardrail may relax the requested resolution, and the response says
    so rather than hiding it.

    **The response shape follows what the backend actually returned**, rather
    than a fixed schema it has to satisfy. A boundary-element solve carries
    surface statistics and no `dx_path`, because there is no volume to write; an
    analytic one carries an energy and nothing else, because it computed nothing
    else. `sashimi_validate_inputs` answers whether a given backend can take a
    request before it is run.

    **A parameter the chosen backend cannot use is refused, not ignored.**
    `resolution` and `padding` describe a grid; `mesh_density` describes a
    triangulation; no backend has both. Silently accepting one that does nothing
    is how a caller comes to believe it made a 450-second mesh cheaper by
    halving a resolution the mesher never reads — and quietly-wrong parameters
    are the failure this project has hit three times over (DelPhi reading
    `temper` as Celsius, DelPhi reading a different PQR radius column,
    Generalized Born being handed the wrong radius dialect). Every one produced
    a plausible number from an input that was not what the caller thought.

    Maps are files, not inline data — a default-resolution grid is megabytes.
    """
    source = Path(pqr_path).expanduser()
    try:
        pqr = read_pqr(source)
    except (OSError, ValueError) as exc:
        raise ToolError(f"could not read PQR {source}: {exc}") from exc

    selected_because = ""
    if prefer is not None:
        # An explicit `backend` wins: a preference is a convenience for a caller
        # who knows what they want from the answer, never something that
        # second-guesses one who named the solver.
        try:
            backend, selected_because = resolve_preference(prefer, surface_model)
        except SashimiError as exc:
            raise _fail(exc) from exc

    try:
        entry = get_backend(backend)
    except SashimiError as exc:
        raise _fail(exc) from exc

    grid_parameters = {"resolution": resolution, "padding": padding}
    named = {k: v for k, v in grid_parameters.items() if v is not None}
    if named and entry.family is not SolverFamily.FINITE_DIFFERENCE:
        raise ToolError(
            f"{backend} builds no grid, so {', '.join(sorted(named))} would do nothing. "
            + (
                "Use mesh_density, which is what this backend's cost depends on."
                if entry.family is SolverFamily.BOUNDARY_ELEMENT
                else "It solves in process and has no discretization to set."
            )
        )
    if mesh_density is not None and entry.family is not SolverFamily.BOUNDARY_ELEMENT:
        raise ToolError(
            f"mesh_density describes a triangulated surface and {backend} does not "
            "build one. Use resolution for a grid backend."
        )

    field_expected = entry.family is not SolverFamily.ANALYTIC
    if not field_expected and not compute_energy:
        raise ToolError(
            f"{backend} computes a solvation energy and no field, so this request "
            "asks it for nothing. Set compute_energy=true, or choose a backend "
            "that returns a map."
        )

    system = System(
        structure=pqr,
        solvent=SolventModel(
            solvent_dielectric=solvent_dielectric,
            solute_dielectric=solute_dielectric,
            ionic_strength=ionic_strength,
            surface_model=_surface_model(surface_model),
        ),
        # Unset means the protocol's own default rather than a number this
        # layer invents; `System` and `BoundaryElementRequest` disagree about
        # the mesh one and §14 has that open.
        grid=_grid_spec(resolution, padding),
        want_energy=compute_energy,
        want_potential=field_expected,
        **({} if mesh_density is None else {"mesh_density": mesh_density}),
    )

    try:
        result = entry.solver().solve(system.request_for(entry.family))
    except SashimiError as exc:
        raise _fail(exc) from exc

    energy = (
        f" Polar solvation energy {result.energy_kj_mol:.3f} kJ/mol."
        if result.energy_kj_mol is not None
        else ""
    )
    response: dict[str, Any] = {
        "energy_kj_mol": result.energy_kj_mol,
        "backend": result.provenance.summary(),
        "backend_name": entry.name,
        "family": entry.family.value,
        "resolved_parameters": result.provenance.resolved_parameters,
        "diagnostics": result.diagnostics,
    }
    if selected_because:
        # Why this backend and not another. A caller who asked for `stable` and
        # got APBS should not have to guess that pyDelPhi was skipped for the
        # surface model rather than for being absent.
        response["selected_because"] = selected_because

    potential = result.potential
    if isinstance(potential, PotentialGrid):
        # Content-addressed by default: two solves that differ in anything that
        # changes the answer get different files, so re-solving cannot silently
        # overwrite a previous map. An explicit output_dx is honored as given.
        address = content_address(pqr, result.provenance.resolved_parameters)
        destination = Path(output_dx).expanduser() if output_dx else map_path(source, address)
        destination.parent.mkdir(parents=True, exist_ok=True)
        potential.to_dx(destination)

        stats = potential.stats()
        relaxed = result.diagnostics.get("resolution_relaxed", False)
        note = " Requested resolution was relaxed to fit max_points." if relaxed else ""
        response |= {
            "dx_path": str(destination),
            "content_address": address,
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
            "summary": (
                f"Solved {source.name} on a {'x'.join(str(n) for n in stats['shape'])} grid "
                f"at {result.diagnostics['spacing_achieved'][0]:.3f} A.{energy}{note} "
                f"Map written to {destination.name} ({result.provenance.backend})."
            ),
        }
        return response

    if isinstance(potential, SurfacePotential):
        stats = potential.stats()
        response |= {
            "surface": {
                "n_vertices": potential.n_vertices,
                "potential_kT_e": {key: stats[key] for key in ("min", "max", "mean", "std")},
            },
            "summary": (
                f"Solved {source.name} on a {potential.n_vertices:,}-vertex dielectric "
                f"surface.{energy} No map written: a boundary-element solver has no "
                f"volume to sample ({result.provenance.backend})."
            ),
        }
        return response

    response["summary"] = (
        f"Solved {source.name} in process.{energy} No map written: "
        f"{entry.name} returns an energy and no field ({result.provenance.backend})."
    )
    return response


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

    On a multi-chain structure the residue name is prefixed, because `SER 58`
    alone names two different residues. `A:SER 58` is a chain ID the file
    stated. `#2:SER 58` means the file named no chains and the second block is
    an inference from a numbering restart — treat it as "some other chain",
    not as a chain called 2. Prepare structures through
    `sashimi_prepare_structure`, which keeps the chain IDs that pdb2pqr would
    otherwise drop.
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


@mcp.tool
def sashimi_capabilities() -> dict[str, Any]:
    """What this installation can do: backends, surface models, units, limits.

    Ask this before planning work rather than discovering the answer by failing.
    A missing backend is reported here, with the reason — this tool is
    deliberately the one that still works when nothing else does.
    """
    return describe_capabilities()


@mcp.tool
def sashimi_validate_inputs(
    *,
    pqr_path: Annotated[str, Field(description="Path to a PQR file.")],
    resolution: Annotated[
        float, Field(description="Target fine-grid spacing, angstroms.", gt=0, le=5.0)
    ] = 0.5,
    padding: Annotated[
        float,
        Field(description="Minimum solute-surface to grid-edge distance, angstroms.", ge=0),
    ] = 10.0,
    surface_model: Annotated[
        str, Field(description="Dielectric boundary definition, as for sashimi_solve.")
    ] = "molecular",
    max_points: Annotated[
        int | None,
        Field(description="Grid point budget. Omit for the default guardrail.", gt=0),
    ] = None,
    backend: Annotated[
        str, Field(description="Which solver the request is being checked against.")
    ] = "apbs",
    mesh_density: Annotated[
        float | None,
        Field(description="Vertices per square angstrom, for boundary-element backends.", gt=0),
    ] = None,
) -> dict[str, Any]:
    """Check a solve before paying for it: grid shape, map size, whether it works.

    Runs no solver. Grid sizing is arithmetic, so the point count, the achieved
    spacing, the estimated map size on disk and any blocking problem are all
    knowable in milliseconds — worth asking first when a solve can take a minute
    and write tens of megabytes.

    Check the `backend` you actually intend to run. The likeliest refusal is a
    surface model it does not support: the default `molecular` is the one every
    backend answers on, but three of the four refuse an explicit
    `smoothed-molecular`, and that refusal is free here and expensive after a
    structure has been prepared. The cost estimate is a grid one, so for
    a boundary-element or analytic backend it is omitted rather than invented,
    and the report says what governs cost instead.

    `ok` is false only for problems that would prevent the solve. A relaxed
    resolution is reported as a warning, because the solve would still run and
    the result would say it happened.
    """
    try:
        structure = read_pqr(Path(pqr_path).expanduser())
    except (OSError, ValueError) as exc:
        raise ToolError(f"could not read PQR {pqr_path}: {exc}") from exc

    solvent = SolventModel(surface_model=_surface_model(surface_model))

    spec = GridSpec(
        resolution=resolution,
        padding=padding,
        **({"max_points": max_points} if max_points is not None else {}),
    )
    try:
        return validate_request(
            structure, spec, solvent, backend=backend, mesh_density=mesh_density
        )
    except SashimiError as exc:
        raise _fail(exc) from exc


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
