"""TABI-PB behaviour that needs the real executables.

Gated behind `@pytest.mark.tabipb` and skipped where TABI-PB is absent, which is
most places: it is built from source and needs a NanoShaper beside it.

What these check is not the physics — `tests/test_cross_validation.py` does that
against the other backends — but that a boundary-element solver travels through
the protocol unchanged. `SurfacePotential` and `BoundaryElementRequest` were
designed in phase 4 for a solver that did not exist, and until now only a stub
exercised them.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from sashimi.errors import InputError, SolverError, UnsupportedRequest
from sashimi.pqr import parse_pqr, read_pqr
from sashimi.protocol import (
    BoundaryElementRequest,
    EnergyTerm,
    PotentialGrid,
    SolventModel,
    SurfaceModel,
)
from sashimi.tabipb import TabipbOptions, TabipbSolver, discover_tabipb
from sashimi.tabipb.discover import TabipbNotFound
from tests.helpers import surface

pytestmark = pytest.mark.tabipb

BORN_ION_PQR = "ATOM      1  I   ION     1       0.000   0.000  0.000  1.00  3.00\n"


@pytest.fixture(scope="module")
def binary():
    """The installed TABI-PB, or a skip that never hides a broken install."""
    try:
        return discover_tabipb()
    except TabipbNotFound as exc:
        if os.environ.get("SASHIMI_TABIPB_PATH"):
            raise
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def peptide():
    return read_pqr(Path(__file__).resolve().parent / "data" / "ala-gly.pqr")


def _request(structure, **kwargs) -> BoundaryElementRequest:
    solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR, ionic_strength=0.0)
    return BoundaryElementRequest(
        structure=structure,
        solvent=dataclasses.replace(solvent, **kwargs.pop("solvent", {})),
        mesh_density=kwargs.pop("mesh_density", 2.0),
        **kwargs,
    )


def test_discovery_reports_the_solver_and_its_mesher(binary):
    """The mesher is part of a result's identity, not an implementation detail."""
    assert binary.path.is_file()
    assert binary.mesher_path.is_file()
    assert binary.label.startswith("tabipb")
    assert "nanoshaper" in binary.label
    assert len(binary.sha256) == 64


def test_a_boundary_element_solve_returns_a_surface_not_a_volume(binary, peptide):
    """ROADMAP.md section 2's acid test, with a real solver rather than a stub."""
    result = TabipbSolver().solve(_request(peptide))
    mesh = surface(result)

    assert not isinstance(result.potential, PotentialGrid)
    assert mesh.n_vertices > 100
    assert mesh.vertices.shape == (mesh.n_vertices, 3)
    assert mesh.triangles is not None
    assert mesh.stats()["kind"] == "surface"


def test_the_energy_is_already_in_protocol_units(binary, peptide):
    """TABI-PB reports kJ/mol directly — the one backend needing no conversion.

    A sanity band rather than a pinned value: ALA-GLY's polar solvation energy
    is a couple of hundred kJ/mol and negative, and anything outside that is a
    unit error rather than a modelling difference.
    """
    result = TabipbSolver().solve(_request(peptide))

    assert result.energy_kj_mol is not None
    assert -400 < result.energy_kj_mol < -100
    assert result.provenance.energy_term is EnergyTerm.POLAR_SOLVATION


def test_the_energy_moves_with_salt(binary, peptide):
    """Which is what `EnergyTerm.POLAR_SOLVATION` claims, so it must be true.

    `sashimi.validate` trusts the declared term when deciding whether a spread
    is meaningful, so a backend whose number contradicts its own label would
    corrupt every comparison downstream.
    """
    plain = TabipbSolver().solve(_request(peptide))
    salted = TabipbSolver().solve(_request(peptide, solvent={"ionic_strength": 0.5}))

    assert salted.energy_kj_mol is not None
    assert plain.energy_kj_mol is not None
    assert salted.energy_kj_mol < plain.energy_kj_mol


def test_a_denser_mesh_produces_more_vertices(binary, peptide):
    """`mesh_density` is the BEM analogue of grid resolution and must act like it."""
    coarse = surface(TabipbSolver().solve(_request(peptide, mesh_density=1.5)))
    fine = surface(TabipbSolver().solve(_request(peptide, mesh_density=3.0)))

    assert fine.n_vertices > coarse.n_vertices


def test_a_too_coarse_mesh_names_mesh_density_as_the_cause(binary, peptide):
    """The protocol's default mesh_density is 1.0, and TABI-PB aborts below 1.5.

    The abort is an uncaught C++ exception carrying no clue, so the likeliest
    first call a caller makes would otherwise fail unintelligibly.
    """
    with pytest.raises(SolverError, match="mesh_density"):
        TabipbSolver().solve(_request(peptide, mesh_density=1.0))


def test_the_born_ion_is_refused_because_it_cannot_be_triangulated(binary):
    """The one case with a closed form is the one BEM cannot take.

    A single sphere has no mesh NanoShaper will build, so the analytic
    calibration that anchors the finite-difference backends is unavailable here.
    Saying so plainly beats a C++ exception from inside the mesher.
    """
    with pytest.raises(InputError, match="at least 4 atoms"):
        TabipbSolver().solve(_request(parse_pqr(BORN_ION_PQR)))


def test_a_grid_only_surface_model_is_refused(binary, peptide):
    with pytest.raises(UnsupportedRequest, match="smoothed-molecular"):
        TabipbSolver().solve(
            _request(peptide, solvent={"surface_model": SurfaceModel.SMOOTHED_MOLECULAR})
        )


def test_provenance_records_the_mesh_and_the_mesher(binary, peptide):
    result = TabipbSolver().solve(_request(peptide))
    resolved = result.provenance.resolved_parameters

    assert result.provenance.backend == binary.label
    assert resolved["tabipb"]["mesh"] == "ses"
    assert result.diagnostics["family"] == "boundary-element"
    assert result.diagnostics["mesher"] == binary.mesher_path.name


def test_the_skin_surface_is_reachable_through_options(binary, peptide):
    """TABI-PB's other mesh has no portable name; asking for it must be explicit."""
    result = TabipbSolver(options=TabipbOptions(mesh_override="skin")).solve(_request(peptide))

    assert result.provenance.resolved_parameters["tabipb"]["mesh"] == "skin"
    assert result.energy_kj_mol is not None


def test_energy_only_request_skips_the_mesh_output(binary, peptide):
    result = TabipbSolver().solve(_request(peptide, want_potential=False))

    assert result.energy_kj_mol is not None
    assert result.potential is None
