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
from pathlib import Path

import numpy as np
import pytest

from sashimi.analytic import born_potential
from sashimi.corpus import MANIFEST, load_summary, verify_case
from sashimi.errors import InputError, SolverError, UnsupportedRequest
from sashimi.pqr import parse_pqr, read_pqr
from sashimi.protocol import (
    BoundaryElementRequest,
    EnergyTerm,
    PotentialGrid,
    PQRData,
    SolventModel,
    SolverFamily,
    SurfaceModel,
)
from sashimi.tabipb import TabipbOptions, TabipbSolver, discover_tabipb
from tests.helpers import installed_or_skip, surface

pytestmark = pytest.mark.tabipb

BORN_ION_PQR = "ATOM      1  I   ION     1       0.000   0.000  0.000  1.00  3.00\n"


@pytest.fixture(scope="module")
def binary():
    """The installed TABI-PB, or a skip that never hides a broken install."""
    return installed_or_skip(discover_tabipb, "SASHIMI_TABIPB_PATH")


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


# --- the golden corpus, for a backend whose binary is built from source -------


TABIPB_DIRECTORY = Path("tests/corpus/tabipb")

# What the boundary-element tier can be asked, and what it costs. Measured on
# osx-arm64, 2026-08-12; every one of these is a corpus case with an APBS
# recording of the same question beside it.
#
# The tier a case declares is its APBS cost, which says nothing about this
# backend: `fas2-molecular` is `standard` and meshes in 48 s, while
# `ion-protein-complex-molecular` is a third the atoms and takes 450 s, because
# the cost is the mesh and not the solute. So what `pytest` re-verifies is named
# here rather than filtered from the manifest.
TABIPB_PER_PUSH = (
    "peptide-molecular",  # 0.1 s, 1,034 vertices
    "peptide-molecular-no-salt",  # 0.1 s — the salt arm, as a surface solver sees it
    "peptide-molecular-high-salt",  # 0.1 s
    "peptide-molecular-cold",  # 0.1 s — temperature, the axis DelPhi's Celsius bug lived on
)
TABIPB_ON_DEMAND = (
    "fas2-molecular",  # 48 s, 21,850 vertices, 1.26% from APBS
    "ion-protein-complex-molecular",  # 450 s, 68,054 vertices, 1.02% from APBS
)


@pytest.mark.parametrize("name", TABIPB_PER_PUSH)
def test_tabipb_reproduces_its_recorded_corpus_answer(binary, name):
    """Boundary-element summaries, recorded and re-verified.

    What is compared is the vertex count and the statistics over the surface —
    not pinned probe coordinates, because the vertices are the mesher's choice
    and move when it is rebuilt. That matters for this backend in particular:
    CI compiles it, and its mesher's version is part of a result's identity.

    Not every corpus case can be asked. `born-ion-molecular` has one atom and
    `methanol-molecular` three, below NanoShaper's four; `acetate-molecular` at
    eight atoms does not finish inside the backend's own 600 s timeout, twice
    measured; and `aspartate-residue-molecular` at twelve atoms fails in the
    mesher immediately and differently, on `stoul: no conversion` after it has
    already reported building the surface. Three small-molecule failures, three
    mechanisms, none of them size alone.
    """
    case = next(c for c in MANIFEST if c.name == name)
    recorded = load_summary(case, TABIPB_DIRECTORY)

    found = verify_case(TabipbSolver(), case, recorded, family=SolverFamily.BOUNDARY_ELEMENT)

    assert found == []
    assert recorded["family"] == "boundary-element"
    assert "geometry" not in recorded  # there is no volume to record
    assert recorded["surface"]["n_vertices"] > 0


# `TABIPB_ON_DEMAND` is re-solved by
# `sashimi corpus verify --backend tabipb --directory tests/corpus/tabipb
# --case <name>`, which is eight minutes of meshing a per-push suite has no
# business repeating. `tests/test_corpus_manifest.py` checks those recordings
# are present and where they landed, without needing the binary at all.


# --- units --------------------------------------------------------------------

# NanoShaper refuses fewer than four atoms, so the sphere is four coincident ones
# at the vertices of a tiny regular tetrahedron: their union is a sphere to
# within the offset, and splitting the charge equally cancels the dipole by
# symmetry, leaving the centred monopole the closed form describes.
_SPHERE_RADIUS = 3.0
_SPHERE_OFFSET = 0.01
_TETRAHEDRON = np.array(
    [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
) / np.sqrt(3.0)


def test_the_surface_potential_crosses_the_protocol_boundary_in_kt_per_e(binary):
    """Grade TABI-PB's field against the one closed form that fixes its unit.

    This is here because the energy self-test cannot do it. TABI-PB reports its
    energy in kJ/mol and its potential in kJ/mol/e, so for six recordings the
    field was RT — a factor of 2.48 — too large while every energy stayed right
    and every corpus check stayed green.

    A sphere is what pins it: outside a centred monopole the potential is exactly
    q / (4 pi eps0 eps_s r), so every vertex has a known answer.

    The unit is one scalar factor, so the mean ratio is what estimates it and the
    3% band is on that. The spread is graded separately and far more loosely,
    because it is the mesh's error rather than the unit's: at `sdens` 3 the mean
    sits 1.75% high and falls as h^2 (`studies/tabipb_units/born_sphere.py`
    walks it to 0.44%), while individual vertices reach 6.7%. Neither tolerance
    is delicate — dropping the conversion puts the mean at 2.479.
    """
    structure = PQRData(
        coords=_TETRAHEDRON * _SPHERE_OFFSET,
        charges=np.full(4, 0.25),
        radii=np.full(4, _SPHERE_RADIUS),
    )
    request = _request(structure, mesh_density=3.0)
    mesh = surface(TabipbSolver().solve(request))

    radii = np.linalg.norm(mesh.vertices, axis=1)
    exact = np.array(
        [
            born_potential(r, 1.0, request.solvent.solvent_dielectric, request.solvent.temperature)
            for r in radii
        ]
    )
    ratio = mesh.values / exact
    assert float(np.mean(ratio)) == pytest.approx(1.0, rel=0.03), (
        f"surface potential is {float(np.mean(ratio)):.3f}x the closed form; "
        f"1.0 is kT/e and 2.479 is the unconverted kJ/mol/e"
    )
    assert float(np.max(np.abs(ratio - 1.0))) < 0.10, (
        "the field is off the closed form vertex by vertex, which a wrong unit "
        "would not do — suspect the mesh or the geometry, not the conversion"
    )


def test_the_surface_potential_is_temperature_dependent_as_kt_per_e_must_be(binary):
    """The property that caught it: kT/e carries a 1/T and kJ/mol/e does not.

    TABI-PB's own output is byte-identical at these two temperatures at zero
    salt, so anything that varies here is the conversion doing its job. Solving
    at both is what makes this a check on the unit rather than on a constant.
    """
    structure = PQRData(
        coords=_TETRAHEDRON * _SPHERE_OFFSET,
        charges=np.full(4, 0.25),
        radii=np.full(4, _SPHERE_RADIUS),
    )
    hot = surface(TabipbSolver().solve(_request(structure, mesh_density=3.0)))
    cold = surface(
        TabipbSolver().solve(_request(structure, mesh_density=3.0, solvent={"temperature": 277.0}))
    )
    assert float(np.mean(cold.values)) / float(np.mean(hot.values)) == pytest.approx(
        298.15 / 277.0, rel=1e-6
    )
