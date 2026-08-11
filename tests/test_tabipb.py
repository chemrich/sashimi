"""TABI-PB backend, binary-free tier.

VTK parsing, the mesh mapping, input generation and the meshability guard — all
pure text and arithmetic, so this runs anywhere. The real solve is
`tests/test_tabipb_solver.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from sashimi.errors import InputError, MalformedStructure, UnsupportedRequest
from sashimi.protocol import SolventModel, SurfaceModel
from sashimi.tabipb.input import build_input, resolved_parameters
from sashimi.tabipb.options import SUPPORTED_SURFACES, TabipbOptions, resolve_mesh
from sashimi.tabipb.run import MIN_ATOMS, check_meshable
from sashimi.tabipb.vtk import parse_vtk

VTK = """\
# vtk DataFile Version 1.0
vtk file output.vtk
ASCII
DATASET POLYDATA

POINTS 4 double
0.000000 0.000000 0.000000
1.000000 0.000000 0.000000
0.000000 1.000000 0.000000
0.000000 0.000000 1.000000

POLYGONS 2 8
3 0 1 2
3 0 1 3

POINT_DATA 4
SCALARS Potential double
LOOKUP_TABLE default
1.5
-2.5
3.5
-4.5
SCALARS NormalPotential double
LOOKUP_TABLE default
10.0
20.0
30.0
40.0
"""


def solvent(model: SurfaceModel = SurfaceModel.MOLECULAR, **kwargs) -> SolventModel:
    return SolventModel(surface_model=model, **kwargs)


# --- VTK -> SurfacePotential -------------------------------------------------


def test_vtk_becomes_a_surface_potential():
    parsed = parse_vtk(VTK)
    surface = parsed.potential

    assert surface.n_vertices == 4
    assert surface.vertices.shape == (4, 3)
    assert surface.values == pytest.approx([1.5, -2.5, 3.5, -4.5])
    assert surface.triangles is not None
    assert surface.triangles.tolist() == [[0, 1, 2], [0, 1, 3]]


def test_the_normal_derivative_is_kept_rather_than_dropped():
    """The other half of what a BEM solver produces, with nowhere in the protocol
    to live yet. Parsing it and handing it back beats discarding it silently."""
    parsed = parse_vtk(VTK)
    assert parsed.normal_derivative == pytest.approx([10.0, 20.0, 30.0, 40.0])


def test_surface_stats_report_a_surface_not_a_volume():
    stats = parse_vtk(VTK).potential.stats()
    assert stats["kind"] == "surface"
    assert stats["n_vertices"] == 4


def test_a_truncated_vtk_is_rejected():
    with pytest.raises(MalformedStructure, match="truncated"):
        parse_vtk(VTK.replace("0.000000 0.000000 1.000000\n", ""))


def test_a_missing_scalars_block_is_named():
    with pytest.raises(MalformedStructure, match="NormalPotential"):
        parse_vtk(VTK.replace("SCALARS NormalPotential double", "SCALARS Other double"))


def test_a_non_triangular_mesh_is_refused():
    """A quad mesh would otherwise parse into silently wrong connectivity."""
    quads = VTK.replace("POLYGONS 2 8\n3 0 1 2\n3 0 1 3", "POLYGONS 1 5\n4 0 1 2 3")
    with pytest.raises(MalformedStructure, match="triangular"):
        parse_vtk(quads)


def test_a_file_without_points_is_refused():
    with pytest.raises(MalformedStructure, match="POINTS"):
        parse_vtk("# vtk DataFile Version 1.0\nASCII\n")


# --- the mesh mapping --------------------------------------------------------


def test_molecular_maps_to_the_solvent_excluded_mesh():
    assert resolve_mesh(SurfaceModel.MOLECULAR, TabipbOptions()) == "ses"


@pytest.mark.parametrize("model", [SurfaceModel.SMOOTHED_MOLECULAR, SurfaceModel.GAUSSIAN])
def test_grid_only_surface_models_are_refused_as_grid_concepts(model):
    """Not a missing feature: they describe a dielectric varying over a volume."""
    with pytest.raises(UnsupportedRequest, match="does not have"):
        resolve_mesh(model, TabipbOptions())


def test_van_der_waals_is_refused_for_its_own_reason():
    with pytest.raises(UnsupportedRequest, match="zero probe radius"):
        resolve_mesh(SurfaceModel.VAN_DER_WAALS, TabipbOptions())


def test_the_skin_surface_is_reachable_only_by_asking_for_it():
    """It has no solver-neutral name, so it lives behind options like spl2 does."""
    assert resolve_mesh(SurfaceModel.MOLECULAR, TabipbOptions(mesh_override="skin")) == "skin"


def test_an_unknown_mesh_override_is_rejected():
    with pytest.raises(ValueError, match="mesh_override"):
        TabipbOptions(mesh_override="blancmange")


def test_tree_theta_must_be_a_fraction():
    with pytest.raises(ValueError, match="tree_theta"):
        TabipbOptions(tree_theta=1.5)


def test_tabipb_shares_the_molecular_surface_with_the_grid_backends():
    """The precondition for cross-family validation being possible at all."""
    from sashimi.apbs.options import SURFACE_KEYWORD  # noqa: PLC0415

    assert SurfaceModel.MOLECULAR in (set(SURFACE_KEYWORD) & SUPPORTED_SURFACES)


# --- input generation --------------------------------------------------------


def test_the_input_carries_the_physical_request():
    text = build_input(solvent(ionic_strength=0.15, surface_radius=1.4), mesh_density=2.5)

    assert "mesh              ses" in text
    assert "sdens             2.500000" in text
    assert "srad              1.4000" in text
    assert "bulk              0.150000" in text
    assert "temp              298.1500" in text


def test_the_vtk_is_requested_only_when_a_potential_is_wanted():
    """The CSV is a summary row; only the VTK carries the per-vertex field."""
    assert "outdata           vtk" in build_input(solvent(), 2.0, write_potential=True)
    assert "outdata" not in build_input(solvent(), 2.0, write_potential=False)


def test_mesh_density_reaches_the_file_as_sdens():
    """`mesh_density` is the BEM analogue of grid resolution, which is why it
    lives on `BoundaryElementRequest` rather than in `GridSpec`."""
    assert "sdens             4.000000" in build_input(solvent(), mesh_density=4.0)


def test_resolved_parameters_record_the_mapped_mesh():
    resolved = resolved_parameters(solvent(), 2.0, TabipbOptions())

    assert resolved["surface_model"] == "molecular"
    assert resolved["tabipb"]["mesh"] == "ses"
    assert resolved["tabipb"]["sdens"] == 2.0
    # tree_theta = 0 is the exact setting; a faster default would surface as a
    # solver disagreement and be blamed on the solver.
    assert resolved["tabipb"]["tree_theta"] == 0.0


# --- meshability -------------------------------------------------------------


@pytest.mark.parametrize("n_atoms", [1, 2, 3])
def test_a_solute_too_small_to_triangulate_is_refused(n_atoms):
    """The Born ion has no boundary-element answer; say so, in advance.

    Without this the failure is `stoul: no conversion` from inside a C++ binary,
    several layers below anything the caller can act on.
    """
    with pytest.raises(InputError, match="at least 4 atoms"):
        check_meshable(n_atoms)


def test_a_meshable_solute_passes():
    check_meshable(MIN_ATOMS)


def test_the_refusal_points_at_a_backend_that_can_answer():
    with pytest.raises(InputError, match="finite-difference"):
        check_meshable(1)


def test_vertices_and_values_must_agree():
    """`SurfacePotential` enforces its own shape; a mismatched VTK must not slip through."""
    broken = VTK.replace("POINT_DATA 4", "POINT_DATA 3")
    with pytest.raises(MalformedStructure):
        parse_vtk(broken.replace("1.5\n-2.5\n3.5\n-4.5", "1.5\n-2.5\n3.5"))


def test_vertices_are_read_in_order():
    vertices = parse_vtk(VTK).potential.vertices
    assert np.array_equal(vertices[1], np.array([1.0, 0.0, 0.0]))
