import numpy as np
import pytest

from sashimi.apbs.grid import CFAC, LEGAL_DIME, legal_dime, size_grid
from sashimi.apbs.input import build_input
from sashimi.apbs.options import ApbsOptions
from sashimi.errors import GridTooLarge, UnsupportedRequest
from sashimi.protocol import Equation, GridSpec, PQRData, SolventModel, SurfaceModel


def ion(radius=3.0):
    return PQRData(
        coords=np.zeros((1, 3)),
        charges=np.array([1.0]),
        radii=np.array([radius]),
    )


def test_legal_dime_lattice_is_multigrid_compatible():
    """APBS requires n = c * 2^(l+1) + 1; with 4 levels that is 32c + 1."""
    for n in LEGAL_DIME:
        assert (n - 1) % 32 == 0
    assert LEGAL_DIME[:4] == (33, 65, 97, 129)


def test_legal_dime_rounds_up():
    assert legal_dime(33) == 33
    assert legal_dime(34) == 65
    assert legal_dime(65) == 65
    assert legal_dime(100) == 129


def test_fine_grid_honors_padding():
    grid = size_grid(ion(), GridSpec(resolution=0.5, padding=10.0))
    # extent is 6 A (a 3 A sphere); padding adds 10 A on each side.
    np.testing.assert_allclose(grid.fglen, [26.0, 26.0, 26.0])


def test_coarse_grid_is_larger_than_fine():
    """Otherwise the Debye-Huckel boundary sits on the fine-grid edge."""
    grid = size_grid(ion(), GridSpec())
    assert all(c > f for c, f in zip(grid.cglen, grid.fglen, strict=True))
    np.testing.assert_allclose(grid.cglen, np.array(grid.fglen) * CFAC)


def test_resolution_target_is_met():
    spec = GridSpec(resolution=0.5, padding=10.0)
    grid = size_grid(ion(), spec)
    assert all(s <= spec.resolution for s in grid.spacing)
    assert grid.dime == (65, 65, 65)


def test_finer_resolution_gives_more_points():
    coarse = size_grid(ion(), GridSpec(resolution=0.8))
    fine = size_grid(ion(), GridSpec(resolution=0.2))
    assert fine.n_points > coarse.n_points


def test_max_points_caps_the_grid_and_relaxes_resolution():
    spec = GridSpec(resolution=0.1, padding=10.0, max_points=65**3)
    grid = size_grid(ion(), spec)
    assert grid.n_points <= spec.max_points
    # The budget bit, so the achieved spacing is coarser than requested.
    assert max(grid.spacing) > spec.resolution


def test_unsatisfiable_budget_raises_with_actionable_message():
    spec = GridSpec(resolution=0.5, padding=50.0, max_points=1000)
    with pytest.raises(GridTooLarge, match="Raise max_points or reduce padding"):
        size_grid(ion(), spec)


def test_grid_is_centered_on_the_molecule():
    off_center = PQRData(
        coords=np.array([[10.0, -5.0, 2.0]]),
        charges=np.array([1.0]),
        radii=np.array([3.0]),
    )
    grid = size_grid(off_center, GridSpec())
    np.testing.assert_allclose(grid.center, [10.0, -5.0, 2.0])


def test_input_uses_the_single_supported_template():
    text = build_input(size_grid(ion(), GridSpec()), SolventModel())
    for required in ("mg-auto", "lpbe", "bcfl sdh", "chgm spl4", "write pot dx"):
        assert required in text
    # Out-of-scope solvers must never appear.
    for forbidden in ("fe-manual", "geoflow", "pbam", "pbsam", "npbe"):
        assert forbidden not in text


def test_energy_request_adds_a_reference_block_and_print_statement():
    """Solvation energy is a difference; a single block reports self-energy."""
    without = build_input(size_grid(ion(), GridSpec()), SolventModel())
    with_energy = build_input(size_grid(ion(), GridSpec()), SolventModel(), compute_energy=True)

    assert without.count("elec name") == 1
    assert "print elecEnergy" not in without
    assert "calcenergy no" in without

    assert with_energy.count("elec name") == 2
    assert "elec name reference" in with_energy
    assert "print elecEnergy solvated - reference end" in with_energy
    assert "calcenergy total" in with_energy


def test_reference_block_has_no_mobile_ions():
    solvent = SolventModel(ionic_strength=0.15, solute_dielectric=2.0)
    text = build_input(size_grid(ion(), GridSpec()), solvent, compute_energy=True)
    solvated, reference = text.split("elec name reference")
    assert "ion charge" in solvated
    assert "ion charge" not in reference
    # Reference is a uniform dielectric at the solute value.
    assert "sdie 2.0000" in reference


def test_zero_ionic_strength_emits_no_ion_lines():
    text = build_input(size_grid(ion(), GridSpec()), SolventModel(ionic_strength=0.0))
    assert "ion charge" not in text


class TestSurfaceMapping:
    """The portable enum maps onto APBS, and refuses what APBS cannot do."""

    def test_smoothed_molecular_maps_to_smol(self):
        text = build_input(
            size_grid(ion(), GridSpec()),
            SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
        )
        assert "srfm smol" in text

    def test_molecular_maps_to_mol(self):
        text = build_input(
            size_grid(ion(), GridSpec()), SolventModel(surface_model=SurfaceModel.MOLECULAR)
        )
        assert "srfm mol" in text

    def test_van_der_waals_collapses_the_probe(self):
        """APBS has no vdW srfm; it is the molecular surface with srad 0."""
        text = build_input(
            size_grid(ion(), GridSpec()),
            SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS, surface_radius=1.4),
        )
        assert "srfm mol" in text
        assert "srad 0.0000" in text

    def test_gaussian_is_refused_with_an_actionable_message(self):
        """DelPhi's model has no APBS equivalent; guessing would be worse."""
        with pytest.raises(UnsupportedRequest, match="no equivalent"):
            build_input(
                size_grid(ion(), GridSpec()),
                SolventModel(surface_model=SurfaceModel.GAUSSIAN),
            )

    def test_spline_surfaces_are_reachable_only_by_asking_for_them(self):
        """spl2 is for forces; using it for energy moved a dipeptide by 25%."""
        default = build_input(size_grid(ion(), GridSpec()), SolventModel())
        assert "spl2" not in default

        explicit = build_input(
            size_grid(ion(), GridSpec()),
            SolventModel(),
            ApbsOptions(srfm_override="spl2"),
        )
        assert "srfm spl2" in explicit

    def test_no_surface_model_maps_to_a_spline_by_default(self):
        for model in (SurfaceModel.MOLECULAR, SurfaceModel.SMOOTHED_MOLECULAR):
            text = build_input(size_grid(ion(), GridSpec()), SolventModel(surface_model=model))
            assert "srfm spl" not in text


class TestEquation:
    def test_linear_emits_lpbe(self):
        text = build_input(size_grid(ion(), GridSpec()), SolventModel(), equation=Equation.LINEAR)
        assert "\n    lpbe" in text
        assert "npbe" not in text

    def test_nonlinear_is_expressible_in_the_template(self):
        """Representable: the mapping exists even though the backend declines."""
        text = build_input(
            size_grid(ion(), GridSpec()), SolventModel(), equation=Equation.NONLINEAR
        )
        assert "\n    npbe" in text
