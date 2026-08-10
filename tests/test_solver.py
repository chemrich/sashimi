"""Backend behaviour that needs the real binary: discovery, error mapping,
guardrails, and the DX contract as APBS actually writes it."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from sashimi.apbs import ApbsSolver, discover, discover_apbs
from sashimi.apbs.discover import ApbsNotFound, _probe_version
from sashimi.apbs.grid import size_grid
from sashimi.apbs.input import build_input
from sashimi.apbs.run import ApbsCrash, run_apbs
from sashimi.dx import read_dx
from sashimi.errors import GridTooLarge, UnsupportedRequest
from sashimi.pqr import format_pqr
from sashimi.protocol import (
    Equation,
    FiniteDifferenceRequest,
    GridSpec,
    PQRData,
    SolventModel,
    SurfaceModel,
)
from tests.helpers import volume

pytestmark = pytest.mark.apbs


@pytest.fixture(scope="module")
def ion():
    return PQRData(
        coords=np.zeros((1, 3)),
        charges=np.array([1.0]),
        radii=np.array([3.0]),
    )


@pytest.fixture(scope="module")
def coarse():
    """Cheap grid — these tests exercise plumbing, not physics."""
    return GridSpec(resolution=1.0, padding=6.0)


def test_discovery_reports_a_version_and_path():
    binary = discover_apbs()
    assert binary.version.startswith("3.")
    assert binary.path.is_file()
    assert binary.label == f"apbs-{binary.version}"


def test_explicit_path_outranks_everything_else(monkeypatch, tmp_path):
    """$SASHIMI_APBS_PATH is the escape hatch when the system binary is wrong."""
    real = discover_apbs()
    discover._discover_cached.cache_clear()
    monkeypatch.setenv("SASHIMI_APBS_PATH", str(real.path))
    try:
        assert discover_apbs().path == real.path
    finally:
        discover._discover_cached.cache_clear()


def test_version_probe_leaves_no_io_mc(tmp_path, monkeypatch):
    """APBS writes io.mc into cwd on every invocation, --version included."""
    binary = discover_apbs()
    monkeypatch.chdir(tmp_path)
    assert _probe_version(binary.path) is not None
    assert not (tmp_path / "io.mc").exists()


def test_solve_leaves_no_files_behind(tmp_path, monkeypatch, ion, coarse):
    monkeypatch.chdir(tmp_path)
    ApbsSolver().solve(FiniteDifferenceRequest(structure=ion, grid=coarse))
    assert list(tmp_path.iterdir()) == []


def test_solve_without_energy_returns_none(ion, coarse):
    result = ApbsSolver().solve(
        FiniteDifferenceRequest(structure=ion, grid=coarse, want_energy=False)
    )
    assert result.energy_kj_mol is None
    assert volume(result).values.size > 0


def test_potential_grid_geometry_matches_the_request(ion, coarse):
    result = ApbsSolver().solve(FiniteDifferenceRequest(structure=ion, grid=coarse))
    grid, diag = volume(result), result.diagnostics
    assert list(grid.shape) == diag["dime"]
    np.testing.assert_allclose(grid.spacing, diag["spacing_achieved"], atol=1e-6)
    # Grid is centered on the molecule.
    center = grid.origin + (np.array(grid.shape) - 1) * grid.spacing / 2
    np.testing.assert_allclose(center, ion.center(), atol=1e-6)


def test_relaxed_resolution_is_reported_not_hidden(ion):
    """The budget may bite; the caller must be able to tell that it did."""
    spec = GridSpec(resolution=0.05, padding=6.0, max_points=65**3)
    result = ApbsSolver().solve(FiniteDifferenceRequest(structure=ion, grid=spec))
    assert result.diagnostics["resolution_relaxed"] is True
    assert max(result.diagnostics["spacing_achieved"]) > spec.resolution


def test_impossible_grid_fails_before_launching_apbs(ion):
    with pytest.raises(GridTooLarge):
        ApbsSolver().solve(
            FiniteDifferenceRequest(
                structure=ion, grid=GridSpec(resolution=0.5, padding=80.0, max_points=1000)
            )
        )


def test_missing_binary_raises_with_an_install_hint(monkeypatch):
    discover._discover_cached.cache_clear()
    monkeypatch.setenv("SASHIMI_APBS_PATH", "/nonexistent/apbs")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    # String targets rather than reaching through the module's imported names.
    monkeypatch.setattr("shutil.which", lambda _: None)
    monkeypatch.setattr("pathlib.Path.cwd", staticmethod(lambda: Path("/")))
    try:
        with pytest.raises(ApbsNotFound, match="brew install apbs"):
            discover.discover_apbs()
    finally:
        discover._discover_cached.cache_clear()


def test_malformed_input_is_caught_despite_exit_code(ion, coarse):
    """APBS exits 0 on some failures, so success is verified structurally."""
    binary = discover_apbs()
    with pytest.raises(ApbsCrash):
        run_apbs(binary, pqr_text=format_pqr(ion), input_text="this is not apbs input\nquit\n")


def test_timeout_is_reported_as_a_crash(ion):
    binary = discover_apbs()
    text = build_input(size_grid(ion, GridSpec(resolution=0.2, padding=10.0)), SolventModel())
    with pytest.raises(ApbsCrash, match="timed out"):
        run_apbs(binary, pqr_text=format_pqr(ion), input_text=text, timeout=0.05)


def test_written_dx_round_trips_through_our_own_reader(tmp_path, ion, coarse):
    """The writer must produce something we — and PyMOL — can read back."""
    result = ApbsSolver().solve(FiniteDifferenceRequest(structure=ion, grid=coarse))
    path = tmp_path / "potential.dx"
    volume(result).to_dx(path)

    again = read_dx(path)
    assert again.shape == volume(result).shape
    np.testing.assert_allclose(again.origin, volume(result).origin)
    np.testing.assert_allclose(again.spacing, volume(result).spacing)
    np.testing.assert_allclose(again.values, volume(result).values, rtol=1e-6)


def test_ionic_strength_screens_the_potential(ion, coarse):
    """Salt must reduce the potential at distance — a physics smoke test."""
    solver = ApbsSolver()
    probe = np.array([[8.0, 0.0, 0.0]])
    no_salt = solver.solve(
        FiniteDifferenceRequest(
            structure=ion, grid=coarse, solvent=SolventModel(ionic_strength=0.0)
        )
    )
    salty = solver.solve(
        FiniteDifferenceRequest(
            structure=ion, grid=coarse, solvent=SolventModel(ionic_strength=0.5)
        )
    )
    assert volume(salty).value_at(probe)[0] < volume(no_salt).value_at(probe)[0]


class TestPhase4Contract:
    """Provenance and request handling introduced by the protocol rewrite."""

    def test_provenance_carries_a_binary_checksum(self, ion, coarse):
        """Version is not provenance: distributions patch, and three builds all
        call themselves 3.4.1 while differing observably."""
        result = ApbsSolver().solve(FiniteDifferenceRequest(structure=ion, grid=coarse))
        sha = result.provenance.binary_sha256
        assert sha is not None
        assert len(sha) == 64
        assert result.provenance.summary().startswith("apbs-3.4.1 sha256:")

    def test_checksum_matches_the_binary_on_disk(self):
        binary = discover_apbs()
        expected = hashlib.sha256(binary.path.read_bytes()).hexdigest()
        assert binary.sha256 == expected

    def test_resolved_parameters_record_the_mapping_not_the_request(self, ion, coarse):
        """A surface model that was mapped must be auditable after the fact."""
        result = ApbsSolver().solve(
            FiniteDifferenceRequest(
                structure=ion,
                grid=coarse,
                solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS),
            )
        )
        resolved = result.provenance.resolved_parameters
        assert resolved["surface_model"] == "van-der-waals"
        assert resolved["apbs"]["srfm"] == "mol"
        assert resolved["apbs"]["srad"] == 0.0

    def test_nonlinear_is_refused_rather_than_silently_approximated(self, ion, coarse):
        """Representable, not implemented: returning untested npbe numbers
        would be worse than declining."""
        request = FiniteDifferenceRequest(structure=ion, grid=coarse, equation=Equation.NONLINEAR)
        with pytest.raises(UnsupportedRequest, match="does not yet"):
            ApbsSolver().solve(request)

    def test_energy_can_be_declined(self, ion, coarse):
        result = ApbsSolver().solve(
            FiniteDifferenceRequest(structure=ion, grid=coarse, want_energy=False)
        )
        assert result.energy_kj_mol is None
        assert result.potential is not None

    def test_potential_can_be_declined(self, ion, coarse):
        """Skipping the DX write is the cheap path for an energy-only solve."""
        result = ApbsSolver().solve(
            FiniteDifferenceRequest(structure=ion, grid=coarse, want_potential=False)
        )
        assert result.potential is None
        assert result.energy_kj_mol is not None

    def test_gaussian_dielectric_is_refused_before_launching_apbs(self, ion, coarse):
        request = FiniteDifferenceRequest(
            structure=ion,
            grid=coarse,
            solvent=SolventModel(surface_model=SurfaceModel.GAUSSIAN),
        )
        with pytest.raises(UnsupportedRequest, match="no equivalent"):
            ApbsSolver().solve(request)
