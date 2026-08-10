"""Capability reporting and dry-run validation.

Mostly binary-free. The point of `validate_request` is that it answers expensive
questions cheaply, so a test that needed a solver would undercut it.
"""

import numpy as np
import pytest

from sashimi.apbs import discover
from sashimi.apbs.options import ApbsOptions
from sashimi.capabilities import UNITS, describe_capabilities, validate_request
from sashimi.delphi import discover as delphi_discover
from sashimi.protocol import Equation, GridSpec, PQRData, SolventModel, SurfaceModel


@pytest.fixture
def hide_backends(monkeypatch):
    """Make every backend undiscoverable, whatever this machine has installed.

    Both discovery layers cache, and both read the environment directly, so the
    caches have to be cleared on the way in *and* out — otherwise a test that
    hides APBS poisons the lookup for every test after it.
    """
    caches = (discover._discover_cached, delphi_discover._discover_cached)
    for cache in caches:
        cache.cache_clear()
    monkeypatch.setenv("SASHIMI_APBS_PATH", "/nonexistent/apbs")
    monkeypatch.setenv("SASHIMI_DELPHI_PATH", "/nonexistent/delphi")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    yield
    for cache in caches:
        cache.cache_clear()


def ion(radius: float = 3.0) -> PQRData:
    return PQRData(
        coords=np.zeros((1, 3)),
        charges=np.array([1.0]),
        radii=np.array([radius]),
        labels=("ION 1 I",),
    )


class TestCapabilities:
    def test_reports_units_so_nobody_has_to_guess(self):
        caps = describe_capabilities()
        assert caps["units"] == UNITS
        assert caps["units"]["potential"] == "kT/e"
        assert caps["units"]["energy"] == "kJ/mol"

    def test_lists_portable_surface_models(self):
        portable = describe_capabilities()["surface_models"]["portable"]
        assert set(portable) == {m.value for m in SurfaceModel}

    def test_spline_surfaces_are_named_as_deliberately_absent(self):
        """Otherwise their absence looks like an oversight."""
        note = describe_capabilities()["surface_models"]["note"]
        assert "Spline" in note
        assert "force" in note

    def test_distinguishes_solved_equations_from_representable_ones(self):
        backend = describe_capabilities()["backends"][0]
        assert backend["equations"] == ["linear"]
        if backend["available"]:
            assert "nonlinear" in backend["representable_equations"]

    def test_names_what_is_not_supported(self):
        unsupported = " ".join(describe_capabilities()["not_supported"])
        assert "nonlinear" in unsupported
        assert "passthrough" in unsupported

    @pytest.mark.usefixtures("hide_backends")
    def test_a_missing_backend_is_a_report_not_an_exception(self):
        """This tool must survive exactly the situation it exists to explain."""
        caps = describe_capabilities()

        assert caps["available_backends"] == []
        assert all(backend["available"] is False for backend in caps["backends"])
        assert "brew install apbs" in caps["backends"][0]["detail"]
        # Every backend must explain its own absence, not just the first one.
        assert "compbio.clemson.edu" in caps["backends"][1]["detail"]
        assert "0/2 backend" in caps["summary"]

    def test_reports_both_finite_difference_backends(self):
        names = [backend["name"] for backend in describe_capabilities()["backends"]]
        assert names == ["apbs", "delphi"]

    @pytest.mark.usefixtures("hide_backends")
    def test_comparable_surface_models_are_reported(self):
        """Which models two backends share decides whether they can be compared.

        With nothing installed the honest answer is an empty list, and an empty
        list is a real answer here rather than a missing one — APBS and pyDelPhi
        genuinely share no surface model.
        """
        models = describe_capabilities()["surface_models"]
        assert models["comparable_across_available_backends"] == []


class TestValidateRequest:
    def test_reports_the_grid_that_would_be_used(self):
        report = validate_request(ion(), GridSpec(resolution=0.5, padding=10.0))
        assert report["grid"]["dime"] == [65, 65, 65]
        assert report["grid"]["n_points"] == 65**3

    def test_estimates_the_map_size_on_disk(self):
        """A caller should know it is about to write 12 MB before it happens."""
        report = validate_request(ion(), GridSpec(resolution=0.25, padding=10.0))
        assert report["grid"]["estimated_map_mb"] == pytest.approx(29.0, rel=0.05)

    def test_a_relaxed_resolution_warns_but_does_not_block(self):
        """The solve would still run, and the result would say it was relaxed."""
        report = validate_request(ion(), GridSpec(resolution=0.05, max_points=65**3))
        assert report["grid"]["resolution_relaxed"] is True
        assert any("caps the grid" in p for p in report["problems"])
        assert report["ok"] is True

    def test_an_impossible_grid_blocks(self):
        report = validate_request(ion(), GridSpec(resolution=0.5, padding=80.0, max_points=1000))
        assert report["ok"] is False
        assert any("max_points" in p for p in report["problems"])
        assert "Would not run" in report["summary"]

    def test_an_unsupported_surface_model_blocks(self):
        report = validate_request(ion(), solvent=SolventModel(surface_model=SurfaceModel.GAUSSIAN))
        assert report["ok"] is False
        assert any("no equivalent" in p for p in report["problems"])

    def test_reports_the_resolved_surface_keyword(self):
        report = validate_request(
            ion(), solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS)
        )
        assert report["surface"]["resolved_keyword"] == "mol"
        assert report["surface"]["resolved_probe_radius_a"] == 0.0

    def test_nonlinear_blocks_with_an_explanation(self):
        report = validate_request(ion(), equation=Equation.NONLINEAR)
        assert report["ok"] is False
        assert any("no solver path" in p for p in report["problems"])

    def test_the_escape_hatch_is_honoured(self):
        """An explicit srfm_override must validate, not be second-guessed."""
        report = validate_request(ion(), options=ApbsOptions(srfm_override="spl2"))
        assert report["surface"]["resolved_keyword"] == "spl2"

    def test_describes_the_structure_it_was_given(self):
        report = validate_request(ion(radius=5.0))
        assert report["n_atoms"] == 1
        assert report["total_charge"] == pytest.approx(1.0)
        assert report["extent_a"] == [10.0, 10.0, 10.0]
