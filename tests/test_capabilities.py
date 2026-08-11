"""Capability reporting and dry-run validation.

Mostly binary-free. The point of `validate_request` is that it answers expensive
questions cheaply, so a test that needed a solver would undercut it.
"""

import numpy as np
import pytest

from sashimi.apbs import discover
from sashimi.apbs.options import ApbsOptions
from sashimi.capabilities import (
    UNITS,
    BackendReport,
    comparable_surface_models,
    describe_capabilities,
    validate_request,
)
from sashimi.delphi import discover as delphi_discover
from sashimi.protocol import Equation, GridSpec, PQRData, SolventModel, SurfaceModel
from sashimi.tabipb import discover as tabipb_discover


@pytest.fixture
def hide_backends(monkeypatch):
    """Make every backend undiscoverable, whatever this machine has installed.

    Both discovery layers cache, and both read the environment directly, so the
    caches have to be cleared on the way in *and* out — otherwise a test that
    hides APBS poisons the lookup for every test after it.
    """
    caches = (
        discover._discover_cached,
        delphi_discover._discover_cached,
        tabipb_discover._discover_cached,
    )
    for cache in caches:
        cache.cache_clear()
    monkeypatch.setenv("SASHIMI_APBS_PATH", "/nonexistent/apbs")
    monkeypatch.setenv("SASHIMI_DELPHI_PATH", "/nonexistent/delphi")
    monkeypatch.setenv("SASHIMI_TABIPB_PATH", "/nonexistent/tabipb")
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
        assert "Treecodes/TABI-PB" in caps["backends"][2]["detail"]
        assert "0/3 backend" in caps["summary"]

    def test_reports_every_backend_with_its_solver_family(self):
        """The family is what tells a caller why TABI-PB answers different
        questions: a boundary-element solve has no volume to interpolate."""
        backends = describe_capabilities()["backends"]

        assert [b["name"] for b in backends] == ["apbs", "delphi", "tabipb"]
        assert [b["family"] for b in backends] == [
            "finite-difference",
            "finite-difference",
            "boundary-element",
        ]

    def test_one_backend_is_comparable_with_nothing(self, monkeypatch):
        """A lone backend trivially shares every model with itself.

        Reporting those as comparable would tell a caller cross-validation is
        available when there is nothing installed to validate against — which is
        exactly what `tests/test_cross_validation.py` tripped over.
        """

        def which_apbs_only(name: str) -> str | None:
            return f"/usr/bin/{name}" if name == "apbs" else None

        caches = (delphi_discover._discover_cached, tabipb_discover._discover_cached)
        for cache in caches:
            cache.cache_clear()
        monkeypatch.setenv("SASHIMI_DELPHI_PATH", "/nonexistent/delphi")
        monkeypatch.setenv("SASHIMI_TABIPB_PATH", "/nonexistent/tabipb")
        monkeypatch.setattr("shutil.which", which_apbs_only)
        try:
            assert comparable_surface_models() == []
        finally:
            for cache in caches:
                cache.cache_clear()

    def test_a_backend_nobody_shares_a_model_with_does_not_empty_the_list(self, monkeypatch):
        """The regression that adding a fourth backend would otherwise cause.

        This intersected every available backend, so a Generalized Born tier
        supporting only `van-der-waals` — which no surface solver can mesh —
        would empty the result. That is not a smaller answer: an empty list
        stops `sashimi validate` outright and skips the entire cross-validation
        tier, so adding a backend would have silently switched off the
        comparisons between the other three while CI stayed green.
        """
        reports = [
            BackendReport("apbs", True, "finite-difference", surface_models=("molecular", "smol")),
            BackendReport("delphi", True, "finite-difference", surface_models=("molecular",)),
            BackendReport("gb", True, "analytic", surface_models=("van-der-waals",)),
        ]
        monkeypatch.setattr("sashimi.capabilities._apbs_report", lambda: reports[0])
        monkeypatch.setattr("sashimi.capabilities._delphi_report", lambda: reports[1])
        monkeypatch.setattr("sashimi.capabilities._tabipb_report", lambda: reports[2])

        assert comparable_surface_models() == ["molecular"]

    def test_a_model_only_one_backend_supports_is_not_comparable(self, monkeypatch):
        """Two or more, still — the property the intersection was there for."""
        monkeypatch.setattr(
            "sashimi.capabilities._apbs_report",
            lambda: BackendReport("apbs", True, "fd", surface_models=("molecular", "smol")),
        )
        monkeypatch.setattr(
            "sashimi.capabilities._delphi_report",
            lambda: BackendReport("delphi", True, "fd", surface_models=("gaussian",)),
        )
        monkeypatch.setattr(
            "sashimi.capabilities._tabipb_report",
            lambda: BackendReport("tabipb", False, "bem", surface_models=("molecular",)),
        )

        assert comparable_surface_models() == []

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
