"""Capability reporting and dry-run validation.

Mostly binary-free. The point of `validate_request` is that it answers expensive
questions cheaply, so a test that needed a solver would undercut it.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from sashimi import backends
from sashimi.apbs import discover
from sashimi.apbs.options import ApbsOptions
from sashimi.backends import get as get_backend
from sashimi.backends import names as backend_names
from sashimi.capabilities import (
    UNITS,
    BackendReport,
    comparable_surface_models,
    describe_capabilities,
    validate_request,
)
from sashimi.delphi import discover as delphi_discover
from sashimi.pqr import parse_pqr, read_pqr
from sashimi.protocol import Equation, GridSpec, PQRData, SolventModel, SurfaceModel
from sashimi.tabipb import discover as tabipb_discover


def substitute(monkeypatch, **replacements: BackendReport) -> None:
    """Make named backends report something else, at the registry.

    The registry is the seam `describe_capabilities`, `--backend` and
    `sashimi_solve` all read, so substituting an entry here is the one place
    that changes every caller's view at once. Patching the private report
    functions, which is what this used to do, only changed the ones that
    happened to import them.

    **Every registered backend is replaced, named or not.** An unnamed one used
    to keep reporting whatever this machine actually has, so a test constructing
    a three-backend world silently got a fourth — which is exactly what happened
    when `delphi-cpp` and `pydelphi` were registered and a comparability test
    started intersecting a `van-der-waals` it had never asked for. Anything not
    named is made unavailable, so the world a test builds is the whole world.
    """
    for name in backends.names():
        if name in replacements:
            continue
        replacements[name] = BackendReport(name, False, "finite-difference")

    for name, report in replacements.items():
        entry = backends.REGISTRY[name]

        def describe(captured: BackendReport = report) -> BackendReport:
            return captured

        monkeypatch.setitem(backends.REGISTRY, name, dataclasses.replace(entry, describe=describe))


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

        # Every backend that needs installing is absent; `gb` and `debye` need
        # none, so "nothing installed" is no longer the same thing as "nothing
        # usable" — and since M5 it is not even the same thing as "no reference
        # tier", which is the whole point of a clean-room solver.
        assert caps["available_backends"] == ["gb", "debye"]
        assert all(
            backend["available"] is False
            for backend in caps["backends"]
            if backend["name"] not in ("gb", "debye")
        )
        # Keyed by name, not by position. These were indexed `[0]`, `[1]`,
        # `[2]` and broke the moment `delphi-cpp` and `pydelphi` were registered
        # between them — a test that asserts registry *order* where it means
        # registry *content*.
        detail = {backend["name"]: backend["detail"] for backend in caps["backends"]}
        assert "brew install apbs" in detail["apbs"]
        # Every backend must explain its own absence, not just the first one.
        assert "compbio.clemson.edu" in detail["delphi"]
        assert "Treecodes/TABI-PB" in detail["tabipb"]
        # Both pinned flavours explain themselves too, rather than inheriting
        # the auto entry's message.
        assert detail["delphi-cpp"]
        assert detail["pydelphi"]
        assert f"2/{len(caps['backends'])} backend" in caps["summary"]

    def test_reports_every_backend_with_its_solver_family(self):
        """The family is what tells a caller why TABI-PB answers different
        questions: a boundary-element solve has no volume to interpolate."""
        backends = describe_capabilities()["backends"]

        assert [b["name"] for b in backends] == list(backend_names())
        assert [b["family"] for b in backends] == [
            get_backend(b["name"]).family.value for b in backends
        ]

    def test_only_gb_reports_itself_as_an_approximation(self):
        """The tier is what stops a triage number being read as an answer."""
        tiers = {b["name"]: b["accuracy_tier"] for b in describe_capabilities()["backends"]}

        # Named rather than exhaustive: the claim is that `gb` is the only
        # approximation, which stays true as backends are added. An exhaustive
        # dict would have to be edited every time one is, and the two DelPhi
        # flavours would add nothing to it — they are the same solver.
        assert tiers["gb"] == "approximate"
        assert {name for name, tier in tiers.items() if tier == "approximate"} == {"gb"}
        for name in ("apbs", "delphi", "delphi-cpp", "pydelphi", "tabipb", "debye"):
            assert tiers[name] == "reference"

    def test_one_backend_is_comparable_with_nothing(self, monkeypatch):
        """A lone backend trivially shares every model with itself.

        Reporting those as comparable would tell a caller cross-validation is
        available when there is nothing installed to validate against — which is
        exactly what `tests/test_cross_validation.py` tripped over.

        Reaching this state now takes hiding `gb` as well, since it is always
        available. That is the rule still holding rather than going away: a real
        installation has a comparison partner whether or not it installed one.
        """

        def which_apbs_only(name: str) -> str | None:
            return f"/usr/bin/{name}" if name == "apbs" else None

        caches = (delphi_discover._discover_cached, tabipb_discover._discover_cached)
        for cache in caches:
            cache.cache_clear()
        monkeypatch.setenv("SASHIMI_DELPHI_PATH", "/nonexistent/delphi")
        monkeypatch.setenv("SASHIMI_TABIPB_PATH", "/nonexistent/tabipb")
        monkeypatch.setattr("shutil.which", which_apbs_only)
        substitute(
            monkeypatch,
            gb=BackendReport("gb", False, "analytic", detail="hidden for this test"),
            debye=BackendReport("debye", False, "finite-difference", detail="hidden for this test"),
        )
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
        substitute(
            monkeypatch,
            apbs=reports[0],
            delphi=reports[1],
            tabipb=reports[2],
            gb=BackendReport("unused", False, "analytic"),
            debye=BackendReport("unused", False, "finite-difference"),
        )

        assert comparable_surface_models() == ["molecular"]

    def test_a_model_only_one_backend_supports_is_not_comparable(self, monkeypatch):
        """Two or more, still — the property the intersection was there for."""
        substitute(
            monkeypatch,
            apbs=BackendReport("apbs", True, "fd", surface_models=("molecular", "smol")),
            delphi=BackendReport("delphi", True, "fd", surface_models=("gaussian",)),
            tabipb=BackendReport("tabipb", False, "bem", surface_models=("molecular",)),
            gb=BackendReport("unused", False, "analytic"),
            debye=BackendReport("unused", False, "finite-difference"),
        )

        assert comparable_surface_models() == []

    @pytest.mark.usefixtures("hide_backends")
    def test_a_bare_machine_can_still_compare_two_backends(self):
        """Which models two backends share decides whether they can be compared.

        **This answer changed at M5, and the change is the milestone.** With
        nothing installed the list used to be empty — an honest answer, since
        the only always-present backend was `gb` and one backend is comparable
        with nothing. Registering debye puts a second binary-free solver in the
        registry, and the two share `molecular`, so cross-validation now works
        on a machine with no APBS, no DelPhi and no TABI-PB.

        What they are *not* is two of a kind: debye discretizes the equation and
        `gb` approximates it, so `sashimi validate` reports the pair as a
        reference answer with a deviation beside it rather than as a spread.
        That partition is `AccuracyTier`'s job and is why this is a real
        comparison rather than an average of two guesses.
        """
        models = describe_capabilities()["surface_models"]
        assert models["comparable_across_available_backends"] == ["molecular"]

    def test_what_a_defaulted_request_will_actually_be_solved_on_is_discoverable(self):
        """The grid half of this was reported and the physics half was not.

        An agent could learn the spacing a defaulted solve would use but not the
        boundary — and the boundary is the larger modelling choice of the two,
        as this module's own note says. It moved once, on 2026-08-13, so it is
        published rather than left to be assumed.
        """
        described = describe_capabilities()
        defaults = SolventModel()

        assert described["surface_models"]["default"] == defaults.surface_model.value
        assert described["solvent_defaults"]["surface_model"] == defaults.surface_model.value
        assert described["solvent_defaults"]["ionic_strength"] == defaults.ionic_strength
        # Every field, so a new one cannot arrive undocumented.
        assert set(described["solvent_defaults"]) == {f.name for f in dataclasses.fields(defaults)}


class TestValidateRequest:
    def test_reports_the_grid_that_would_be_used(self):
        report = validate_request(ion(), GridSpec(resolution=0.5, padding=10.0))
        assert report["cost"]["grid"]["native"]["dime"] == [65, 65, 65]
        assert report["cost"]["grid"]["n_points"] == 65**3

    def test_estimates_the_map_size_on_disk(self):
        """A caller should know it is about to write 12 MB before it happens."""
        report = validate_request(ion(), GridSpec(resolution=0.25, padding=10.0))
        assert report["cost"]["grid"]["estimated_map_mb"] == pytest.approx(29.0, rel=0.05)

    def test_a_relaxed_resolution_is_reported(self):
        """Pure arithmetic, so it holds on a machine with nothing installed."""
        report = validate_request(ion(), GridSpec(resolution=0.05, max_points=65**3))
        assert report["cost"]["grid"]["resolution_relaxed"] is True
        assert any("caps the grid" in p for p in report["problems"])

    @pytest.mark.apbs
    def test_a_relaxed_resolution_warns_but_does_not_block(self):
        """The solve would still run, and the result would say it was relaxed.

        Marked `apbs` for what looks like a dry run, because `ok` folds in
        whether the backend is *present*: with APBS absent this is false for a
        reason that has nothing to do with the grid. The claim being made —
        a relaxation is a warning rather than a refusal — can only be seen when
        nothing else is blocking.
        """
        report = validate_request(ion(), GridSpec(resolution=0.05, max_points=65**3))
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


class TestPerBackendCost:
    """What a request would cost, in the terms the chosen backend works in.

    `validate_request` grew a `backend` parameter and kept costing everything
    with APBS's grid sizer, which is a confident wrong number rather than a
    missing one: a point count, a map size and a "resolution was relaxed"
    warning belonging to a solver the caller is not running.
    """

    def peptide(self) -> PQRData:
        return read_pqr(Path(__file__).resolve().parent / "data" / "ala-gly.pqr")

    def molecular(self) -> SolventModel:
        return SolventModel(surface_model=SurfaceModel.MOLECULAR)

    def test_two_grid_backends_do_not_share_a_grid(self):
        """APBS sizes a multigrid `dime` lattice, DelPhi an odd cubic one.

        Measured against the real solves: APBS runs ALA-GLY at 65^3 / 0.467 A,
        DelPhi at 61^3 / 0.498 A. Costing the second with the first's sizer is
        what this asserts is no longer happening.
        """
        apbs = validate_request(self.peptide(), solvent=self.molecular(), backend="apbs")
        delphi = validate_request(self.peptide(), solvent=self.molecular(), backend="delphi")

        assert apbs["cost"]["grid"]["native"]["dime"] == [65, 65, 65]
        assert delphi["cost"]["grid"]["native"]["gsize"] == 61
        assert apbs["cost"]["grid"]["n_points"] != delphi["cost"]["grid"]["n_points"]

    @pytest.mark.parametrize("backend", ["apbs", "delphi", "tabipb", "gb", "debye"])
    def test_the_cost_is_always_under_the_same_key(self, backend: str):
        """A caller should not have to know the family to know which key exists.

        The estimate used to land under `report["grid"]` for finite difference
        and `report["cost"]` for everything else, with no discriminator in the
        response and a `KeyError` for reading the wrong one.
        """
        report = validate_request(self.peptide(), solvent=self.molecular(), backend=backend)

        assert "grid" not in report
        assert report["cost"]["family"] == report["backend"]["family"]
        assert "note" in report["cost"]

    @pytest.mark.parametrize("backend", ["tabipb", "gb"])
    def test_a_backend_with_no_grid_estimates_no_grid(self, backend: str):
        """True whether or not the backend is installed, which is the point.

        `validate_request` needs no binary: an absent backend still reports its
        family and still has no grid. What changes when it is missing is the
        *summary*, which becomes the install message — see the test below.
        """
        report = validate_request(self.peptide(), solvent=self.molecular(), backend=backend)

        assert report["cost"]["grid"] is None
        assert "no grid" in report["cost"]["note"]
        assert "unknown grid" not in report["summary"]

    def test_the_summary_of_a_gridless_backend_names_what_it_costs_instead(self):
        """It used to read "Would run on a unknown grid" — inviting a caller to
        lower `max_points` for a solver that has none.

        Asked of `gb` specifically, because it is the only backend that is always
        available. This test first asked TABI-PB and failed on the APBS-only CI
        leg within an hour of being written: an absent backend's summary is its
        install instructions, which is a different and correct answer to a
        different question.
        """
        report = validate_request(self.peptide(), solvent=self.molecular(), backend="gb")

        assert report["ok"] is True
        assert "no grid" in report["summary"]

    def test_the_mesher_floor_is_refused_before_a_solve_rather_than_inside_one(self):
        """A one-atom Born ion is the corpus's own anchor case, and TABI-PB
        cannot triangulate it. `min_atoms` was already published in the backend's
        report and simply never checked, so this validated `ok` and then failed
        in the mesher after the structure had been prepared."""
        ion = parse_pqr("ATOM      1  I   ION     1       0.000   0.000  0.000  1.00  3.00\n")

        report = validate_request(ion, solvent=self.molecular(), backend="tabipb")

        assert report["ok"] is False
        assert any("at least 4 atoms" in p for p in report["problems"])

    def test_a_mesh_density_the_solver_aborts_on_is_refused(self):
        """TABI-PB's abort below 1.5 is an uncaught C++ exception with no cause,
        and `BoundaryElementRequest` still defaults to 1.0 — so the most obvious
        first call is the one that fails unintelligibly."""
        report = validate_request(
            self.peptide(), solvent=self.molecular(), backend="tabipb", mesh_density=1.0
        )

        assert report["ok"] is False
        assert any("mesh_density" in p for p in report["problems"])
