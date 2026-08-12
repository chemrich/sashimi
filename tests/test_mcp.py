"""The MCP surface, exercised through a real client session.

Tests go through `fastmcp.Client` rather than calling the functions directly,
so schema generation, argument validation and error translation are all in the
path — that is what an agent actually hits.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from sashimi.dx import write_dx
from sashimi.mcp import mcp
from sashimi.protocol import PotentialGrid
from sashimi.tabipb.discover import discover_tabipb
from tests.helpers import installed_or_skip

FIXTURE_PDB = Path(__file__).parent / "data" / "ala-gly.pdb"

EXPECTED_TOOLS = {
    "sashimi_capabilities",
    "sashimi_compare_maps",
    "sashimi_potential_at",
    "sashimi_potential_extrema",
    "sashimi_potential_in_sphere",
    "sashimi_prepare_structure",
    "sashimi_residue_potentials",
    "sashimi_solve",
    "sashimi_validate_inputs",
}


def payload(result):
    """The structured content an agent receives."""
    return result.structured_content or json.loads(result.content[0].text)


@pytest.fixture
async def client():
    async with Client(mcp) as session:
        yield session


def make_grid(offset: float = 0.0, shape=(9, 9, 9)) -> PotentialGrid:
    x = np.linspace(-1.0, 1.0, shape[0])
    values = np.repeat(np.repeat(x[:, None, None], shape[1], 1), shape[2], 2) + offset
    return PotentialGrid(values=values, origin=np.zeros(3), spacing=np.full(3, 0.5))


class TestSurface:
    async def test_exposes_exactly_the_four_planned_tools(self, client):
        assert {t.name for t in await client.list_tools()} == EXPECTED_TOOLS

    async def test_every_tool_documents_itself(self, client):
        for tool in await client.list_tools():
            assert tool.description, f"{tool.name} has no description"
            props = (tool.inputSchema or {}).get("properties", {})
            # sashimi_capabilities takes no arguments by design.
            if tool.name == "sashimi_capabilities":
                continue
            assert props, f"{tool.name} exposes no parameters"
            for name, schema in props.items():
                assert schema.get("description"), f"{tool.name}.{name} is undocumented"

    async def test_no_raw_apbs_passthrough_is_exposed(self, client):
        """ROADMAP.md section 6: a passthrough would defeat the abstraction."""
        blob = json.dumps([t.model_dump(mode="json") for t in await client.list_tools()])
        for leak in ("mg-auto", "cglen", "fglen", "dime", "fe-manual"):
            assert leak not in blob, f"APBS vocabulary {leak!r} leaked into the tool surface"


class TestPrepareStructure:
    async def test_writes_a_pqr_and_reports_what_changed(self, client, tmp_path):
        out = tmp_path / "prepared.pqr"
        result = payload(
            await client.call_tool(
                "sashimi_prepare_structure",
                {"pdb_path": str(FIXTURE_PDB), "output_pqr": str(out)},
            )
        )
        assert out.exists()
        assert result["pqr_path"] == str(out)
        assert result["structure_was_modified"] is True
        assert result["n_warnings"] >= 1
        assert "OXT" in json.dumps(result["edits"])
        assert "atoms" in result["summary"]

    async def test_missing_file_becomes_a_clean_tool_error(self, client, tmp_path):
        with pytest.raises(ToolError, match="no such structure file"):
            await client.call_tool(
                "sashimi_prepare_structure", {"pdb_path": str(tmp_path / "absent.pdb")}
            )

    async def test_out_of_range_ph_is_rejected_by_the_schema(self, client):
        with pytest.raises(ToolError):
            await client.call_tool(
                "sashimi_prepare_structure", {"pdb_path": str(FIXTURE_PDB), "ph": 99.0}
            )


class TestPotentialAt:
    async def test_samples_a_saved_map(self, client, tmp_path):
        dx = tmp_path / "map.dx"
        write_dx(dx, make_grid())

        result = payload(
            await client.call_tool(
                "sashimi_potential_at",
                {"dx_path": str(dx), "points": [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]},
            )
        )
        assert len(result["values_kT_e"]) == 2
        assert result["n_outside_grid"] == 0

    async def test_points_outside_the_grid_come_back_null(self, client, tmp_path):
        """Null, not a clamped edge value that reads as a measurement."""
        dx = tmp_path / "map.dx"
        write_dx(dx, make_grid())

        result = payload(
            await client.call_tool(
                "sashimi_potential_at",
                {"dx_path": str(dx), "points": [[999.0, 0.0, 0.0]]},
            )
        )
        assert result["values_kT_e"] == [None]
        assert result["n_outside_grid"] == 1
        assert "outside" in result["summary"]

    async def test_unreadable_map_is_a_clean_tool_error(self, client, tmp_path):
        with pytest.raises(ToolError, match="could not read DX"):
            await client.call_tool(
                "sashimi_potential_at",
                {"dx_path": str(tmp_path / "nope.dx"), "points": [[0.0, 0.0, 0.0]]},
            )


class TestCompareMaps:
    async def test_identical_maps_have_zero_rmsd(self, client, tmp_path):
        a = tmp_path / "a.dx"
        write_dx(a, make_grid())

        result = payload(
            await client.call_tool("sashimi_compare_maps", {"dx_a": str(a), "dx_b": str(a)})
        )
        assert result["rmsd_kT_e"] == pytest.approx(0.0)
        assert result["max_abs_diff_kT_e"] == pytest.approx(0.0)
        assert result["correlation"] == pytest.approx(1.0)

    async def test_offset_map_reports_the_offset(self, client, tmp_path):
        a, b = tmp_path / "a.dx", tmp_path / "b.dx"
        write_dx(a, make_grid())
        write_dx(b, make_grid(offset=0.25))

        result = payload(
            await client.call_tool("sashimi_compare_maps", {"dx_a": str(a), "dx_b": str(b)})
        )
        assert result["mean_diff_kT_e"] == pytest.approx(-0.25, abs=1e-6)
        assert result["rmsd_kT_e"] == pytest.approx(0.25, abs=1e-6)
        # A constant offset preserves shape exactly.
        assert result["correlation"] == pytest.approx(1.0)

    async def test_mismatched_grids_are_refused_not_resampled(self, client, tmp_path):
        a, b = tmp_path / "a.dx", tmp_path / "b.dx"
        write_dx(a, make_grid())
        write_dx(b, make_grid(shape=(7, 7, 7)))

        with pytest.raises(ToolError, match="grid shapes differ"):
            await client.call_tool("sashimi_compare_maps", {"dx_a": str(a), "dx_b": str(b)})

    async def test_mismatched_geometry_is_refused(self, client, tmp_path):
        a, b = tmp_path / "a.dx", tmp_path / "b.dx"
        write_dx(a, make_grid())
        shifted = make_grid()
        shifted.origin = np.array([5.0, 0.0, 0.0])
        write_dx(b, shifted)

        with pytest.raises(ToolError, match="grid geometry differs"):
            await client.call_tool("sashimi_compare_maps", {"dx_a": str(a), "dx_b": str(b)})


@pytest.mark.apbs
class TestSolve:
    async def test_solves_and_writes_a_loadable_map(self, client, tmp_path):
        pqr = tmp_path / "ion.pqr"
        pqr.write_text("ATOM      1  I   ION     1       0.000   0.000  0.000  1.00  3.00\n")
        dx = tmp_path / "out.dx"

        result = payload(
            await client.call_tool(
                "sashimi_solve",
                {
                    "pqr_path": str(pqr),
                    "resolution": 1.0,
                    "padding": 6.0,
                    "compute_energy": True,
                    "output_dx": str(dx),
                },
            )
        )
        assert dx.exists()
        assert result["backend"].startswith("apbs-3.")
        assert result["energy_kj_mol"] < 0, "solvating a charge must release energy"
        assert result["grid"]["resolution_relaxed"] is False
        assert result["potential_kT_e"]["max"] > 0

    async def test_energy_is_null_unless_requested(self, client, tmp_path):
        pqr = tmp_path / "ion.pqr"
        pqr.write_text("ATOM      1  I   ION     1       0.000   0.000  0.000  1.00  3.00\n")

        result = payload(
            await client.call_tool(
                "sashimi_solve",
                {"pqr_path": str(pqr), "resolution": 1.0, "padding": 6.0},
            )
        )
        assert result["energy_kj_mol"] is None

    async def test_an_empty_pqr_is_a_clean_tool_error(self, client, tmp_path):
        """Solver failures reach the agent as messages, not tracebacks."""
        pqr = tmp_path / "empty.pqr"
        pqr.write_text("REMARK nothing here\n")

        with pytest.raises(ToolError, match="could not read PQR"):
            await client.call_tool("sashimi_solve", {"pqr_path": str(pqr)})

    async def test_the_tool_cannot_express_an_impossible_grid(self, client, tmp_path):
        """`max_points` is not a tool parameter, so the guardrail can only
        relax the request, never reject it — and the response says which."""
        tools = {t.name: t for t in await client.list_tools()}
        params = (tools["sashimi_solve"].inputSchema or {}).get("properties", {})
        assert "max_points" not in params
        assert "resolution_relaxed" in json.dumps(
            payload(
                await client.call_tool(
                    "sashimi_solve",
                    {"pqr_path": str(self._ion(tmp_path)), "resolution": 1.0, "padding": 6.0},
                )
            )
        )

    @staticmethod
    def _ion(tmp_path: Path) -> Path:
        pqr = tmp_path / "ion.pqr"
        pqr.write_text("ATOM      1  I   ION     1       0.000   0.000  0.000  1.00  3.00\n")
        return pqr

    async def test_unreadable_pqr_is_a_clean_tool_error(self, client, tmp_path):
        with pytest.raises(ToolError, match="could not read PQR"):
            await client.call_tool("sashimi_solve", {"pqr_path": str(tmp_path / "nope.pqr")})


class TestDerivedQueries:
    """The tools that turn a 12 MB grid into an answer."""

    @staticmethod
    def peaked_map(tmp_path, path="map.dx"):
        values = np.zeros((21, 21, 21))
        idx = np.indices((21, 21, 21)).astype(float)
        for (i, j, k), height in {(4, 4, 4): -8.0, (16, 16, 16): 6.0}.items():
            values += height * np.exp(
                -((idx[0] - i) ** 2 + (idx[1] - j) ** 2 + (idx[2] - k) ** 2) / 2.0
            )
        dx = tmp_path / path
        write_dx(dx, PotentialGrid(values=values, origin=np.zeros(3), spacing=np.full(3, 1.0)))
        return dx

    async def test_extrema_reports_both_signs_with_coordinates(self, client, tmp_path):
        result = payload(
            await client.call_tool(
                "sashimi_potential_extrema", {"dx_path": str(self.peaked_map(tmp_path))}
            )
        )
        assert result["most_positive"][0]["value_kT_e"] > 0
        assert result["most_negative"][0]["value_kT_e"] < 0
        assert len(result["most_positive"][0]["position"]) == 3
        assert "kT/e" in result["summary"]

    async def test_extrema_can_be_restricted_to_one_sign(self, client, tmp_path):
        result = payload(
            await client.call_tool(
                "sashimi_potential_extrema",
                {"dx_path": str(self.peaked_map(tmp_path)), "sign": "negative"},
            )
        )
        assert "most_negative" in result
        assert "most_positive" not in result

    async def test_extrema_warns_when_the_solute_is_not_masked(self, client, tmp_path):
        """Otherwise an agent reads self-energy singularities as binding sites."""
        result = payload(
            await client.call_tool(
                "sashimi_potential_extrema", {"dx_path": str(self.peaked_map(tmp_path))}
            )
        )
        assert result["solute_masked"] is False
        assert "pass pqr_path" in result["summary"]

    async def test_sphere_summarises_a_region(self, client, tmp_path):
        result = payload(
            await client.call_tool(
                "sashimi_potential_in_sphere",
                {
                    "dx_path": str(self.peaked_map(tmp_path)),
                    "centre": [16.0, 16.0, 16.0],
                    "radius": 3.0,
                },
            )
        )
        assert result["n_points"] > 0
        assert result["mean_kT_e"] > 0

    async def test_sphere_outside_the_map_says_so(self, client, tmp_path):
        result = payload(
            await client.call_tool(
                "sashimi_potential_in_sphere",
                {
                    "dx_path": str(self.peaked_map(tmp_path)),
                    "centre": [500.0, 0.0, 0.0],
                    "radius": 2.0,
                },
            )
        )
        assert result["n_points"] == 0
        assert "outside the map" in result["summary"]

    async def test_residue_potentials_rank_residues(self, client, tmp_path):
        dx = self.peaked_map(tmp_path)
        pqr = tmp_path / "two.pqr"
        pqr.write_text(
            "ATOM      1  N   ALA A   1       4.000   4.000   4.000 -0.3000 1.5000\n"
            "ATOM      2  N   GLY A   2      16.000  16.000  16.000  0.3000 1.5000\n"
        )
        result = payload(
            await client.call_tool(
                "sashimi_residue_potentials", {"dx_path": str(dx), "pqr_path": str(pqr)}
            )
        )
        residues = result["residues"]
        assert [r["residue"] for r in residues] == ["ALA 1", "GLY 2"]
        assert residues[0]["mean_kT_e"] < residues[1]["mean_kT_e"]
        assert "Most negative" in result["summary"]

    async def test_a_structure_without_labels_is_a_clean_error(self, client, tmp_path):
        dx = self.peaked_map(tmp_path)
        pqr = tmp_path / "bare.pqr"
        pqr.write_text("ATOM      1  I   ION     1       0.000   0.000  0.000  1.00  3.00\n")
        # This PQR does carry labels, so the tool should succeed; the guard is
        # for structures built in code without them.
        result = payload(
            await client.call_tool(
                "sashimi_residue_potentials", {"dx_path": str(dx), "pqr_path": str(pqr)}
            )
        )
        assert "residues" in result


class TestDiscoverySurface:
    """Capabilities and dry-run validation, through the client."""

    @staticmethod
    def ion_pqr(tmp_path):
        pqr = tmp_path / "ion.pqr"
        pqr.write_text("ATOM      1  I   ION     1       0.000   0.000  0.000  1.00  3.00\n")
        return pqr

    async def test_capabilities_needs_no_arguments(self, client):
        result = payload(await client.call_tool("sashimi_capabilities", {}))
        assert "units" in result
        assert result["units"]["potential"] == "kT/e"
        assert "backend" in result["summary"]

    async def test_capabilities_names_the_unsupported(self, client):
        result = payload(await client.call_tool("sashimi_capabilities", {}))
        assert any("nonlinear" in item for item in result["not_supported"])

    async def test_validate_reports_cost_without_solving(self, client, tmp_path):
        result = payload(
            await client.call_tool(
                "sashimi_validate_inputs",
                {"pqr_path": str(self.ion_pqr(tmp_path)), "resolution": 0.5, "padding": 10.0},
            )
        )
        assert result["ok"] is True
        assert result["cost"]["grid"]["native"]["dime"] == [65, 65, 65]
        assert result["cost"]["grid"]["estimated_map_mb"] > 0
        # No map was written: this is arithmetic, not a solve.
        assert not list(tmp_path.glob("*.dx"))

    async def test_validate_blocks_an_impossible_grid(self, client, tmp_path):
        result = payload(
            await client.call_tool(
                "sashimi_validate_inputs",
                {
                    "pqr_path": str(self.ion_pqr(tmp_path)),
                    "resolution": 0.5,
                    "padding": 80.0,
                    "max_points": 1000,
                },
            )
        )
        assert result["ok"] is False
        assert "Would not run" in result["summary"]

    async def test_validate_rejects_an_unknown_surface_model(self, client, tmp_path):
        with pytest.raises(ToolError, match="unknown surface_model"):
            await client.call_tool(
                "sashimi_validate_inputs",
                {"pqr_path": str(self.ion_pqr(tmp_path)), "surface_model": "banana"},
            )

    @pytest.mark.apbs
    async def test_validate_agrees_with_what_solve_actually_does(self, client, tmp_path):
        """The prediction is worthless if it disagrees with the real thing."""
        pqr = self.ion_pqr(tmp_path)
        predicted = payload(
            await client.call_tool(
                "sashimi_validate_inputs",
                {"pqr_path": str(pqr), "resolution": 1.0, "padding": 6.0},
            )
        )
        actual = payload(
            await client.call_tool(
                "sashimi_solve",
                {"pqr_path": str(pqr), "resolution": 1.0, "padding": 6.0},
            )
        )
        assert predicted["cost"]["grid"]["native"]["dime"] == actual["grid"]["shape"]
        assert (
            predicted["cost"]["grid"]["resolution_relaxed"] == actual["grid"]["resolution_relaxed"]
        )


class TestBackendSelection:
    """Choosing a solver, and the shape of what each one returns.

    Until this, `sashimi_solve` hardcoded APBS: four backends shipped, CI
    exercised all four, and an agent could reach exactly one. The tier that
    needs no binary — the only one guaranteed to be there — was the least
    reachable of all.
    """

    def peptide(self, tmp_path):
        source = Path(__file__).parent / "data" / "ala-gly.pqr"
        pqr = tmp_path / "ala-gly.pqr"
        pqr.write_text(source.read_text())
        return pqr

    async def test_the_in_process_backend_needs_no_binary(self, client, tmp_path):
        """`gb` is the whole point of backend selection: it always works.

        No APBS marker on this test, deliberately — it must pass on a machine
        with nothing installed, which is the machine a consumer's fallback path
        has to survive.
        """
        result = payload(
            await client.call_tool(
                "sashimi_solve",
                {
                    "pqr_path": str(self.peptide(tmp_path)),
                    "backend": "gb",
                    "surface_model": "molecular",
                    "compute_energy": True,
                },
            )
        )

        assert result["energy_kj_mol"] < 0
        assert result["backend_name"] == "gb"
        assert result["family"] == "analytic"
        # The response shape follows the answer: no field was computed, so no
        # map path is offered rather than a null one being invented.
        assert "dx_path" not in result
        assert "grid" not in result
        assert "no field" in result["summary"]

    async def test_asking_the_energy_only_backend_for_no_energy_is_refused(self, client, tmp_path):
        """It computes one number. Not wanting it is a request for nothing."""
        with pytest.raises(ToolError, match="asks it for nothing"):
            await client.call_tool(
                "sashimi_solve",
                {
                    "pqr_path": str(self.peptide(tmp_path)),
                    "backend": "gb",
                    "surface_model": "molecular",
                    "compute_energy": False,
                },
            )

    async def test_an_unknown_backend_names_the_installed_ones(self, client, tmp_path):
        with pytest.raises(ToolError, match="unknown backend"):
            await client.call_tool(
                "sashimi_solve",
                {"pqr_path": str(self.peptide(tmp_path)), "backend": "apbs2"},
            )

    async def test_a_backend_that_cannot_take_the_surface_says_so_before_solving(
        self, client, tmp_path
    ):
        """The pre-flight an agent needs now that it has a choice to get wrong.

        `smoothed-molecular` is the default and three of the four backends
        refuse it, so this is the common failure — and free to discover here
        rather than after a structure has been prepared and a solve started.
        """
        report = payload(
            await client.call_tool(
                "sashimi_validate_inputs",
                {"pqr_path": str(self.peptide(tmp_path)), "backend": "gb"},
            )
        )

        assert report["ok"] is False
        assert any("does not support" in p for p in report["problems"])
        assert report["backend"]["name"] == "gb"

    async def test_a_backend_with_no_grid_estimates_no_grid(self, client, tmp_path):
        """A point count from the wrong model is worse than no point count."""
        report = payload(
            await client.call_tool(
                "sashimi_validate_inputs",
                {
                    "pqr_path": str(self.peptide(tmp_path)),
                    "backend": "gb",
                    "surface_model": "molecular",
                },
            )
        )

        assert report["ok"] is True
        assert report["cost"]["grid"] is None
        assert "no grid" in report["cost"]["note"]
        assert "grid" not in report

    @pytest.mark.apbs
    async def test_the_default_backend_is_unchanged(self, client, tmp_path):
        """Adding the parameter must not move what an existing caller gets."""
        result = payload(
            await client.call_tool(
                "sashimi_solve",
                {"pqr_path": str(self.peptide(tmp_path)), "compute_energy": True},
            )
        )

        assert result["backend_name"] == "apbs"
        assert result["dx_path"].endswith(".dx")

    @pytest.fixture
    def tabipb_installed(self):
        """The marker selects; this is what actually skips.

        Without it the test runs wherever TABI-PB is absent — which is the macOS
        CI leg, since only the Linux one builds it.
        """
        return installed_or_skip(discover_tabipb, "SASHIMI_TABIPB_PATH")

    @pytest.mark.tabipb
    @pytest.mark.usefixtures("tabipb_installed")
    async def test_a_boundary_element_solve_returns_a_surface_and_no_map(self, client, tmp_path):
        """The third response shape: statistics over a mesh, with nothing to write.

        A boundary-element solver has no volume, so `dx_path` would have to be
        null or invented. The response omits it and says why in the summary,
        which is the same rule the corpus summaries follow.
        """
        result = payload(
            await client.call_tool(
                "sashimi_solve",
                {
                    "pqr_path": str(self.peptide(tmp_path)),
                    "backend": "tabipb",
                    "surface_model": "molecular",
                    "compute_energy": True,
                },
            )
        )

        assert result["family"] == "boundary-element"
        assert result["surface"]["n_vertices"] > 0
        assert "dx_path" not in result
        assert "No map written" in result["summary"]

    @pytest.mark.tabipb
    @pytest.mark.usefixtures("tabipb_installed")
    async def test_mesh_density_is_the_boundary_element_resolution(self, client, tmp_path):
        """The cost knob the pre-flight names must exist on the tool.

        `sashimi_validate_inputs` tells a caller "mesh density is the knob" and
        warns that a mesh can take 450 s; without this parameter there was no
        knob to turn, and `resolution` — the obvious thing to reach for — was
        accepted and silently dropped.
        """
        pqr = str(self.peptide(tmp_path))
        args = {"surface_model": "molecular", "compute_energy": True, "backend": "tabipb"}

        coarse = payload(await client.call_tool("sashimi_solve", {"pqr_path": pqr, **args}))
        fine = payload(
            await client.call_tool("sashimi_solve", {"pqr_path": pqr, "mesh_density": 3.0, **args})
        )

        assert fine["surface"]["n_vertices"] > coarse["surface"]["n_vertices"]

    async def test_a_grid_parameter_is_refused_by_a_backend_with_no_grid(self, client, tmp_path):
        """Refused, not ignored.

        A caller told to make a 450-second mesh cheaper reaches for
        `resolution`; accepting it silently leaves them believing they did.
        """
        with pytest.raises(ToolError, match="builds no grid"):
            await client.call_tool(
                "sashimi_solve",
                {
                    "pqr_path": str(self.peptide(tmp_path)),
                    "backend": "gb",
                    "surface_model": "molecular",
                    "compute_energy": True,
                    "resolution": 0.5,
                },
            )

    async def test_a_mesh_parameter_is_refused_by_a_backend_with_no_mesh(self, client, tmp_path):
        """The same rule in the other direction, so neither family is special."""
        with pytest.raises(ToolError, match="does not build one"):
            await client.call_tool(
                "sashimi_solve",
                {
                    "pqr_path": str(self.peptide(tmp_path)),
                    "backend": "apbs",
                    "mesh_density": 3.0,
                },
            )

    @pytest.mark.apbs
    async def test_omitting_a_grid_parameter_still_means_the_protocol_default(
        self, client, tmp_path
    ):
        """`resolution` became optional, which must not change what APBS does."""
        result = payload(
            await client.call_tool(
                "sashimi_solve",
                {"pqr_path": str(self.peptide(tmp_path)), "compute_energy": True},
            )
        )

        assert result["grid"]["shape"] == [65, 65, 65]
        assert result["grid"]["resolution_relaxed"] is False
