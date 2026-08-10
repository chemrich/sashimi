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

FIXTURE_PDB = Path(__file__).parent / "data" / "ala-gly.pdb"

EXPECTED_TOOLS = {
    "sashimi_compare_maps",
    "sashimi_potential_at",
    "sashimi_prepare_structure",
    "sashimi_solve",
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
