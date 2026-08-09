import numpy as np
import pytest

from sashimi.dx import parse_dx, read_dx, write_dx
from sashimi.protocol import PotentialGrid

# The header layout verified against APBS 3.4.1 output in Phase 0.
APBS_HEADER = """\
# Data from APBS 3.4.1
#
# POTENTIAL (kT/e)
#
object 1 class gridpositions counts 2 2 2
origin -6.000000e+00 -6.000000e+00 -6.000000e+00
delta 1.875000e-01 0.000000e+00 0.000000e+00
delta 0.000000e+00 1.875000e-01 0.000000e+00
delta 0.000000e+00 0.000000e+00 1.875000e-01
object 2 class gridconnections counts 2 2 2
object 3 class array type double rank 0 items 8 data follows
1.000000e+00 2.000000e+00 3.000000e+00
4.000000e+00 5.000000e+00 6.000000e+00
7.000000e+00 8.000000e+00
attribute "dep" string "positions"
object "regular positions regular connections" class field
component "positions" value 1
component "connections" value 2
component "data" value 3
"""


def make_grid(shape=(5, 6, 7)):
    rng = np.random.default_rng(0)
    return PotentialGrid(
        values=rng.normal(size=shape),
        origin=np.array([-3.0, -4.0, -5.0]),
        spacing=np.array([0.25, 0.5, 0.75]),
    )


def test_parses_apbs_header_in_c_order():
    grid = parse_dx(APBS_HEADER)
    assert grid.shape == (2, 2, 2)
    np.testing.assert_allclose(grid.origin, [-6.0, -6.0, -6.0])
    np.testing.assert_allclose(grid.spacing, [0.1875, 0.1875, 0.1875])
    # C order: last index varies fastest.
    assert grid.values[0, 0, 0] == 1.0
    assert grid.values[0, 0, 1] == 2.0
    assert grid.values[1, 1, 1] == 8.0


def test_round_trip_is_lossless_to_written_precision(tmp_path):
    original = make_grid()
    path = tmp_path / "p.dx"
    write_dx(path, original)
    again = read_dx(path)
    assert again.shape == original.shape
    np.testing.assert_allclose(again.origin, original.origin)
    np.testing.assert_allclose(again.spacing, original.spacing)
    np.testing.assert_allclose(again.values, original.values, rtol=1e-6)


def test_written_header_matches_apbs_conventions(tmp_path):
    path = tmp_path / "p.dx"
    write_dx(path, make_grid(shape=(2, 3, 4)))
    text = path.read_text()
    assert "object 1 class gridpositions counts 2 3 4" in text
    assert "items 24 data follows" in text
    assert 'component "data" value 3' in text


def test_rejects_truncated_data():
    truncated = APBS_HEADER.replace("7.000000e+00 8.000000e+00\n", "")
    with pytest.raises(ValueError, match="truncated"):
        parse_dx(truncated)


def test_rejects_inconsistent_item_count():
    bad = APBS_HEADER.replace("items 8", "items 9")
    with pytest.raises(ValueError, match="inconsistent"):
        parse_dx(bad)


def test_rejects_non_axis_aligned():
    skewed = APBS_HEADER.replace(
        "delta 0.000000e+00 1.875000e-01 0.000000e+00",
        "delta 1.000000e-02 1.875000e-01 0.000000e+00",
    )
    with pytest.raises(ValueError, match="non-axis-aligned"):
        parse_dx(skewed)


def test_rejects_missing_header():
    with pytest.raises(ValueError, match="malformed DX header"):
        parse_dx("# nothing but a comment\n")


def test_value_at_interpolates_a_linear_ramp():
    """A field linear in x must interpolate exactly."""
    nx, ny, nz = 9, 4, 4
    x = np.arange(nx) * 0.5 - 2.0
    values = np.repeat(np.repeat(x[:, None, None], ny, axis=1), nz, axis=2)
    grid = PotentialGrid(values=values, origin=np.array([-2.0, 0.0, 0.0]), spacing=np.full(3, 0.5))

    probe = np.array([[-1.75, 0.5, 0.5], [0.0, 0.25, 0.75], [1.1, 1.0, 1.0]])
    np.testing.assert_allclose(grid.value_at(probe), probe[:, 0], atol=1e-12)


def test_value_at_returns_nan_outside_the_grid():
    grid = make_grid()
    out = grid.value_at(np.array([[1e3, 0.0, 0.0], [-3.0, -4.0, -5.0]]))
    assert np.isnan(out[0])
    assert not np.isnan(out[1])


def test_stats_reports_geometry_and_range():
    stats = make_grid(shape=(3, 3, 3)).stats()
    assert stats["shape"] == [3, 3, 3]
    assert stats["min"] <= stats["mean"] <= stats["max"]
