import numpy as np
import pytest

from sashimi.pqr import format_pqr, parse_pqr

BORN_ION = "ATOM      1  I   ION     1       0.000   0.000  0.000  1.00  3.00\n"

# Chain ID present — one extra field before the coordinates.
WITH_CHAIN = (
    "ATOM      1  N   MET A   1      -1.245   2.100   0.000 -0.3000 1.8240\n"
    "ATOM      2  CA  MET A   1       0.000   0.000   1.500  0.0900 1.9080\n"
)


def test_parses_born_ion():
    pqr = parse_pqr(BORN_ION)
    assert pqr.n_atoms == 1
    assert pqr.total_charge == pytest.approx(1.0)
    np.testing.assert_allclose(pqr.coords, [[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(pqr.radii, [3.0])


def test_parses_with_and_without_chain_id():
    """The chain field is optional, so fields are read from the end of the line."""
    pqr = parse_pqr(WITH_CHAIN)
    assert pqr.n_atoms == 2
    np.testing.assert_allclose(pqr.charges, [-0.30, 0.09])
    np.testing.assert_allclose(pqr.radii, [1.8240, 1.9080])
    np.testing.assert_allclose(pqr.coords[1], [0.0, 0.0, 1.5])


def test_ignores_non_atom_records():
    text = "REMARK something\n" + BORN_ION + "TER\nEND\n"
    assert parse_pqr(text).n_atoms == 1


def test_round_trip_preserves_numbers():
    original = parse_pqr(WITH_CHAIN)
    again = parse_pqr(format_pqr(original))
    np.testing.assert_allclose(again.coords, original.coords, atol=1e-3)
    np.testing.assert_allclose(again.charges, original.charges, atol=1e-4)
    np.testing.assert_allclose(again.radii, original.radii, atol=1e-4)


def test_rejects_empty():
    with pytest.raises(ValueError, match="no ATOM/HETATM"):
        parse_pqr("REMARK nothing here\n")


def test_rejects_short_line():
    with pytest.raises(ValueError, match="at least 9 fields"):
        parse_pqr("ATOM 1 I ION 1 0.0 0.0\n")


def test_rejects_non_numeric():
    with pytest.raises(ValueError, match="non-numeric"):
        parse_pqr("ATOM      1  I   ION     1       0.000   0.000  0.000  1.00  abc\n")


def test_extent_and_center_include_radii():
    pqr = parse_pqr(BORN_ION)
    np.testing.assert_allclose(pqr.extent(), [6.0, 6.0, 6.0])
    np.testing.assert_allclose(pqr.center(), [0.0, 0.0, 0.0])
