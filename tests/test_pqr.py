import numpy as np
import pytest

from sashimi.pqr import format_pqr, parse_pqr
from sashimi.protocol import PQRData

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


class TestFixedColumns:
    """The writer's columns, which a whitespace-splitting reader cannot check.

    `format_pqr` used minimum-width fields, so a four-character residue name
    pushed every field after it one column right. Every reader sashimi owns
    splits on whitespace and APBS's is lenient, so nothing noticed for a year.
    DelPhi reads fixed columns: acetate arrived as two charged atoms carrying
    +80.84 e where the file says seven and -1, and the solve returned
    -865,205 kJ/mol against APBS's -196.90.
    """

    def structure(self, res_name: str, atom_name: str = "C1") -> PQRData:
        return PQRData(
            coords=np.zeros((1, 3)),
            charges=np.array([-1.0]),
            radii=np.array([1.88]),
            labels=(f"{res_name} 1 {atom_name}",),
        )

    @pytest.mark.parametrize("res_name", ["ALA", "TARG", "MEOH", "A", "HOH"])
    def test_the_line_is_the_same_length_whatever_the_residue_is_called(self, res_name: str):
        """Column stability is the property; the name is not read by anything."""
        line = format_pqr(self.structure(res_name)).splitlines()[0]

        assert len(line) == 69

    @pytest.mark.parametrize("res_name", ["ALA", "TARG", "MEOH"])
    def test_the_numbers_land_in_the_same_columns(self, res_name: str):
        """What a fixed-column reader depends on: coordinates at 31, and the
        charge and radius where they were."""
        line = format_pqr(self.structure(res_name)).splitlines()[0]

        assert line[30:38].strip() == "0.000"
        assert line[54:62].strip() == "-1.0000"
        assert line[62:].strip() == "1.8800"

    @pytest.mark.parametrize("atom_name", ["C", "CA", "CH3", "HG12"])
    def test_a_long_atom_name_does_not_shift_the_line_either(self, atom_name: str):
        line = format_pqr(self.structure("ALA", atom_name)).splitlines()[0]

        assert len(line) == 69

    def test_a_three_character_residue_renders_exactly_as_it_used_to(self):
        """The fix must move no recorded number, and this is why it does not:
        for names that fit, exact widths and minimum widths are the same bytes."""
        line = format_pqr(self.structure("ALA", "N")).splitlines()[0]

        assert line == ("ATOM      1    N ALA     1       0.000   0.000   0.000 -1.0000 1.8800")

    def test_what_is_written_is_what_is_read_back(self):
        """Round-tripping is necessary and was never sufficient: the old writer
        round-tripped perfectly through sashimi's own whitespace reader while
        being unreadable to a fixed-column one."""
        original = self.structure("TARG")
        recovered = parse_pqr(format_pqr(original))

        assert recovered.n_atoms == 1
        np.testing.assert_allclose(recovered.charges, original.charges)
        np.testing.assert_allclose(recovered.radii, original.radii)
