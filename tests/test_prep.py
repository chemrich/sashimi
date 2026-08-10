"""Structure preparation.

No `apbs` marker: pdb2pqr is a Python dependency, so these run in the
binary-free tier alongside the parser tests.
"""

from pathlib import Path

import pytest

from sashimi.prep import PreparationFailed, prepare_structure

FIXTURE = Path(__file__).parent / "data" / "ala-gly.pdb"


@pytest.fixture(scope="module")
def prepared():
    """One pdb2pqr run for the whole module; it takes a second or two."""
    return prepare_structure(FIXTURE)


def test_produces_a_usable_structure(prepared):
    pqr = prepared.pqr
    assert pqr.n_atoms > 9, "hydrogens should have been added"
    assert (pqr.radii > 0).all(), "every atom needs a radius for the solver"
    assert prepared.pqr_text.startswith("ATOM")


def test_neutral_dipeptide_comes_out_near_neutral(prepared):
    """A sanity check on charge assignment, not a physics claim."""
    assert abs(prepared.pqr.total_charge) < 0.01


def test_rebuilt_atoms_are_surfaced_not_buried(prepared):
    """The fixture is missing OXT; an agent must be told it was added."""
    assert prepared.structure_was_modified
    assert any("OXT" in w for w in prepared.warnings)
    assert any("OXT" in entry for entry in prepared.edits["added_atoms"])


def test_warnings_are_deduplicated(prepared):
    """pdb2pqr emits each record on both streams; the caller sees it once."""
    assert len(prepared.warnings) == len(set(prepared.warnings))


def test_only_structural_warnings_reach_the_caller(prepared):
    """The fixture must stay standards-conforming.

    An earlier version of it used REMARK lines without the required remark
    number, which made pdb2pqr report the file as non-standard and buried the
    one warning that matters under parser noise.
    """
    assert prepared.warnings == ("Missing atom OXT in residue GLY A 2",)


def test_summary_is_json_shaped(prepared):
    summary = prepared.summary()
    assert summary["n_atoms"] == prepared.pqr.n_atoms
    assert summary["structure_was_modified"] is True
    assert isinstance(summary["edits"], dict)
    assert isinstance(summary["warnings"], list)
    # Only non-empty edit categories are reported.
    assert all(v for v in summary["edits"].values())


def test_missing_file_fails_before_launching_pdb2pqr(tmp_path):
    with pytest.raises(PreparationFailed, match="no such structure file"):
        prepare_structure(tmp_path / "nope.pdb")


def test_unparseable_input_is_reported(tmp_path):
    junk = tmp_path / "junk.pdb"
    junk.write_text("this is not a PDB file\n")
    with pytest.raises(PreparationFailed):
        prepare_structure(junk)


def test_forcefield_choice_changes_the_charges(tmp_path):
    """Different parameter sets must actually reach pdb2pqr."""
    amber = prepare_structure(FIXTURE, forcefield="AMBER")
    parse = prepare_structure(FIXTURE, forcefield="PARSE")
    assert amber.pqr.n_atoms > 0 and parse.pqr.n_atoms > 0
    assert not (
        amber.pqr.radii.shape == parse.pqr.radii.shape
        and (amber.pqr.radii == parse.pqr.radii).all()
        and (amber.pqr.charges == parse.pqr.charges).all()
    ), "AMBER and PARSE should not produce identical charges and radii"
