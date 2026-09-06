"""Derived queries.

Binary-free: every function here is pure, over a synthetic field whose answers
are known by construction. That is the point of the module — the questions an
agent actually asks are cheap and testable anywhere.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from sashimi.analysis import (
    _labelled,
    _residue_groups,
    _residue_of,
    _solute_mask,
    potential_extrema,
    potential_in_sphere,
    residue_potentials,
)
from sashimi.pqr import read_pqr
from sashimi.protocol import PotentialGrid, PQRData

SPACING = 1.0


def grid_with_peaks(peaks: dict[tuple[int, int, int], float], shape=(21, 21, 21)):
    """A flat field with Gaussian bumps at named grid indices."""
    values = np.zeros(shape)
    idx = np.indices(shape).astype(float)
    for (i, j, k), height in peaks.items():
        r2 = (idx[0] - i) ** 2 + (idx[1] - j) ** 2 + (idx[2] - k) ** 2
        values += height * np.exp(-r2 / 2.0)
    return PotentialGrid(values=values, origin=np.zeros(3), spacing=np.full(3, SPACING))


class TestExtrema:
    def test_finds_a_single_peak_where_it_was_put(self):
        grid = grid_with_peaks({(10, 10, 10): 5.0})
        [peak] = potential_extrema(grid, n=1)
        np.testing.assert_allclose(peak.position, [10.0, 10.0, 10.0], atol=SPACING)
        assert peak.value == pytest.approx(5.0, rel=0.01)

    def test_separates_distinct_peaks(self):
        grid = grid_with_peaks({(4, 4, 4): 5.0, (16, 16, 16): 4.0})
        peaks = potential_extrema(grid, n=2, min_separation=5.0)
        assert len(peaks) == 2
        assert peaks[0].value > peaks[1].value
        separation = np.linalg.norm(np.array(peaks[0].position) - np.array(peaks[1].position))
        assert separation >= 5.0

    def test_suppression_prevents_reporting_one_peak_n_times(self):
        """Without it the top-n are all neighbours of the same maximum."""
        grid = grid_with_peaks({(10, 10, 10): 5.0})
        assert len(potential_extrema(grid, n=5, min_separation=8.0)) == 1

        crowded = potential_extrema(grid, n=5, min_separation=0.0)
        assert len(crowded) == 5, "with no suppression, neighbours fill the list"

    def test_noise_is_not_reported_as_a_peak(self):
        """One real feature must yield one answer, not n padded with 1e-14."""
        grid = grid_with_peaks({(10, 10, 10): 5.0})
        assert len(potential_extrema(grid, n=5, min_separation=8.0)) == 1

    def test_min_fraction_zero_restores_the_padding(self):
        """The floor is a policy, not a hard-coded truth."""
        grid = grid_with_peaks({(10, 10, 10): 5.0})
        assert len(potential_extrema(grid, n=5, min_separation=8.0, min_fraction=0.0)) > 1

    def test_genuinely_weaker_peaks_still_appear(self):
        """The floor must not swallow a real secondary feature."""
        grid = grid_with_peaks({(4, 4, 4): 5.0, (16, 16, 16): 1.0})
        assert len(potential_extrema(grid, n=5, min_separation=5.0)) == 2

    def test_solute_masking_finds_the_solvent_side_feature(self):
        """Unmasked, the answer is always "at the atoms" — true and useless."""
        grid = grid_with_peaks({(10, 10, 10): 50.0, (16, 16, 16): 3.0})
        atom = PQRData(
            coords=np.array([[10.0, 10.0, 10.0]]),
            charges=np.array([1.0]),
            radii=np.array([2.0]),
            labels=("ION 1 I",),
        )
        unmasked = potential_extrema(grid, n=1)
        assert unmasked[0].value == pytest.approx(50.0, rel=0.01)

        masked = potential_extrema(grid, n=1, exclude_near=atom, exclusion_margin=3.0)
        assert masked[0].value == pytest.approx(3.0, rel=0.05)
        np.testing.assert_allclose(masked[0].position, [16.0, 16.0, 16.0], atol=SPACING)

    def test_masking_everything_returns_nothing_rather_than_noise(self):
        grid = grid_with_peaks({(10, 10, 10): 5.0})
        blanket = PQRData(
            coords=np.array([[10.0, 10.0, 10.0]]),
            charges=np.array([1.0]),
            radii=np.array([100.0]),
            labels=("BIG 1 X",),
        )
        assert potential_extrema(grid, n=5, exclude_near=blanket) == []

    def test_negative_mode_finds_troughs(self):
        grid = grid_with_peaks({(5, 5, 5): 3.0, (15, 15, 15): -7.0})
        [trough] = potential_extrema(grid, n=1, most_positive=False)
        assert trough.value == pytest.approx(-7.0, rel=0.01)
        np.testing.assert_allclose(trough.position, [15.0, 15.0, 15.0], atol=SPACING)

    def test_positions_are_in_angstroms_not_indices(self):
        grid = PotentialGrid(
            values=grid_with_peaks({(10, 10, 10): 1.0}).values,
            origin=np.array([-5.0, -5.0, -5.0]),
            spacing=np.full(3, 0.5),
        )
        [peak] = potential_extrema(grid, n=1)
        np.testing.assert_allclose(peak.position, [0.0, 0.0, 0.0], atol=0.5)

    def test_rejects_nonsense_arguments(self):
        grid = grid_with_peaks({(10, 10, 10): 1.0})
        with pytest.raises(ValueError, match="n must be positive"):
            potential_extrema(grid, n=0)
        with pytest.raises(ValueError, match="min_separation"):
            potential_extrema(grid, min_separation=-1.0)


class TestSphere:
    def test_averages_only_what_is_inside(self):
        values = np.zeros((21, 21, 21))
        values[10, 10, 10] = 10.0
        grid = PotentialGrid(values=values, origin=np.zeros(3), spacing=np.full(3, 1.0))

        tight = potential_in_sphere(grid, np.array([10.0, 10.0, 10.0]), radius=0.5)
        assert tight["n_points"] == 1
        assert tight["mean_kT_e"] == pytest.approx(10.0)

        loose = potential_in_sphere(grid, np.array([10.0, 10.0, 10.0]), radius=5.0)
        assert loose["n_points"] > 1
        assert loose["mean_kT_e"] < 10.0

    def test_reports_emptiness_rather_than_dividing_by_zero(self):
        grid = grid_with_peaks({(10, 10, 10): 1.0})
        stats = potential_in_sphere(grid, np.array([500.0, 0.0, 0.0]), radius=1.0)
        assert stats["n_points"] == 0
        assert "no grid points" in stats["note"]

    def test_point_count_is_reported_so_a_thin_sample_is_visible(self):
        grid = grid_with_peaks({(10, 10, 10): 1.0})
        stats = potential_in_sphere(grid, np.array([10.0, 10.0, 10.0]), radius=1.0)
        assert "n_points" in stats

    def test_rejects_a_nonpositive_radius(self):
        grid = grid_with_peaks({(10, 10, 10): 1.0})
        with pytest.raises(ValueError, match="radius must be positive"):
            potential_in_sphere(grid, np.zeros(3), radius=0.0)

    def test_an_unmasked_pocket_mean_is_the_atom_and_not_the_field(self):
        """The trap `potential_extrema` was given `exclude_near` for, in a mean.

        A pocket contains atoms by definition, and the value at an atom centre
        is that charge's own self-energy. Masking has to move the mean by more
        than rounding or the argument does nothing.
        """
        grid = grid_with_peaks({(10, 10, 10): 500.0, (14, 10, 10): 3.0})
        atom = PQRData(
            coords=np.array([[10.0, 10.0, 10.0]]),
            charges=np.array([1.0]),
            radii=np.array([2.0]),
            labels=("ION 1 I",),
        )
        centre = np.full(3, 10.0 * SPACING)

        unmasked = potential_in_sphere(grid, centre, radius=6.0)
        masked = potential_in_sphere(
            grid, centre, radius=6.0, exclude_near=atom, exclusion_margin=3.0
        )

        assert unmasked["solute_masked"] is False
        assert masked["solute_masked"] is True
        assert masked["n_points_excluded_as_solute"] > 0
        assert masked["n_points"] < unmasked["n_points"]
        # The singularity dominates the unmasked mean; removing it must show.
        assert abs(masked["mean_kT_e"]) < abs(unmasked["mean_kT_e"])
        assert unmasked["max_kT_e"] > 100.0
        assert masked["max_kT_e"] < 100.0

    def test_a_buried_sphere_says_it_is_buried_rather_than_dividing_by_zero(self):
        grid = grid_with_peaks({(10, 10, 10): 5.0})
        blanket = PQRData(
            coords=np.array([[10.0 * SPACING, 10.0 * SPACING, 10.0 * SPACING]]),
            charges=np.array([1.0]),
            radii=np.array([100.0]),
            labels=("BIG 1 B",),
        )
        stats = potential_in_sphere(
            grid, np.full(3, 10.0 * SPACING), radius=2.0, exclude_near=blanket
        )
        assert stats["n_points"] == 0
        assert stats["n_points_in_sphere"] > 0
        assert "buried" in stats["note"]
        # Distinct from the empty-sphere case, which reports no points at all.
        assert "no grid points fall inside" not in stats["note"]

    def test_the_atom_filter_is_answer_preserving(self):
        """The mask is built from nearby atoms only; that must change nothing.

        An atom further than `radius + r_i + margin` from the centre cannot
        contain a point inside the sphere, so restricting to those is exact
        rather than approximate. It is worth ~20x on a protein, and an
        optimisation that moved a reported number would not be worth anything.
        """
        rng = np.random.default_rng(11)
        n = 400
        structure = PQRData(
            coords=rng.uniform(0.0, 12.0, size=(n, 3)),
            charges=rng.normal(size=n),
            radii=rng.uniform(1.2, 2.2, size=n),
        )
        grid = PotentialGrid(
            values=rng.normal(size=(31, 31, 31)),
            origin=np.zeros(3),
            spacing=np.full(3, 0.5),
        )
        for centre in (np.full(3, 6.0), np.full(3, 2.0), np.array([9.0, 3.0, 11.0])):
            for radius in (1.5, 4.0):
                got = potential_in_sphere(grid, centre, radius=radius, exclude_near=structure)
                axes = [
                    grid.origin[a] + np.arange(grid.values.shape[a]) * grid.spacing[a]
                    for a in range(3)
                ]
                xx, yy, zz = np.meshgrid(*axes, indexing="ij")
                in_sphere = (
                    (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (zz - centre[2]) ** 2
                ) <= radius**2
                expected = in_sphere & ~_solute_mask(grid, structure, 1.4)

                assert got["n_points"] == int(expected.sum())
                if expected.any():
                    values = grid.values[expected]
                    assert got["mean_kT_e"] == float(values.mean())
                    assert got["min_kT_e"] == float(values.min())
                    assert got["max_kT_e"] == float(values.max())

    def test_the_filter_carries_optional_fields_rather_than_dropping_them(self):
        """`chains` is length-validated, so a subset that keeps the parent's raises."""
        structure = PQRData(
            coords=np.array([[1.0, 1.0, 1.0], [40.0, 40.0, 40.0]]),
            charges=np.array([1.0, -1.0]),
            radii=np.array([2.0, 2.0]),
            labels=("ALA 1 CA", "GLY 2 CA"),
            chains=("A", "B"),
        )
        grid = PotentialGrid(
            values=np.ones((21, 21, 21)), origin=np.zeros(3), spacing=np.full(3, 0.5)
        )
        stats = potential_in_sphere(
            grid, np.array([1.0, 1.0, 1.0]), radius=3.0, exclude_near=structure
        )
        assert stats["n_points_excluded_as_solute"] > 0

    def test_masking_is_off_by_default_so_the_old_answer_is_unchanged(self):
        grid = grid_with_peaks({(10, 10, 10): 4.0})
        centre = np.full(3, 10.0 * SPACING)
        assert potential_in_sphere(grid, centre, radius=3.0)["mean_kT_e"] == pytest.approx(
            potential_in_sphere(grid, centre, radius=3.0, exclude_near=None)["mean_kT_e"]
        )


class TestResiduePotentials:
    @staticmethod
    def two_residues() -> PQRData:
        return PQRData(
            coords=np.array([[4.0, 10.0, 10.0], [5.0, 10.0, 10.0], [16.0, 10.0, 10.0]]),
            charges=np.array([0.1, -0.1, 0.2]),
            radii=np.array([1.5, 1.5, 1.5]),
            labels=("ALA 1 N", "ALA 1 CA", "GLY 2 N"),
        )

    def test_groups_atoms_into_residues(self):
        grid = grid_with_peaks({(4, 10, 10): -6.0, (16, 10, 10): 6.0})
        results = residue_potentials(grid, self.two_residues())
        assert {r.label for r in results} == {"ALA 1", "GLY 2"}
        assert {r.n_atoms for r in results} == {2, 1}

    def test_sorted_most_negative_first(self):
        grid = grid_with_peaks({(4, 10, 10): -6.0, (16, 10, 10): 6.0})
        results = residue_potentials(grid, self.two_residues())
        assert results[0].label == "ALA 1"
        assert results[0].value < results[-1].value

    def test_top_limits_the_result(self):
        grid = grid_with_peaks({(4, 10, 10): -6.0, (16, 10, 10): 6.0})
        assert len(residue_potentials(grid, self.two_residues(), top=1)) == 1

    def test_under_sampling_is_counted_not_hidden(self):
        """A residue at the box edge must be visibly partial."""
        edge = PQRData(
            coords=np.array([[0.0, 0.0, 0.0]]),
            charges=np.array([1.0]),
            radii=np.array([1.5]),
            labels=("ARG 9 NZ",),
        )
        grid = grid_with_peaks({(10, 10, 10): 1.0})
        results = residue_potentials(grid, edge)
        assert results == [] or results[0].n_sampled <= results[0].n_atoms

    def test_requires_labels(self):
        unlabelled = PQRData(
            coords=np.zeros((1, 3)), charges=np.array([1.0]), radii=np.array([1.5])
        )
        grid = grid_with_peaks({(10, 10, 10): 1.0})
        with pytest.raises(ValueError, match="no per-atom labels"):
            residue_potentials(grid, unlabelled)

    def test_probe_offset_moves_the_sample_away_from_the_atom(self):
        """Sampling at atom centres reports self-energy, not environment."""
        grid = grid_with_peaks({(10, 10, 10): 20.0})
        atom = PQRData(
            coords=np.array([[10.0, 10.0, 10.0]]),
            charges=np.array([1.0]),
            radii=np.array([1.0]),
            labels=("LYS 1 NZ",),
        )
        near = residue_potentials(grid, atom, probe_offset=0.0)[0].value
        far = residue_potentials(grid, atom, probe_offset=4.0)[0].value
        assert near > far, "moving the probe outward must leave the peak behind"


class TestSoluteMask:
    """The mask is an optimisation, so it needs an oracle and a scale guard.

    On hen lysozyme the naive form — every atom against the whole grid — took
    64 s, three times longer than the solve it was analysing. Restricting each
    atom to its own bounding box is ~50x faster in practice and must produce a
    bit-identical result.
    """

    @staticmethod
    def naive_mask(grid: PotentialGrid, structure: PQRData, margin: float) -> np.ndarray:
        """The obvious implementation, kept as an oracle."""
        axes = [
            grid.origin[axis] + np.arange(grid.values.shape[axis]) * grid.spacing[axis]
            for axis in range(3)
        ]
        xx, yy, zz = np.meshgrid(*axes, indexing="ij")
        mask = np.zeros(grid.values.shape, dtype=bool)
        for index in range(structure.n_atoms):
            cx, cy, cz = structure.coords[index]
            cutoff = structure.radii[index] + margin
            mask |= ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) <= cutoff**2
        return mask

    @pytest.mark.parametrize("margin", [0.0, 1.4, 6.0])
    @pytest.mark.parametrize("seed", range(4))
    def test_matches_the_naive_implementation(self, margin, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 25))
        grid = PotentialGrid(
            values=np.zeros((17, 19, 23)),
            origin=rng.uniform(-5, 5, 3),
            spacing=rng.uniform(0.3, 1.2, 3),
        )
        structure = PQRData(
            coords=rng.uniform(-8, 8, (n, 3)),
            charges=rng.normal(size=n),
            radii=rng.uniform(0.5, 3.0, n),
        )
        np.testing.assert_array_equal(
            _solute_mask(grid, structure, margin),
            self.naive_mask(grid, structure, margin),
        )

    def test_atoms_entirely_off_the_grid_are_skipped_not_wrapped(self):
        """A negative index slice would silently mask the wrong corner."""
        grid = grid_with_peaks({(10, 10, 10): 1.0})
        far = PQRData(
            coords=np.array([[-500.0, -500.0, -500.0]]),
            charges=np.array([1.0]),
            radii=np.array([2.0]),
        )
        assert not _solute_mask(grid, far, 1.4).any()

    def test_scales_to_a_real_protein(self):
        """Catches a return to O(atoms x grid points).

        Sized like hen lysozyme: ~2,000 atoms on a 129x161x129 grid. The naive
        form takes ~64 s; this budget is 8x the optimised time, so it is loose
        against a noisy CI runner and still an order of magnitude under the
        behaviour it exists to prevent.
        """
        rng = np.random.default_rng(1)
        n_atoms = 2000
        grid = PotentialGrid(
            values=np.zeros((129, 161, 129)),
            origin=np.zeros(3),
            spacing=np.full(3, 0.475),
        )
        structure = PQRData(
            coords=rng.uniform(5, 55, (n_atoms, 3)),
            charges=rng.normal(size=n_atoms),
            radii=np.full(n_atoms, 1.9),
        )
        started = time.monotonic()
        mask = _solute_mask(grid, structure, 1.4)
        elapsed = time.monotonic() - started
        assert mask.any(), "a protein-sized structure must mask something"
        assert elapsed < 10.0, (
            f"masking took {elapsed:.1f}s for {n_atoms} atoms; the per-atom bounding "
            "box has probably been lost and this is O(atoms x grid points) again"
        )


class TestResidueGrouping:
    """`"<resName> <resSeq>"` is not a residue identifier, and real files prove it."""

    @staticmethod
    def albumin() -> PQRData:
        return read_pqr(Path(__file__).parent / "data" / "1ao6.pqr.gz")

    def test_a_two_chain_protein_is_not_collapsed_to_one(self):
        """Serum albumin is a dimer: 1,156 residues, not 578.

        The file carries no chain column — pdb2pqr dropped it — so this is the
        case a chain-aware fix alone would not have touched.
        """
        structure = self.albumin()
        assert structure.chains == (), "the fixture's premise: no chain column"

        merged = {_residue_of(label) for label in structure.labels}
        assert len(merged) == 578, "the defect, kept as the thing being fixed"
        assert len(_residue_groups(structure)) == 1156

    def test_no_residue_spans_more_than_a_residue(self):
        """The observable behind the count: 22 atoms over 115 A reported as one."""
        structure = self.albumin()
        worst = 0.0
        for group in _residue_groups(structure):
            coords = structure.coords[group.indices]
            if len(coords) < 2:
                continue
            spread = float(np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1).max())
            worst = max(worst, spread)
        assert worst < 15.0, f"a group spans {worst:.1f} A, so it is not one residue"

    def test_every_residue_gets_its_own_name(self):
        structure = self.albumin()
        labels = _labelled(_residue_groups(structure))
        assert len(set(labels)) == len(labels) == 1156

    def test_a_synthesised_segment_is_not_dressed_up_as_a_chain(self):
        """The file names no chains, so neither may we — `#2` is visibly an inference."""
        groups = _residue_groups(self.albumin())
        labels = _labelled(groups)
        assert all(group.chain is None for group in groups)
        assert {label.split(":")[0] for label in labels} == {"#1", "#2"}

    def test_a_single_chain_structure_keeps_the_labels_it_had(self):
        """Nothing to disambiguate, so nothing changes — recordings depend on this."""
        structure = read_pqr(Path(__file__).parent / "data" / "apbs-examples" / "fas2.pqr")
        labels = _labelled(_residue_groups(structure))
        assert labels[:3] == ["NTE 544", "THR 544", "MET 545"]
        assert all(":" not in label for label in labels)

    def test_a_named_chain_is_reported_as_read(self):
        structure = read_pqr(Path(__file__).parent / "data" / "apbs-examples" / "barnase.pqr")
        groups = _residue_groups(structure)
        assert {group.chain for group in groups} == {"A", "B"}
        assert _labelled(groups)[0] == "B:ALA 1"

    def test_naming_the_only_chain_disambiguates_nothing(self):
        """barstar is all chain D. Prefixing every label with it is pure churn."""
        structure = read_pqr(Path(__file__).parent / "data" / "apbs-examples" / "barstar.pqr")
        groups = _residue_groups(structure)
        assert {group.chain for group in groups} == {"D"}
        assert _labelled(groups)[0] == "LYS 1"

    def test_interleaved_chains_cannot_share_a_label(self):
        """A...B...A with reused numbering: the last resort suffix, exercised."""
        interleaved = PQRData(
            coords=np.array([[0.0, 0, 0], [20.0, 0, 0], [40.0, 0, 0]]),
            charges=np.zeros(3),
            radii=np.full(3, 1.5),
            labels=("ALA 1 N", "ALA 1 N", "ALA 1 N"),
            chains=("A", "B", "A"),
        )
        labels = _labelled(_residue_groups(interleaved))
        assert len(set(labels)) == 3, labels
        assert labels == ["A:ALA 1~1", "B:ALA 1", "A:ALA 1~2"]


class TestResiduePotentialsAcrossChains:
    @staticmethod
    def two_chains() -> PQRData:
        """Two copies of residues 1-2, 11 A apart, with no chain column."""
        return PQRData(
            coords=np.array(
                [
                    [4.0, 10.0, 10.0],
                    [5.0, 10.0, 10.0],
                    [6.0, 10.0, 10.0],
                    [15.0, 10.0, 10.0],
                    [16.0, 10.0, 10.0],
                    [17.0, 10.0, 10.0],
                ]
            ),
            charges=np.array([0.1, -0.1, 0.2, 0.1, -0.1, 0.2]),
            radii=np.full(6, 1.5),
            labels=("ALA 1 N", "ALA 1 CA", "GLY 2 N", "ALA 1 N", "ALA 1 CA", "GLY 2 N"),
        )

    def test_the_same_numbers_in_two_chains_are_four_residues(self):
        grid = grid_with_peaks({(4, 10, 10): -6.0, (16, 10, 10): 6.0})
        results = residue_potentials(grid, self.two_chains())
        assert len(results) == 4
        assert len({r.label for r in results}) == 4
        assert {r.segment for r in results} == {1, 2}

    def test_the_chain_reaches_the_caller_when_the_file_has_one(self):
        structure = self.two_chains()
        named = PQRData(
            coords=structure.coords,
            charges=structure.charges,
            radii=structure.radii,
            labels=structure.labels,
            chains=("A", "A", "A", "B", "B", "B"),
        )
        grid = grid_with_peaks({(4, 10, 10): -6.0, (16, 10, 10): 6.0})
        results = residue_potentials(grid, named)
        assert {r.chain for r in results} == {"A", "B"}
        assert {r.as_dict()["residue"] for r in results} == {
            "A:ALA 1",
            "A:GLY 2",
            "B:ALA 1",
            "B:GLY 2",
        }

    def test_two_single_residue_chains_are_indistinguishable_without_a_chain_column(self):
        """The limit of the fix, asserted so it is a known boundary and not a surprise.

        One residue per chain, adjacent in the file, identically numbered: the
        run boundary coincides with nothing. Only the coordinates say there are
        two, and grouping on distance would invent a threshold the file does not
        justify. `--keep-chain` is what actually closes this case.
        """
        ambiguous = PQRData(
            coords=np.array([[4.0, 10.0, 10.0], [16.0, 10.0, 10.0]]),
            charges=np.array([0.1, 0.1]),
            radii=np.full(2, 1.5),
            labels=("ALA 1 N", "ALA 1 N"),
        )
        grid = grid_with_peaks({(4, 10, 10): -6.0, (16, 10, 10): 6.0})
        assert len(residue_potentials(grid, ambiguous)) == 1

        named = PQRData(
            coords=ambiguous.coords,
            charges=ambiguous.charges,
            radii=ambiguous.radii,
            labels=ambiguous.labels,
            chains=("A", "B"),
        )
        assert len(residue_potentials(grid, named)) == 2
