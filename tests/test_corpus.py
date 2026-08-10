"""The golden corpus, and proof that it catches what it exists to catch.

ROADMAP.md phase 3's exit criterion is that a deliberate unit-conversion bug is
caught by `corpus verify`. That is what `TestCatchesUnitBugs` demonstrates: a
solver wrapper that scales energies, potentials or lengths by a plausible wrong
factor, and a verification pass that names the field that moved.

This is also the guard on the system APBS, which no lockfile pins any more.
"""

from dataclasses import dataclass, replace

import numpy as np
import pytest

from sashimi.apbs import ApbsSolver, discover_apbs
from sashimi.cli import main
from sashimi.corpus import (
    MANIFEST,
    Tolerances,
    build_case,
    load_summary,
    verify_case,
    verify_manifest,
    write_summary,
)
from sashimi.protocol import GridSpec, PQRData, SolventModel, Solver, SolveResult
from tests.born_reference import born_solvation_energy

pytestmark = pytest.mark.apbs

# CLAUDE.md freezes the backend here. conda-forge, brew and apt all ship it.
EXPECTED_APBS_VERSION = "3.4.1"

# Cheapest case in the manifest; the fault-injection tests re-solve repeatedly.
CHEAP = next(c for c in MANIFEST if c.name == "born-ion-coarse")

# Plausible wrong factors, not arbitrary ones.
KJ_PER_KCAL = 4.184
MV_PER_KT_E = 25.693  # kT/e at 298.15 K, in millivolts
ANGSTROM_PER_NM = 10.0


@dataclass
class Perturbed:
    """A `Solver` that corrupts its inner solver's output in one specific way.

    Stands in for the bug class this corpus exists to catch: arithmetic that
    looks right and produces a plausible grid with the wrong units.
    """

    inner: Solver
    energy_factor: float = 1.0
    potential_factor: float = 1.0
    spacing_factor: float = 1.0

    def solve_lpbe(
        self,
        pqr: PQRData,
        grid: GridSpec,
        solvent: SolventModel = SolventModel(),  # noqa: B008 — frozen dataclass
        *,
        compute_energy: bool = False,
    ) -> SolveResult:
        result = self.inner.solve_lpbe(pqr, grid, solvent, compute_energy=compute_energy)
        if result.energy_kj_mol is not None:
            result.energy_kj_mol *= self.energy_factor
        result.potential.values = result.potential.values * self.potential_factor
        result.potential.spacing = result.potential.spacing * self.spacing_factor
        return result


@pytest.fixture(scope="module")
def solver():
    return ApbsSolver()


@pytest.fixture(scope="module")
def recorded():
    return load_summary(CHEAP)


class TestBackend:
    def test_version_is_the_pinned_one(self):
        """A drifted APBS must fail here, not silently change every result."""
        binary = discover_apbs()
        assert binary.version == EXPECTED_APBS_VERSION, (
            f"expected APBS {EXPECTED_APBS_VERSION}, found {binary.version} at {binary.path}. "
            "Pin the package back, or rebuild the corpus deliberately and update this constant."
        )

    def test_recorded_summaries_name_the_expected_backend(self):
        for case in MANIFEST:
            assert load_summary(case)["backend"] == f"apbs-{EXPECTED_APBS_VERSION}"


class TestReproduces:
    def test_the_whole_manifest_verifies_clean(self, solver):
        assert verify_manifest(solver) == []

    def test_born_ion_converges_toward_the_closed_form(self):
        """The two Born cases exist as a pair; refining must reduce the error."""
        exact = born_solvation_energy(3.0, solute_dielectric=1.0)
        coarse = abs(load_summary(MANIFEST[0])["energy_kj_mol"] - exact)
        fine = abs(load_summary(MANIFEST[1])["energy_kj_mol"] - exact)
        assert fine < coarse, f"error grew when refining: {coarse:.4f} -> {fine:.4f} kJ/mol"

    def test_probe_placement_is_deterministic(self, solver):
        """Probes must land on the same coordinates on every build, or the
        recorded values mean nothing."""
        a = build_case(solver, CHEAP)["probes"]["points"]
        b = build_case(solver, CHEAP)["probes"]["points"]
        np.testing.assert_array_equal(np.array(a), np.array(b))
        np.testing.assert_allclose(np.array(a), np.array(load_summary(CHEAP)["probes"]["points"]))


class TestCatchesUnitBugs:
    """Phase 3's exit criterion."""

    def test_catches_kj_mistaken_for_kcal(self, solver, recorded):
        faulty = Perturbed(solver, energy_factor=1 / KJ_PER_KCAL)
        found = verify_case(faulty, CHEAP, recorded)
        assert any(d.field == "energy_kj_mol" for d in found), (
            "a 4.184x energy error must be caught"
        )

    def test_catches_potential_reported_in_millivolts(self, solver, recorded):
        faulty = Perturbed(solver, potential_factor=MV_PER_KT_E)
        found = verify_case(faulty, CHEAP, recorded)
        fields = {d.field for d in found}
        assert "probes.values_kT_e" in fields
        assert any(f.startswith("potential_stats.") for f in fields)

    def test_catches_angstroms_mistaken_for_nanometres(self, solver, recorded):
        """A length-unit bug shifts the grid, so probes sample the wrong place."""
        faulty = Perturbed(solver, spacing_factor=1 / ANGSTROM_PER_NM)
        found = verify_case(faulty, CHEAP, recorded)
        assert any(d.field == "geometry.spacing" for d in found)

    def test_catches_a_sign_flip(self, solver, recorded):
        faulty = Perturbed(solver, energy_factor=-1.0)
        found = verify_case(faulty, CHEAP, recorded)
        assert any(d.field == "energy_kj_mol" for d in found)

    def test_a_discrepancy_reads_usefully(self, solver, recorded):
        faulty = Perturbed(solver, energy_factor=1 / KJ_PER_KCAL)
        message = str(next(d for d in verify_case(faulty, CHEAP, recorded)))
        assert CHEAP.name in message
        assert "energy_kj_mol" in message
        assert "%" in message, "an energy discrepancy should state how far off it is"


class TestTolerances:
    def test_drift_below_tolerance_passes(self, solver, recorded):
        """Otherwise the corpus fails on ordinary cross-platform float noise."""
        tolerances = Tolerances()
        faulty = Perturbed(solver, energy_factor=1 + tolerances.energy_rtol / 10)
        assert not [d for d in verify_case(faulty, CHEAP, recorded) if d.field == "energy_kj_mol"]

    def test_drift_above_tolerance_fails(self, solver, recorded):
        tolerances = Tolerances()
        faulty = Perturbed(solver, energy_factor=1 + tolerances.energy_rtol * 100)
        assert [d for d in verify_case(faulty, CHEAP, recorded) if d.field == "energy_kj_mol"]


class TestCli:
    def test_build_then_verify_round_trips(self, tmp_path, capsys):
        args = ["corpus", "build", "--case", CHEAP.name, "--directory", str(tmp_path), "--force"]
        assert main(args) == 0
        assert (tmp_path / f"{CHEAP.name}.json").exists()

        assert main(["corpus", "verify", "--case", CHEAP.name, "--directory", str(tmp_path)]) == 0
        assert "reproduce" in capsys.readouterr().out

    def test_verify_exits_nonzero_on_a_missing_summary(self, tmp_path, capsys):
        assert main(["corpus", "verify", "--case", CHEAP.name, "--directory", str(tmp_path)]) == 1
        assert "MISS" in capsys.readouterr().out

    def test_verify_exits_nonzero_when_the_numbers_moved(self, tmp_path, solver, capsys):
        corrupted = build_case(solver, CHEAP)
        corrupted["energy_kj_mol"] /= KJ_PER_KCAL
        write_summary(corrupted, tmp_path / f"{CHEAP.name}.json")

        assert main(["corpus", "verify", "--case", CHEAP.name, "--directory", str(tmp_path)]) == 1
        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "investigate before rebuilding" in out

    def test_build_refuses_to_overwrite_without_force(self, tmp_path, capsys):
        args = ["corpus", "build", "--case", CHEAP.name, "--directory", str(tmp_path)]
        assert main(args) == 0
        assert main(args) == 0
        assert "skip" in capsys.readouterr().out

    def test_unknown_case_is_rejected(self, tmp_path):
        with pytest.raises(SystemExit, match="unknown case"):
            main(["corpus", "verify", "--case", "no-such-case", "--directory", str(tmp_path)])

    def test_tolerances_are_overridable_from_the_command_line(self, tmp_path, solver):
        """An operator investigating drift needs to widen the gate temporarily."""
        loosened = build_case(solver, CHEAP)
        loosened["energy_kj_mol"] *= 1.001
        write_summary(loosened, tmp_path / f"{CHEAP.name}.json")

        base = ["corpus", "verify", "--case", CHEAP.name, "--directory", str(tmp_path)]
        assert main(base) == 1
        assert main([*base, "--energy-rtol", "0.01"]) == 0


def test_every_manifest_case_has_a_recorded_summary():
    """A case added without a summary would silently never be verified."""
    for case in MANIFEST:
        summary = load_summary(case)
        assert summary["name"] == case.name
        assert len(summary["probes"]["values_kT_e"]) == len(summary["probes"]["points"])


def test_manifest_names_are_unique():
    names = [case.name for case in MANIFEST]
    assert len(names) == len(set(names))


def test_cases_are_frozen():
    """Mutating a case mid-run would decouple the summary from what was solved."""
    with pytest.raises(Exception, match=r"frozen|immutable|cannot assign"):
        replace(CHEAP, name="x").name = "y"  # type: ignore[misc]
