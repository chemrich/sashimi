"""Born ion against the closed form.

This is the test that catches unit-conversion bugs, which are the entire
failure mode of a wrapper project: every layer here has a unit convention, and
a wrong factor produces a plausible-looking grid.
"""

import numpy as np
import pytest

from born_reference import born_potential, born_solvation_energy
from sashimi.apbs import ApbsSolver
from sashimi.protocol import GridSpec, PQRData, SolventModel

pytestmark = pytest.mark.apbs

RADIUS = 3.0
TOLERANCE = 0.01  # 1%; see PLAN.md section 7 for why not 0.5%

# Matches the classic Born setup: vacuum reference, no mobile ions.
BORN_SOLVENT = SolventModel(
    solvent_dielectric=78.54,
    solute_dielectric=1.0,
    ionic_strength=0.0,
    surface_method="smol",
    surface_radius=1.4,
    temperature=298.15,
)


@pytest.fixture(scope="module")
def born_ion():
    return PQRData(
        coords=np.zeros((1, 3)),
        charges=np.array([1.0]),
        radii=np.array([RADIUS]),
        labels=("ION 1 I",),
    )


@pytest.fixture(scope="module")
def solved(born_ion):
    """Solve once per resolution; APBS runs are seconds, not milliseconds."""
    solver = ApbsSolver()
    return {
        res: solver.solve_lpbe(
            born_ion,
            GridSpec(resolution=res, padding=10.0),
            BORN_SOLVENT,
            compute_energy=True,
        )
        for res in (0.5, 0.25)
    }


def test_solvation_energy_matches_closed_form(solved):
    expected = born_solvation_energy(RADIUS, solute_dielectric=1.0)
    for res, result in solved.items():
        error = abs(result.energy_kj_mol - expected) / abs(expected)
        assert error < TOLERANCE, (
            f"resolution {res} A: got {result.energy_kj_mol:.4f} kJ/mol, "
            f"expected {expected:.4f} ({error:.2%} off)"
        )


def test_error_shrinks_with_spacing(solved):
    """Discretization error, not a systematic unit bug."""
    expected = born_solvation_energy(RADIUS, solute_dielectric=1.0)
    coarse = abs(solved[0.5].energy_kj_mol - expected)
    fine = abs(solved[0.25].energy_kj_mol - expected)
    assert fine < coarse, f"error grew when refining: {coarse:.4f} -> {fine:.4f} kJ/mol"


def test_potential_outside_the_ion_matches_closed_form(solved):
    """Probes stay at r >= 1.25a: at the dielectric boundary the smoothed
    surface diverges from the closed form by ~70%, and r = 0 is singular."""
    grid = solved[0.25].potential
    radii = np.array([1.25, 1.5, 1.75, 2.0]) * RADIUS
    points = np.zeros((len(radii), 3))
    points[:, 0] = radii

    got = grid.value_at(points)
    expected = np.array([born_potential(r) for r in radii])
    assert not np.isnan(got).any(), "probe points fell outside the grid"
    np.testing.assert_allclose(got, expected, rtol=0.03)


def test_potential_is_positive_and_decays(solved):
    grid = solved[0.25].potential
    radii = np.array([1.25, 1.5, 2.0, 2.5]) * RADIUS
    points = np.zeros((len(radii), 3))
    points[:, 0] = radii
    values = grid.value_at(points)
    assert np.all(values > 0), "a +1 ion must produce positive potential"
    assert np.all(np.diff(values) < 0), "potential must decay with distance"


def test_provenance_and_diagnostics_travel(solved):
    result = solved[0.5]
    assert result.backend.startswith("apbs-3.")
    assert result.diagnostics["dime"] == [65, 65, 65]
    assert result.diagnostics["n_points"] == 65**3
    assert result.diagnostics["resolution_relaxed"] is False
    assert "binary_path" in result.diagnostics
