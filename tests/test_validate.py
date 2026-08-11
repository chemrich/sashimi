"""Cross-solver validation, binary-free.

The comparison engine is pure arithmetic over `SolveResult`s, so stub solvers
exercise all of it — including the paths that matter most, which are the ones
that refuse to answer. A real two-backend run is covered by
`tests/test_cross_validation.py`; what is tested here is the judgement, not the
physics.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sashimi.protocol import (
    EnergyTerm,
    Equation,
    FiniteDifferenceRequest,
    PotentialGrid,
    PQRData,
    Provenance,
    SolventModel,
    SolveResult,
    SurfaceModel,
)
from sashimi.validate import (
    BackendRun,
    Incomparable,
    compare_results,
    overlap_probe_points,
    validate,
    with_surface_model,
)


def ion() -> PQRData:
    return PQRData(
        coords=np.zeros((1, 3)),
        charges=np.array([1.0]),
        radii=np.array([3.0]),
    )


def grid(origin=(0.0, 0.0, 0.0), spacing=0.5, n=9, fill=1.0) -> PotentialGrid:
    return PotentialGrid(
        values=np.full((n, n, n), fill, dtype=float),
        origin=np.array(origin, dtype=float),
        spacing=np.full(3, spacing, dtype=float),
    )


class StubSolver:
    """A solver that returns whatever it was constructed with."""

    def __init__(self, energy, term=EnergyTerm.POLAR_SOLVATION, potential=None):
        self.energy = energy
        self.term = term
        self.potential = potential

    def solve(self, request: FiniteDifferenceRequest) -> SolveResult:  # noqa: ARG002 — the protocol's signature
        return SolveResult(
            provenance=Provenance(backend="stub", energy_term=self.term),
            energy_kj_mol=self.energy,
            potential=self.potential,
        )


def run(  # noqa: PLR0917 — a builder for a 7-field record
    name="a",
    energy=-100.0,
    term=EnergyTerm.POLAR_SOLVATION,
    surface=SurfaceModel.MOLECULAR,
    equation=Equation.LINEAR,
    potential=None,
    ionic_strength=0.0,
) -> BackendRun:
    return BackendRun(
        name=name,
        energy_kj_mol=energy,
        energy_term=term,
        surface_model=surface,
        equation=equation,
        potential=potential,
        ionic_strength=ionic_strength,
    )


def request(**kwargs) -> FiniteDifferenceRequest:
    solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR, **kwargs.pop("solvent", {}))
    return FiniteDifferenceRequest(structure=ion(), solvent=solvent, **kwargs)


# --- agreement ---------------------------------------------------------------


def test_close_energies_agree():
    comparison = compare_results([run("a", -100.0), run("b", -102.0)])

    assert comparison.agrees
    assert comparison.energy_spread == pytest.approx(2.0 / 102.0)
    assert comparison.energy_range_kj_mol == (-102.0, -100.0)


def test_distant_energies_disagree():
    comparison = compare_results([run("a", -100.0), run("b", -160.0)])

    assert not comparison.agrees
    assert comparison.energy_spread == pytest.approx(60.0 / 160.0)


def test_the_spread_is_relative_to_the_largest_magnitude():
    """Otherwise a near-zero member makes every spread look enormous."""
    comparison = compare_results([run("a", -0.001), run("b", -100.0)])
    assert comparison.energy_spread == pytest.approx(99.999 / 100.0)


def test_one_backend_is_not_a_comparison():
    with pytest.raises(Incomparable, match="at least two"):
        compare_results([run("a")])


# --- refusals ----------------------------------------------------------------


def test_mismatched_surface_models_are_refused():
    """The 25.7% confounder of ROADMAP.md section 5, guarded."""
    with pytest.raises(Incomparable, match="surface models differ"):
        compare_results(
            [
                run("a", surface=SurfaceModel.MOLECULAR),
                run("b", surface=SurfaceModel.VAN_DER_WAALS),
            ]
        )


def test_mismatched_equations_are_refused():
    with pytest.raises(Incomparable, match="equations differ"):
        compare_results([run("a", equation=Equation.LINEAR), run("b", equation=Equation.NONLINEAR)])


def test_an_unstated_energy_term_is_refused():
    """Silence is not agreement: a backend must say what it reports."""
    with pytest.raises(Incomparable, match="did not state"):
        compare_results([run("a"), run("b", term=None)])


def test_mismatched_energy_terms_are_refused_when_salt_is_present():
    with pytest.raises(Incomparable, match="mobile-ion contribution"):
        compare_results(
            [
                run("a", term=EnergyTerm.POLAR_SOLVATION, ionic_strength=0.15),
                run("b", term=EnergyTerm.REACTION_FIELD, ionic_strength=0.15),
            ]
        )


def test_mismatched_energy_terms_are_allowed_at_zero_salt():
    """With no mobile ions there is no osmotic term, so the two coincide.

    Refusing here would make the tool useless on exactly the cases where it is
    most trustworthy — which is what it did before this rule was refined from
    "same term" to "same term, or terms that provably coincide".
    """
    comparison = compare_results(
        [
            run("a", -100.0, term=EnergyTerm.POLAR_SOLVATION, ionic_strength=0.0),
            run("b", -101.0, term=EnergyTerm.REACTION_FIELD, ionic_strength=0.0),
        ]
    )

    assert comparison.agrees
    assert any("coincide at zero ionic strength" in note for note in comparison.notes)


def test_override_downgrades_a_refusal_to_a_note():
    comparison = compare_results(
        [
            run("a", surface=SurfaceModel.MOLECULAR),
            run("b", surface=SurfaceModel.VAN_DER_WAALS),
        ],
        allow_mismatch=True,
    )

    assert comparison.energy_spread is not None
    assert any(note.startswith("OVERRIDDEN") for note in comparison.notes)


# --- potentials across incompatible grids ------------------------------------


def test_probe_points_fall_inside_every_grid():
    """The whole point: grids that share no shape can still share coordinates."""
    a = grid(origin=(0.0, 0.0, 0.0), spacing=0.5, n=21)  # 0 -> 10 A
    b = grid(origin=(2.0, 2.0, 2.0), spacing=0.3, n=21)  # 2 -> 8 A

    points = overlap_probe_points([a, b])

    assert len(points) > 0
    assert np.all(points >= 2.0)
    assert np.all(points <= 8.0)
    assert not np.any(np.isnan(a.value_at(points)))
    assert not np.any(np.isnan(b.value_at(points)))


def test_probe_points_are_deterministic():
    grids = [grid(), grid(spacing=0.25, n=17)]
    assert np.array_equal(overlap_probe_points(grids), overlap_probe_points(grids))


def test_disjoint_grids_yield_no_points():
    a = grid(origin=(0.0, 0.0, 0.0), n=5)
    b = grid(origin=(100.0, 100.0, 100.0), n=5)
    assert len(overlap_probe_points([a, b])) == 0


def test_potentials_are_compared_on_differing_grids():
    """Different shapes and spacings, same physical field: no difference found."""
    a = grid(origin=(0.0, 0.0, 0.0), spacing=0.5, n=21, fill=3.0)
    b = grid(origin=(0.0, 0.0, 0.0), spacing=0.25, n=41, fill=3.0)
    assert a.shape != b.shape

    comparison = compare_results([run("a", potential=a), run("b", potential=b)])

    assert comparison.n_probes > 0
    assert comparison.potential_rmsd_kt_e == pytest.approx(0.0, abs=1e-12)


def test_a_constant_offset_shows_up_in_the_potential_rmsd():
    a = grid(fill=1.0)
    b = grid(fill=1.5, spacing=0.25, n=17)

    comparison = compare_results([run("a", potential=a), run("b", potential=b)])

    assert comparison.potential_rmsd_kt_e == pytest.approx(0.5, abs=1e-9)
    assert comparison.potential_max_abs_kt_e == pytest.approx(0.5, abs=1e-9)


def test_disjoint_grids_are_noted_not_crashed():
    a = grid(origin=(0.0, 0.0, 0.0), n=5)
    b = grid(origin=(100.0, 100.0, 100.0), n=5)

    comparison = compare_results([run("a", potential=a), run("b", potential=b)])

    assert comparison.potential_rmsd_kt_e is None
    assert any("do not overlap" in note for note in comparison.notes)


# --- the top-level entry point -----------------------------------------------


def test_validate_solves_once_per_backend_and_compares():
    solvers = {
        "x": StubSolver(-100.0, potential=grid(fill=1.0)),
        "y": StubSolver(-101.0, potential=grid(fill=1.0)),
    }
    comparison = validate(solvers, request())

    assert [r.name for r in comparison.runs] == ["x", "y"]
    assert comparison.agrees
    assert comparison.n_probes > 0


def test_validate_carries_the_requests_ionic_strength_into_the_check():
    """The salt lives on the request, not the result, so it must be read there."""
    solvers = {
        "x": StubSolver(-100.0, term=EnergyTerm.POLAR_SOLVATION),
        "y": StubSolver(-101.0, term=EnergyTerm.REACTION_FIELD),
    }
    salted = request(solvent={"ionic_strength": 0.15})

    with pytest.raises(Incomparable, match=r"0\.15 M"):
        validate(solvers, salted)


def test_with_surface_model_changes_only_the_surface():
    original = request()
    changed = with_surface_model(original, SurfaceModel.VAN_DER_WAALS)

    assert changed.solvent.surface_model is SurfaceModel.VAN_DER_WAALS
    assert changed.structure is original.structure
    assert (
        dataclasses.replace(changed.solvent, surface_model=original.solvent.surface_model)
        == original.solvent
    )


def test_missing_energies_are_reported_rather_than_guessed():
    comparison = compare_results([run("a", energy=None), run("b", energy=-100.0)])

    assert comparison.energy_spread is None
    assert not comparison.agrees
    assert any("fewer than two" in note for note in comparison.notes)
