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
    AccuracyTier,
    BoundaryElementRequest,
    EnergyTerm,
    Equation,
    FiniteDifferenceRequest,
    GridSpec,
    PotentialGrid,
    PQRData,
    Provenance,
    SolventModel,
    SolveResult,
    SurfaceModel,
)
from sashimi.validate import (
    DEFAULT_APPROXIMATION_TOLERANCE,
    Backend,
    BackendRun,
    Incomparable,
    SolverFamily,
    System,
    compare_grids,
    compare_results,
    overlap_probe_points,
    validate,
    validate_system,
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


def run(  # noqa: PLR0917 — a builder for an 8-field record
    name="a",
    energy=-100.0,
    term=EnergyTerm.POLAR_SOLVATION,
    surface=SurfaceModel.MOLECULAR,
    equation=Equation.LINEAR,
    potential=None,
    ionic_strength=0.0,
    tier=AccuracyTier.REFERENCE,
) -> BackendRun:
    return BackendRun(
        name=name,
        energy_kj_mol=energy,
        energy_term=term,
        surface_model=surface,
        equation=equation,
        potential=potential,
        ionic_strength=ionic_strength,
        accuracy_tier=tier,
    )


def approximation(name="gb", energy=-120.0, **kwargs) -> BackendRun:
    return run(name, energy, tier=AccuracyTier.APPROXIMATE, **kwargs)


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


# --- accuracy tiers ----------------------------------------------------------
#
# An approximation is expected to disagree. The whole point of the partition is
# that its disagreement is reported without either being excused or being
# allowed to contaminate what the reference solvers said about each other.


def test_an_approximation_does_not_widen_the_reference_spread():
    """The number that would otherwise be destroyed by averaging."""
    reference_only = compare_results([run("a", -100.0), run("b", -102.0)])
    with_gb = compare_results([run("a", -100.0), run("b", -102.0), approximation("gb", -125.0)])

    assert with_gb.energy_spread == reference_only.energy_spread
    assert with_gb.energy_range_kj_mol == (-102.0, -100.0)
    assert with_gb.tolerance == reference_only.tolerance


def test_an_approximation_is_measured_against_the_reference_consensus():
    comparison = compare_results([run("a", -100.0), run("b", -102.0), approximation("gb", -111.1)])

    # Consensus is the mean of the reference energies, -101.0.
    assert comparison.approximation_deviation == {"gb": pytest.approx(10.1 / 101.0)}
    assert comparison.approximations_agree
    assert comparison.agrees


def test_an_approximation_beyond_its_own_tolerance_disagrees():
    """The 31%-wrong-surface and 35%-wrong-radii mistakes, in miniature."""
    comparison = compare_results([run("a", -100.0), run("b", -100.0), approximation("gb", -200.0)])

    assert comparison.approximation_deviation["gb"] == pytest.approx(1.0)
    assert not comparison.approximations_agree
    assert not comparison.agrees
    # ...but the reference solvers still agreed, and that is still visible.
    assert comparison.energy_spread == pytest.approx(0.0)


def test_a_reference_disagreement_is_not_excused_by_a_well_behaved_approximation():
    comparison = compare_results([run("a", -100.0), run("b", -160.0), approximation("gb", -130.0)])

    assert not comparison.agrees
    assert comparison.approximations_agree


def test_a_lone_reference_still_calibrates_an_approximation():
    comparison = compare_results([run("a", -100.0), approximation("gb", -110.0)])

    assert comparison.energy_spread is None  # one backend has nothing to spread against
    assert comparison.approximation_deviation == {"gb": pytest.approx(0.1)}
    assert comparison.agrees
    assert any("no reference spread" in note for note in comparison.notes)


def test_approximations_alone_are_compared_but_not_called_accurate():
    comparison = compare_results([approximation("gb", -100.0), approximation("gb2", -110.0)])

    assert comparison.energy_spread == pytest.approx(10.0 / 110.0)
    assert comparison.tolerance == DEFAULT_APPROXIMATION_TOLERANCE
    assert comparison.agrees
    assert any("establishes accuracy" in note for note in comparison.notes)


def test_an_unstated_tier_is_a_reference_tier():
    """Every backend predating the field discretizes the equation."""
    assert (
        BackendRun(
            name="a",
            energy_kj_mol=-100.0,
            energy_term=EnergyTerm.POLAR_SOLVATION,
            surface_model=SurfaceModel.MOLECULAR,
            equation=Equation.LINEAR,
            potential=None,
        ).accuracy_tier
        is AccuracyTier.REFERENCE
    )
    assert Provenance(backend="stub").accuracy_tier is AccuracyTier.REFERENCE


def test_an_approximation_is_still_held_to_the_comparability_checks():
    """Less accurate is allowed; answering a different question is not."""
    with pytest.raises(Incomparable, match="surface models differ"):
        compare_results(
            [
                run("a", surface=SurfaceModel.MOLECULAR),
                approximation("gb", surface=SurfaceModel.VAN_DER_WAALS),
            ]
        )


def test_the_summary_names_the_approximation_separately():
    comparison = compare_results([run("a", -100.0), run("b", -102.0), approximation("gb", -111.1)])
    summary = comparison.summary()

    assert "2 backends agree" in summary  # the reference pair, not all three
    assert "gb 10.00%" in summary


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


def test_compare_grids_pairs_every_node_when_the_lattice_is_shared():
    a, b = grid(fill=1.0), grid(fill=1.25)
    out = compare_grids(a, b)

    assert out["method"] == "lattice"
    assert out["n_points"] == 9**3
    assert out["mean_diff_kT_e"] == pytest.approx(-0.25)
    assert out["rmsd_kT_e"] == pytest.approx(0.25)
    assert "note" not in out


def test_compare_grids_samples_the_shared_region_when_the_lattice_is_not():
    """The solver-versus-solver case: two boxes no caller chose to align."""
    a = grid(origin=(0.0, 0.0, 0.0), spacing=0.5, n=21, fill=2.0)
    b = grid(origin=(2.0, 2.0, 2.0), spacing=0.3, n=21, fill=2.0)

    out = compare_grids(a, b)

    assert out["method"] == "sampled"
    assert out["n_points"] > 0
    assert out["n_points"] not in (a.values.size, b.values.size)
    # Same constant field sampled two ways: interpolation must not invent a difference.
    assert out["rmsd_kT_e"] == pytest.approx(0.0, abs=1e-9)
    assert "differ in geometry" in out["note"]
    # Trilinear sampling of a constant field gives std ~1e-16, not 0. An
    # exact-zero guard would hand corrcoef rounding noise and report r = 0.023
    # beside an RMSD of 2e-16 — a wrong number, stated confidently.
    assert out["correlation"] is None
    # A maximum over samples is a lower bound, so it must not wear the exact key.
    assert "max_abs_diff_kT_e" not in out
    assert out["max_abs_diff_over_samples_kT_e"] == pytest.approx(0.0, abs=1e-9)


def test_a_real_field_still_correlates_on_the_sampled_path():
    """The relative guard must not suppress a correlation that genuinely exists."""
    n = 21
    ramp = np.linspace(-1.0, 1.0, n)
    values = np.repeat(np.repeat(ramp[:, None, None], n, 1), n, 2)
    a = PotentialGrid(values=values, origin=np.zeros(3), spacing=np.full(3, 0.5))
    b = PotentialGrid(values=values * 2.0, origin=np.full(3, 0.2), spacing=np.full(3, 0.45))
    out = compare_grids(a, b)
    assert out["method"] == "sampled"
    assert out["correlation"] is not None
    assert out["correlation"] == pytest.approx(1.0, abs=1e-6)


def test_compare_grids_refuses_when_there_is_no_shared_volume():
    a = grid(origin=(0.0, 0.0, 0.0), n=5)
    b = grid(origin=(100.0, 100.0, 100.0), n=5)
    with pytest.raises(Incomparable, match="no common region"):
        compare_grids(a, b)


def test_a_sampled_comparison_is_never_mistaken_for_an_exact_one():
    """`method` is the whole guard: the two RMSDs are not the same quantity."""
    shared = compare_grids(grid(fill=1.0), grid(fill=1.5))
    crossed = compare_grids(
        grid(origin=(0.0, 0.0, 0.0), spacing=0.5, n=21, fill=1.0),
        grid(origin=(1.0, 1.0, 1.0), spacing=0.3, n=21, fill=1.5),
    )
    assert shared["method"] != crossed["method"]
    assert shared["n_points"] != crossed["n_points"]
    assert ("note" in crossed) and ("note" not in shared)
    # The exact maximum exists on one path only, so a caller cannot read an
    # estimate as though it were exact — it is a KeyError, not a smaller number.
    assert "max_abs_diff_kT_e" in shared
    assert "max_abs_diff_kT_e" not in crossed


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


# --- cross-family comparison -------------------------------------------------


def test_a_system_produces_either_familys_request():
    """The seam `SolveRequest` was designed for, used for the first time.

    One physical question, two dialects: the finite-difference request carries a
    grid, the boundary-element one a mesh density, and neither can read the
    other's.
    """
    system = System(structure=ion(), grid=GridSpec(resolution=0.4), mesh_density=3.0)

    fd = system.request_for(SolverFamily.FINITE_DIFFERENCE)
    be = system.request_for(SolverFamily.BOUNDARY_ELEMENT)

    assert isinstance(fd, FiniteDifferenceRequest)
    assert isinstance(be, BoundaryElementRequest)
    assert fd.grid.resolution == pytest.approx(0.4)
    assert be.mesh_density == pytest.approx(3.0)
    # The physics is shared, which is what makes the comparison legitimate.
    assert fd.structure is be.structure
    assert fd.solvent == be.solvent


def test_validate_system_compares_across_families():
    """A grid solver and a surface solver, one question, one spread."""
    from sashimi.bem_stub import StubBemSolver  # noqa: PLC0415

    system = System(structure=ion(), solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR))
    comparison = validate_system(
        system,
        [
            Backend("grid", StubSolver(-100.0), SolverFamily.FINITE_DIFFERENCE),
            Backend("surface", StubSolver(-104.0), SolverFamily.BOUNDARY_ELEMENT),
        ],
    )

    assert comparison.agrees
    assert comparison.energy_spread == pytest.approx(4.0 / 104.0)
    assert [r.name for r in comparison.runs] == ["grid", "surface"]
    assert StubBemSolver is not None  # the shipped stub still satisfies the protocol


def test_a_boundary_element_request_is_linear_by_construction():
    """`equation` lives on the FD request only, so the comparison must not
    assume it is there — a nonlinear BEM request is unrepresentable, not
    rejected."""
    system = System(structure=ion())
    comparison = validate_system(
        system,
        [
            Backend("a", StubSolver(-100.0), SolverFamily.FINITE_DIFFERENCE),
            Backend("b", StubSolver(-100.0), SolverFamily.BOUNDARY_ELEMENT),
        ],
    )

    assert {r.equation for r in comparison.runs} == {Equation.LINEAR}


def test_cross_family_still_refuses_a_mismatched_energy_term():
    """Crossing families does not relax the rules; it only widens who can play."""
    system = System(structure=ion(), solvent=SolventModel(ionic_strength=0.15))
    with pytest.raises(Incomparable, match="mobile-ion contribution"):
        validate_system(
            system,
            [
                Backend("a", StubSolver(-100.0, term=EnergyTerm.POLAR_SOLVATION)),
                Backend(
                    "b",
                    StubSolver(-101.0, term=EnergyTerm.REACTION_FIELD),
                    SolverFamily.BOUNDARY_ELEMENT,
                ),
            ],
        )
