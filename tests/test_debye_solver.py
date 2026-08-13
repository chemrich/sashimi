"""debye as a solver: the M1 milestone, and the contract it has to keep.

Nothing here is marked and nothing here can skip. That is the property debye
exists for — ROADMAP.md section 12 wants a field on a machine with no binary
installed — so a test suite that quietly skipped it on such a machine would be
testing the opposite of the claim. `sashimi.gb` established the pattern and
`tests/conftest.py` enforces it for everything that *is* marked.

**The gate tolerances are read from `sashimi.corpus`, not restated here.** A
number in two places is a number that drifts, and the specific failure it
avoids is the one ROADMAP.md section 7 records against `per_backend_rtol`: a
tolerance was added to the corpus and nothing re-solved the cases carrying it,
so every tight bound sat there inert. These tests are what make debye's entry
in that table do something before M5 wires it into `corpus verify`.
"""

from __future__ import annotations

import numpy as np
import pytest

from sashimi.analytic import born_potential, born_solvation_energy
from sashimi.corpus import MANIFEST, Case
from sashimi.debye import BACKEND_VERSION, DebyeOptions, DebyeSolver
from sashimi.debye.grid import size_grid
from sashimi.errors import ConvergenceFailure, UnsupportedRequest
from sashimi.protocol import (
    AccuracyTier,
    EnergyTerm,
    Equation,
    FiniteDifferenceRequest,
    GridSpec,
    PotentialGrid,
    PQRData,
    SolventModel,
    SolveResult,
    SurfaceModel,
)

VDW = SolventModel(
    solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.VAN_DER_WAALS
)

# The two van der Waals cases carrying a closed form. M1's gate is the fine one;
# the coarse one is its convergence partner.
GATE_CASES = ("born-ion-vdw", "born-ion-vdw-fine")


def case_named(name: str) -> Case:
    return next(case for case in MANIFEST if case.name == name)


def _energy(result: SolveResult) -> float:
    """The energy a request asked for, as a float rather than an optional one."""
    assert result.energy_kj_mol is not None
    return result.energy_kj_mol


def born_ion(radius: float = 3.0, charge: float = 1.0) -> PQRData:
    return PQRData(coords=np.zeros((1, 3)), charges=np.array([charge]), radii=np.array([radius]))


def born_request(
    resolution: float, radius: float = 3.0, **kwargs: object
) -> FiniteDifferenceRequest:
    return FiniteDifferenceRequest(
        structure=born_ion(radius),
        solvent=VDW,
        grid=GridSpec(resolution=resolution, padding=10.0),
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("name", GATE_CASES)
def test_the_born_ion_lands_inside_the_tolerance_the_corpus_sets_for_debye(name):
    """M1's exit criterion, graded by the corpus's own per-backend number.

    The shared tolerance on these cases is 5% and 1.6%, set by what APBS needs
    on a sharp boundary. Grading debye at those would be a check that cannot
    fail: a solver four percent from the closed form would pass a milestone
    whose stated criterion is one.
    """
    case = case_named(name)
    solver = DebyeSolver()
    reference = case.analytic
    assert reference is not None

    tolerance = reference.rtol_for(solver.label)
    assert tolerance < reference.rtol, (
        f"{name} grades debye at the shared tolerance {reference.rtol}, which is set by "
        "the least accurate backend that runs it. Give it a `debye_rtol`."
    )

    result = solver.solve(case.request())
    assert result.energy_kj_mol is not None
    error = abs(result.energy_kj_mol - reference.energy_kj_mol) / abs(reference.energy_kj_mol)
    assert error <= tolerance, (
        f"{name}: debye is {error:.4%} from the Born closed form, past the {tolerance:.2%} "
        "ROADMAP.md section 12 M1 holds it to."
    )


REFINEMENT_LADDER = (1.0, 0.5, 0.25)


@pytest.fixture(scope="module")
def refinement_ladder():
    """One Born ion at three resolutions, solved once and read by three tests.

    Shared because the 0.25 A solve is the expensive one in this file and the
    three claims below are about the same sequence — the error falling, its sign
    holding, and the iteration count not growing.
    """
    return [
        DebyeSolver().solve(born_request(resolution, want_potential=False))
        for resolution in REFINEMENT_LADDER
    ]


def test_the_born_error_falls_under_refinement(refinement_ladder):
    """The other half of M1: within 1%, *converging monotonically*.

    One accurate number can be a cancellation of two errors. A sequence that
    falls at every step is a discretization behaving like one — and it is the
    claim that catches the specific way this solver could be subtly wrong, a
    charge assignment and a potential interpolation that are not adjoint, whose
    residue grows as the grid refines.
    """
    exact = born_solvation_energy(3.0, 1.0, 1.0, 78.54)
    errors = [abs(_energy(result) - exact) / abs(exact) for result in refinement_ladder]

    assert errors == sorted(errors, reverse=True), f"not monotonic: {errors}"
    assert errors[-1] < 0.01


def test_every_solve_overshoots_the_closed_form_rather_than_straddling_it(refinement_ladder):
    """The sign of the error is the discretization's, and it should not wander.

    A sharp boundary sampled at face centres puts the dielectric interface on a
    staircase, and the staircase always encloses slightly the wrong volume in
    the same direction. An error that changed sign between resolutions would
    say the two solves differ by something other than the grid.
    """
    exact = born_solvation_energy(3.0, 1.0, 1.0, 78.54)
    for result in refinement_ladder:
        assert result.energy_kj_mol is not None
        assert result.energy_kj_mol < exact


def test_the_iteration_count_does_not_grow_with_the_grid(refinement_ladder):
    """Multigrid's defining property, and the one a wrong answer would not reveal.

    A restriction operator off by a constant factor still restricts: CG still
    converges, the energies are unchanged to four decimals, and the only
    symptom is the cycle count climbing with the grid — 20, 33 and 55 cycles as
    this solver was first written, against 8, 8 and 9 once the finite-volume
    scaling was right. Asserting the count is how that class of defect is
    visible at all.
    """
    counts = [result.diagnostics["solvated_solve"]["iterations"] for result in refinement_ladder]

    assert max(counts) <= min(counts) + 2, (
        f"cycle counts {counts} grow with refinement; the multigrid preconditioner is "
        "not doing its job and CG is carrying the solve."
    )
    assert max(counts) < 20


def test_the_potential_field_matches_the_closed_form_outside_the_boundary():
    """M1b's measurement, taken here so M1 records where the field stands.

    Sampled at a + k*h, never on the interface: phi is continuous there and its
    normal derivative is not, so interpolating across a 78-fold gradient jump is
    O(1) wrong for every solver and is nobody's defect to fix.
    """
    result = DebyeSolver().solve(born_request(0.25))
    grid = result.potential
    assert isinstance(grid, PotentialGrid)

    spacing = float(np.max(grid.spacing))
    for k in (2, 4, 8):
        radius = 3.0 + k * spacing
        exact = born_potential(radius, 1.0, VDW.solvent_dielectric)
        got = float(grid.value_at([[radius, 0.0, 0.0]])[0])
        assert abs(got - exact) / abs(exact) < 0.02


def test_a_lone_sphere_is_the_same_boundary_whether_or_not_a_probe_rolls_over_it():
    """debye's van der Waals surface, against the one geometric fact about it.

    A probe cannot carve a re-entrant surface out of a single sphere, so the
    molecular and van der Waals boundaries coincide there. debye builds only the
    van der Waals one, and what this asserts is the consequence: `surface_radius`
    is not read, so changing it cannot move the answer.
    """
    baseline = DebyeSolver().solve(born_request(0.5, want_potential=False))
    from dataclasses import replace  # noqa: PLC0415 — local to the one test that needs it

    probed = DebyeSolver().solve(
        FiniteDifferenceRequest(
            structure=born_ion(),
            solvent=replace(VDW, surface_radius=3.0),
            grid=GridSpec(resolution=0.5, padding=10.0),
            want_potential=False,
        )
    )
    assert probed.energy_kj_mol == baseline.energy_kj_mol


def test_it_refuses_a_boundary_it_cannot_build():
    """Every surface but van der Waals, by name, with the milestone that adds it."""
    for model in (
        SurfaceModel.MOLECULAR,
        SurfaceModel.SMOOTHED_MOLECULAR,
        SurfaceModel.GAUSSIAN,
    ):
        with pytest.raises(UnsupportedRequest, match="van-der-waals"):
            DebyeSolver().solve(
                FiniteDifferenceRequest(
                    structure=born_ion(),
                    solvent=SolventModel(surface_model=model),
                    grid=GridSpec(resolution=1.0),
                )
            )


def test_it_refuses_the_nonlinear_equation():
    with pytest.raises(UnsupportedRequest, match="nonlinear"):
        DebyeSolver().solve(
            FiniteDifferenceRequest(
                structure=born_ion(),
                solvent=VDW,
                grid=GridSpec(resolution=1.0),
                equation=Equation.NONLINEAR,
            )
        )


def test_it_refuses_before_it_solves():
    """A refusal that arrives after the arithmetic is a refusal that cost the caller.

    Both checks land before the grid is sized, which `sashimi.gb` and every
    subprocess backend also do. Measured rather than asserted structurally: a
    request that would take seconds to solve returns immediately.
    """
    import time  # noqa: PLC0415

    started = time.perf_counter()
    with pytest.raises(UnsupportedRequest):
        DebyeSolver().solve(
            FiniteDifferenceRequest(
                structure=born_ion(),
                solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
                grid=GridSpec(resolution=0.1, padding=20.0),
            )
        )
    assert time.perf_counter() - started < 0.1


def test_it_reports_no_binary_because_there_is_none():
    """The second backend to exercise `Provenance`'s optional binary, and the first
    in the reference tier to do it. That combination is debye's entire purpose."""
    result = DebyeSolver().solve(born_request(1.0, want_potential=False))
    provenance = result.provenance

    assert provenance.backend == f"debye-{BACKEND_VERSION}"
    assert provenance.binary_path is None
    assert provenance.binary_sha256 is None
    assert provenance.accuracy_tier is AccuracyTier.REFERENCE
    assert provenance.energy_term is EnergyTerm.POLAR_SOLVATION


def test_it_reports_the_grid_it_actually_used():
    """`resolution` is a request; the achieved spacing is what the answer is on."""
    result = DebyeSolver().solve(born_request(0.25, want_potential=False))
    grid = result.provenance.resolved_parameters["grid"]

    assert grid["shape"] == [105, 105, 105]
    assert grid["spacing_achieved"] == [0.25, 0.25, 0.25]
    assert grid["multigrid_levels"] >= 3


def test_a_request_for_no_energy_skips_the_reference_solve():
    """Two solves are the price of an energy, and only of an energy."""
    field_only = DebyeSolver().solve(
        FiniteDifferenceRequest(
            structure=born_ion(),
            solvent=VDW,
            grid=GridSpec(resolution=1.0, padding=10.0),
            want_energy=False,
            want_potential=True,
        )
    )
    assert field_only.energy_kj_mol is None
    assert "uniform_dielectric_solve" not in field_only.diagnostics
    assert isinstance(field_only.potential, PotentialGrid)


def test_the_potential_is_the_solvated_state_on_the_grid_it_reports():
    result = DebyeSolver().solve(born_request(1.0))
    grid = result.potential
    assert isinstance(grid, PotentialGrid)

    sized = size_grid(born_ion(), GridSpec(resolution=1.0, padding=10.0))
    assert grid.shape == sized.shape
    assert np.allclose(grid.origin, sized.origin)
    assert np.allclose(grid.spacing, sized.spacing)
    # A positive charge makes a positive potential everywhere around it.
    assert float(grid.values.max()) > 0.0
    assert np.isfinite(grid.values).all()


def test_an_impossible_budget_says_so_rather_than_returning_the_best_it_managed():
    """`ConvergenceFailure` is in the taxonomy because a half-solved field is not
    a less accurate answer, it is a different system's answer."""
    with pytest.raises(ConvergenceFailure, match="max_cycles"):
        DebyeSolver(options=DebyeOptions(tolerance=1e-14, max_cycles=2)).solve(
            born_request(0.5, want_potential=False)
        )


def test_charge_scales_the_energy_quadratically_and_sign_does_not_matter():
    """Born goes as q^2. The one arm where an analytic identity is exact on the grid:
    the dielectric map does not depend on the charge, so the two solves differ by a
    scale factor and nothing else."""
    single = DebyeSolver().solve(born_request(0.5, want_potential=False))
    doubled = DebyeSolver().solve(
        FiniteDifferenceRequest(
            structure=born_ion(charge=2.0),
            solvent=VDW,
            grid=GridSpec(resolution=0.5, padding=10.0),
            want_potential=False,
        )
    )
    negative = DebyeSolver().solve(
        FiniteDifferenceRequest(
            structure=born_ion(charge=-1.0),
            solvent=VDW,
            grid=GridSpec(resolution=0.5, padding=10.0),
            want_potential=False,
        )
    )
    assert single.energy_kj_mol is not None
    assert doubled.energy_kj_mol == pytest.approx(4.0 * single.energy_kj_mol, rel=1e-9)
    assert negative.energy_kj_mol == pytest.approx(single.energy_kj_mol, rel=1e-9)


@pytest.mark.apbs
def test_it_agrees_with_apbs_on_a_real_structure():
    """The lesson this project keeps relearning: fixtures are spheres, bugs are proteins.

    `peptide-vdw` is the corpus's only van der Waals case with a real structure
    in the fast tier, and it is the one that exercises what the Born ion cannot:
    twenty atoms, two of them with a radius of 0.6 A, a net charge of zero, a
    non-cubic grid, and 0.15 M salt — so it is also the only test here in which
    the Boltzmann term is not identically zero.
    """
    from sashimi.apbs import ApbsSolver  # noqa: PLC0415

    case = case_named("peptide-vdw")
    request = case.request()
    assert request.solvent.ionic_strength > 0.0

    debye = DebyeSolver().solve(request)
    apbs = ApbsSolver().solve(request)
    assert debye.energy_kj_mol is not None
    assert apbs.energy_kj_mol is not None

    # Both report the same term, so this is a discretization comparison and the
    # reference tier's own spread is the yardstick: section 7 measures the
    # families at 1.0-1.6% apart.
    deviation = abs(debye.energy_kj_mol - apbs.energy_kj_mol) / abs(apbs.energy_kj_mol)
    assert deviation < 0.02, f"debye and APBS are {deviation:.3%} apart on {case.name}"
    assert debye.provenance.energy_term is apbs.provenance.energy_term


def test_the_smoother_cannot_be_made_asymmetric_by_a_caller():
    """One sweep count, because two would let a caller break CG's precondition.

    The V-cycle is a legal preconditioner only if it is its own adjoint, which
    needs the post-smoothing sweeps to match the pre-smoothing ones. This
    carried separate `pre_smooth` and `post_smooth` fields until a review asked
    what stopped them differing — nothing did, and an unequal pair makes the
    preconditioner nonsymmetric with no error: measured on the Born ion at
    0.5 A, (2,2) converged in 8 cycles where (2,1) took 13 and (0,2) took 14.
    The failure would have surfaced as a `ConvergenceFailure` advising a larger
    `max_cycles`, which would not have been the cause.

    Made unrepresentable rather than validated, which ROADMAP.md section 7 ranks
    as the deeper fix: there is one field, so there is no unequal pair to reject.
    """
    assert not hasattr(DebyeOptions(), "pre_smooth")
    assert not hasattr(DebyeOptions(), "post_smooth")
    assert DebyeOptions().smoothing_sweeps == 2

    with pytest.raises(ValueError, match="smoothing_sweeps"):
        DebyeOptions(smoothing_sweeps=0)


def test_provenance_carries_every_knob_that_decides_the_answer():
    """`max_cycles` decides whether the answer exists at all, so it travels.

    Two solves of the same case — one at the default budget, one at a raised
    one — produced byte-identical `resolved_parameters` until a review asked.
    ROADMAP.md section 4 wants provenance to be enough to reproduce a number.
    """
    default = DebyeSolver().solve(born_request(1.0, want_potential=False))
    raised = DebyeSolver(options=DebyeOptions(max_cycles=500)).solve(
        born_request(1.0, want_potential=False)
    )

    assert default.provenance.resolved_parameters != raised.provenance.resolved_parameters
    knobs = default.provenance.resolved_parameters["debye"]
    assert {"tolerance", "max_cycles", "smoothing_sweeps"} <= set(knobs)
    assert knobs["max_cycles"] == DebyeOptions().max_cycles
