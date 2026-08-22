"""The properties debye's discretization has to have, checked as properties.

Every assertion here is exact or near-exact algebra rather than a recorded
number, which is the point: these are the invariants that make the physics come
out, and each one has a specific way of being wrong that produces a *plausible*
answer rather than an obviously broken one.

The transpose test is the example worth reading. `_restrict` was written with
the textbook full-weighting stencil, which carries a factor of a half per axis
because the textbook operator is a differenced Laplacian and its residual is
pointwise. debye's operator is a finite-volume flux balance, so its residual is
an integral over a control volume and eight fine cells make one coarse one: the
restriction has to be exactly the transpose of prolongation, with no averaging
factor. With the factor in place every V-cycle applied an eighth of the coarse
correction it should — and the *energies were unchanged to four decimals*,
because CG on top still converged. The only symptom was the cycle count growing
with the grid. An exact-transpose assertion catches it; no amount of comparing
answers to closed forms does.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sashimi.debye.dielectric import (
    bjerrum_length_a,
    dielectric_faces,
    inside_union_of_spheres,
    screening_nodes,
)
from sashimi.debye.grid import DebyeGrid, axis_coordinates, coarsen, size_grid
from sashimi.debye.linear import INNER, _prolong, _restrict, build_levels
from sashimi.debye.sources import (
    assign_charges,
    boundary_mask,
    debye_huckel_boundary,
    interpolate_at_atoms,
)
from sashimi.errors import GridTooLarge, InputError
from sashimi.protocol import GridSpec, PQRData, SolventModel, SurfaceModel

VDW = SolventModel(
    solute_dielectric=1.0, ionic_strength=0.0, surface_model=SurfaceModel.VAN_DER_WAALS
)


def two_atoms() -> PQRData:
    """Off-centre and asymmetric, so an axis mix-up cannot hide behind symmetry."""
    return PQRData(
        coords=np.array([[0.4, -0.7, 1.1], [2.3, 1.0, -0.6]]),
        charges=np.array([0.8, -1.3]),
        radii=np.array([1.8, 1.2]),
    )


def small_grid() -> DebyeGrid:
    """Deliberately non-cubic, with a different spacing on every axis.

    `size_grid` happens to return 17^3 for this structure, and a cubic grid with
    isotropic spacing is the one shape on which swapping two axes is invisible:
    the arrays are interchangeable, so a stencil that reads the y face for its
    x neighbour still passes a symmetry test. Every property here is a statement
    about axes, so the fixture has to make the axes distinguishable.
    """
    return DebyeGrid(
        shape=(13, 9, 11), origin=(-6.0, -5.0, -5.5), spacing=(0.9, 1.1, 1.3), center=(0.0,) * 3
    )


def test_restriction_is_exactly_the_transpose_of_prolongation():
    """<R f, c> == <f, P c>, which is what makes the coarse correction the right size.

    Any scalar multiple of `_restrict` still restricts, still converges under
    CG and still gives the same energy. Only this identity notices.
    """
    rng = np.random.default_rng(20260813)
    fine_shape = (17, 9, 9)
    coarse_shape = (9, 5, 5)

    fine = rng.standard_normal(fine_shape)
    coarse = rng.standard_normal(coarse_shape)
    # Both operators are defined on vectors with homogeneous Dirichlet data.
    fine[boundary_mask(fine_shape)] = 0.0
    coarse[boundary_mask(coarse_shape)] = 0.0

    assert float(np.vdot(_restrict(fine), coarse)) == pytest.approx(
        float(np.vdot(fine, _prolong(coarse))), rel=1e-12
    )


def test_the_operator_is_symmetric():
    """<A x, y> == <x, A y>. A face indexed on the wrong side breaks this and little else."""
    rng = np.random.default_rng(7)
    grid = small_grid()
    level = build_levels(grid, two_atoms(), VDW)[0]

    x = np.zeros(grid.shape)
    y = np.zeros(grid.shape)
    x[INNER] = rng.standard_normal(tuple(n - 2 for n in grid.shape))
    y[INNER] = rng.standard_normal(tuple(n - 2 for n in grid.shape))

    assert float(np.vdot(level.apply(x), y)) == pytest.approx(
        float(np.vdot(x, level.apply(y))), rel=1e-12
    )


def test_the_operator_is_positive_definite():
    """x.Ax > 0 for x != 0, which is what CG and the multigrid smoother both assume."""
    rng = np.random.default_rng(11)
    grid = small_grid()
    level = build_levels(grid, two_atoms(), VDW)[0]

    for _ in range(5):
        x = np.zeros(grid.shape)
        x[INNER] = rng.standard_normal(tuple(n - 2 for n in grid.shape))
        assert float(np.vdot(x, level.apply(x))) > 0.0


def test_charge_assignment_conserves_charge():
    """A leaked charge is a solver quietly answering about a different molecule."""
    structure = two_atoms()
    rho = assign_charges(small_grid(), structure)
    assert float(rho.sum()) == pytest.approx(structure.total_charge, abs=1e-12)


def test_interpolation_is_the_adjoint_of_charge_assignment():
    """<assign(q), phi> == <q, interpolate(phi)>, exactly.

    The identity the solvation energy rests on. The grid self-energy of a point
    charge is enormous and diverges as h shrinks; it cancels between the
    solvated and uniform-dielectric solves *only* because the charge is spread
    and the potential read back with the same weights. Read it back with a
    fancier interpolation and the residue scales like 1/h, so the Born ion gets
    worse under refinement — which is what M1's monotonicity claim would catch,
    one measurement later than this does.
    """
    rng = np.random.default_rng(3)
    grid = small_grid()
    structure = two_atoms()
    phi = rng.standard_normal(grid.shape)

    spread = float(np.vdot(assign_charges(grid, structure), phi))
    gathered = float(np.dot(structure.charges, interpolate_at_atoms(grid, phi, structure)))
    assert spread == pytest.approx(gathered, rel=1e-12)


def test_a_sphere_marks_the_volume_it_encloses():
    """The dielectric map's geometry, against the volume the sphere actually has."""
    structure = PQRData(coords=np.zeros((1, 3)), charges=np.array([1.0]), radii=np.array([4.0]))
    grid = size_grid(structure, GridSpec(resolution=0.2, padding=4.0))
    axes = axis_coordinates(grid)
    inside = inside_union_of_spheres(axes, structure.coords, structure.radii)

    cell = float(np.prod(grid.spacing))
    counted = float(inside.sum()) * cell
    exact = 4.0 / 3.0 * np.pi * 4.0**3
    assert counted == pytest.approx(exact, rel=0.01)


def test_a_zero_radius_atom_bounds_no_volume():
    """Kirkwood's charge carrier has radius zero, and it is not a dielectric body.

    The fourth appearance of "a radius is not always a radius" in this project:
    `sashimi.gb` divides by one and cannot take a zero, DelPhi reads a different
    column than pdb2pqr writes. Here the honest reading is that the atom encloses
    nothing, and the enclosing sphere is a different atom.
    """
    structure = PQRData(
        coords=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        charges=np.array([0.0, 1.0]),
        radii=np.array([3.0, 0.0]),
    )
    grid = size_grid(structure, GridSpec(resolution=0.5, padding=4.0))
    axes = axis_coordinates(grid)

    with_charge_carrier = inside_union_of_spheres(axes, structure.coords, structure.radii)
    sphere_only = inside_union_of_spheres(axes, structure.coords[:1], structure.radii[:1])
    assert np.array_equal(with_charge_carrier, sphere_only)


def test_the_dielectric_is_low_inside_and_high_outside():
    structure = two_atoms()
    grid = small_grid()
    faces = dielectric_faces(grid, structure, VDW)

    for axis, eps in enumerate(faces):
        expected_shape = tuple(n - 1 if a == axis else n for a, n in enumerate(grid.shape))
        assert eps.shape == expected_shape
        assert set(np.unique(eps)) <= {VDW.solute_dielectric, VDW.solvent_dielectric}
        # The box is the solute plus 4 A of padding, so its corner is solvent.
        assert eps[0, 0, 0] == VDW.solvent_dielectric


def test_the_ion_exclusion_layer_is_thicker_than_the_dielectric_boundary():
    """Mobile ions have a radius of their own, and it is not the solvent probe.

    Invisible at zero salt, which is every case M1 is graded on — so it is
    asserted now rather than discovered at M3.
    """
    structure = two_atoms()
    grid = small_grid()
    salty = SolventModel(
        solute_dielectric=1.0,
        ionic_strength=0.15,
        ion_radius=2.0,
        surface_model=SurfaceModel.VAN_DER_WAALS,
    )
    screening, bulk = screening_nodes(grid, structure, salty)
    assert bulk > 0.0

    axes = axis_coordinates(grid)
    dielectric_interior = inside_union_of_spheres(axes, structure.coords, structure.radii)
    ion_free = screening == 0.0
    # Every point inside the dielectric body is ion-free, and strictly more
    # points are ion-free than are inside it.
    assert np.all(ion_free[dielectric_interior])
    assert int(ion_free.sum()) > int(dielectric_interior.sum())


def test_there_is_no_screening_without_salt():
    screening, bulk = screening_nodes(small_grid(), two_atoms(), VDW)
    assert bulk == 0.0
    assert not screening.any()


def test_the_boundary_carries_the_coulomb_tail_and_the_interior_is_untouched():
    structure = two_atoms()
    grid = small_grid()
    field = debye_huckel_boundary(grid, structure, VDW)

    mask = boundary_mask(grid.shape)
    assert not field[~mask].any()

    # A corner node, against the closed-form sum it is built from.
    corner = np.asarray(grid.origin)
    distances = np.linalg.norm(structure.coords - corner, axis=1)
    expected = (bjerrum_length_a(VDW.temperature) / VDW.solvent_dielectric) * float(
        np.sum(structure.charges / distances)
    )
    assert field[0, 0, 0] == pytest.approx(expected, rel=1e-12)


def test_salt_screens_the_boundary_values_towards_zero():
    """Each atom's tail shrinks with salt — asserted where "each" and "the sum" agree.

    Pointwise on a lone charge, in aggregate on the dipole. The distinction is
    a real one and cost this test a first draft: `two_atoms` carries +0.8 and
    -1.3, so its two tails partly cancel, and screening reweights them by
    distance and exclusion radius. On 21 of 594 boundary nodes the *magnitude
    of the sum* therefore rises while every term in it falls. Asserting a
    per-term property on a sum is how a guard comes to fail on correct physics.
    """
    grid = small_grid()
    salty = SolventModel(
        solute_dielectric=1.0, ionic_strength=0.15, surface_model=SurfaceModel.VAN_DER_WAALS
    )
    mask = boundary_mask(grid.shape)

    lone = PQRData(
        coords=np.array([[0.4, -0.7, 1.1]]), charges=np.array([1.0]), radii=np.array([1.8])
    )
    unscreened = np.abs(debye_huckel_boundary(grid, lone, VDW)[mask])
    screened = np.abs(debye_huckel_boundary(grid, lone, salty)[mask])
    assert np.all(screened < unscreened)

    dipole = two_atoms()
    assert (
        np.abs(debye_huckel_boundary(grid, dipole, salty)[mask]).sum()
        < np.abs(debye_huckel_boundary(grid, dipole, VDW)[mask]).sum()
    )


def test_the_homogeneous_reference_boundary_uses_the_solute_dielectric():
    """The reference state is the solute's dielectric everywhere, and no ions."""
    structure = two_atoms()
    grid = small_grid()
    solvent = SolventModel(
        solute_dielectric=2.0,
        solvent_dielectric=78.54,
        ionic_strength=0.15,
        surface_model=SurfaceModel.VAN_DER_WAALS,
    )
    solvated = debye_huckel_boundary(grid, structure, solvent)
    reference = debye_huckel_boundary(grid, structure, solvent, homogeneous=True)
    mask = boundary_mask(grid.shape)
    # Same geometry, dielectric 2 rather than 78.54 and no screening, so the
    # reference tail is the larger of the two everywhere it is non-zero.
    assert np.all(np.abs(reference[mask]) >= np.abs(solvated[mask]) - 1e-15)


def test_coarsening_preserves_the_box():
    """A coarse node sits on a fine node, which is what licenses re-discretization."""
    grid = size_grid(two_atoms(), GridSpec(resolution=0.5, padding=6.0))
    coarse = coarsen(grid)

    assert coarse.origin == grid.origin
    for axis in range(3):
        fine_span = grid.spacing[axis] * (grid.shape[axis] - 1)
        coarse_span = coarse.spacing[axis] * (coarse.shape[axis] - 1)
        assert coarse_span == pytest.approx(fine_span, rel=1e-12)
        assert coarse.spacing[axis] == pytest.approx(2.0 * grid.spacing[axis], rel=1e-12)


def test_the_bjerrum_length_is_the_familiar_one():
    """7.14 A in water at 298.15 K, which is this divided by the dielectric."""
    assert bjerrum_length_a(298.15) / 78.54 == pytest.approx(7.14, abs=0.01)


def test_the_point_budget_survives_a_grid_too_large_to_count():
    """`max_points` is a documented guardrail, and it silently stopped guarding.

    `int(np.prod(shape))` returns int64 and wraps. `apbs.grid` is safe from this
    only because `LEGAL_DIME` stops at 1281; debye's lattice has no ceiling, so
    a request at 1e-6 A produced 26,000,001 nodes per axis, a *negative* product,
    a comparison that was false, and a 1.8e22-point grid returned in 3 ms — to
    die later in a numpy allocation rather than with the error the taxonomy has
    for exactly this. Found by review, not by the suite.
    """
    structure = PQRData(coords=np.zeros((1, 3)), charges=np.array([1.0]), radii=np.array([3.0]))

    # The contract is to *relax* the resolution until the budget is met — the
    # same rule `apbs.grid` follows — and to raise only when even the smallest
    # grid will not fit. The overflow broke the first half, not the second: the
    # comparison was false, so nothing relaxed and a 1.8e22-point grid came back.
    for resolution in (1e-6, 0.01, 0.25):
        grid = size_grid(structure, GridSpec(resolution=resolution, padding=10.0))
        assert math.prod(grid.shape) <= GridSpec().max_points

    grid = size_grid(structure, GridSpec(resolution=0.25, padding=10.0, max_points=60**3))
    assert math.prod(grid.shape) <= 60**3

    with pytest.raises(GridTooLarge, match="max_points"):
        size_grid(structure, GridSpec(resolution=0.25, padding=10.0, max_points=1000))


def test_a_charge_on_the_box_face_is_refused_rather_than_clipped():
    """The boundary condition's own singularity, which used to be clipped and solved on.

    A zero-radius atom at the low corner of its bounding box with `padding=0`
    lands exactly on a boundary node. It is legally inside the grid, so
    `trilinear_weights` raises nothing; before this, a `np.maximum(d, 1e-6)`
    clip turned the singularity into 7.1e6 kT/e and the solver carried on to a
    confident wrong energy. The clip's own comment said the case could not
    arise.
    """
    structure = PQRData(
        coords=np.array([[0.0, 0.0, 0.0], [6.0, 6.0, 6.0]]),
        charges=np.array([1.0, 0.0]),
        radii=np.array([0.0, 3.0]),
    )
    grid = size_grid(structure, GridSpec(resolution=1.0, padding=0.0))
    assert np.allclose(grid.origin, structure.coords[0]), "the atom should sit on the corner"

    with pytest.raises(InputError, match="padding"):
        debye_huckel_boundary(grid, structure, VDW)

    # The same structure with room around it is ordinary, and still solvable.
    padded = size_grid(structure, GridSpec(resolution=1.0, padding=5.0))
    assert np.isfinite(debye_huckel_boundary(padded, structure, VDW)).all()


def _born(radius: float) -> PQRData:
    return PQRData(coords=np.zeros((1, 3)), charges=np.array([1.0]), radii=np.array([radius]))


def _born_energy(radius: float, spacing: float, smoothing: float) -> float:
    from sashimi.debye import DebyeSolver  # noqa: PLC0415
    from sashimi.debye.options import DebyeOptions  # noqa: PLC0415
    from sashimi.protocol import FiniteDifferenceRequest  # noqa: PLC0415

    request = FiniteDifferenceRequest(
        structure=_born(radius),
        solvent=SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS, ionic_strength=0.0),
        grid=GridSpec(resolution=spacing, padding=10.0),
        want_potential=False,
    )
    answer = DebyeSolver(options=DebyeOptions(dielectric_smoothing=smoothing)).solve(request)
    assert answer.energy_kj_mol is not None
    return float(answer.energy_kj_mol)


def test_the_shipped_scheme_is_the_hard_one_and_stays_the_default():
    """The knob must be off, or every recorded corpus energy is a different quantity."""
    from sashimi.debye.options import DebyeOptions  # noqa: PLC0415

    assert DebyeOptions().dielectric_smoothing == 0.0


@pytest.mark.parametrize(("radius", "spacing"), [(2.0, 0.5), (3.0, 0.25), (3.0, 0.125)])
def test_a_sub_cell_ramp_beats_the_hard_assignment_against_the_closed_form(radius, spacing):
    """Graded on the Born energy, which is exact — not against another solver.

    Both finite-difference incumbents make the same face-centre assignment this
    replaces, so neither can referee it; `sashimi.debye.dielectric` says so at
    length. The closed form has no such conflict.

    Measured 4-17x better across the ladder. The bar here is 2x, which is far
    enough below the measurement to survive a solver-tolerance change and far
    enough above 1 to catch the scheme being silently disabled.
    """
    from sashimi.analytic import born_solvation_energy  # noqa: PLC0415

    solvent = SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS, ionic_strength=0.0)
    exact = born_solvation_energy(
        radius_a=radius,
        charge_e=1.0,
        solute_dielectric=solvent.solute_dielectric,
        solvent_dielectric=solvent.solvent_dielectric,
    )
    hard = abs(_born_energy(radius, spacing, 0.0) - exact)
    ramped = abs(_born_energy(radius, spacing, 0.5) - exact)
    assert ramped * 2.0 < hard, (
        f"the ramp is {hard / ramped:.1f}x better at a={radius}, h={spacing}, "
        "where it was measured at 4-17x"
    )


def test_a_band_wider_than_a_cell_is_worse_than_no_ramp_at_all():
    """The negative result that decided the scheme, pinned so it is not re-run.

    The first implementation averaged the *indicator* over a box of whole cells
    and blended harmonically — which is what "smoothing over a band of w cells"
    naturally reads as. It smears the interface across three cells and on this
    case it is **worse than the hard assignment**. The ramp is sub-cell for that
    reason, and a future widening would be caught here rather than in a
    quarter's worth of drifted energies.
    """
    from sashimi.analytic import born_solvation_energy  # noqa: PLC0415

    solvent = SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS, ionic_strength=0.0)
    exact = born_solvation_energy(
        radius_a=3.0,
        charge_e=1.0,
        solute_dielectric=solvent.solute_dielectric,
        solvent_dielectric=solvent.solvent_dielectric,
    )
    hard = abs(_born_energy(3.0, 0.25, 0.0) - exact)
    wide = abs(_born_energy(3.0, 0.25, 2.0) - exact)
    assert wide > hard, "a two-cell band is no longer worse than hard, so the finding moved"


def test_the_ramp_refuses_the_surface_it_cannot_measure_a_distance_to():
    """`molecular` is the default surface and the one this does not yet cover.

    The ramp is built from `min(|x - c| - r)`, which is the distance to a union
    of spheres and *not* to the solvent-excluded surface. Refusing is the whole
    point: ramping against the wrong surface would return a number that looked
    like an improvement and was not.
    """
    from sashimi.debye.dielectric import dielectric_faces  # noqa: PLC0415
    from sashimi.errors import UnsupportedRequest  # noqa: PLC0415

    structure = _born(3.0)
    grid = size_grid(structure, GridSpec(resolution=1.0, padding=10.0))
    solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR)
    with pytest.raises(UnsupportedRequest, match="van-der-waals"):
        dielectric_faces(grid, structure, solvent, smoothing=0.5)
