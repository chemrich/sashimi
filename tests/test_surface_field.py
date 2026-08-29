"""The off-surface Green evaluator, graded where a closed form exists.

Two of these gates need no solver at all: the solid-angle identity is refereed by
the divergence theorem, and the null-field identity by the representation itself.
That matters more than convenience — this module exists to be a referee, and a
referee whose own correctness rests on the codes it is meant to check is worth
nothing.

The failure modes are *run* rather than cited, because the two defects this
expression is prone to — an inverted sign and a spurious `1/eps_s` — both produce
a smooth, plausible field of the wrong magnitude, and ROADMAP.md section 12
measured them at -0.50395 and -0.01283 of exact before anyone wrote the code.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sashimi.analytic import born_potential, kirkwood_potential
from sashimi.protocol import BoundaryElementRequest, PQRData, SolventModel, SurfacePotential
from sashimi.surface_field import evaluate_exterior, faces_of, solid_angle
from sashimi.tabipb import TabipbSolver, discover_tabipb
from tests.helpers import installed_or_skip, surface

pytestmark = pytest.mark.tabipb

RADIUS, OFFSET = 3.0, 0.01
TETRAHEDRON = np.array(
    [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
) / np.sqrt(3.0)
SPHERE = PQRData(coords=TETRAHEDRON * OFFSET, charges=np.full(4, 0.25), radii=np.full(4, RADIUS))
UNSCREENED = SolventModel(solute_dielectric=2.0, solvent_dielectric=78.54, ionic_strength=0.0)


def directions(count: int = 200) -> np.ndarray:
    """Quasi-uniform directions. A single ray measures whichever symmetry class
    it happens to lie in — `sashimi.field` records that costing a factor of 2.6
    on DelPhi's worst case."""
    rng = np.random.default_rng(0)
    d = rng.normal(size=(count, 3))
    return d / np.linalg.norm(d, axis=1)[:, None]


def solve(mesh_density: float = 5.0, solvent: SolventModel = UNSCREENED) -> SurfacePotential:
    installed_or_skip(discover_tabipb, "SASHIMI_TABIPB")
    return surface(
        TabipbSolver().solve(
            BoundaryElementRequest(structure=SPHERE, solvent=solvent, mesh_density=mesh_density)
        )
    )


def test_the_orientation_identity_holds_inside_and_outside():
    """B-G1, and the only gate here refereed by nothing but the divergence theorem.

    `tabipb.vtk` leaves `normals` at None and NanoShaper's winding convention is
    asserted nowhere, so a global flip would invert every evaluated field by a
    route no magnitude check would attribute correctly. A flip reads +1 against
    -1 — a hundred times the bar — and mixed winding on a non-convex surface reads
    something in between.
    """
    mesh = solve(mesh_density=3.0)
    faces = faces_of(mesh)
    rng = np.random.default_rng(1)

    for point in rng.normal(size=(20, 3)) * 0.4:  # well inside
        assert solid_angle(point, faces) == pytest.approx(-1.0, abs=0.02)
    for point in directions(20) * (RADIUS + 6.0):
        assert solid_angle(point, faces) == pytest.approx(0.0, abs=0.02)

    # Recorded rather than assumed: NanoShaper's winding already points outward,
    # so the correction below is a guard and not a routine fix-up.
    assert faces.outward_from_winding


def test_a_flipped_winding_is_invisible_in_the_field_and_only_the_identity_catches_it():
    """Why B-G1 is not redundant with the magnitude gates — measured, not argued.

    Flipping the winding negates `n`, which appears in the double layer and not
    in the single layer, so it does **not** simply invert the field. And on a
    sphere with a centred charge the exterior double layer is *identically zero*,
    so a flip moves the evaluated potential by essentially nothing: 0.06% here,
    three times smaller than the discretization residual it hides inside.

    So on the one fixture with an exact closed form, the defect that would invert
    a real molecule's field is undetectable by comparing against that closed
    form. The solid-angle identity reads +1 against -1 regardless, because it is
    a statement about the surface rather than about the solution on it.
    """
    mesh = solve(mesh_density=3.0)
    faces = faces_of(mesh)
    flipped = faces_of(mesh)
    flipped.normals = -flipped.normals

    # The identity catches it outright.
    assert solid_angle(np.zeros(3), faces) == pytest.approx(-1.0, abs=0.02)
    assert solid_angle(np.zeros(3), flipped) == pytest.approx(+1.0, abs=0.02)

    # The field does not.
    shell = directions(40) * (RADIUS + 1.5)
    right = evaluate_exterior(mesh, shell, UNSCREENED, faces=faces)
    wrong = evaluate_exterior(mesh, shell, UNSCREENED, faces=flipped)
    drift = abs(float(np.mean(wrong / right)) - 1.0)
    assert drift < 0.005, (
        f"the flip moved the field by {drift:.2%}; if it ever moves it by more "
        "than the monopole residual, this argument needs re-measuring"
    )


@pytest.mark.parametrize("mesh_density", [5.0, 8.0])
def test_the_born_exterior_potential_comes_back(mesh_density):
    """B-G2. `ionic_strength = 0` is load-bearing: `born_potential` is unscreened
    and overstates by ~30% two cells out at 0.15 M, which is the default.

    Reachable per ROADMAP.md section 12's measured ladder — 1.278 / 0.790 / 0.502
    / 0.338 % at sdens 3 / 5 / 8 / 12 — which this reproduces to the digit. That
    residual is *not* quadrature error; it is a spurious monopole in the discrete
    surface data, so the bar cannot be tightened by standing further off.
    """
    mesh = solve(mesh_density=mesh_density)
    shell = directions() * (RADIUS + 1.5)
    got = evaluate_exterior(mesh, shell, UNSCREENED)
    want = born_potential(RADIUS + 1.5, 1.0, 78.54, 298.15)

    assert got.mean() / want == pytest.approx(1.0, abs=0.02)
    # Angular scatter is far below the monopole: the error is a level, not a shape.
    assert float(np.std(got / want)) <= 0.005


def test_feeding_the_interior_side_is_wrong_by_the_dielectric_ratio():
    """The defect the protocol's field name exists to prevent, priced here.

    On a sphere with a centred charge the constant-density double layer vanishes
    outside, so the single layer carries the whole answer — and the side error is
    therefore *fully* exposed rather than hidden. That is why this gate lives
    beside B-G2 rather than replacing it.
    """
    mesh = solve()
    shell = directions(40) * (RADIUS + 1.5)
    right = evaluate_exterior(mesh, shell, UNSCREENED)

    # A caller that reaches past `exterior_normal_derivative` and uses the stored
    # interior array directly.
    stored = mesh.interior_normal_derivative
    assert stored is not None
    naive = dataclasses.replace(
        mesh,
        interior_normal_derivative=stored
        * (UNSCREENED.solvent_dielectric / UNSCREENED.solute_dielectric),
    )
    wrong = evaluate_exterior(naive, shell, UNSCREENED)
    ratio = float(np.mean(wrong / right))
    assert ratio == pytest.approx(78.54 / 2.0, rel=0.02), (
        f"expected the side error to show as {78.54 / 2.0:.2f}x, got {ratio:.2f}x"
    )


def test_the_screened_kernel_lowers_the_field_and_the_unscreened_limit_recovers_it():
    """`kappa > 0` is a different kernel, not a scaling, so it needs its own check.

    Salt screens, so the potential falls; and as the ionic strength goes to zero
    the screened expression must return to the unscreened one rather than merely
    approach it.
    """
    salted = dataclasses.replace(UNSCREENED, ionic_strength=0.15)
    shell = directions(40) * (RADIUS + 1.5)

    mesh_salt = solve(solvent=salted)
    screened = evaluate_exterior(mesh_salt, shell, salted)
    unscreened_on_same_mesh = evaluate_exterior(mesh_salt, shell, UNSCREENED)
    assert screened.mean() < unscreened_on_same_mesh.mean()

    # The limit, on one mesh so only the kernel varies.
    nearly_zero = dataclasses.replace(UNSCREENED, ionic_strength=1e-9)
    limit = evaluate_exterior(mesh_salt, shell, nearly_zero)
    np.testing.assert_allclose(limit, unscreened_on_same_mesh, rtol=1e-4)


def test_a_surface_without_triangles_or_a_derivative_is_refused_by_name():
    """Both are real: `bem_stub` emits normals and no triangles, and every
    finite-difference backend emits neither."""
    mesh = solve(mesh_density=3.0)
    with pytest.raises(ValueError, match="triangles"):
        faces_of(dataclasses.replace(mesh, triangles=None))
    with pytest.raises(ValueError, match="interior_normal_derivative"):
        evaluate_exterior(
            dataclasses.replace(mesh, interior_normal_derivative=None),
            directions(4) * (RADIUS + 1.5),
            UNSCREENED,
        )


# --- B-G4: the only gate that can see the double layer ------------------------
#
# On a sphere with a *centred* charge the exterior double layer is identically
# zero, so every gate above is blind to the term that does all the work on a real
# molecule: measured, it carries 3.4e-5 of `a_0` on the centred fixture and
# 0.3335 of `a_1` on this one. Kirkwood's off-centre sphere is the smallest
# geometry with `l >= 1` structure and an exact closed form.
#
# The corpus's own Kirkwood cases cannot be used. `corpus.kirkwood_pqr` builds
# each as two atoms — one sphere plus a *zero-radius* charge — and `tabipb.run`
# refuses fewer than four. So the sphere is the same four-atom tetrahedron used
# above, with a fifth charged atom small enough to sit strictly inside it.

CHARGE_OFFSET, CHARGE_RADIUS, SHELL = 1.5, 0.5, 5.0
KIRKWOOD = PQRData(
    coords=np.vstack([TETRAHEDRON * OFFSET, [[CHARGE_OFFSET, 0.0, 0.0]]]),
    charges=np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
    radii=np.array([RADIUS, RADIUS, RADIUS, RADIUS, CHARGE_RADIUS]),
)


def legendre_coefficients(
    values: np.ndarray, mu: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """`a_0` and `a_1` of `phi = a_0 + a_1 P_1 + ...` on a shell."""
    return 0.5 * float(np.sum(weights * values)), 1.5 * float(np.sum(weights * values * mu))


def shell_quadrature(polar_nodes: int = 16, azimuthal: int = 12):
    """Gauss-Legendre in `cos(theta)` about the charge's axis, averaged over
    azimuth. The solution is axially symmetric, so the azimuthal average only
    suppresses mesh noise."""
    mu, w = np.polynomial.legendre.leggauss(polar_nodes)
    points, weights, cosines = [], [], []
    for m, weight in zip(mu, w, strict=True):
        sin = np.sqrt(max(0.0, 1.0 - m * m))
        for k in range(azimuthal):
            angle = 2.0 * np.pi * k / azimuthal
            points.append([SHELL * m, SHELL * sin * np.cos(angle), SHELL * sin * np.sin(angle)])
            weights.append(weight / azimuthal)
            cosines.append(m)
    return np.array(points), np.array(weights), np.array(cosines)


def exact_dipole_coefficient() -> float:
    """`a_1` from `analytic.kirkwood_potential`, projected rather than re-derived.

    Using the shipped closed form keeps this from becoming a second transcription
    of the series that has to be kept in step with the first.
    """
    mu, w = np.polynomial.legendre.leggauss(48)
    phi = np.array(
        [
            kirkwood_potential(
                SHELL,
                float(m),
                RADIUS,
                CHARGE_OFFSET,
                1.0,
                solute_dielectric=UNSCREENED.solute_dielectric,
                solvent_dielectric=UNSCREENED.solvent_dielectric,
            )
            for m in mu
        ]
    )
    return legendre_coefficients(phi, mu, w)[1]


def test_the_off_centre_dipole_term_comes_back():
    """B-G4. The fixture is checked before the physics is graded on it.

    Measured +1.046% at sdens 5 against a 3% bar, and +1.53% at sdens 3 — so a
    mesher regression to coarse output still passes, and the bar is grading the
    evaluator rather than NanoShaper.
    """
    installed_or_skip(discover_tabipb, "SASHIMI_TABIPB")
    mesh = surface(
        TabipbSolver().solve(
            BoundaryElementRequest(structure=KIRKWOOD, solvent=UNSCREENED, mesh_density=5.0)
        )
    )

    # Precondition: the fifth atom must not poke through or spawn a component.
    radii = np.linalg.norm(mesh.vertices, axis=1)
    assert float(radii.max() - radii.min()) <= 0.15
    assert abs(float(radii.mean()) - RADIUS) <= 0.05
    assert solid_angle(np.zeros(3), faces_of(mesh)) == pytest.approx(-1.0, abs=0.02)

    points, weights, cosines = shell_quadrature()
    _, dipole = legendre_coefficients(evaluate_exterior(mesh, points, UNSCREENED), cosines, weights)
    assert dipole / exact_dipole_coefficient() == pytest.approx(1.0, abs=0.03)


def test_the_dipole_term_is_what_the_double_layer_carries():
    """Why B-G4 is not redundant with the monopole gates.

    Deleting the double layer leaves `a_1` at 0.67 of its value and flipping its
    sign leaves 0.34 — both far outside the 3% bar, where on the *centred*
    fixture the same mutations move the answer by parts in a hundred thousand.
    """
    installed_or_skip(discover_tabipb, "SASHIMI_TABIPB")
    mesh = surface(
        TabipbSolver().solve(
            BoundaryElementRequest(structure=KIRKWOOD, solvent=UNSCREENED, mesh_density=5.0)
        )
    )
    points, weights, cosines = shell_quadrature()

    def dipole_of(surface_potential, **kwargs):
        values = evaluate_exterior(surface_potential, points, UNSCREENED, **kwargs)
        return legendre_coefficients(values, cosines, weights)[1]

    intact = dipole_of(mesh)
    no_double_layer = dipole_of(dataclasses.replace(mesh, values=np.zeros_like(mesh.values)))
    flipped_faces = faces_of(mesh)
    flipped_faces.normals = -flipped_faces.normals
    flipped = dipole_of(mesh, faces=flipped_faces)

    assert abs(no_double_layer / intact - 1.0) > 0.03
    assert abs(flipped / intact - 1.0) > 0.03
