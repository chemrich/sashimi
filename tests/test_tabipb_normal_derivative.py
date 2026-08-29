"""The normal derivative's side and its unit, against a real TABI-PB solve.

`NormalPotential` is the **interior** derivative, `eps_p`'s side. That was
measured rather than argued: sweeping the solute dielectric on the four-atom Born
sphere moves the file's values as `1/eps_p` exactly and leaves them invariant to
`eps_s`, which are opposite, discretization-free predictions. The two sides
differ by `eps_s/eps_p` = 39.27 at the protocol's defaults, so a gate written
against the exterior closed form is red on a *correct* implementation by that
factor — a trap ROADMAP names in as many words.

**What each gate can and cannot see** is stated on each, because the failure this
file exists to catch survives every energy check and every single-dielectric one.
"""

from __future__ import annotations

import numpy as np
import pytest

from sashimi.analytic import born_potential
from sashimi.protocol import BoundaryElementRequest, PQRData, SolventModel
from sashimi.tabipb import TabipbSolver, discover_tabipb
from tests.helpers import installed_or_skip, surface

pytestmark = pytest.mark.tabipb

RADIUS = 3.0
# NanoShaper refuses fewer than four atoms, so the sphere is four coincident ones
# on a tiny regular tetrahedron: their union is a sphere to within the offset and
# the split charge cancels the dipole exactly. Same construction as
# `studies/tabipb_units/born_sphere.py`, which is where the calibration lives.
OFFSET = 0.01
TETRAHEDRON = np.array(
    [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
) / np.sqrt(3.0)
SPHERE = PQRData(coords=TETRAHEDRON * OFFSET, charges=np.full(4, 0.25), radii=np.full(4, RADIUS))


def solve(*, mesh_density=5.0, temperature=298.15, pdie=2.0, sdie=78.54):
    """A zero-salt Born sphere. `ionic_strength = 0` is load-bearing: the closed
    forms below are unscreened, and at 0.15 M `born_potential` overstates by
    ~30% two cells out."""
    installed_or_skip(discover_tabipb, "SASHIMI_TABIPB")
    solvent = SolventModel(
        solute_dielectric=pdie,
        solvent_dielectric=sdie,
        ionic_strength=0.0,
        temperature=temperature,
    )
    result = TabipbSolver().solve(
        BoundaryElementRequest(structure=SPHERE, solvent=solvent, mesh_density=mesh_density)
    )
    return surface(result), solvent


def exact_exterior_at(radii, sdie=78.54, temperature=298.15):
    """`phi = C/r` outside, so `dphi/dr = -phi(r)/r`, in kT/(e A).

    Evaluated at **each vertex's own radius**, not at the nominal `a`. The mesh
    sits slightly outside the sphere it approximates — <r> is 3.0045 A at sdens 5
    and 3.0065 at 8 — and because the derivative goes as `1/r^2` that excess
    enters as a -0.3% bias that *grows* with density. Grading the mean against
    the closed form at `a` therefore mixes the mesh's geometry into the
    discretization residual and makes it cross zero; per-vertex grading is what
    isolates the quantity these gates are about. Same convention as
    `studies/tabipb_units/born_sphere.py`.
    """
    return np.array([-born_potential(float(r), 1.0, sdie, temperature) / float(r) for r in radii])


def vertex_radii(mesh):
    return np.linalg.norm(mesh.vertices, axis=1)


def test_the_stored_side_tracks_the_solute_dielectric_and_the_exterior_one_does_not():
    """A-G1, first leg — the gate the whole change turns on.

    The exact *exterior* derivative is independent of `eps_p`, so a correct
    conversion is flat across it. Storing the raw interior side instead gives a
    4.000x spread over `eps_p` in {1, 2, 4}, since the file's values go as
    `1/eps_p`. Note what this leg CANNOT see: a hardcoded 1/39.27 also cancels
    the `eps_p` dependence and is equally flat here. That is what the second leg
    is for.
    """
    interior, exterior = [], []
    for pdie in (1.0, 2.0, 4.0):
        mesh, solvent = solve(pdie=pdie)
        assert mesh.interior_normal_derivative is not None
        interior.append(float(mesh.interior_normal_derivative.mean()))
        converted = mesh.exterior_normal_derivative(solvent)
        assert converted is not None
        exterior.append(float(converted.mean()))

    # The premise: the stored side really does go as 1/eps_p. Magnitudes, because
    # every value is negative and max/min on negatives reads the reciprocal.
    interior = [abs(v) for v in interior]
    exterior = [abs(v) for v in exterior]
    spread_interior = max(interior) / min(interior)
    assert spread_interior == pytest.approx(4.0, rel=0.02), (
        f"interior derivative should span 4x over eps_p 1->4, got {spread_interior:.4f}"
    )
    # The gate: converted to the solvent side, it stops moving.
    spread = max(exterior) / min(exterior)
    assert spread <= 1.01, (
        f"exterior derivative moved {spread:.4f}x across eps_p; it must not move "
        "at all — the stored side is interior and was not converted"
    )


def test_the_conversion_reads_the_solvent_dielectric_rather_than_the_default_ratio():
    """A-G1, second leg — what separates reading the request from a hardcode.

    The stored interior value is invariant to `eps_s` (measured: 0.0009% over
    78.54 -> 40), so the exterior one must move by exactly `eps_s` 's ratio. A
    hardcoded 1/39.27 leaves it flat at 1.000 and passes the first leg.
    """
    mesh_hi, solvent_hi = solve(sdie=78.54)
    mesh_lo, solvent_lo = solve(sdie=40.0)

    raw_hi = float(mesh_hi.interior_normal_derivative.mean())
    raw_lo = float(mesh_lo.interior_normal_derivative.mean())
    assert raw_lo / raw_hi == pytest.approx(1.0, rel=5e-4), (
        "the stored interior side must not depend on eps_s"
    )

    ext_hi = float(mesh_hi.exterior_normal_derivative(solvent_hi).mean())
    ext_lo = float(mesh_lo.exterior_normal_derivative(solvent_lo).mean())
    assert ext_lo / ext_hi == pytest.approx(78.54 / 40.0, rel=0.01), (
        f"exterior derivative moved {ext_lo / ext_hi:.4f}x for a 1.9635x change in "
        "eps_s — the dielectric ratio is not being read from the request"
    )


def test_the_divisor_is_rt_and_carries_the_temperature():
    """A-G2 — a divisor check, and explicitly not a correctness criterion.

    Omitting the `RT` division reads 1.0000 exactly, because the raw file values
    are temperature-independent to every printed digit. **This is blind to the
    dielectric half**, which is the half that is wrong by 39.27x and is
    temperature-independent — gates one and two above are what cover it.
    """
    warm, _ = solve(temperature=298.15, mesh_density=3.0)
    cold, _ = solve(temperature=277.0, mesh_density=3.0)
    ratio = float(cold.interior_normal_derivative.mean()) / float(
        warm.interior_normal_derivative.mean()
    )
    assert ratio == pytest.approx(298.15 / 277.0, rel=1e-3), (
        f"got {ratio:.6f}; 1.0000 means the RT divisor was never applied"
    )


def test_the_conversion_is_exactly_rt_times_the_side_and_not_a_fitted_constant():
    """A-G3 — the h^2 signature, which catches a *small* constant offset.

    Grading the converted exterior value against the closed form, the residual
    must fall as h^2, i.e. `error x density` is constant down the ladder. That is
    the same signature that proved the potential's factor was exactly RT. Fold a
    constant k = 1 + delta into a correct implementation and this goes red for
    delta above ~0.03%, roughly thirty times tighter than any magnitude bar.

    `sdens 3` is deliberately excluded and named as excluded: it is the recorded
    coarse-mesh outlier on both the potential and the derivative.
    """
    errors = {}
    for density in (5.0, 8.0):
        mesh, solvent = solve(mesh_density=density)
        got = mesh.exterior_normal_derivative(solvent)
        want = exact_exterior_at(vertex_radii(mesh))
        errors[density] = float(np.mean(got / want)) - 1.0

    # Level: the exterior value is right to about a percent at these densities.
    assert abs(errors[5.0]) <= 0.02, f"exterior derivative off by {errors[5.0]:.4%} at sdens 5"

    # Order: excess x density constant is the h^2 fingerprint.
    product = (errors[5.0] * 5.0) / (errors[8.0] * 8.0)
    assert product == pytest.approx(1.0, abs=0.25), (
        f"error x density moved {product:.4f} between sdens 5 and 8; the residual "
        "is not pure h^2, so the conversion carries a constant that is not RT"
    )


def test_the_derivative_survives_the_protocol_boundary_at_all():
    """The regression this change exists to prevent: the value was parsed,
    carried through `run.py`, and then dropped where the `SolveResult` was built.
    """
    mesh, _ = solve(mesh_density=3.0)
    assert mesh.interior_normal_derivative is not None
    assert mesh.interior_normal_derivative.shape == (mesh.n_vertices,)
    assert np.all(mesh.interior_normal_derivative < 0.0), (
        "a positive charge must pull the potential down along the outward normal"
    )
    # And it is not silently the potential block over again.
    assert not np.allclose(mesh.interior_normal_derivative, mesh.values)
