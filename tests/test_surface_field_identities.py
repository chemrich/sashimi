"""The evaluator's two identities, on a mesh built from a closed form.

These need no solver. The surface data is Born's exact answer written onto an
analytically generated icosphere, so the only error in play is the quadrature —
which is the point: a referee whose correctness is established by the codes it is
meant to referee is worth nothing, and this file is where that circularity is cut.

**The null-field identity is the sharper of the two.** For a target inside the
solute the exterior representation is identically zero, exactly, because the
single and double layers cancel term for term. Nothing has to be known about the
solution to assert it, and each way of getting the expression wrong breaks the
cancellation by a factor of hundreds.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from sashimi.analytic import born_potential
from sashimi.protocol import SolventModel, SurfacePotential
from sashimi.surface_field import Faces, evaluate_exterior, faces_of, solid_angle

RADIUS = 3.0
SOLVENT = SolventModel(solute_dielectric=2.0, solvent_dielectric=78.54, ionic_strength=0.0)

GATE_MESH = 4  # 5,120 faces; the whole file runs in well under a second
GATE_BAR = 0.005


def icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """A geodesic sphere of unit radius: icosahedron, subdivided, renormalised.

    Analytic, so the vertices lie on the sphere to machine precision and the only
    mesh error is the flat-triangle chord — unlike a mesher's output, where
    geometry and algorithm are entangled.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts: list[np.ndarray] = list(
        np.array(
            [
                [-1, phi, 0],
                [1, phi, 0],
                [-1, -phi, 0],
                [1, -phi, 0],
                [0, -1, phi],
                [0, 1, phi],
                [0, -1, -phi],
                [0, 1, -phi],
                [phi, 0, -1],
                [phi, 0, 1],
                [-phi, 0, -1],
                [-phi, 0, 1],
            ],
            dtype=np.float64,
        )
    )
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int64,
    )
    for _ in range(subdivisions):
        midpoint: dict[tuple[int, int], int] = {}
        new_faces: list[list[int]] = []
        for a, b, c in faces:
            corner: list[int] = []
            for i, j in ((a, b), (b, c), (c, a)):
                key = (min(int(i), int(j)), max(int(i), int(j)))
                if key not in midpoint:
                    midpoint[key] = len(verts)
                    verts.append((verts[int(i)] + verts[int(j)]) / 2.0)
                corner.append(midpoint[key])
            ab, bc, ca = corner
            new_faces += [[int(a), ab, ca], [int(b), bc, ab], [int(c), ca, bc], [ab, bc, ca]]
        faces = np.array(new_faces, dtype=np.int64)
    points = np.array(verts)
    return points / np.linalg.norm(points, axis=1)[:, None], faces


def born_surface(subdivisions: int) -> SurfacePotential:
    """Born's exact Cauchy data on an icosphere.

    `phi = C/a` on the surface and, on the *interior* side,
    `dphi/dn = -C/(eps_p a^2)` — the exterior value scaled by `eps_s/eps_p`,
    which is the relation the protocol type inverts.
    """
    unit, faces = icosphere(subdivisions)
    phi = np.full(len(unit), born_potential(RADIUS, 1.0, SOLVENT.solvent_dielectric))
    exterior = -phi / RADIUS
    ratio = SOLVENT.solvent_dielectric / SOLVENT.solute_dielectric
    return SurfacePotential(
        vertices=unit * RADIUS,
        values=phi,
        triangles=faces,
        interior_normal_derivative=exterior * ratio,
    )


def interior_points() -> np.ndarray:
    """Twenty points well inside the solute, one pinned at the centre."""
    rng = np.random.default_rng(7)
    directions = rng.normal(size=(19, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    radii = rng.uniform(0.0, RADIUS / 2.0, size=(19, 1))
    return np.vstack([np.zeros((1, 3)), directions * radii])


def residual(
    mesh: SurfacePotential, scale: float | None = None, faces: Faces | None = None
) -> float:
    """Max |evaluated| inside the solute, as a fraction of the surface potential.

    `scale` is separable so a mutation that zeroes the surface potential can
    still be graded against the unmutated magnitude.
    """
    values = evaluate_exterior(mesh, interior_points(), SOLVENT, faces=faces)
    denominator = scale if scale is not None else float(np.mean(np.abs(mesh.values)))
    return float(np.max(np.abs(values))) / denominator


def test_the_interior_null_field_holds():
    """B-G3, refereed by the representation itself rather than by a solver.

    Measured 9.6e-4 at 5,120 faces, so the bar sits about five times above it.
    The residual is the flat-triangle chord: a face centroid lies inside the
    sphere its vertices are on, which biases the single layer against the double
    one and leaves the cancellation slightly incomplete.
    """
    got = residual(born_surface(GATE_MESH))
    assert got <= GATE_BAR, f"interior residual {got:.2e} exceeds {GATE_BAR}"


def test_the_null_field_residual_is_quadrature_and_converges():
    """The stronger statement, and what says the bar above is not luck.

    A residual that is quadrature error falls four-fold per subdivision, because
    each splits every edge; one that is a defect in the expression does not fall
    at all. Measured 6.2e-2 / 1.5e-2 / 3.9e-3 / 9.6e-4 over 80 / 320 / 1,280 /
    5,120 faces.
    """
    ladder = [residual(born_surface(n)) for n in (1, 2, 3, 4)]
    assert ladder == sorted(ladder, reverse=True)
    for coarse, fine in pairwise(ladder):
        assert coarse / fine >= 3.0, (
            f"residual fell only {coarse / fine:.2f}x per subdivision ({ladder}); "
            "h^2 quadrature error falls 4x"
        )


def test_each_way_of_breaking_the_cancellation_is_caught():
    """The three defects the identity exists for, run rather than cited.

    Only that each mutation clears the bar is asserted, never what it reads: the
    values are artifacts of a sphere — a deleted double layer happens to leave
    about 1.0 here because the single layer alone reproduces the monopole — and
    pinning them would turn this into a characterisation test of the fixture.
    """
    mesh = born_surface(GATE_MESH)
    scale = float(np.mean(np.abs(mesh.values)))
    assert residual(mesh) <= GATE_BAR

    flipped = faces_of(mesh)
    flipped.normals = -flipped.normals
    assert residual(mesh, scale, faces=flipped) > GATE_BAR

    no_double_layer = SurfacePotential(
        vertices=mesh.vertices,
        values=np.zeros_like(mesh.values),
        triangles=mesh.triangles,
        interior_normal_derivative=mesh.interior_normal_derivative,
    )
    assert residual(no_double_layer, scale) > GATE_BAR

    # The side error: the stored array used as if it were the solvent-side one.
    stored = mesh.interior_normal_derivative
    assert stored is not None
    unconverted = SurfacePotential(
        vertices=mesh.vertices,
        values=mesh.values,
        triangles=mesh.triangles,
        interior_normal_derivative=stored
        * (SOLVENT.solvent_dielectric / SOLVENT.solute_dielectric),
    )
    assert residual(unconverted, scale) > GATE_BAR


def test_the_orientation_identity_needs_no_solver_either():
    """B-G1's binary-free half. Exact in the continuum for any closed surface,
    so the only error is the chord of a flat triangle."""
    faces = faces_of(born_surface(GATE_MESH))
    assert solid_angle(np.zeros(3), faces) == pytest.approx(-1.0, abs=0.02)
    assert solid_angle(np.array([0.0, 0.0, 30.0]), faces) == pytest.approx(0.0, abs=0.02)


def test_the_exterior_field_comes_back_from_exact_surface_data():
    """With no discretization in the Cauchy data the evaluator reproduces Born to
    the quadrature's own accuracy — which is what says the residual on a real mesh
    belongs to that mesh and not to this expression."""
    mesh = born_surface(GATE_MESH)
    rng = np.random.default_rng(3)
    directions = rng.normal(size=(60, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    got = evaluate_exterior(mesh, directions * (RADIUS + 2.0), SOLVENT)
    want = born_potential(RADIUS + 2.0, 1.0, SOLVENT.solvent_dielectric)
    assert got.mean() / want == pytest.approx(1.0, abs=0.005)
