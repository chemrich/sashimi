"""Exterior potential from a boundary-element answer, off the surface it was solved on.

**Why this exists.** Closed forms referee a field exactly, and this project has
two of them — `analytic.born_potential` and `analytic.kirkwood_potential` — but
both describe *one sphere with one charge*. Above two atoms the only referee
sashimi has ever used for a near field is either the same solver on a finer
lattice or another code that assigns the dielectric at the same face centres, and
APBS, DelPhi C++ and debye all do the latter, so none of them is independent of
the others. A boundary-element answer has no volumetric lattice at all. Evaluated
away from its own surface it is the one prospective referee here that shares no
construction with the finite-difference family — which is the still-open half of
the referee gap ROADMAP.md section 12 records.

**The representation.** With `n` outward from the solute, for a target `x` in the
solvent, Green's second identity on the *solvent* domain gives

    phi(x) = closed integral over Gamma of
             [ phi(s) dG/dn(s) - G(x, s) dphi_out/dn(s) ] dS

    G_0 = 1/(4 pi R)          G_kappa = exp(-kappa R)/(4 pi R)      R = |x - s|
    dG_kappa/dn(s) = (1 + kappa R) exp(-kappa R) [(x - s).n(s)] / (4 pi R^3)

Two things about that expression are worth stating because both were got wrong
before they were got right, and both fail *quietly*:

The sign. The identity holds with `n` outward from the domain containing `x`, and
the solvent domain's outward normal on the interface is the negative of the
solute's. Written with the solute's normals and the interior ordering
`[G dphi/dn - phi dG/dn]`, the whole evaluated field inverts.

The kernel carries no `eps_s`. The exterior problem is homogeneous — no free
charge, so `(grad^2 - kappa^2) phi = 0` — and the representation is therefore
purely geometric. The dielectric enters only through the Cauchy data. A
`1/(4 pi eps_s R)` kernel divides the answer by a further 78.54.

ROADMAP.md section 12 measured all three candidates against the exact Born
potential: the sign-flipped `eps_s`-weighted kernel reads -0.50395, that same
kernel with only the derivative convention repaired reads -0.01283, and the
expression above reads +1.00790. The first two are `-1/eps_p` and a
cancellation; a formula of this shape either reproduces a monopole or it does not.

**Orientation is derived and then checked, never assumed.** `tabipb.vtk` builds
its `SurfacePotential` with vertices, values and triangles and leaves `normals`
at None, and NanoShaper's winding convention is asserted nowhere. A globally
flipped winding inverts the field by the same order as the sign error above and
by a route nothing else here would catch. `solid_angle` is the check, and it is
refereed by the divergence theorem rather than by another solver.

**What the accuracy is limited by, which is not quadrature.** The evaluated
field's error equals TABI-PB's own Gauss-law flux error to three significant
figures, is bit-identical under a 16x-refined quadrature rule, and is flat from
`a + 0.5 A` to `a + 30 A` with the same sign in every direction. It is a spurious
monopole fixed by the discrete surface data: 1.278%, 0.790%, 0.502% and 0.338%
at mesh densities 3, 5, 8 and 12. **Neither refining the quadrature nor standing
further off does anything about it — the only knob is mesh density**, and a
consumer that needs a tighter referee has to pay for it there.
"""

from __future__ import annotations

import numpy as np

from sashimi.analytic import debye_length_a
from sashimi.protocol import DIMENSIONS, FloatArray, SolventModel, SurfacePotential

__all__ = ["Faces", "evaluate_exterior", "faces_of", "solid_angle"]


class Faces:
    """Per-triangle centroids, outward unit normals and areas.

    A small class rather than a tuple because the orientation is a *property that
    was established*, not an input, and `outward_from_winding` records whether the
    mesh's own winding already pointed out of the solute.
    """

    __slots__ = ("areas", "centroids", "normals", "outward_from_winding")

    def __init__(
        self,
        centroids: FloatArray,
        normals: FloatArray,
        areas: FloatArray,
        outward_from_winding: bool,
    ) -> None:
        self.centroids = centroids
        self.normals = normals
        self.areas = areas
        self.outward_from_winding = outward_from_winding


def solid_angle(point: FloatArray, faces: Faces) -> float:
    """`sum_t A_t (x - c_t).n_t / (4 pi R_t^3)` — the divergence theorem, discretized.

    Exactly `-1` for a point enclosed by a closed surface whose normals point
    outward, and exactly `0` for a point outside it. Nothing refereed it but the
    identity itself, which is what makes it the right check for a mesh whose
    winding convention is documented nowhere: a global flip reads `+1` against
    `-1`, mixed winding on a non-convex surface reads something in between, and a
    hole reads a fractional angle.
    """
    offset = point - faces.centroids
    distance = np.linalg.norm(offset, axis=1)
    projected = np.einsum("ij,ij->i", offset, faces.normals)
    return float(np.sum(faces.areas * projected / (4.0 * np.pi * distance**3)))


def faces_of(potential: SurfacePotential) -> Faces:
    """Triangle geometry with the normals oriented out of the solute.

    The winding gives a normal up to sign; `solid_angle` at an interior point
    fixes which sign that is. The mesh's vertex centroid is used as that interior
    point, which is sound for the closed surfaces a mesher produces — and if it
    ever is not, the identity reports a value near zero rather than near -1 and
    the caller sees a mesh it should not trust.
    """
    if potential.triangles is None:
        raise ValueError(
            "surface_field needs a triangulated surface; this SurfacePotential "
            "carries no `triangles`"
        )
    vertices = potential.vertices
    corners = potential.triangles
    a, b, c = (vertices[corners[:, i]] for i in range(DIMENSIONS))
    cross = np.cross(b - a, c - a)
    lengths = np.linalg.norm(cross, axis=1)
    if not np.all(lengths > 0.0):
        raise ValueError("surface_field found a degenerate triangle with zero area")

    faces = Faces(
        centroids=(a + b + c) / 3.0,
        normals=cross / lengths[:, None],
        areas=0.5 * lengths,
        outward_from_winding=True,
    )
    if solid_angle(vertices.mean(axis=0), faces) > 0.0:
        faces.normals = -faces.normals
        faces.outward_from_winding = False
    return faces


def evaluate_exterior(
    potential: SurfacePotential,
    targets: FloatArray,
    solvent: SolventModel,
    *,
    faces: Faces | None = None,
) -> FloatArray:
    """Potential at `targets`, kT/e, from the surface data alone.

    `targets` is `(N, 3)` in angstroms and must lie in the *solvent*: the
    representation returns the null field inside the solute, not the interior
    potential. Centroid quadrature with piecewise-constant data, so accuracy
    degrades as a target approaches the surface and the kernel's `1/R` sharpens
    — stand off at least a couple of angstroms, which is where the error was
    characterized and where it is flat.

    The exterior normal derivative comes from `SurfacePotential`, which stores the
    interior one, through the flux continuity the protocol type applies. A caller
    that reaches past that and feeds the stored array directly is wrong by
    `eps_s/eps_p`, and on a sphere that error is fully exposed — the
    constant-density double layer vanishes outside, so the single layer carries
    the whole answer and scales with it.
    """
    exterior = potential.exterior_normal_derivative(solvent)
    if exterior is None:
        raise ValueError(
            "surface_field needs the normal derivative; this SurfacePotential "
            "carries no `interior_normal_derivative`"
        )
    corners = potential.triangles
    assert corners is not None  # faces_of has already refused None
    geometry = faces if faces is not None else faces_of(potential)

    phi = potential.values[corners].mean(axis=1)
    dphi_dn = exterior[corners].mean(axis=1)
    kappa = (
        0.0
        if solvent.ionic_strength <= 0.0
        else 1.0
        / debye_length_a(solvent.ionic_strength, solvent.solvent_dielectric, solvent.temperature)
    )

    points = np.atleast_2d(np.asarray(targets, dtype=np.float64))
    out = np.empty(len(points), dtype=np.float64)
    for index, point in enumerate(points):
        offset = point - geometry.centroids
        distance = np.linalg.norm(offset, axis=1)
        decay = np.exp(-kappa * distance)
        projected = np.einsum("ij,ij->i", offset, geometry.normals)
        double_layer = (1.0 + kappa * distance) * decay * projected / (4.0 * np.pi * distance**3)
        single_layer = decay / (4.0 * np.pi * distance)
        out[index] = float(np.sum(geometry.areas * (phi * double_layer - dphi_dn * single_layer)))
    return out
