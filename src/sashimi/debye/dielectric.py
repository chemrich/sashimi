"""Geometry to coefficients: where the solvent is, and how strongly it screens.

Two maps, and they are not the same map. The **dielectric** boundary is the van
der Waals surface — the union of the atomic spheres — and it lives on the faces
between nodes, because a flux through a face is what the finite-volume operator
integrates. The **ion-accessible** region is further out: mobile ions have a
radius of their own and cannot approach the solute closer than that, so the
Boltzmann term switches on at the union of spheres inflated by `ion_radius`.
Both incumbents make this distinction and it is invisible at zero salt, which
is exactly the configuration M1 is graded on — so it is written now rather than
discovered at M3.

The dielectric is sampled at face centres rather than averaged over the face.
**That is the same choice APBS makes with `srfm mol`, and remembering so is not
trivia — it disqualifies APBS as the reference for any experiment on this line.**
From M1 until M1c this docstring called a volume-fraction average "the obvious
place to look". M1c looked; ROADMAP.md section 12, "M1c — the spike ran", carries
the numbers and the retraction. The short version:

- Smoothing the dielectric over a band of `w` cells damps the grid-phase
  oscillation M1b measured far less than the summary statistic suggests. The
  *swing ratio* falls 5.35x -> 1.56x mostly because the best configurations get
  worse (0.773% -> 1.975%); the **worst** near-field error, which is what a
  consumer sees, moves only 4.138% -> 3.085%. **That is the whole reason M4a was
  dropped** — a quarter is not worth two to three days on an axis where debye
  already matches both incumbents.
- Blended arithmetically it wrecks the energy: 0.853% -> 3.545% at 0.25 A.
  Blended **harmonically** — the textbook mean for flux normal to a layered
  interface — it made the Born energy **8x better**: 0.853% -> 0.107% at 0.25 A,
  monotonic, no sign flips, across the whole ladder.
- That energy result is **open, and the evidence for it has since strengthened.**
  It was first written up as a fixture artifact, on the strength of real-structure
  energies drifting from APBS (ALA-GLY -0.409% -> +3.565%, barnase -1.102% ->
  +5.153% at w = 1) — and APBS makes the same hard assignment under test, so that
  comparison cannot referee it. A reference-free refinement study on ALA-GLY then
  found all three schemes extrapolating to within 0.25% of one limit, with
  harmonic reaching it several times sooner: at h = 0.13 A hard is still 1.1% out
  where harmonic is inside 0.16%.

Both bullets above are graded against exact references — the Born potential and
the Born energy respectively. What separates them is that the *field* result
needs no other evidence, while the *energy* result's real-geometry check had to
be redone once the first reference turned out to share the bias under test.

So the face-centre sample stays, on the field axis, which is the axis debye's
consumer reads. **Anything that revisits this needs a reference that does not
itself discretize a volumetric dielectric** — TABI-PB, a closed form, or a truly
converged grid. Reaching for APBS or DelPhi here measures a shared bias.
"""

from __future__ import annotations

import math

import numpy as np

from sashimi.analytic import debye_length_a
from sashimi.constants import (
    ANGSTROM,
    BOLTZMANN,
    ELEMENTARY_CHARGE,
    VACUUM_PERMITTIVITY,
)
from sashimi.debye.grid import DebyeGrid, axis_coordinates
from sashimi.debye.surface import ReducedSurface, dilate, inside_union_of_spheres
from sashimi.protocol import DIMENSIONS, FloatArray, PQRData, SolventModel, SurfaceModel

__all__ = [
    "bjerrum_length_a",
    "dielectric_faces",
    "inside_union_of_spheres",
    "screening_nodes",
]


def bjerrum_length_a(temperature: float) -> float:
    """e^2 / (4 pi eps0 kT), in angstroms: the vacuum Bjerrum length.

    The one constant that converts this module's charges into this module's
    potentials. In water at 298.15 K the familiar number is 7.14 A, which is
    this divided by 78.54 — the dielectric is not folded in here because the
    solver carries it in the operator, where it varies with position.

    Built from `sashimi.constants` rather than quoted, for the reason that
    module exists: the Born closed form is computed from the same CODATA 2018
    values, so a solver that agrees with it to six digits is agreeing about the
    physics rather than about a rounding.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    metres = ELEMENTARY_CHARGE**2 / (4.0 * math.pi * VACUUM_PERMITTIVITY * BOLTZMANN * temperature)
    return metres / ANGSTROM


def _surface_gap(axes: list[FloatArray], structure: PQRData, reach: float) -> FloatArray:
    """Signed distance to the van der Waals surface, clamped beyond `reach`.

    `min_i(|x - c_i| - r_i)`, which is exact for a union of spheres and is why
    this is offered on that boundary and refused on the other one. Computed over
    each sphere's own index window, like `inside_union_of_spheres` — anything
    further than `reach` from every atom is solvent by a margin the ramp does
    not care about, so it keeps the clamp and is never visited.
    """
    shape = tuple(len(axis) for axis in axes)
    gap = np.full(shape, reach, dtype=np.float64)
    for centre, radius in zip(structure.coords, structure.radii, strict=True):
        window = []
        for axis in range(DIMENSIONS):
            span = radius + reach
            lo = int(np.searchsorted(axes[axis], centre[axis] - span, side="left"))
            hi = int(np.searchsorted(axes[axis], centre[axis] + span, side="right"))
            window.append(slice(lo, hi))
        if any(w.start >= w.stop for w in window):
            continue
        offsets = [(axes[axis][window[axis]] - centre[axis]) ** 2 for axis in range(DIMENSIONS)]
        squared = offsets[0][:, None, None] + offsets[1][None, :, None] + offsets[2][None, None, :]
        np.minimum(gap[tuple(window)], np.sqrt(squared) - radius, out=gap[tuple(window)])
    return gap


def dielectric_faces(
    grid: DebyeGrid,
    structure: PQRData,
    solvent: SolventModel,
    surface: ReducedSurface | None = None,
    smoothing: float = 0.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Dielectric at the face centres, one array per axis.

    `faces[axis]` has the grid's shape with `axis` one shorter: entry (i, j, k)
    of `faces[0]` is the dielectric halfway between nodes (i, j, k) and
    (i+1, j, k), which is the coefficient of the flux the operator sums there.

    Which boundary is asked for is `surface.inside_solute`'s business as of M4.
    This function knew it was a union of spheres from M1 until then, and the
    swap is one call because ROADMAP.md section 12 said to build the seam early.

    **The uniform-dielectric state never asks where the solute is.** Every
    energy is two solves differenced, and the reference one sets
    `solvent_dielectric = solute_dielectric` — so `np.where` below would pick
    between two equal numbers at every face. Returning the constant directly is
    an identity rather than an approximation, and it matters because the
    geometry is no longer free: at M4 the solvent-excluded surface was being
    built and thrown away on three staggered lattices at every multigrid level
    of a state whose answer cannot depend on it.
    """
    if solvent.solute_dielectric == solvent.solvent_dielectric:
        return tuple(  # type: ignore[return-value]
            np.full(
                tuple(n - 1 if axis == staggered else n for axis, n in enumerate(grid.shape)),
                solvent.solute_dielectric,
                dtype=np.float64,
            )
            for staggered in range(DIMENSIONS)
        )

    # One surface for the three staggered lattices: they differ in where the
    # nodes are, not in where the solute is.
    surface = surface or ReducedSurface(structure, solvent)
    faces = []
    for axis in range(DIMENSIONS):
        axes = axis_coordinates(grid, staggered=axis)
        if smoothing > 0.0:
            # **Harmonic, not arithmetic**, and **sub-cell, not a band.** The
            # mean is harmonic because that is the textbook one for flux normal
            # to a layered interface, which is what a face coefficient in a
            # finite-volume operator is; blended arithmetically M1c measured the
            # Born energy going 0.853% -> 3.545%.
            #
            # The width is where a first attempt at this went wrong and the
            # measurement is in ROADMAP.md section 12. Averaging the *indicator*
            # over a box of whole cells smears the interface across three of
            # them, and on the Born closed form that is **worse than the hard
            # assignment at every rung but one**. Ramping the fraction across a
            # single cell from the exact signed distance is 5-8x *better* than
            # hard instead.
            # **Clamped, because `smoothing` is in cells and a coarse level's
            # cell is not the fine one's.** `build_levels` re-discretizes at
            # every multigrid level and hands each the same cell count, so at a
            # 1.0 A request the coarsest level's spacing is 6.4 A and a 0.25-cell
            # ramp is 1.6 A wide — wider than the probe, past which the gap field
            # saturates and the ramp stops being a ramp: measured, solvent faces
            # came back at 23.0 instead of 78.54. The probe is the range the
            # distance carries, so it is the width's ceiling.
            width = min(smoothing * float(min(grid.spacing)), solvent.surface_radius)
            # **The distance is the surface's own, not a stand-in for it.** M8
            # shipped this for `van-der-waals` only, where `min(|x - c| - r)` is
            # exact; M8a gives the solvent-excluded surface a signed distance of
            # its own, out of the same three families that decide `inside`. Both
            # go through `signed_gap`, so the ramp no longer knows or cares which
            # boundary it is ramping across.
            gap = surface.signed_gap(axes)
            fraction = np.clip(0.5 - gap / (2.0 * width), 0.0, 1.0)
            eps = 1.0 / (
                fraction / solvent.solute_dielectric + (1.0 - fraction) / solvent.solvent_dielectric
            )
        else:
            eps = np.where(
                surface.inside(axes), solvent.solute_dielectric, solvent.solvent_dielectric
            )
        faces.append(np.ascontiguousarray(eps, dtype=np.float64))
    return faces[0], faces[1], faces[2]


def screening_nodes(
    grid: DebyeGrid,
    structure: PQRData,
    solvent: SolventModel,
    surface: ReducedSurface | None = None,
) -> tuple[FloatArray, float]:
    """The Boltzmann term's coefficient at each node, and the bulk value it takes.

    Returns `eps_s * kappa^2` in 1/A^2 — zero inside the ion-exclusion region,
    bulk outside it. Zero everywhere at zero ionic strength, which is every case
    M1 is graded on; the array is still built, because a solver that only works
    at zero salt is not a Poisson-Boltzmann solver and would not say so.

    The exclusion radius is the atomic radius plus `ion_radius`, not plus the
    solvent probe: the ion is the thing being excluded. `sashimi.analytic`'s
    screened Born expression evaluates its screening term at `a + ion_radius`
    for the same reason, so the two agree about what the Stern layer is. M3
    graded that: on the van der Waals sphere debye's ionic contribution is
    within 0.14% of the closed form and 0.22% of APBS.

    **M4 changes what the Stern layer sits around.** It is the region within
    `ion_radius` of the *solute*, and on a molecular surface the solute is no
    longer the union of spheres — so the exclusion is the solvent-excluded
    volume dilated by `ion_radius`, not a union inflated by it. The two
    coincide exactly on `van-der-waals`, which is what keeps M3's measurements
    describing the same quantity, and `tests/test_debye_m4.py` asserts the
    coincidence rather than assuming it. The union path stays for that surface
    because it is exact and needs no dilation, where the general one quantises
    the offset to the lattice.
    """
    if solvent.ionic_strength <= 0.0:
        return np.zeros(grid.shape, dtype=np.float64), 0.0

    kappa = 1.0 / debye_length_a(
        solvent.ionic_strength, solvent.solvent_dielectric, solvent.temperature
    )
    bulk = solvent.solvent_dielectric * kappa * kappa  # 1/A^2

    axes = axis_coordinates(grid)
    if solvent.surface_model is SurfaceModel.MOLECULAR:
        solute = (surface or ReducedSurface(structure, solvent)).inside(axes)
        excluded = dilate(solute, grid.spacing, solvent.ion_radius)
    else:
        excluded = inside_union_of_spheres(
            axes, structure.coords, structure.radii + solvent.ion_radius
        )
    return np.where(excluded, 0.0, bulk), bulk
