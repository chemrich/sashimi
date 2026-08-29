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
- That energy result was **open for two milestones and is now settled the ramp's
  way.** It was first written up as a fixture artifact, on the strength of
  real-structure energies drifting from APBS (ALA-GLY -0.409% -> +3.565%,
  barnase -1.102% -> +5.153% at w = 1) — and APBS makes the same hard assignment
  under test, so that comparison cannot referee it. A reference-free refinement
  study on ALA-GLY then found all three schemes extrapolating to within 0.25% of
  one limit, with harmonic reaching it several times sooner. Measured properly
  on 2026-08-25 against the shared limit of a four-rung ALA-GLY ladder: at
  0.4545 A the hard scheme is 10.34 kJ/mol out and the ramp at w = 0.5 is 2.13,
  **4.9x closer**, and 8.8x closer at 0.2432 A.

**And the field axis, which this file has named as the precondition since M1,
has now been measured — and the ramp does not win it.** Same fixture, same
surface, same lattices, RMS over a shell 2-3 A outside the solvent-accessible
surface against each scheme's own refined solve: on ALA-GLY, refereed at 4x the
coarse spacing, the ramp is **2.4-13x further from the referee than the hard
assignment at w >= 0.75**, growing monotonically with the width. `fas2` agrees in
direction and its referees are too coarse to count. At w = 0.5 every referee at
h <= 0.15 says worse by 1.3-2.9x and a merely 2x-finer one says better, so that
width is *bounded rather than settled* — a referee sharing a construction with a
candidate sits nearest that candidate. And on the Born sphere with an exact
reference the two summaries disagree: the worst-direction error at a radius
improves 12-23% while by shell RMS the median is unchanged and the floor
doubles, so there is no clean gain there either.

**The axis is bounded rather than discharged**, and what would discharge it is a
reference that is not debye at all.

**So the two axes disagree, and this file's own sentence is why that matters:**
the field is the axis debye's consumer reads. `DebyeOptions.dielectric_smoothing`
stays off by default for that reason rather than for a coverage one — and the
face-centre sample stays the default with it. **Anything that revisits this needs
a reference that does not itself discretize a volumetric dielectric** — TABI-PB,
a closed form, or a truly converged grid; refinement is admissible here only
because the ramp's band is `w*h` and vanishes with `h`, so the two schemes share
a continuum limit. Reaching for APBS or DelPhi measures a shared bias.
ROADMAP.md section 12, "The field axis, measured", carries the tables.
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
            # shipped this for `van-der-waals` only, on `min(|x - c| - r)`; M8a
            # gives the solvent-excluded surface a signed distance of its own,
            # out of the same three families that decide `inside`. Both go
            # through `signed_gap`, so the ramp no longer knows or cares which
            # boundary it is ramping across.
            # *`min(|x - c| - r)` was called exact here and is not: it is exact
            # outside a union of spheres and an upper bound inside it, so the
            # van der Waals ramp was reading a bound as a depth on 19% of its
            # interior band faces. `ReducedSurface._union_signed_gap` measures
            # to the union's own rims and seats instead.*
            # `band=width` because the `clip` on the next line is what makes the
            # field outside it unobservable — the two numbers must stay the same
            # number, which is why it is passed rather than defaulted.
            gap = surface.signed_gap(axes, band=width)
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
    volume dilated by `ion_radius`, not a union inflated by it.

    **M3a: above the probe those are the same region, so the exact test is
    taken on both.** For any dilation radius `r >= probe`,
    `dilate(SES, r) == dilate(vdw_union, r)` exactly, in the continuum. *Proof.*
    Take `x` within `r` of an SES point `s`. Because `s` is solute, every `w`
    with `|w - s| <= probe` has `dist(w, vdw) < probe` — otherwise `w` is a legal
    probe centre whose ball covers `s`, making `s` solvent. If `|x - s| <=
    probe`, take `w = x`. Otherwise take `w = s + probe*(x - s)/|x - s|`, so
    `|x - w| <= r - probe`; `dist(., vdw)` is 1-Lipschitz, so
    `dist(x, vdw) < probe + (r - probe) = r`. Either way `x` is in
    `dilate(vdw, r)`, and the reverse inclusion is free because vdw is inside
    SES. **`r >= probe` is used exactly once and is sharp.**

    So the union test — exact, and needing no lattice dilation — is correct for
    `molecular` too whenever `ion_radius >= surface_radius`, which every shipped
    case satisfies at 2.0 against 1.4. What that buys is not tidiness: `dilate`
    quantises its reach to the lattice, so its accuracy is decided by whether
    `ion_radius` happens to be commensurate with the achieved spacing. Measured
    across nineteen spacings between 0.82 and 1.00 A, the dilated path lands
    outside 2% on twelve of them and the union path on none; two spacings 0.8%
    apart, 0.8929 and 0.9000 A, differ four-fold in error because 2.0 A is
    representable on one lattice and not the other. `dilate` also degrades to
    the identity once `ion_radius < min(spacing)`, and `build_levels`
    re-discretizes at every level, so the coarse levels carried no Stern layer
    at all.

    **Below the probe the dilated path stays, because there the union is not
    less accurate but wrong.** At `ion_radius = 0` — the standard "no Stern
    layer" request, which `dilate` serves exactly, since a zero-radius ball is
    the identity — the union would leave 13.5% of fas2's solute nodes carrying
    bulk screening *inside* the dielectric body. Falling back rather than
    raising is deliberate: raising would turn a working request into an
    exception.

    *An earlier version of this docstring said the two constructions "coincide
    exactly on `van-der-waals`" and credited `tests/test_debye_m4.py` with
    asserting it. The second half was false — no test in that file calls this
    function — and the first was true only in the continuum, where the shipped
    lattice `dilate` disagreed with the exact union on 18.8% of `ala-gly`'s
    excluded nodes at 0.87 A. `tests/test_debye_m3a.py` asserts what this one
    now claims.*
    """
    if solvent.ionic_strength <= 0.0:
        return np.zeros(grid.shape, dtype=np.float64), 0.0

    kappa = 1.0 / debye_length_a(
        solvent.ionic_strength, solvent.solvent_dielectric, solvent.temperature
    )
    bulk = solvent.solvent_dielectric * kappa * kappa  # 1/A^2

    axes = axis_coordinates(grid)
    if (
        solvent.surface_model is SurfaceModel.MOLECULAR
        and solvent.ion_radius < solvent.surface_radius
    ):
        # Below the probe the two constructions genuinely diverge and this is
        # the correct one. Above it they are the same region, so the branch
        # below is taken instead: exact, and free of the lattice quantisation.
        solute = (surface or ReducedSurface(structure, solvent)).inside(axes)
        excluded = dilate(solute, grid.spacing, solvent.ion_radius)
    else:
        excluded = inside_union_of_spheres(
            axes, structure.coords, structure.radii + solvent.ion_radius
        )
    return np.where(excluded, 0.0, bulk), bulk
