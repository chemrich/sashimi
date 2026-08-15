"""Closed-form solutions, for the handful of geometries that have them.

Almost nothing in continuum electrostatics is analytically solvable, which is
why Poisson-Boltzmann solvers exist. The exceptions are worth a module: they are
the only references in this project that are *right* rather than merely
*recorded*, and they are what distinguishes "APBS has not changed" from "APBS is
correct". The golden corpus needs both — a backend can reproduce a wrong number
forever.

Charged sphere (Born, 1920). A point charge at the centre of a sphere of radius
`a` and dielectric `eps_p`, in a solvent of dielectric `eps_s`:

    dG = -(q^2 / 8 pi eps0 a) (1/eps_p - 1/eps_s)

With mobile ions the Debye-Huckel treatment adds a screening term. The two agree
at zero ionic strength by construction, so a case at zero salt exercises the
same expression either way.

These are cheap to evaluate and expensive to get wrong, so each is derived in a
comment rather than quoted, and `tests/test_analytic.py` checks them against the
solvers rather than the other way round. That file needs APBS, so it says nothing
on a bare checkout and nothing about whether two expressions here agree with
*each other*; `tests/test_analytic_closed_forms.py` is the internal half, and it
exists because M3 added a second salted expression that has to be the same
matching condition as the first.
"""

from __future__ import annotations

import math

from sashimi.constants import (
    ANGSTROM,
    AVOGADRO,
    BOLTZMANN,
    ELEMENTARY_CHARGE,
    JOULES_PER_KJ,
    VACUUM_PERMITTIVITY,
)

__all__ = [
    "born_potential",
    "born_solvation_energy",
    "debye_length_a",
    "kirkwood_solvation_energy",
    "screened_born_potential",
    "screened_born_solvation_energy",
]


def born_solvation_energy(
    radius_a: float,
    charge_e: float = 1.0,
    solute_dielectric: float = 1.0,
    solvent_dielectric: float = 78.54,
) -> float:
    """Polar solvation free energy of a charged sphere, kJ/mol.

    dG = -(q^2 / 8 pi eps0 a) (1/eps_p - 1/eps_s)

    The APBS `examples/born` README states -230.62 kJ/mol for the 3 A, +1e case;
    this expression with CODATA 2018 constants gives -228.61. The 0.87% gap is
    the constants, not the physics, which is why sashimi pins its own value from
    named constants rather than quoting a number.
    """
    if radius_a <= 0:
        raise ValueError(f"radius must be positive, got {radius_a}")
    q = charge_e * ELEMENTARY_CHARGE
    a = radius_a * ANGSTROM
    joules = -(q**2 / (8 * math.pi * VACUUM_PERMITTIVITY * a)) * (
        1.0 / solute_dielectric - 1.0 / solvent_dielectric
    )
    return joules * AVOGADRO / JOULES_PER_KJ


def debye_length_a(
    ionic_strength: float, solvent_dielectric: float = 78.54, temperature: float = 298.15
) -> float:
    """Debye screening length in angstroms for a 1:1 salt; infinite at zero salt.

    kappa^2 = 2 N_A e^2 I / (eps0 eps_s k T). At 0.15 M and 298.15 K this is
    7.86 A, which is the number a units mistake fails on.
    """
    if ionic_strength <= 0:
        return math.inf
    number_density = ionic_strength * 1000.0 * AVOGADRO  # per cubic metre
    kappa_squared = (
        2.0
        * number_density
        * ELEMENTARY_CHARGE**2
        / (VACUUM_PERMITTIVITY * solvent_dielectric * BOLTZMANN * temperature)
    )
    return 1.0 / (math.sqrt(kappa_squared) * ANGSTROM)


def screened_born_solvation_energy(
    radius_a: float,
    charge_e: float = 1.0,
    solute_dielectric: float = 1.0,
    solvent_dielectric: float = 78.54,
    *,
    ionic_strength: float = 0.0,
    temperature: float = 298.15,
    ion_radius: float = 2.0,
) -> float:
    """Born energy with the Debye-Huckel ionic term, kJ/mol.

    Mobile ions cannot approach closer than `a + ion_radius` — the ion-exclusion
    radius the solvers call the Stern layer — so the screening term is evaluated
    there rather than at the dielectric boundary:

        dG_ion = -(q^2 / 8 pi eps0 eps_s) * kappa / (1 + kappa b),  b = a + r_ion

    Reduces to `born_solvation_energy` exactly at zero ionic strength, where
    kappa is zero. This is the *linearized* result, so it is only a reference for
    the linearized equation — which is the only one sashimi solves.

    **How tight a reference this is depends on the surface model, which is not
    what it looked like until M3 measured it.** On the 3 A, +1e ion at 0.15 M
    the ionic contribution -- G(I) - G(0), the only part of the total this
    expression's screening term describes -- reads, against an exact -0.6880:

        APBS `molecular`            -0.6878 / -0.7766 / -0.7155  at 0.5/0.25/0.125 A
        APBS `smoothed-molecular`   -0.6880 / -0.7087 / -0.7520
        APBS `van-der-waals`        -0.6878 / -0.6878 / -0.6879
        DelPhi C++ `van-der-waals`  -0.4958 at every resolution
        debye                       -0.6887 / -0.6882

    So the ~10% scatter this docstring used to attribute to the *quantity* --
    "a small difference of two large numbers, so it carries grid noise" -- is a
    property of the **probe-based surfaces**. Where the probe is zero, APBS
    reproduces the closed form to 0.03% at every spacing, and debye to 0.14%.
    The likely mechanism, offered as a suggestion rather than a measurement: with
    a probe the dielectric boundary and the ion-exclusion boundary are two
    differently-constructed surfaces whose discretizations move apart with h,
    where at `srad 0` both are bare staircases that scale together.

    DelPhi's stays resolution-independent to four decimals and lands 39% away,
    which says it computes the ionic term semi-analytically rather than from the
    grid. Both codes report `polar-solvation`, so this is not the `EnergyTerm`
    gap of section 12; it is a different ion-exclusion convention underneath the
    same declared quantity. That disagreement is real and is why the corpus
    declines a closed form for `born-ion-salt` -- but it is DelPhi's alone, not
    a coin flip between two conventions: two codes sharing no source land on this
    expression to better than 0.2%.

    **What it is still not a reference for is the total.** The ionic term is
    0.3% of the solvation energy where discretization is 1.6%, so a check of the
    *total* against this expression cannot see the salt at all: every mutation of
    debye's screening tried at M3 -- including deleting the Boltzmann term
    outright -- leaves the total within -1.40% to -1.76% of it, inside a band
    APBS itself needs 2.4% for. A salted case is graded on the difference between
    two recordings, and section 12's M3 records why.
    """
    unscreened = born_solvation_energy(radius_a, charge_e, solute_dielectric, solvent_dielectric)
    if ionic_strength <= 0:
        return unscreened

    kappa = 1.0 / debye_length_a(ionic_strength, solvent_dielectric, temperature)  # 1/A
    exclusion = (radius_a + ion_radius) * ANGSTROM
    kappa_si = kappa / ANGSTROM
    q = charge_e * ELEMENTARY_CHARGE
    joules = -(q**2 / (8 * math.pi * VACUUM_PERMITTIVITY * solvent_dielectric)) * (
        kappa_si / (1.0 + kappa_si * exclusion)
    )
    return unscreened + joules * AVOGADRO / JOULES_PER_KJ


def kirkwood_solvation_energy(
    radius_a: float,
    offset_a: float,
    charge_e: float = 1.0,
    solute_dielectric: float = 1.0,
    solvent_dielectric: float = 78.54,
    *,
    terms: int = 400,
) -> float:
    """Charge off-centre in a sphere: the Kirkwood series (1934), kJ/mol.

    A point charge at distance `d` from the centre of a sphere of radius `a`.
    Born is the d = 0 special case, and the rest of the series is the whole
    multipole structure of the reaction field — which a centred charge cannot
    exercise, because every term above the monopole vanishes:

        dG = (q^2 / 8 pi eps0 a) * sum_n
                 (n+1)(eps_p - eps_s) / (eps_p (n eps_p + (n+1) eps_s)) * (d/a)^2n

    Why it earns its place: the Born ion tests that a solver gets one number
    right, and it is symmetric in every way a solver could be wrong about
    direction. This tests charge *placement* — the term that grows as the charge
    approaches the boundary, where the discretization is worst and where real
    solutes keep their charges.

    Verified against APBS at 0.25 A on a 3 A sphere: 0.113%, 0.097%, 0.473% and
    0.114% at d/a of 0, 0.3, 0.5 and 0.7. The n = 0 term reproduces
    `born_solvation_energy` exactly, which is the check that the series was
    transcribed correctly.

    The series converges geometrically in (d/a)^2, so `terms` only matters as
    the charge nears the surface; 400 is far past convergence for d/a <= 0.9.
    """
    if not 0.0 <= offset_a < radius_a:
        raise ValueError(
            f"the charge must sit inside the sphere: got offset {offset_a} A "
            f"in a sphere of radius {radius_a} A"
        )
    ratio = (offset_a / radius_a) ** 2
    series = sum(
        (n + 1)
        * (solute_dielectric - solvent_dielectric)
        / (solute_dielectric * (n * solute_dielectric + (n + 1) * solvent_dielectric))
        * ratio**n
        for n in range(terms)
    )
    q = charge_e * ELEMENTARY_CHARGE
    joules = q**2 / (8 * math.pi * VACUUM_PERMITTIVITY * radius_a * ANGSTROM) * series
    return joules * AVOGADRO / JOULES_PER_KJ


def born_potential(
    r_a: float,
    charge_e: float = 1.0,
    solvent_dielectric: float = 78.54,
    temperature: float = 298.15,
) -> float:
    """Potential at distance r outside the sphere, in kT/e.

    Valid only for r > a: at the dielectric boundary itself the smoothed surface
    makes the grid value diverge from this by ~70%, and at r = 0 the point
    charge is singular.

    Unscreened, so it describes a case only at zero ionic strength — at 0.15 M
    it overstates the potential two cells outside a 3 A sphere by about 30% and
    eight cells out by about half. Sampling is at `a + k*h`, so the exact figure
    depends on the backend's achieved spacing: 29.7% and 47.9% on APBS's
    0.40625 A, 31.1% and 52.6% on DelPhi C++'s 0.5 A.
    `screened_born_potential` is the salted expression, and
    `sashimi.corpus.AnalyticField` picks between them from the case's solvent
    rather than leaving the choice to a caller.
    """
    q = charge_e * ELEMENTARY_CHARGE
    r = r_a * ANGSTROM
    volts = q / (4 * math.pi * VACUUM_PERMITTIVITY * solvent_dielectric * r)
    kt_over_e = BOLTZMANN * temperature / ELEMENTARY_CHARGE
    return volts / kt_over_e


def screened_born_potential(
    r_a: float,
    charge_e: float = 1.0,
    solvent_dielectric: float = 78.54,
    temperature: float = 298.15,
    *,
    radius_a: float,
    ionic_strength: float = 0.0,
    ion_radius: float = 2.0,
) -> float:
    """Potential outside a charged sphere in salt, with a Stern layer, in kT/e.

    The third closed form in this project, and the first on the *field* under
    salt. Mobile ions are excluded inside `b = radius_a + ion_radius`, so there
    are two regions and they are not the same expression. Solving Poisson
    between the dielectric boundary and the Stern radius, the linearized PB
    equation beyond it, and matching phi and eps dphi/dr at b:

        a < r <= b:   phi(r) = q / (4 pi eps0 eps_s) * [1/r - kappa/(1 + kappa b)]
            r >= b:   phi(r) = q / (4 pi eps0 eps_s) * exp(-kappa (r - b))
                                                      / ((1 + kappa b) r)

    Both branches agree at r = b, and so do their derivatives: eps does not jump
    there — only the Boltzmann coefficient does — so the kink is in the second
    derivative. That is why a sample near the Stern radius is a well-posed
    question where a sample near the *dielectric* boundary is the O(1)-wrong one
    `sashimi.corpus.AnalyticField` documents.

    The inner branch is the unscreened Born potential shifted by a constant, and
    that constant is exactly twice the ionic term of
    `screened_born_solvation_energy` — the reaction potential at the charge is
    the same quantity read at a point rather than integrated.
    `tests/test_analytic_closed_forms.py` asserts that identity, which is the
    check that both expressions were transcribed from the same matching
    conditions. Not `tests/test_analytic.py`, which is marked `apbs` and grades
    this module against a solver: an identity between two expressions here is
    exactly what a solver cannot arbitrate.

    Measured against the solvers on the 3 A van der Waals sphere at 0.15 M,
    worst over the eight sampled directions, two cells out: debye 6.38%, APBS
    4.51%, DelPhi C++ 1.72%. Those relative numbers are larger than the
    zero-salt ones mostly because the screened potential they divide by is
    smaller: in **absolute** terms, at the sample nearest the boundary, debye
    reads 0.0812 kT/e at zero salt, 0.0805 at 0.15 M and 0.0798 at 0.5 M — the
    same discretization error, on the same lattice, at the same radius. The
    agreement loosens with distance (-6.6% and -14.3% at eight cells out, on
    errors of 0.006 kT/e), so the statement to carry is the near-boundary one:
    where the discretization error lives, the Boltzmann term adds none of its
    own.
    """
    # Before the zero-salt shortcut, not after: a guard that only runs on one
    # branch means the same invalid radius raises `ValueError` at 0.15 M and
    # `ZeroDivisionError` at zero, which is two failure modes chosen by a
    # parameter unrelated to what is wrong.
    if r_a <= 0:
        raise ValueError(f"radius must be positive, got {r_a}")
    if ionic_strength <= 0:
        return born_potential(r_a, charge_e, solvent_dielectric, temperature)

    kappa = 1.0 / debye_length_a(ionic_strength, solvent_dielectric, temperature)  # 1/A
    exclusion = radius_a + ion_radius
    q = charge_e * ELEMENTARY_CHARGE
    # Everything below is in angstroms, so the SI prefactor carries one 1/ANGSTROM.
    prefactor = q / (4 * math.pi * VACUUM_PERMITTIVITY * solvent_dielectric * ANGSTROM)
    if r_a <= exclusion:
        volts = prefactor * (1.0 / r_a - kappa / (1.0 + kappa * exclusion))
    else:
        volts = prefactor * math.exp(-kappa * (r_a - exclusion)) / ((1.0 + kappa * exclusion) * r_a)
    kt_over_e = BOLTZMANN * temperature / ELEMENTARY_CHARGE
    return volts / kt_over_e
