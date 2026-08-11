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
solvers rather than the other way round.
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

__all__ = ["born_potential", "born_solvation_energy", "screened_born_solvation_energy"]


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

    **Not a tight reference, and the reason is a real disagreement between the
    solvers rather than a defect in this expression.** On the 3 A, +1e ion at
    0.15 M, the measured ionic contribution is:

        this expression   -0.688 kJ/mol
        APBS              -0.688 / -0.777 / -0.694 at 0.5 / 0.25 / 0.125 A
        DelPhi C++        -0.496 at all three resolutions

    APBS straddles it — the shift is a small difference of two large numbers, so
    it carries grid noise of order 10% even where the energies themselves have
    converged. DelPhi's is resolution-independent to four decimal places, which
    says it computes the ionic term semi-analytically rather than from the grid,
    and lands 39% away. Both codes report `polar-solvation`, so this is not the
    `EnergyTerm` gap of section 12; it is a different ion-exclusion convention
    underneath the same declared quantity.

    Use it as a ~10% sanity anchor for the APBS convention, not as a corpus
    reference. Zero-salt cases are exact and are what the corpus checks against.
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
    """
    q = charge_e * ELEMENTARY_CHARGE
    r = r_a * ANGSTROM
    volts = q / (4 * math.pi * VACUUM_PERMITTIVITY * solvent_dielectric * r)
    kt_over_e = BOLTZMANN * temperature / ELEMENTARY_CHARGE
    return volts / kt_over_e
