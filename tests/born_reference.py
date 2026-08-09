"""Closed-form Born ion, computed here rather than quoted.

The APBS `examples/born` README states an analytic solvation energy of
-230.62 kJ/mol; the expression below with current CODATA constants gives
-228.61. That 0.87% spread straddles a tight tolerance, so sashimi pins its own
value from named constants and asserts at 1%. See PLAN.md section 7.
"""

from __future__ import annotations

import math

# CODATA 2018.
ELEMENTARY_CHARGE = 1.602176634e-19  # C
VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m
AVOGADRO = 6.02214076e23  # 1/mol
BOLTZMANN = 1.380649e-23  # J/K

ANGSTROM = 1e-10  # m


def born_solvation_energy(
    radius_a: float,
    charge_e: float = 1.0,
    solute_dielectric: float = 1.0,
    solvent_dielectric: float = 78.54,
) -> float:
    """Polar solvation free energy of a charged sphere, kJ/mol.

    dG = -(q^2 / 8 pi eps0 a) (1/eps_p - 1/eps_s)
    """
    q = charge_e * ELEMENTARY_CHARGE
    a = radius_a * ANGSTROM
    joules = -(q**2 / (8 * math.pi * VACUUM_PERMITTIVITY * a)) * (
        1.0 / solute_dielectric - 1.0 / solvent_dielectric
    )
    return joules * AVOGADRO / 1000.0


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
