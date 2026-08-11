"""Still's equation, and the Debye screening that puts salt into it.

Given effective radii, the polar solvation energy is a double sum over atom
pairs with an interpolating denominator that reduces to the Born formula for a
single atom and to Coulomb's law at long range:

    f_GB(i,j) = sqrt(r_ij^2 + R_i R_j exp(-r_ij^2 / 4 R_i R_j))

    dG_pol = -1/(8 pi eps0) sum_ij q_i q_j (1/eps_in - exp(-kappa f) / eps_out) / f

The i = j terms are the Born self-energies, which dominate; the cross terms are
the charge-charge screening. The salt factor is the Debye-Huckel form: it enters
inside the sum rather than outside because the screening depends on the
effective separation, not the real one.
"""

from __future__ import annotations

import math

import numpy as np

from sashimi.constants import (
    ANGSTROM,
    AVOGADRO,
    BOLTZMANN,
    ELEMENTARY_CHARGE,
    JOULES_PER_KJ,
    VACUUM_PERMITTIVITY,
)
from sashimi.protocol import FloatArray, SolventModel

__all__ = ["debye_kappa", "polar_solvation_energy"]

# Molar to particles per cubic metre.
LITRES_PER_CUBIC_METRE = 1000.0


def debye_kappa(solvent: SolventModel) -> float:
    """Inverse Debye length in inverse angstroms; zero at zero ionic strength.

    kappa^2 = 2 N_A e^2 I / (eps0 eps_s k T) for a 1:1 salt. At the 0.15 M and
    298.15 K defaults this is a Debye length of 7.86 A, which is the number to
    check a units mistake against.
    """
    if solvent.ionic_strength <= 0:
        return 0.0
    number_density = solvent.ionic_strength * LITRES_PER_CUBIC_METRE * AVOGADRO  # 1/m^3
    kappa_squared = (
        2.0
        * number_density
        * ELEMENTARY_CHARGE**2
        / (VACUUM_PERMITTIVITY * solvent.solvent_dielectric * BOLTZMANN * solvent.temperature)
    )
    return math.sqrt(kappa_squared) * ANGSTROM  # 1/m -> 1/A


def polar_solvation_energy(
    coords: FloatArray,
    charges: FloatArray,
    radii: FloatArray,
    solvent: SolventModel,
    chunk_size: int = 512,
) -> float:
    """Polar solvation free energy in kJ/mol, from effective Born radii.

    Chunked over rows for the same reason `descreening_integral` is: the pair
    matrix is N^2 and nothing here runs in another process.
    """
    kappa = debye_kappa(solvent)
    inverse_solute = 1.0 / solvent.solute_dielectric
    inverse_solvent = 1.0 / solvent.solvent_dielectric

    total = 0.0
    for start in range(0, len(coords), chunk_size):
        stop = min(start + chunk_size, len(coords))
        deltas = coords[start:stop, None, :] - coords[None, :, :]
        r_squared = np.einsum("ijk,ijk->ij", deltas, deltas)

        radius_product = radii[start:stop, None] * radii[None, :]
        f_gb = np.sqrt(r_squared + radius_product * np.exp(-r_squared / (4.0 * radius_product)))

        screening = inverse_solute - np.exp(-kappa * f_gb) * inverse_solvent
        pairs = charges[start:stop, None] * charges[None, :] * screening / f_gb
        total += float(np.sum(pairs))

    # -1/(8 pi eps0) with charges in e and lengths in angstroms, to kJ/mol.
    prefactor = -(ELEMENTARY_CHARGE**2) / (8.0 * math.pi * VACUUM_PERMITTIVITY * ANGSTROM)
    return total * prefactor * AVOGADRO / JOULES_PER_KJ
