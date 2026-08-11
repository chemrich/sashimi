"""DelPhi's output unit conventions, converted at the backend boundary.

Two conversions, both of which exist because DelPhi reports in units the
protocol does not use:

- **Energies in kT.** APBS prints kJ/mol directly; DelPhi prints multiples of
  kT, so the temperature the solve actually ran at is part of the conversion.
  Reading a DelPhi energy without it is a silent factor-of-2.5 error.
- **Cube geometry in Bohr.** The Gaussian Cube format is atomic units by
  definition, so a map read as angstroms would be 89% too small in every
  dimension and its potentials would land in the wrong place.

Constants are CODATA 2018, matching `tests/born_reference.py`, so an energy
converted here and a closed form computed there are on the same footing.
"""

from __future__ import annotations

__all__ = ["BOHR_TO_ANGSTROM", "GAS_CONSTANT", "kt_to_kj_per_mol"]

AVOGADRO = 6.02214076e23  # 1/mol
BOLTZMANN = 1.380649e-23  # J/K
GAS_CONSTANT = BOLTZMANN * AVOGADRO  # J/(mol K)

# CODATA 2018 Bohr radius in metres, expressed in angstroms.
BOHR_TO_ANGSTROM = 0.529177210903


def kt_to_kj_per_mol(energy_kt: float, temperature: float) -> float:
    """Convert an energy in units of kT to kJ/mol at the given temperature.

    At the 298.15 K default one kT is 2.4789 kJ/mol.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return energy_kt * GAS_CONSTANT * temperature / 1000.0
