"""Physical constants, in one place so two backends cannot disagree about them.

CODATA 2018 throughout, matching `tests/born_reference.py` — which is the point.
The Born ion is the only case in this project with a closed form, so a backend
computing it from constants that differ in the sixth digit from the reference's
would fail a comparison for a reason that has nothing to do with the physics.
"""

from __future__ import annotations

ELEMENTARY_CHARGE = 1.602176634e-19  # C
VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m
AVOGADRO = 6.02214076e23  # 1/mol
BOLTZMANN = 1.380649e-23  # J/K
GAS_CONSTANT = BOLTZMANN * AVOGADRO  # J/(mol K)

ANGSTROM = 1e-10  # m
JOULES_PER_KJ = 1000.0

__all__ = [
    "ANGSTROM",
    "AVOGADRO",
    "BOLTZMANN",
    "ELEMENTARY_CHARGE",
    "GAS_CONSTANT",
    "JOULES_PER_KJ",
    "VACUUM_PERMITTIVITY",
]
