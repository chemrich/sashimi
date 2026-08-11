"""Closed-form Born ion — now a shipped module, re-exported here.

These lived here while they were test-only. The golden corpus checks cases
against closed forms too, so they moved to `sashimi.analytic`, which is also
where the derivations and the measured disagreements are recorded. This shim
keeps the existing imports working and is the one place that knows they moved.
"""

from __future__ import annotations

from sashimi.analytic import born_potential, born_solvation_energy
from sashimi.constants import (
    ANGSTROM,
    AVOGADRO,
    BOLTZMANN,
    ELEMENTARY_CHARGE,
    VACUUM_PERMITTIVITY,
)

__all__ = [
    "ANGSTROM",
    "AVOGADRO",
    "BOLTZMANN",
    "ELEMENTARY_CHARGE",
    "VACUUM_PERMITTIVITY",
    "born_potential",
    "born_solvation_energy",
]
