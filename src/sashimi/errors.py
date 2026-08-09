"""Backend-neutral failure modes.

These sit above the protocol boundary, so they carry no APBS vocabulary: a
native solver hits "grid too large" and "did not converge" too. Backend-specific
subclasses (`ApbsNotFound`, `ApbsCrash`) live under `sashimi.apbs`, which is the
only layer allowed to know what APBS is.
"""

from __future__ import annotations

__all__ = [
    "ConvergenceFailure",
    "GridTooLarge",
    "SashimiError",
    "SolverCrash",
    "SolverNotFound",
]


class SashimiError(Exception):
    """Base for every error sashimi raises deliberately."""


class SolverNotFound(SashimiError):
    """No usable solver backend could be located."""


class GridTooLarge(SashimiError):
    """The requested resolution and padding cannot fit within `max_points`."""


class ConvergenceFailure(SashimiError):
    """The solver ran but did not reach a usable solution."""


class SolverCrash(SashimiError):
    """The solver exited abnormally, timed out, or produced no usable output."""
