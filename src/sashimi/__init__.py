"""sashimi — thinly sliced Poisson.

A maintained wrapper around APBS for biomolecular electrostatics. Import the
protocol types from here; import a concrete backend from `sashimi.apbs`.
"""

from sashimi.errors import (
    ConvergenceFailure,
    GridTooLarge,
    SashimiError,
    SolverCrash,
    SolverNotFound,
)
from sashimi.protocol import (
    GridSpec,
    PotentialGrid,
    PQRData,
    SolventModel,
    Solver,
    SolveResult,
)

__version__ = "0.1.0"

__all__ = [
    "ConvergenceFailure",
    "GridSpec",
    "GridTooLarge",
    "PQRData",
    "PotentialGrid",
    "SashimiError",
    "SolveResult",
    "SolventModel",
    "Solver",
    "SolverCrash",
    "SolverNotFound",
    "__version__",
]
