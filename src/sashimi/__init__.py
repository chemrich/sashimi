"""sashimi — thinly sliced Poisson.

A maintained wrapper around Poisson-Boltzmann electrostatics solvers. Import the
protocol types from here; import a concrete backend from `sashimi.apbs`.
"""

from sashimi.errors import (
    BackendUnavailable,
    ConvergenceFailure,
    GridTooLarge,
    InputError,
    MalformedStructure,
    PreparationFailed,
    SashimiError,
    SolverCrash,
    SolverError,
    UnsupportedRequest,
)
from sashimi.protocol import (
    BoundaryElementRequest,
    Equation,
    FiniteDifferenceRequest,
    GridSpec,
    Potential,
    PotentialGrid,
    PQRData,
    Provenance,
    SolventModel,
    Solver,
    SolveRequest,
    SolveResult,
    SurfaceModel,
    SurfacePotential,
)

__version__ = "0.1.0"

__all__ = [
    "BackendUnavailable",
    "BoundaryElementRequest",
    "ConvergenceFailure",
    "Equation",
    "FiniteDifferenceRequest",
    "GridSpec",
    "GridTooLarge",
    "InputError",
    "MalformedStructure",
    "PQRData",
    "Potential",
    "PotentialGrid",
    "PreparationFailed",
    "Provenance",
    "SashimiError",
    "SolveRequest",
    "SolveResult",
    "SolventModel",
    "Solver",
    "SolverCrash",
    "SolverError",
    "SurfaceModel",
    "SurfacePotential",
    "UnsupportedRequest",
    "__version__",
]
