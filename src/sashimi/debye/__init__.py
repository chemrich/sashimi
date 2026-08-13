"""debye: a clean-room Poisson-Boltzmann solver, in process and in pure Python.

In-repo as `sashimi.debye` by the decision recorded in ROADMAP.md sections 10,
12 and 14: the packaging cycle a separate repo would avoid does not bite until
there is a consumer other than sashimi, and the corpus, the registry and the
protocol all live here. `tests/test_protocol_boundary.py` is what keeps the
extraction mechanical when that changes — debye is the first module that would
want to reach into `sashimi.apbs` for a shortcut, and the layering test is the
thing that says no.

Deliberately not registered in `sashimi.backends` yet. Registry integration is
M5, and `tests/test_backends.py` holds the line: every registered backend has to
answer the default surface model, which debye cannot until the solvent-excluded
surface arrives at M4. Registering a solver that refuses the default would make
`sashimi_solve` fail on an unremarkable request, and narrowing the default to
suit a half-built backend is the argument that test exists to force.
"""

from __future__ import annotations

from sashimi.debye.backend import BACKEND_VERSION, DebyeSolver
from sashimi.debye.options import SUPPORTED_SURFACES, DebyeOptions

__all__ = ["BACKEND_VERSION", "SUPPORTED_SURFACES", "DebyeOptions", "DebyeSolver"]
