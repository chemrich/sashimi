"""TABI-PB backend — the boundary-element solver, and the protocol's acid test.

ROADMAP.md section 2 names the FD/BEM split as the constraint that shaped the
protocol: finite-difference solvers take a grid and return a volume, while
boundary-element solvers take a mesh and return values on a surface, and a
protocol assuming the former forecloses half the landscape. Phase 4 built
`BoundaryElementRequest` and `SurfacePotential` for a solver that did not exist
yet, and guarded them with a stub. This is that solver.

TABI-PB (treecode-accelerated boundary integral, BSD-3-Clause, University of
Michigan) is driven as a subprocess like every other backend. It differs from
the finite-difference pair in ways the protocol already anticipated — no grid,
a triangulated interface instead of a volume, `mesh_density` instead of
`GridSpec` — and in one it did not: it shells out to a second executable,
NanoShaper, to build the triangulation. See `discover.py`.

Serves ROADMAP.md phase 7.
"""

from __future__ import annotations

from sashimi.tabipb.backend import TabipbSolver
from sashimi.tabipb.discover import TabipbBinary, TabipbNotFound, discover_tabipb
from sashimi.tabipb.options import TabipbOptions

__all__ = [
    "TabipbBinary",
    "TabipbNotFound",
    "TabipbOptions",
    "TabipbSolver",
    "discover_tabipb",
]
