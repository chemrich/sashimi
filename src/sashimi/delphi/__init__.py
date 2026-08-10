"""DelPhi backend — the second finite-difference solver behind the protocol.

Two executables, one backend. DelPhi ships as a C++ program built from source
(`delphicpp`) and as a Python/Numba reimplementation from the same lab
(`pydelphi-static`); they read the same parameter dialect and the same structure
statement, so one input generator drives both and `DelphiFlavour` names the
handful of places they genuinely differ.

Why both: the C++ build is the reference implementation and runs a Born ion in
0.04 s, but it has to be compiled from source and no package manager carries it,
so CI cannot have it. pyDelPhi is `pip install`-able on every platform,
including the `linux-aarch64` gap ROADMAP.md section 9 calls the real platform
debt, so it is the flavour CI can actually exercise. Both are driven as
subprocesses, never imported — the same boundary section 9 draws around APBS,
which for pyDelPhi also keeps its AGPL-3.0 licence on its own side of a process
boundary and its `numpy<2.3` pin out of sashimi's environment.

Serves ROADMAP.md phase 7.
"""

from __future__ import annotations

from sashimi.delphi.backend import DelphiSolver
from sashimi.delphi.discover import (
    DelphiBinary,
    DelphiFlavour,
    DelphiNotFound,
    discover_delphi,
)
from sashimi.delphi.options import DelphiOptions

__all__ = [
    "DelphiBinary",
    "DelphiFlavour",
    "DelphiNotFound",
    "DelphiOptions",
    "DelphiSolver",
    "discover_delphi",
]
