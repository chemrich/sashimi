"""What `DebyeSolver` can be asked for, and what it refuses.

The refusals are the interesting half. debye is being built up the ladder in
ROADMAP.md section 12, so at M1 it solves the linearized equation on a van der
Waals boundary and nothing else. Saying so through `UnsupportedRequest` — the
same mechanism every other backend uses for a surface model it has no
equivalent of — is what keeps a half-built solver from returning a confident
number for physics it has not implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from sashimi.errors import UnsupportedRequest
from sashimi.protocol import Equation, SurfaceModel

__all__ = [
    "SUPPORTED_EQUATIONS",
    "SUPPORTED_SURFACES",
    "DebyeOptions",
    "check_equation",
    "check_surface",
]

# M1's surface, and only M1's surface. The solvent-excluded (`MOLECULAR`)
# boundary is M4: it needs a probe rolled over the union of spheres, which is
# the hardest single construction on the ladder and is not something to
# approximate quietly with a union of inflated spheres — that is a different
# surface with the same name, which is the mistake `sashimi.gb` made once and
# ROADMAP.md section 7 records twice.
SUPPORTED_SURFACES: frozenset[SurfaceModel] = frozenset({SurfaceModel.VAN_DER_WAALS})

# The nonlinear equation is representable in the protocol and solved by nobody
# here; `backends.IMPLEMENTED_EQUATIONS` says the same thing for the shipped
# backends. debye's operator is linear by construction — the Boltzmann term
# enters as a diagonal, which is what makes the system symmetric positive
# definite and the multigrid preconditioner valid.
SUPPORTED_EQUATIONS: frozenset[Equation] = frozenset({Equation.LINEAR})


@dataclass(frozen=True)
class DebyeOptions:
    """Solver knobs. Every default is measured rather than guessed.

    `tolerance` is on the relative residual ||b - A phi|| / ||b||, not on the
    energy. An energy tolerance would be the quantity the caller cares about
    and the wrong thing to iterate on: the solvation energy is a difference of
    two large solves, so it converges long before the field does, and stopping
    when it stops moving would hand back a potential that is still wrong in
    the third digit — which is precisely the quantity M1b grades and the one
    the consumer displays.
    """

    tolerance: float = 1e-8
    max_cycles: int = 200
    # Smoothing sweeps per multigrid level, down and up. Two of each is the
    # textbook V(2,2) and is what the convergence numbers in the module
    # docstring of `linear.py` were measured with.
    pre_smooth: int = 2
    post_smooth: int = 2

    # There is deliberately no `relaxation` here. A damping factor is the
    # obvious next knob and the first draft carried one — unread by the
    # smoother, because red-black Gauss-Seidel does not need damping to smooth
    # an M-matrix. A parameter that validates its input and changes no answer is
    # the same shape as the checks ROADMAP.md section 7 keeps finding: it reads
    # as a capability and is silence. Add it when a measurement wants it.

    def __post_init__(self) -> None:
        if self.tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {self.tolerance}")
        if self.max_cycles < 1:
            raise ValueError(f"max_cycles must be at least 1, got {self.max_cycles}")
        if self.pre_smooth < 0 or self.post_smooth < 0:
            raise ValueError("smoothing sweep counts must be non-negative")


def check_surface(model: SurfaceModel) -> None:
    """Refuse a boundary debye cannot build yet, naming the milestone."""
    if model not in SUPPORTED_SURFACES:
        supported = ", ".join(sorted(m.value for m in SUPPORTED_SURFACES))
        raise UnsupportedRequest(
            f"debye builds the {supported} boundary and was asked for "
            f"{model.value!r}. The solvent-excluded surface is ROADMAP.md "
            "section 12 M4; harmonic averaging is APBS's and debye will not "
            "have one. Ask APBS or DelPhi for those, or request "
            "surface_model='van-der-waals'."
        )


def check_equation(equation: Equation) -> None:
    """Refuse the nonlinear equation, which debye does not discretize."""
    if equation not in SUPPORTED_EQUATIONS:
        raise UnsupportedRequest(
            f"debye solves the linearized Poisson-Boltzmann equation and was "
            f"asked for {equation.value!r}. No sashimi backend solves the "
            "nonlinear equation; see ROADMAP.md section 14 Q1."
        )
