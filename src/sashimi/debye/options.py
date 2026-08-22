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

# Both sharp boundaries, as of M4. The solvent-excluded (`MOLECULAR`) surface is
# built in `debye.surface` by rolling a probe over the union of spheres — not by
# inflating the spheres, which is a different surface wearing the same name and
# the mistake `sashimi.gb` made once and ROADMAP.md section 7 records twice.
#
# `SMOOTHED_MOLECULAR` will never join them: harmonic averaging over a 9-point
# stencil is APBS's discretization rather than a boundary, and M1c measured what
# debye would gain by smoothing its own dielectric — the worst near-field error
# moves 4.138% -> 3.085%, which is why M4a was dropped. `GAUSSIAN` is DelPhi's.
SUPPORTED_SURFACES: frozenset[SurfaceModel] = frozenset(
    {SurfaceModel.VAN_DER_WAALS, SurfaceModel.MOLECULAR}
)

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

    # Width in cells of the band the dielectric is blended over at the interface,
    # blended *harmonically*. Zero is the shipped scheme: a hard assignment from
    # the face centre's own side of the surface, which is what APBS does with
    # `srfm mol` and what every recorded corpus energy was measured with.
    #
    # **Non-zero changes the answer**, so it is a knob and not a default. See
    # `sashimi.debye.dielectric` for what M1c measured and ROADMAP.md section 12
    # for the gate it has to pass before the default could move.
    dielectric_smoothing: float = 0.0

    tolerance: float = 1e-8
    max_cycles: int = 200

    # Smoothing sweeps per multigrid level, applied on the way down *and* on the
    # way up. **One count, not two, and that is a correctness constraint rather
    # than a simplification.** The V-cycle is a legal CG preconditioner only if
    # it is symmetric, which needs the post-smoother to be the adjoint of the
    # pre-smoother: red-black Gauss-Seidel run forward has the reverse colour
    # order as its adjoint, and the counts have to match. Separate `pre_smooth`
    # and `post_smooth` knobs — which this carried until a review asked what
    # stopped them differing — let a caller tuning for speed silently make the
    # preconditioner nonsymmetric, and the symptom would be a stall reported as
    # a `ConvergenceFailure` advising a bigger `max_cycles`, which is not the
    # cause. Two of each is the textbook V(2,2) and is what `linear.py`'s
    # measured convergence rate was taken with.
    smoothing_sweeps: int = 2

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
        if self.smoothing_sweeps < 1:
            raise ValueError(
                f"smoothing_sweeps must be at least 1, got {self.smoothing_sweeps}; "
                "a V-cycle with no smoothing transfers error between grids without "
                "removing any"
            )


def check_surface(model: SurfaceModel) -> None:
    """Refuse a boundary debye cannot build yet, naming the milestone."""
    if model not in SUPPORTED_SURFACES:
        supported = ", ".join(sorted(m.value for m in SUPPORTED_SURFACES))
        raise UnsupportedRequest(
            f"debye builds the {supported} boundaries and was asked for "
            f"{model.value!r}. Harmonic averaging is APBS's discretization and a "
            "Gaussian dielectric is DelPhi's; debye will have neither. Ask that "
            "backend for those, or request one of the sharp boundaries."
        )


def check_equation(equation: Equation) -> None:
    """Refuse the nonlinear equation, which debye does not discretize."""
    if equation not in SUPPORTED_EQUATIONS:
        raise UnsupportedRequest(
            f"debye solves the linearized Poisson-Boltzmann equation and was "
            f"asked for {equation.value!r}. No sashimi backend solves the "
            "nonlinear equation; see ROADMAP.md section 14 Q1."
        )
