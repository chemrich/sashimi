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

from sashimi.debye.sources import DEFAULT_BOUNDARY_PITCH_A
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
#
# *debye has since built a dielectric smoothing of its own — a sub-cell ramp
# from a signed distance, `DebyeOptions.dielectric_smoothing` below — and it is
# not this. The refusal stands because a surface model names a **boundary** and
# `SMOOTHED_MOLECULAR` names a discretization of one, which is a difference the
# ramp does not close: a caller asking for it is asking for APBS's stencil.*
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
    # **Non-zero changes the answer**, so it is a knob and not a default — and
    # since 2026-08-25 that is a measurement rather than missing coverage. The
    # ramp buys the **energy** and does not buy the **field**: on `ala-gly` at
    # 0.4545 A it is 4.9x closer to the converged energy than the hard
    # assignment, and 2.4-13x *further* from a refined referee on the potential
    # 2-3 A outside the surface at w >= 0.75.
    #
    # **w = 0.5 was the open case and is now closed, against an exact referee.**
    # Every earlier field measurement used a refined debye run as the yardstick,
    # which shares its construction with the candidate. Graded instead against
    # `analytic.kirkwood_potential` -- a closed form for an off-centre charge in
    # a sphere, sharing no construction with any lattice -- w = 0.5 is **worse
    # on the 2-3 A shell in 269 of 272 paired configurations**, over four charge
    # offsets, five spacings and up to sixteen lattice phases per cell (four to
    # eight at the finest spacings), field and energy taken from the same
    # solves. **w = 0.25 is decided against too**, where it was previously
    # undecided outright.
    #
    # The *sign* is what is settled: it survives every box tested. The
    # *magnitude* is box-dependent and should not be quoted bare -- the same
    # fixture reads 1.6-2.5x at padding 10 A and 5.8x at 24 A, because debye's
    # Dirichlet face is Coulomb in eps_s and its positive error partly cancels
    # the hard scheme's negative one. What is box-free is that the ramp's shell
    # error is **proportional to the physical band width w*h**, at 2.4-2.7 %/A
    # for a charge at d/a = 0.3 rising to 9.0-10.8 %/A at 0.9, near-constant
    # across a 3.1x range in h and a 4x range in w. **That is a modelling error,
    # not a discretization error, which is why refining does not close it** --
    # the hard scheme's field converges at order +1.3 to +1.6 against the ramp's
    # +1.11 to +1.18, so there is no crossover in either direction.
    #
    # **Why the Born sphere said "roughly neutral" and was not merely
    # insensitive.** For any spherically symmetric eps(r) at zero ionic
    # strength, Gauss's law fixes the exterior potential of a *centred* charge
    # regardless of what the interface scheme does inside. A Born gate is
    # therefore **provably** blind to this error rather than just short of
    # resolution -- the same "gate a monopole passes twice" shape as M9. The
    # off-centre fixture supplies the n >= 1 reaction-field multipoles that
    # carry it.
    #
    # **Turn it on for a solvation energy on a coarse grid**, where it is worth
    # 1.7-3.1x in spacing. Do not turn it on to display a field, where the same
    # substitution runs backwards and costs 1.9-2.4x in spacing -- about 14x the
    # nodes. `sashimi.debye.dielectric` carries the reasoning and ROADMAP.md
    # section 12 "The field axis, measured" the tables.
    #
    # **The energy gain is not universal either.** It is 3.2-7.2x at d/a <= 0.7,
    # and at d/a = 0.9 -- the charge about one cell from the interface -- there
    # is no accuracy gain at all (the ramp wins 8 to 12 of 16 phases). What it
    # buys there is *reproducibility*: the peak-to-peak spread over lattice
    # phase collapses from 107.3 to 10.8 percentage points, leaving the value
    # biased by -8 to -14%. A number that is reliably wrong is not a more
    # accurate number.
    #
    # Coverage gaps that remain open. The Kirkwood fixture is **one convex
    # sphere**, so it says nothing about concave or re-entrant geometry, and
    # `molecular` is bitwise the same object as `van-der-waals` on it -- so the
    # molecular surface is still exercised only by two tests on one 20-atom
    # dipeptide. Also untested: non-zero ionic strength, and eps_p > 1.
    #
    # And do not read a pose-dispersion improvement as the accuracy case. A pose
    # spread is the *phase-dependent* half of the discretization error and on
    # debye it is the smaller half — Q0, and `sashimi.invariants` says the same
    # in its header. It is not the half an interface scheme is decided on.
    dielectric_smoothing: float = 0.0

    # How far apart, in angstroms, the box face is sampled before the
    # Debye-Huckel sum is interpolated up to every face node. **A distance, not
    # a node stride** — see `sources.DEFAULT_BOUNDARY_PITCH_A` for why a stride
    # gives three different pitches on the three axes of one face.
    #
    # Zero or negative evaluates every node, which is the pre-M9 scheme and is
    # bit-identical to it. So is any case under `sources.EXACT_FACE_PAIRS`,
    # whatever this is set to: the exact face is already free there, and holding
    # the small cases still is what keeps the closed-form and small-molecule
    # recordings from needing to move.
    boundary_pitch_a: float = DEFAULT_BOUNDARY_PITCH_A

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
