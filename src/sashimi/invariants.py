"""Quality checks that need no reference answer, and why the corpus needed them.

**Every closed form in the corpus is a one- or two-atom solute.** Thirty-seven
cases carry an analytic energy and twelve an analytic field, and all of them are
Born ions or Kirkwood spheres — because those are the geometries a closed form
exists for. Thirty-two cases sit above 500 atoms and **not one has any ground
truth at all.** Above a peptide the corpus grades *agreement*, and agreement is
not accuracy: when the reference-tier backends spread 10.4% on a 1,156-residue
protein (ROADMAP.md section 12), nothing in the suite can say which of them is
closer to right.

What fills that gap is not a better reference. It is the identities the answer
has to satisfy whatever the answer is, which hold at every size and cost one
extra solve each.

**Charge scaling.** The linearized Poisson-Boltzmann equation is linear in the
charge and the dielectric map does not depend on it, so scaling every partial
charge by `lam` scales the potential by `lam` and the polar solvation energy by
exactly `lam**2`. That is an identity, not an approximation, and it holds for
every family here — finite difference, boundary element and analytic alike,
since a Generalized Born radius is a property of the geometry. A backend that
fails it has mis-assigned charge, which is precisely the failure mode section 7
records DelPhi hitting for a year when `format_pqr` shifted a column.

**Rigid-motion invariance.** Solvation energy is a property of the solute, so
translating or rotating it cannot change the answer. On a fixed lattice it does,
and **the spread across poses is the part of that backend's discretization error
that depends on grid phase** — reference-free, available at any size, and the
quantity the corpus had no way to state above two atoms.

The measurement that shows it works is `gb`: an analytic method has no lattice
to be out of phase with, and it reads **0.000%** where the finite-difference
backends read 0.3-3%. A metric whose control comes out at exactly zero is
measuring what it claims to.

**That sentence used to say "is that backend's discretization error", and the
overstatement mattered.** A pose spread sees only the error that *moves* when
the solute shifts against the lattice. The part that does not move is invisible
to it, and on debye it is the larger of the two: on `ala-gly` at 0.5 A the
dispersion is 0.88% while the energy sits about 5% from the limit its own
refinement extrapolates to. A scheme can be beautifully phase-stable about the
wrong answer, and a gate on dispersion alone would call that an improvement.

So this module states both halves. `grade_pose_spread` reports the part that
varies; `grade_refinement` reports where the answer is *going* as the lattice
refines, and how far the current one is from it. Neither needs a reference
backend, which is the property that made this module worth writing — but they
are different numbers and the roadmap should quote both.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np

from sashimi.protocol import FloatArray, PQRData, System
from sashimi.validate import Backend

__all__ = [
    "DEFAULT_LADDER",
    "MIN_POSES",
    "MIN_REFINEMENTS",
    "POSE_SEED",
    "POSE_SHIFT_CELLS",
    "ChargeScaling",
    "PoseSpread",
    "Refinement",
    "grade_charge_scaling",
    "grade_pose_spread",
    "grade_refinement",
    "posed",
    "scaled",
]

# Pinned, for the reason `PROBE_SEED` is: a spread must not depend on which
# poses happened to be drawn, and an unpinned one would make every recorded
# number unreproducible.
POSE_SEED = 20260820

# The translation, as a fraction of the grid spacing rather than in angstroms.
#
# It has to be sub-spacing: a shift long enough to move the solute within the box
# would change the boundary condition too. A fixed 0.5 A was the first draft and
# was only sub-spacing at one resolution — at the 0.5 A the tests use it spans
# two whole cells. Scaled by the spacing it means the same thing everywhere.
#
# **Measured 2026-08-22: on this codebase the translation changes nothing at
# all, and the reasoning it used to carry was wrong.** This comment said grid
# phase was what the shift probed. It is not: `size_grid` derives the box from
# `pqr.center()` and `pqr.extent()`, so translating the solute translates the
# lattice with it and every atom sits exactly where it did relative to its
# nodes. Measured on `ala-gly` at 0.5 A, shifting by 0.25, 0.50 and 0.75 cells
# moves the energy by 0.0, 0.0 and one ulp; the rotation in the same function
# moves it by 0.96 kJ/mol. **So a pose spread on this backend is a rotation
# spread**, and it is the rotation — which changes the bounding box, and so the
# lattice, and so where each atom falls within it — that does all the work.
#
# The shift is kept because it costs nothing and because a backend driven with
# a *fixed* box would see it. `tests/test_invariants.py` pins the fact so the
# next reader does not re-derive the rationale that was here.
#
# One consequence worth having in front of you: a spherically symmetric solute
# has no rotation either, so **the metric is identically zero on a Born ion** —
# which is exactly the geometry the closed forms cover. Pose dispersion and the
# analytic references apply to disjoint sets of structures.
POSE_SHIFT_CELLS = 0.5

# One pose is the original and a spread needs something to spread against.
MIN_POSES = 2

# Richardson needs three energies to solve for both unknowns — the limit and the
# order the error approaches it at. Two would need the order assumed, and
# assuming it is the whole question: debye's is 1.17 on `ala-gly`, not the 2 a
# reader would guess from a second-order operator.
MIN_REFINEMENTS = 3

# Halving, because Richardson's ratio is cleanest on a geometric sequence.
DEFAULT_LADDER = (1.0, 0.5, 0.25)


def scaled(structure: PQRData, factor: float) -> PQRData:
    """The same solute with every partial charge multiplied by `factor`.

    Geometry untouched, so the dielectric boundary — and therefore the whole
    discretization — is bit-identical between the two solves. That is what makes
    the `lam**2` identity exact rather than approximate on a grid.
    """
    return replace(structure, charges=structure.charges * factor)


def posed(structure: PQRData, index: int, *, spacing: float, seed: int = POSE_SEED) -> PQRData:
    """The same solute rigidly moved: pose 0 is the original, the rest are turned.

    A proper rotation about the centroid plus a translation of up to half a grid
    cell. `spacing` is required rather than defaulted because the translation is
    only meaningful relative to it — see `POSE_SHIFT_CELLS`.

    **The rotation's determinant is forced to +1 and that is not cosmetic.** A
    QR factorisation returns an orthogonal matrix, which is a rotation *or* a
    reflection; a reflection is a different molecule, and for a chiral solute a
    different answer. `tests/test_invariants.py` asserts the determinant rather
    than only checking distances, because every distance survives a reflection.
    """
    if index == 0:
        return structure
    rng = np.random.default_rng(seed + index)
    turn, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    turn = turn * np.sign(np.linalg.det(turn))
    reach = POSE_SHIFT_CELLS * spacing
    centre = structure.coords.mean(axis=0)
    shift = rng.uniform(-reach, reach, 3)
    moved: FloatArray = (structure.coords - centre) @ turn.T + centre + shift
    return replace(structure, coords=moved)


@dataclass(frozen=True)
class ChargeScaling:
    """How far a backend is from the exact `lam**2` the linear equation requires."""

    backend: str
    factor: float
    reference_energy: float
    scaled_energy: float

    @property
    def expected(self) -> float:
        return self.reference_energy * self.factor**2

    @property
    def error(self) -> float:
        """Relative deviation from the identity. Zero is the only correct answer."""
        return abs(self.scaled_energy - self.expected) / abs(self.expected)


@dataclass(frozen=True)
class PoseSpread:
    """A backend's energies over rigid poses of one solute, and their spread.

    `relative` is the figure of merit: it is a discretization error bar on a
    structure with no closed form, which is every structure the corpus holds
    above two atoms.
    """

    backend: str
    energies: tuple[float, ...]

    @property
    def mean(self) -> float:
        return float(np.mean(self.energies))

    @property
    def relative(self) -> float:
        """Peak-to-peak spread over the mean. Intuitive, and a noisy estimator.

        The range of a small sample is dominated by its two extremes, so this
        moves a lot with which poses were drawn: debye read 3.01% and 0.60% on
        the same structure under two different five-pose draws. Quote it, but
        gate on `dispersion`.
        """
        return float(np.ptp(self.energies) / abs(self.mean))

    @property
    def dispersion(self) -> float:
        """Standard deviation over the mean — the stable form of the same thing.

        Every pose is an independent draw of the same quantity, so the scatter
        is the statistic and the range is only its most volatile summary.
        """
        return float(np.std(self.energies, ddof=1) / abs(self.mean))


@dataclass(frozen=True)
class Refinement:
    """A backend's energies down a refinement ladder, and where they are heading.

    **What this answers that a pose spread cannot.** A spread sees the error
    that moves with grid phase. This sees the error that does not: solve the
    same system at `h`, `h/2` and `h/4`, fit `E(h) = limit + C h**order`, and
    report both the limit and how far each rung sits from it. No reference
    backend appears anywhere, which is the point — every reference-tier solver
    discretizes a volumetric dielectric the same way, so on this axis they share
    the bias rather than referee it.

    **Richardson is only as good as its assumption**, which is that one error
    term dominates over the ladder. `converging` says whether the data supports
    that: successive differences must shrink, and their ratio is what `order`
    is read from. A ladder that is not converging returns an order and a limit
    that mean nothing, so callers must check it — `tests/test_invariants.py`
    validates the whole extrapolator against the Born closed form, where the
    limit is known in advance and the fit has to find it.
    """

    backend: str
    spacings: tuple[float, ...]
    energies: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.spacings) < MIN_REFINEMENTS:
            raise ValueError(
                f"a refinement needs at least {MIN_REFINEMENTS} spacings, got {len(self.spacings)}"
            )
        if len(self.spacings) != len(self.energies):
            raise ValueError("every spacing needs an energy")
        ratios = [a / b for a, b in pairwise(self.spacings)]
        if not all(r > 1.0 for r in ratios):
            raise ValueError(f"spacings must descend, got {self.spacings}")

    @property
    def _differences(self) -> tuple[float, ...]:
        return tuple(a - b for a, b in pairwise(self.energies))

    @property
    def converging(self) -> bool:
        """Whether successive corrections shrink, which Richardson assumes.

        Without this the fit is happy to report an order from a ladder that is
        diverging or oscillating — and ROADMAP.md section 12 records that
        nothing converges monotonically at `d/a >= 0.5` on a sharp boundary, so
        the case is real rather than defensive.
        """
        steps = self._differences
        return all(abs(a) > abs(b) > 0.0 for a, b in pairwise(steps))

    @property
    def order(self) -> float:
        """The exponent the error approaches the limit at, from the last three rungs.

        First order is the signature of an O(1) error at the dielectric
        interface; second is what the operator would give on a smooth
        coefficient. debye reads 1.17 on `ala-gly`, which is the measurement
        that says the interface treatment, not the solver, is what bounds its
        accuracy.
        """
        first, second = self._differences[-2:]
        step = self.spacings[-3] / self.spacings[-2]
        return float(np.log(abs(first / second)) / np.log(step))

    @property
    def limit(self) -> float:
        """The energy the ladder extrapolates to, by Richardson on the last three."""
        step = self.spacings[-3] / self.spacings[-2]
        correction = self._differences[-1] / (step**self.order - 1.0)
        return float(self.energies[-1] - correction)

    @property
    def bias(self) -> tuple[float, ...]:
        """Each rung's signed relative distance from the limit."""
        limit = self.limit
        return tuple(float((e - limit) / abs(limit)) for e in self.energies)


def grade_refinement(
    backend: Backend, system: System, *, spacings: Sequence[float] | None = None
) -> Refinement:
    """Solve the same system down a refinement ladder and extrapolate.

    The ladder halves, because Richardson's ratio is cleanest on a geometric
    sequence and because `size_grid` lands nearer a halved request than an
    arbitrary one. Costs one solve per rung and the finest rung dominates: on a
    peptide the 0.25 A rung is most of the total.
    """
    ladder = tuple(spacings) if spacings is not None else DEFAULT_LADDER
    energies = tuple(
        _energy(backend, replace(system, grid=replace(system.grid, resolution=h))) for h in ladder
    )
    return Refinement(backend=backend.name, spacings=ladder, energies=energies)


def grade_charge_scaling(backend: Backend, system: System, *, factor: float = 2.0) -> ChargeScaling:
    """Solve once as given and once with every charge scaled; compare to `lam**2`."""
    reference = _energy(backend, system)
    scaled_system = replace(system, structure=scaled(system.structure, factor))
    return ChargeScaling(
        backend=backend.name,
        factor=factor,
        reference_energy=reference,
        scaled_energy=_energy(backend, scaled_system),
    )


def grade_pose_spread(backend: Backend, system: System, *, poses: int = 5) -> PoseSpread:
    """Solve the same solute in several rigid poses and report the spread.

    The translation scales with the grid, so a boundary-element backend — which
    has no grid — is posed against the spacing the system nominally carries. Its
    answer should barely move either way; TABI-PB reads 0.052% where the
    finite-difference backends read 0.4% to 1.4%.
    """
    if poses < MIN_POSES:
        raise ValueError(f"a spread needs at least {MIN_POSES} poses, got {poses}")
    spacing = float(system.grid.resolution)
    energies = [
        _energy(backend, replace(system, structure=posed(system.structure, i, spacing=spacing)))
        for i in range(poses)
    ]
    return PoseSpread(backend=backend.name, energies=tuple(energies))


def _energy(backend: Backend, system: System) -> float:
    result = backend.solver.solve(system.request_for(backend.family))
    if result.energy_kj_mol is None:  # pragma: no cover - every case asks for energy
        raise ValueError(f"{backend.name} returned no energy")
    return float(result.energy_kj_mol)
