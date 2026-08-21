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
and **the spread across poses is that backend's discretization error on that
structure** — reference-free, available at any size, and the quantity the
corpus had no way to state above two atoms.

The measurement that shows it works is `gb`: an analytic method has no lattice
to be out of phase with, and it reads **0.000%** where the finite-difference
backends read 0.3-3%. A metric whose control comes out at exactly zero is
measuring what it claims to.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from sashimi.protocol import FloatArray, PQRData, Solver, SolverFamily, System

__all__ = [
    "MIN_POSES",
    "POSE_SEED",
    "ChargeScaling",
    "PoseSpread",
    "grade_charge_scaling",
    "grade_pose_spread",
    "posed",
    "scaled",
]

# Pinned, for the reason `PROBE_SEED` is: a spread must not depend on which
# poses happened to be drawn, and an unpinned one would make every recorded
# number unreproducible.
POSE_SEED = 20260820

# Sub-spacing, so the solute lands in a different place *within* a cell without
# moving far enough to change which part of the box it occupies. Grid phase is
# what is being probed; a large translation would also change the boundary
# condition and confound the two.
POSE_SHIFT_A = 0.5

# One pose is the original and a spread needs something to spread against.
MIN_POSES = 2


def scaled(structure: PQRData, factor: float) -> PQRData:
    """The same solute with every partial charge multiplied by `factor`.

    Geometry untouched, so the dielectric boundary — and therefore the whole
    discretization — is bit-identical between the two solves. That is what makes
    the `lam**2` identity exact rather than approximate on a grid.
    """
    return replace(structure, charges=structure.charges * factor)


def posed(structure: PQRData, index: int, *, seed: int = POSE_SEED) -> PQRData:
    """The same solute rigidly moved: pose 0 is the original, the rest are turned.

    A proper rotation about the centroid plus a sub-spacing translation. The
    rotation comes from a QR factorisation with its determinant forced to +1 —
    a reflection would be a different molecule, and for a chiral solute a
    different answer.
    """
    if index == 0:
        return structure
    rng = np.random.default_rng(seed + index)
    turn, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    turn = turn * np.sign(np.linalg.det(turn))
    centre = structure.coords.mean(axis=0)
    shift = rng.uniform(-POSE_SHIFT_A, POSE_SHIFT_A, 3)
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


def grade_charge_scaling(
    solver: Solver[Any], family: SolverFamily, system: System, *, factor: float = 2.0
) -> ChargeScaling:
    """Solve once as given and once with every charge scaled; compare to `lam**2`."""
    reference = _energy(solver, family, system)
    scaled_system = replace(system, structure=scaled(system.structure, factor))
    return ChargeScaling(
        backend=_label(solver),
        factor=factor,
        reference_energy=reference,
        scaled_energy=_energy(solver, family, scaled_system),
    )


def grade_pose_spread(
    solver: Solver[Any], family: SolverFamily, system: System, *, poses: int = 5
) -> PoseSpread:
    """Solve the same solute in several rigid poses and report the spread."""
    if poses < MIN_POSES:
        raise ValueError(f"a spread needs at least {MIN_POSES} poses, got {poses}")
    energies = [
        _energy(solver, family, replace(system, structure=posed(system.structure, index)))
        for index in range(poses)
    ]
    return PoseSpread(backend=_label(solver), energies=tuple(energies))


def _energy(solver: Solver[Any], family: SolverFamily, system: System) -> float:
    result = solver.solve(system.request_for(family))
    if result.energy_kj_mol is None:  # pragma: no cover - every case asks for energy
        raise ValueError(f"{_label(solver)} returned no energy")
    return float(result.energy_kj_mol)


def _label(solver: Solver[Any]) -> str:
    return str(getattr(solver, "label", type(solver).__name__))
