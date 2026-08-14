"""Cross-solver validation: run one system through N backends, report the spread.

ROADMAP.md section 8's product feature, and the reason the multi-backend
strategy is worth anything. Disagreement beyond discretization noise flags an
input-generation bug (ours) or genuine parameter sensitivity (the caller's
problem, surfaced honestly rather than averaged away).

**Most of this module is about refusing to answer.** A spread is only a solver
disagreement if everything else was held fixed, and three things can differ
without any of them being visible in the number:

- **Surface model.** Varying only this moves a dipeptide's energy across 25.7%
  (section 5) — two orders of magnitude more than the solver disagreements this
  is meant to detect.
- **Energy term.** APBS reports a difference against a uniform-dielectric,
  ion-free reference and so carries the mobile-ion contribution; DelPhi reports
  the polarization term alone. At zero salt they coincide, which is what makes
  this easy to miss and expensive to miss. Hence `EnergyTerm` in provenance and
  section 14's rule: same *reported term*, not merely same equation.
- **Equation.** The total-energy integral differs between linear and nonlinear.

Each is checked before a spread is computed, and each can be overridden only by
saying so explicitly, which puts the caveat in the caller's hands rather than in
a footnote.

**One difference is partitioned rather than refused.** A backend that
approximates the equation instead of discretizing it — `AccuracyTier` in
provenance — is expected to disagree, by tens of percent, and neither refusing
it nor averaging it in would be honest. The spread stays a statement about the
reference tier, and each approximation is reported separately as its distance
from what that tier agreed on.

**Why this is not `corpus.verify_case`.** The corpus answers "has this backend
changed?" against recorded numbers, and its first act is to compare grid shape
and bail if it moved — correct there, useless here, because two backends have
different legal grids by construction (APBS's dime must be 32c+1; DelPhi's gsize
is any odd integer). Comparing them therefore cannot go through the grid at all.
Energies are grid-independent scalars, and potentials are compared by
interpolating both maps at the *same physical coordinates* inside the box they
share. That is the one substantive difference between the two engines; the
`Reference` protocol and everything else in `corpus` survives unchanged.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sashimi.errors import InputError

# `SolverFamily` and `System` moved to the protocol layer, where a statement
# about request types belongs and where the corpus can reach them without
# depending on this module. Re-exported so existing imports keep working.
from sashimi.protocol import (
    DIMENSIONS,
    AccuracyTier,
    EnergyTerm,
    Equation,
    FiniteDifferenceRequest,
    FloatArray,
    PotentialGrid,
    Solver,
    SolveRequest,
    SolveResult,
    SolverFamily,
    SurfaceModel,
    System,
)

__all__ = [
    "DEFAULT_APPROXIMATION_TOLERANCE",
    "DEFAULT_ENERGY_TOLERANCE",
    "DEFAULT_FIELD_FACTOR",
    "N_PROBES",
    "PROBE_SEED",
    "Backend",
    "BackendRun",
    "Comparison",
    "FieldGrade",
    "Incomparable",
    "SolverFamily",
    "System",
    "compare_results",
    "grade_field",
    "overlap_probe_points",
    "validate",
    "validate_system",
]

N_PROBES = 200
PROBE_SEED = 20260811  # pinned; a spread must not depend on where we sampled
PROBE_INSET = 0.5  # sample the middle 50% of the shared box, away from boundaries

# Two independent finite-difference codes on different legal grids do not agree
# to corpus tolerance and never will. Measured between APBS and DelPhi on shared
# surface models: 2.3% on a Born ion and ALA-GLY, 2.4% on hen lysozyme, 4.0% on
# a van der Waals boundary. This is the line between "discretization" and
# "someone has a bug", and it is deliberately generous.
DEFAULT_ENERGY_TOLERANCE = 0.10

# An approximation is not held to the tolerance above, because it is not trying
# to meet it. This is the line between "the approximation behaved as documented"
# and "it is broken or misconfigured", and it is set from measurement in both
# directions rather than from the literature's loose 10-30%.
#
# Measured for Generalized Born against the APBS/DelPhi consensus on the
# molecular surface: 1.89, 2.77, 2.04, 7.10 and 6.75% across the five corpus
# cases, and 1.48% on hen lysozyme at 1,960 atoms. This is roughly twice the
# worst of those.
#
# The other direction matters more. Both real misconfigurations found while
# building the GB backend land far outside it: declaring the wrong surface model
# costs 31% and feeding it pdb2pqr's radii instead of mbondi costs 35%. A
# tolerance that tolerated those would have let both ship.
DEFAULT_APPROXIMATION_TOLERANCE = 0.15


@dataclass(frozen=True)
class Backend:
    """A solver, its name, and the request family it speaks."""

    name: str
    solver: Solver[Any]
    family: SolverFamily = SolverFamily.FINITE_DIFFERENCE


class Incomparable(InputError):
    """The backends were not asked the same question, so no spread is meaningful.

    An `InputError` because the caller can act on it: change the request so the
    backends agree on surface model, energy term and equation, or override the
    check having understood what it protects.
    """


@dataclass(frozen=True)
class BackendRun:
    """One backend's answer, reduced to what a comparison needs."""

    name: str
    energy_kj_mol: float | None
    energy_term: EnergyTerm | None
    surface_model: SurfaceModel
    equation: Equation
    potential: PotentialGrid | None
    ionic_strength: float = 0.0  # M; decides whether differing terms coincide
    # What a closed form has to be evaluated at to describe this run. Carried
    # rather than passed in beside it: a Born potential taken at a different
    # dielectric or temperature than the solver used shifts the reference by a
    # factor common to every backend, and because `grade_field`'s verdict is a
    # *ratio* of errors, a large common offset drives every ratio towards 1.0
    # and the grade passes. `corpus.AnalyticField.exact_at` refuses the same
    # class of mismatch on ionic strength; these are the other two axes.
    solvent_dielectric: float = 78.54
    temperature: float = 298.15
    wall_seconds: float | None = None
    accuracy_tier: AccuracyTier = AccuracyTier.REFERENCE

    @classmethod
    def from_result(cls, name: str, result: SolveResult, request: SolveRequest) -> BackendRun:
        potential = result.potential if isinstance(result.potential, PotentialGrid) else None
        return cls(
            name=name,
            energy_kj_mol=result.energy_kj_mol,
            energy_term=result.provenance.energy_term,
            surface_model=request.solvent.surface_model,
            # `equation` lives on the FD request only: BEM formulations are
            # built on the linearized operator's Green function, so a nonlinear
            # one is not something to reject but something unrepresentable.
            equation=getattr(request, "equation", Equation.LINEAR),
            potential=potential,
            ionic_strength=request.solvent.ionic_strength,
            solvent_dielectric=request.solvent.solvent_dielectric,
            temperature=request.solvent.temperature,
            wall_seconds=result.provenance.wall_seconds,
            accuracy_tier=result.provenance.accuracy_tier,
        )


@dataclass
class Comparison:
    """What the backends said, and whether it constitutes agreement.

    `energy_spread` is the spread across *reference-tier* backends only. An
    approximation does not widen it; it is reported separately, in
    `approximation_deviation`, as its distance from what the reference solvers
    agreed on. Folding the two together would cost both numbers — see
    `AccuracyTier`.
    """

    runs: list[BackendRun]
    energy_spread: float | None = None  # max pairwise relative difference
    energy_range_kj_mol: tuple[float, float] | None = None
    potential_rmsd_kt_e: float | None = None
    potential_max_abs_kt_e: float | None = None
    n_probes: int = 0
    agrees: bool = False
    tolerance: float = DEFAULT_ENERGY_TOLERANCE
    # Relative distance from the reference consensus, per approximate backend.
    approximation_deviation: dict[str, float] = field(default_factory=dict)
    approximation_tolerance: float = DEFAULT_APPROXIMATION_TOLERANCE
    approximations_agree: bool = True
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backends": [r.name for r in self.runs],
            "energies_kj_mol": {r.name: r.energy_kj_mol for r in self.runs},
            "accuracy_tiers": {r.name: r.accuracy_tier.value for r in self.runs},
            "energy_spread": self.energy_spread,
            "energy_range_kj_mol": list(self.energy_range_kj_mol)
            if self.energy_range_kj_mol
            else None,
            "potential_rmsd_kt_e": self.potential_rmsd_kt_e,
            "potential_max_abs_kt_e": self.potential_max_abs_kt_e,
            "n_probes": self.n_probes,
            "tolerance": self.tolerance,
            "approximation_deviation": self.approximation_deviation,
            "approximation_tolerance": self.approximation_tolerance,
            "approximations_agree": self.approximations_agree,
            "agrees": self.agrees,
            "notes": self.notes,
        }

    def approximation_summary(self) -> str:
        """One line per approximate backend, or empty if there were none."""
        if not self.approximation_deviation:
            return ""
        parts = ", ".join(
            f"{name} {dev:.2%}" for name, dev in sorted(self.approximation_deviation.items())
        )
        verdict = "as documented" if self.approximations_agree else "OUTSIDE"
        return (
            f"approximations {verdict} from the reference: {parts} "
            f"(tolerance {self.approximation_tolerance:.0%})"
        )

    def summary(self) -> str:
        approximations = self.approximation_summary()
        if self.energy_spread is None:
            return approximations or "no energies to compare"
        lo, hi = self.energy_range_kj_mol or (float("nan"), float("nan"))
        verdict = "agree" if self.agrees else "DISAGREE"
        detail = ""
        if self.potential_rmsd_kt_e is not None:
            detail = (
                f", potential RMSD {self.potential_rmsd_kt_e:.4f} kT/e over {self.n_probes} pts"
            )
        n_reference = sum(1 for r in self.runs if r.accuracy_tier is AccuracyTier.REFERENCE)
        counted = n_reference if self.approximation_deviation else len(self.runs)
        line = (
            f"{counted} backends {verdict}: {self.energy_spread:.2%} spread "
            f"({lo:.3f} to {hi:.3f} kJ/mol, tolerance {self.tolerance:.0%}){detail}"
        )
        return f"{line}; {approximations}" if approximations else line


def check_comparable(runs: list[BackendRun], *, allow_mismatch: bool = False) -> list[str]:
    """Verify the backends were asked, and answered, the same question.

    Returns notes worth reporting on success; raises `Incomparable` otherwise.
    `allow_mismatch` downgrades every refusal to a note, because there are
    legitimate reasons to look at a mismatched spread as long as nobody mistakes
    it for a solver disagreement.
    """
    problems: list[str] = []
    notes: list[str] = []

    surfaces = {r.surface_model for r in runs}
    if len(surfaces) > 1:
        problems.append(
            f"surface models differ ({', '.join(sorted(s.value for s in surfaces))}). "
            "Varying only the surface moves a dipeptide's energy by 25.7%, so this "
            "spread would be a modelling difference, not a solver disagreement."
        )

    equations = {r.equation for r in runs}
    if len(equations) > 1:
        problems.append(
            f"equations differ ({', '.join(sorted(e.value for e in equations))}); "
            "the total-energy integral is not the same quantity across them."
        )

    terms = {r.energy_term for r in runs}
    if None in terms:
        unnamed = sorted(r.name for r in runs if r.energy_term is None)
        problems.append(
            f"{', '.join(unnamed)} did not state which energy it reports, so it cannot be "
            "compared. A backend must set Provenance.energy_term."
        )
    elif len(terms) > 1:
        named = ", ".join(f"{r.name}={r.energy_term.value}" for r in runs if r.energy_term)
        salt = max(r.ionic_strength for r in runs)
        if salt > 0:
            problems.append(
                f"energy terms differ ({named}) at {salt} M ionic strength. The difference "
                "between them is exactly the mobile-ion contribution, which is nonzero here, "
                "so the spread would report a definitional gap as a solver disagreement."
            )
        else:
            # Not a concession: with no mobile ions there is no osmotic term, so
            # the polar solvation energy and the reaction-field energy are the
            # same quantity. Refusing here would make the tool useless on
            # exactly the cases where it is most trustworthy.
            notes.append(
                f"energy terms differ ({named}) but coincide at zero ionic strength, "
                "where the mobile-ion contribution that separates them vanishes"
            )

    if problems and not allow_mismatch:
        raise Incomparable(
            "refusing to report a spread:\n  - "
            + "\n  - ".join(problems)
            + "\n\nFix the request so the backends agree, or pass allow_mismatch=True "
            "having understood that the number will not mean what it appears to."
        )
    notes.extend(f"OVERRIDDEN: {p}" for p in problems)
    return notes


def overlap_probe_points(grids: list[PotentialGrid]) -> FloatArray:
    """Deterministic sample coordinates inside the box every grid covers.

    The heart of comparing backends whose grids differ by construction. Both
    `PotentialGrid`s can be interpolated at arbitrary coordinates, so the
    comparison happens in physical space and never touches grid indices.
    Points are kept off the boundary, where a smoothed dielectric surface makes
    values sensitive to details that are not the solver's arithmetic.
    """
    lows = np.array([g.origin for g in grids])
    highs = np.array([g.origin + (np.array(g.shape) - 1) * g.spacing for g in grids])

    lo = lows.max(axis=0)
    hi = highs.min(axis=0)
    if np.any(hi <= lo):
        return np.zeros((0, DIMENSIONS), dtype=np.float64)

    centre = (lo + hi) / 2.0
    half = (hi - lo) * PROBE_INSET / 2.0
    rng = np.random.default_rng(PROBE_SEED)
    offsets = rng.uniform(-1.0, 1.0, size=(N_PROBES, DIMENSIONS)) * half
    return np.asarray(centre + offsets, dtype=np.float64)


def _relative_spread(energies: list[float]) -> tuple[float, tuple[float, float]]:
    """Range over the largest magnitude present, so a near-zero energy cannot blow it up."""
    lo, hi = min(energies), max(energies)
    scale = max(abs(v) for v in energies)
    return ((hi - lo) / scale if scale else 0.0), (lo, hi)


def _compare_energies(
    comparison: Comparison, *, tolerance: float, approximation_tolerance: float
) -> None:
    """Spread across the reference tier; approximations measured against it.

    The partition is what keeps both numbers meaningful. Two finite-difference
    codes agreeing to 2.3% is evidence about *them*, and it disappears if a
    Generalized Born answer 20% away is averaged in. Equally, calling that GB
    answer a disagreement would report the method working as designed as though
    it were a bug.
    """
    reference = [r for r in comparison.runs if r.accuracy_tier is AccuracyTier.REFERENCE]
    approximate = [r for r in comparison.runs if r.accuracy_tier is AccuracyTier.APPROXIMATE]
    ref_energies = [r.energy_kj_mol for r in reference if r.energy_kj_mol is not None]
    approx_energies = [r.energy_kj_mol for r in approximate if r.energy_kj_mol is not None]

    spread_ok: bool | None = None
    consensus: float | None = None

    if len(ref_energies) >= 2:  # noqa: PLR2004
        comparison.energy_spread, comparison.energy_range_kj_mol = _relative_spread(ref_energies)
        spread_ok = comparison.energy_spread <= tolerance
        consensus = float(np.mean(ref_energies))
    elif len(ref_energies) == 1:
        consensus = ref_energies[0]
        if approximate:
            comparison.notes.append(
                "one reference backend, so there is no reference spread; the "
                "approximation is measured against it alone"
            )
    elif len(approx_energies) >= 2:  # noqa: PLR2004
        # Nothing here establishes accuracy: two approximations agreeing means
        # they implement the same approximation, not that either is right. The
        # spread is still worth reporting, at the tolerance that applies to it.
        comparison.tolerance = approximation_tolerance
        comparison.energy_spread, comparison.energy_range_kj_mol = _relative_spread(approx_energies)
        spread_ok = comparison.energy_spread <= approximation_tolerance
        comparison.notes.append(
            "no reference-tier backend ran, so this spread is between "
            "approximations and nothing in it establishes accuracy"
        )

    for run in approximate:
        if run.energy_kj_mol is None or not consensus:
            continue
        comparison.approximation_deviation[run.name] = abs(run.energy_kj_mol - consensus) / abs(
            consensus
        )
    comparison.approximations_agree = all(
        deviation <= approximation_tolerance
        for deviation in comparison.approximation_deviation.values()
    )

    if spread_ok is None and not comparison.approximation_deviation:
        comparison.notes.append("fewer than two backends returned an energy")
    else:
        comparison.agrees = spread_ok is not False and comparison.approximations_agree


def compare_results(
    runs: list[BackendRun],
    *,
    tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    approximation_tolerance: float = DEFAULT_APPROXIMATION_TOLERANCE,
    allow_mismatch: bool = False,
) -> Comparison:
    """Reduce N backend answers to a spread, refusing when that would mislead."""
    if len(runs) < 2:  # noqa: PLR2004 — a spread needs two things to spread between
        raise Incomparable(
            f"cross-validation needs at least two backends, got {len(runs)}. "
            "One backend trivially agrees with itself."
        )

    notes = check_comparable(runs, allow_mismatch=allow_mismatch)
    comparison = Comparison(
        runs=runs,
        tolerance=tolerance,
        approximation_tolerance=approximation_tolerance,
        notes=notes,
    )
    _compare_energies(
        comparison, tolerance=tolerance, approximation_tolerance=approximation_tolerance
    )

    grids = [r.potential for r in runs if r.potential is not None]
    if len(grids) >= 2:  # noqa: PLR2004
        points = overlap_probe_points(grids)
        if len(points) == 0:
            comparison.notes.append("grids do not overlap; no pointwise comparison")
        else:
            sampled = np.array([g.value_at(points) for g in grids])
            usable = ~np.any(np.isnan(sampled), axis=0)
            if not usable.any():
                comparison.notes.append("no probe point fell inside every grid")
            else:
                values = sampled[:, usable]
                spread = values.max(axis=0) - values.min(axis=0)
                comparison.n_probes = int(usable.sum())
                comparison.potential_rmsd_kt_e = float(np.sqrt(np.mean(spread**2)))
                comparison.potential_max_abs_kt_e = float(np.max(np.abs(spread)))

    return comparison


def validate(
    # Mapping rather than dict: dict is invariant in its value type, so a caller
    # holding a `dict[str, ApbsSolver]` could not pass it without a cast.
    solvers: Mapping[str, Solver[FiniteDifferenceRequest]],
    request: FiniteDifferenceRequest,
    *,
    tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    approximation_tolerance: float = DEFAULT_APPROXIMATION_TOLERANCE,
    allow_mismatch: bool = False,
) -> Comparison:
    """Solve one request with every backend and compare the answers.

    The request is shared by construction, which removes the whole class of
    "were they actually asked the same thing" mistakes. What remains to check is
    whether they *answered* the same thing, which `check_comparable` does.
    """
    runs = [
        BackendRun.from_result(name, solver.solve(request), request)
        for name, solver in solvers.items()
    ]
    return compare_results(
        runs,
        tolerance=tolerance,
        approximation_tolerance=approximation_tolerance,
        allow_mismatch=allow_mismatch,
    )


def validate_system(
    system: System,
    backends: Sequence[Backend],
    *,
    tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    approximation_tolerance: float = DEFAULT_APPROXIMATION_TOLERANCE,
    allow_mismatch: bool = False,
) -> Comparison:
    """Compare solvers that cannot read each other's requests.

    Each backend is handed the request its family speaks, built from one
    `System`, so "were they asked the same thing" stays true by construction
    even across the FD/BEM divide. What is compared is the energy: a volumetric
    map and a triangulated surface have no shared representation, and inventing
    one would be the same category error `check_comparable` exists to refuse.
    """
    runs = []
    for backend in backends:
        request = system.request_for(backend.family)
        runs.append(BackendRun.from_result(backend.name, backend.solver.solve(request), request))
    return compare_results(
        runs,
        tolerance=tolerance,
        approximation_tolerance=approximation_tolerance,
        allow_mismatch=allow_mismatch,
    )


def with_surface_model(
    request: FiniteDifferenceRequest, model: SurfaceModel
) -> FiniteDifferenceRequest:
    """The same request on a different surface, for picking a shared model."""
    return dataclasses.replace(
        request, solvent=dataclasses.replace(request.solvent, surface_model=model)
    )


# --- grading a field against the incumbents ----------------------------------
#
# The energy half of this module asks "do the backends agree with each other".
# This asks a narrower and harder question: **is a candidate solver's field as
# good as the best one already installed**, measured against the closed form
# both are approximating. ROADMAP.md section 12 M1b is the caller.
#
# Grading against the incumbents rather than against a round number is what
# makes the bar mean something. debye reproduces DelPhi C++'s discretization to
# three decimal places, so "no worse than the worst incumbent" is a bar it meets
# by construction rather than by merit — a check that cannot fail, in section
# 7's sense. Against the *best* incumbent it is a real measurement: agreeing
# with DelPhi does not make a solver within a factor of APBS.

# How much worse than the best installed reference solver a candidate may be.
# Measured rather than chosen: on the corpus's fine van der Waals case debye is
# 1.77x the best reference at the closest sample, and on the coarse one it is
# 5.2x. A factor of 2 therefore passes what is already as good as the incumbents
# and fails what is not, which is the only useful place to put it.
DEFAULT_FIELD_FACTOR = 2.0

# A candidate and something to grade it against.
MIN_FIELD_BACKENDS = 2


@dataclass(frozen=True)
class FieldGrade:
    """A candidate backend's field against the best reference solver's, per radius.

    Every backend is sampled at the **same physical radii**, which is the whole
    reason this is not `corpus`'s field check. That one samples `a + k*h` on each
    backend's own achieved spacing — correct there, because every sample must
    clear *its* interface cell — and it means a coarser grid is sampled further
    out, where the error is smaller. Comparing those numbers across backends
    reads a sampling difference as an accuracy difference: on the corpus's coarse
    case DelPhi is sampled at r = 4.0 A and APBS at r = 3.81 A.
    """

    radii_a: tuple[float, ...]
    spacing_used_a: float  # the coarsest grid in the comparison, which sets the radii
    errors: Mapping[str, tuple[float, ...]]  # backend -> worst-direction error per radius
    worst_directions: Mapping[str, tuple[str, ...]]
    best_reference: tuple[str, ...]  # which reference solver was best, per radius
    candidate: str
    ratios: tuple[float, ...]  # candidate error / best reference error
    factor: float = DEFAULT_FIELD_FACTOR
    notes: tuple[str, ...] = ()

    @property
    def agrees(self) -> bool:
        return all(ratio <= self.factor for ratio in self.ratios)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "radii_a": list(self.radii_a),
            "spacing_used_a": self.spacing_used_a,
            "errors": {name: list(values) for name, values in self.errors.items()},
            "worst_directions": {name: list(v) for name, v in self.worst_directions.items()},
            "best_reference": list(self.best_reference),
            "ratios": list(self.ratios),
            "factor": self.factor,
            "agrees": self.agrees,
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        verdict = "within" if self.agrees else "OUTSIDE"
        worst = max(range(len(self.ratios)), key=lambda i: self.ratios[i])
        best = self.best_reference[worst]
        return (
            f"{self.candidate} is {verdict} {self.factor:g}x the best reference field: "
            f"worst {self.ratios[worst]:.2f}x at r = {self.radii_a[worst]:.4g} A "
            f"({self.errors[self.candidate][worst]:.3%} against {best}'s "
            f"{self.errors[best][worst]:.3%})"
        )


def check_field_comparable(runs: Sequence[BackendRun]) -> None:
    """Refuse a field comparison across backends that were asked different questions.

    Deliberately narrower than `check_comparable`. Surface model and equation
    must match, because both change the field. **`EnergyTerm` is not checked**,
    and that is the point rather than an oversight: it describes which *energy* a
    backend reports, and DelPhi's differs from APBS's while their fields are
    directly comparable. Requiring it here would refuse the one comparison this
    exists for. That the two axes come apart is the finding M0 recorded — on the
    energy DelPhi is four thousand times sharper than APBS, and on the field they
    are near-peers.
    """
    if len(runs) < MIN_FIELD_BACKENDS:
        raise Incomparable(
            f"a field comparison needs at least {MIN_FIELD_BACKENDS} backends with a "
            f"volumetric map, got {len(runs)}. Boundary-element backends return potentials "
            "on the interface and analytic ones return no field at all."
        )
    for attribute, label in (("surface_model", "surface model"), ("equation", "equation")):
        values = {getattr(run, attribute) for run in runs}
        if len(values) > 1:
            raise Incomparable(
                f"cannot compare fields across differing {label}: {sorted(str(v) for v in values)}"
            )
    if len({run.ionic_strength for run in runs}) > 1:
        raise Incomparable("cannot compare fields across differing ionic strength")
    for attribute, label in (
        ("solvent_dielectric", "solvent dielectric"),
        ("temperature", "temperature"),
    ):
        values = {getattr(run, attribute) for run in runs}
        if len(values) > 1:
            raise Incomparable(
                f"cannot compare fields across differing {label}: {sorted(values)}. The "
                "closed form is evaluated once for every backend, so a mismatch here "
                "moves the reference by a factor common to all of them — and a common "
                "offset drives a ratio of errors towards 1.0, which reads as agreement."
            )


def grade_field(
    runs: Sequence[BackendRun],
    *,
    candidate: str,
    centre: FloatArray,
    radius_a: float,
    charge_e: float,
    cells_out: tuple[int, ...] = (2, 4, 8),
    factor: float = DEFAULT_FIELD_FACTOR,
) -> FieldGrade:
    """Grade `candidate`'s field against the best reference-tier solver present.

    The sample radii come from the **coarsest** grid in the comparison, so every
    backend's sample clears its own interface cell — a backend on a finer grid is
    further from the boundary in its own cells, never closer.
    """
    from sashimi.analytic import born_potential  # noqa: PLC0415 — closed form, not a solver
    from sashimi.field import errors_by_radius, sample_radii  # noqa: PLC0415

    with_field = [run for run in runs if run.potential is not None]
    check_field_comparable(with_field)

    by_name = {run.name: run for run in with_field}
    if candidate not in by_name:
        raise Incomparable(
            f"{candidate!r} produced no volumetric field to grade; present: {sorted(by_name)}"
        )
    references = [
        run
        for run in with_field
        if run.name != candidate and run.accuracy_tier is AccuracyTier.REFERENCE
    ]
    if not references:
        raise Incomparable(
            f"nothing to grade {candidate!r} against: no other reference-tier backend "
            "produced a field. An approximation is not a yardstick for a discretization."
        )
    if by_name[candidate].ionic_strength != 0.0:
        raise Incomparable(
            "the Born potential is unscreened, so a field cannot be graded against it at "
            f"{by_name[candidate].ionic_strength} M"
        )

    spacing = max(float(np.max(run.potential.spacing)) for run in with_field)  # type: ignore[union-attr]
    # Through `sashimi.field`, not inline: that module owns the rule that a
    # sample must clear the interface cell, and an inline `a + k*h` here is the
    # second copy its docstring says there must not be. It also rejects
    # `cells_out=()`, which would otherwise produce no ratios at all — and
    # `agrees` is an `all()`, so no ratios reads as agreement.
    radii = sample_radii(radius_a, spacing, cells_out)

    reference_solvent = with_field[0]

    def exact_at(radius: float) -> float:
        return born_potential(
            radius,
            charge_e,
            reference_solvent.solvent_dielectric,
            reference_solvent.temperature,
        )

    errors: dict[str, tuple[float, ...]] = {}
    directions: dict[str, tuple[str, ...]] = {}
    for run in with_field:
        assert run.potential is not None
        found, where = errors_by_radius(run.potential, centre, radii, exact_at)
        errors[run.name] = tuple(found)
        directions[run.name] = tuple(where)

    best_names, ratios = [], []
    for index in range(len(radii)):
        best = min(references, key=lambda r, i=index: errors[r.name][i])  # type: ignore[misc]
        best_names.append(best.name)
        ratios.append(errors[candidate][index] / errors[best.name][index])

    notes = tuple(
        f"{run.name} solved on h = {float(np.max(run.potential.spacing)):.4f} A"  # type: ignore[union-attr]
        for run in with_field
    )
    return FieldGrade(
        radii_a=tuple(radii),
        spacing_used_a=spacing,
        errors=errors,
        worst_directions=directions,
        best_reference=tuple(best_names),
        candidate=candidate,
        ratios=tuple(ratios),
        factor=factor,
        notes=notes,
    )
