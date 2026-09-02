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
    Diagnostics,
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
    "LATTICE_OFFSET_CELLS",
    "LATTICE_RTOL",
    "MIN_BOX_MARGIN_RATIO",
    "N_PROBES",
    "PROBE_SEED",
    "Backend",
    "BackendRun",
    "Comparison",
    "FieldGrade",
    "Incomparable",
    "SolverFamily",
    "System",
    "check_same_lattice",
    "check_samples_clear_the_box",
    "compare_grids",
    "compare_results",
    "grade_field",
    "overlap_probe_points",
    "validate",
    "validate_system",
]

N_PROBES = 200
# A spread needs a stable mean and 200 draws give one. `compare_grids` also
# reports a maximum, which is an extreme-value statistic that 200 uniform samples
# underestimate badly, so it samples harder. Interpolating both grids costs
# 0.6 ms at 200 points and 6.2 ms at 20,000 — free beside reading the maps.
COMPARE_PROBES = 20_000
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
    # and the grade passes. `corpus.AnalyticField.exact_at` used to refuse the
    # same class of mismatch on ionic strength and now *describes* it, since M3
    # added the screened closed form; these are the other two axes, and they are
    # carried rather than refused because a closed form exists for both.
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


def overlap_probe_points(grids: list[PotentialGrid], *, n: int = N_PROBES) -> FloatArray:
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
    offsets = rng.uniform(-1.0, 1.0, size=(n, DIMENSIONS)) * half
    return np.asarray(centre + offsets, dtype=np.float64)


def compare_grids(a: PotentialGrid, b: PotentialGrid) -> Diagnostics:
    """Compare two potential maps, on a shared lattice or across different ones.

    **Two backends usually do not land on the same lattice, and that is the case
    this exists for.** Each sizes its box by its own rules, so asking two of them
    for 0.5 A on one structure generally gives two boxes with different origins,
    shapes and achieved spacings, and a comparison that demanded an identical
    lattice refused exactly the solver-versus-solver question it is meant to
    answer — with a geometry error, which reads like a caller mistake rather
    than a fact about finite-difference solvers.

    *Usually, not always, and the exception matters more than the rule.* debye
    and the DelPhi backend resolve to a bit-identical lattice on 23 of the
    corpus's 100 cases — every Kirkwood case and most Born-ion ones — which
    `tests/test_kirkwood_field.py` pins. So a `"lattice"` result does **not**
    imply one solver produced both maps, and ROADMAP.md section 12 records what
    follows: where two solvers share a lattice, a near-field comparison between
    them measures the lattice and not the solvers. This function is handed two
    grids and cannot know their provenance, so it cannot make that call for the
    caller — it says which comparison it made and leaves the reading to them.

    So there are two methods and the result always says which one ran:

    - `"lattice"` — identical shape, origin and spacing, so every node pairs
      with its twin and the difference is exact. This is the mutant-versus-
      wildtype case, where one solver produced both maps.
    - `"sampled"` — the boxes differ, so the maps are interpolated at
      deterministic points inside the region both cover and compared there.
      Fewer points, and each carries interpolation error on top of the
      solvers' own.

    *The two numbers are not interchangeable and the method field is how a
    caller tells them apart.* A sampled RMSD is over `n_points` probes in the
    shared box; a lattice RMSD is over every node of a box the two agreed on.
    Reporting one as the other is how a grid-phase artefact gets read as a
    physics disagreement — ROADMAP.md section 12 records a five-figure agreement
    between two solvers that turned out to be a shared lattice rather than
    shared physics.

    Raises `Incomparable` when the boxes do not overlap at all, because there is
    then no region in which the question has an answer.
    """
    same_lattice = (
        a.shape == b.shape and np.allclose(a.spacing, b.spacing) and np.allclose(a.origin, b.origin)
    )

    if same_lattice:
        diff = a.values - b.values
        flat_a, flat_b = a.values.reshape(-1), b.values.reshape(-1)
        n_points = int(a.values.size)
        method = "lattice"
    else:
        points = overlap_probe_points([a, b], n=COMPARE_PROBES)
        if len(points) == 0:
            raise Incomparable(
                "these maps cover no common region, so there is no volume in which "
                "to compare them; re-solve them on boxes that overlap"
            )
        sampled = np.array([g.value_at(points) for g in (a, b)])
        usable = ~np.any(np.isnan(sampled), axis=0)
        if not usable.any():
            raise Incomparable(
                "no sample point fell inside both maps, although their boxes overlap; "
                "the shared region is too thin to compare in"
            )
        flat_a, flat_b = sampled[0, usable], sampled[1, usable]
        diff = flat_a - flat_b
        n_points = int(usable.sum())
        method = "sampled"

    rmsd = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.abs(diff).max())

    # A constant map has undefined correlation, not zero — but "constant" is not
    # `std == 0` once interpolation is involved. Trilinear sampling of a constant
    # field returns values differing in the last bits, so its std is ~1e-16
    # rather than 0, and an exact-zero guard hands `corrcoef` pure rounding noise:
    # two byte-identical fields reported r = 0.023 beside an RMSD of 2e-16. The
    # threshold has to be relative to the values, and a non-finite result is
    # undefined however it arose.
    scale = max(float(np.abs(flat_a).max()), float(np.abs(flat_b).max()), 1.0)
    correlation: float | None = None
    if flat_a.std() > 1e-12 * scale and flat_b.std() > 1e-12 * scale:
        candidate = float(np.corrcoef(flat_a, flat_b)[0, 1])
        correlation = candidate if np.isfinite(candidate) else None

    out: Diagnostics = {
        "method": method,
        "n_points": n_points,
        "rmsd_kT_e": rmsd,
        "mean_diff_kT_e": float(diff.mean()),
        "correlation": correlation,
        "shape": list(a.shape),
    }

    # The maximum is the one statistic sampling cannot estimate. A mean and an
    # RMS converge over the shared region; a maximum over N draws is a lower
    # bound on the true maximum and stays one however many draws there are. So
    # it does not get to share a key with the exact maximum the lattice path
    # computes — a caller reading `max_abs_diff_kT_e` gets the real thing or a
    # KeyError, never a quiet underestimate wearing its name.
    if method == "lattice":
        out["max_abs_diff_kT_e"] = max_abs
    else:
        out["max_abs_diff_over_samples_kT_e"] = max_abs
        out["shape_b"] = list(b.shape)
        out["note"] = (
            "grids differ in geometry, so this is an interpolated comparison over "
            f"{n_points} points in the *interior* of the region both cover — the "
            "middle half of each axis, away from the boundaries where a smoothed "
            "surface makes values sensitive to details that are not the solvers' "
            "arithmetic. Not a node-for-node difference and not comparable to one. "
            "The maximum is over those samples and is a lower bound on the true "
            "maximum, which is why it is not reported under `max_abs_diff_kT_e`"
        )
    return out


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
# Measured rather than chosen. **The number the shipped gate actually reads is
# 1.69x** — debye against APBS on `born-ion-vdw-r1`, a/h = 2 — and the typical one
# is 1.01x, so a factor of 2 leaves 1.18x of headroom above what near-peer
# discretizations do to each other. Quote the gate's number here and not the
# 1.62x from the offline shared-spacing sweep: they measure slightly different
# things (the sweep joins backends wherever they happen to coincide; the gate
# pins one lattice per case), and this comment is the record of where the
# constant came from, so it has to match the assertion that runs. It has already
# drifted once — a review caught it reading 1.62 when the gate said 1.65, and
# then the box-margin fix moved the gate to 1.69.
DEFAULT_FIELD_FACTOR = 2.0

# A candidate and something to grade it against.
MIN_FIELD_BACKENDS = 2

# How closely two backends' achieved spacings must agree to count as the same
# lattice. Not a physical tolerance — it absorbs the round trip through DelPhi's
# seven-significant-figure output and nothing else. Two backends that genuinely
# land on the same grid agree to ~3e-8 relative; the next distinct lattice is
# ~1e-2 away, so there is nothing for this number to be sensitive to.
LATTICE_RTOL = 1e-6

# The same question for *where the solute sits within a cell*, and a separate
# number because it is a separate quantity: a fraction of a cell, not a relative
# error on a length. Both bounds are wide. DelPhi's origin arrives through the
# same seven-figure format, which on a ~10 A coordinate is ~1e-5 A, or ~2e-5 of a
# half-angstrom cell; and the phase differences that matter are tenths of a cell,
# since it takes a face centre crossing the boundary to change the staircase. So
# anything from ~1e-4 to ~1e-2 would behave identically here.
LATTICE_OFFSET_CELLS = 1e-3

# A sample must be at least this many times its own radius clear of the box face.
# The boundary condition is applied on that face, so a sample near it reads the
# approximation there rather than the solver — and it is the **reference** that
# gets inflated, which flatters the candidate. Measured on the M1b cases at fixed
# lattice, as the outermost sample's worst-direction error against margin/r_out:
#
#   margin/r_out   APBS            DelPhi C++      debye
#   0.14           0.637%          0.118%          0.111%     <- 5.4x inflated
#   0.60           0.301-0.413%    0.233-0.370%    0.234-0.380%
#   >= 1.29        0.119-0.396%    converged       converged
#
# Only APBS is affected, which is consistent with mg-auto's focusing carrying its
# coarse grid's boundary treatment inward. A floor of 1.0 is the round number
# below the cheapest clean measurement and above the worst dirty one — "the
# sample is nearer the solute than the box face" — and every M1b case now clears
# it by 29% or more. It is a floor, not a target: prefer paddings that clear it.
MIN_BOX_MARGIN_RATIO = 1.0


@dataclass(frozen=True)
class FieldGrade:
    """A candidate backend's field against the best reference solver's, per radius.

    Every backend is sampled at the same physical radii **on one shared lattice**,
    which is the whole reason this is not `corpus`'s field check. That one samples
    `a + k*h` on each backend's own achieved spacing — correct there, because
    every sample must clear *its* interface cell — and it means a coarser grid is
    sampled further out, where the error is smaller.

    Equal radii were the first half of that and were not enough. The second half
    is `check_same_lattice`, and it is the half M1b was missing: the near-field
    error varies more with grid phase than between solvers, so a comparison at
    equal radii but unequal spacing still reads phase as accuracy. `a/h` is
    carried on the grade for the same reason — it is the number that predicts the
    result, so a verdict that does not state it is not reproducible.
    """

    radii_a: tuple[float, ...]
    spacing_used_a: float  # the one lattice every backend solved on
    cells_across_radius: float  # a/h — what actually predicts the error's size
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
            "cells_across_radius": self.cells_across_radius,
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
            f"{self.errors[best][worst]:.3%}), at h = {self.spacing_used_a:.4g} A, "
            f"a/h = {self.cells_across_radius:.4g}"
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


def check_same_lattice(runs: Sequence[BackendRun], centre: FloatArray) -> float:
    """Refuse a field comparison across backends that solved on different lattices.

    Returns the common spacing. **This is the check M1b needed and did not have,
    and its absence produced a wrong milestone verdict**, so it is worth stating
    what it is for rather than only what it does.

    Sampling every backend at the same physical radii is necessary and is not
    sufficient. The near-field error of a staircase-dielectric solver depends far
    more strongly on *where the lattice falls* than on which solver is running
    it: holding the sample radius fixed at r = 4 A on a 3 A sphere and varying
    only the spacing over 0.43-0.50 A, the worst-direction error swings

        APBS        0.585% .. 3.915%   (6.7x)
        DelPhi C++  0.763% .. 3.837%   (5.0x)
        debye       0.773% .. 4.101%   (5.3x)

    and on the 1 A sphere APBS alone spans 21x. The error collapses wherever a/h
    approaches an integer, because the discretized cavity is a staircase and its
    shape changes discretely as face centres cross the sphere. That is a property
    of hard midpoint dielectric assignment, which all three backends share — not
    a property of any one of them.

    So a grade taken across two lattices reads phase as accuracy, and it does so
    at a magnitude that swamps every real difference between these solvers. M1b
    graded debye at a/h = 6.46 against DelPhi at a/h = 6.00 and read 5.24x; at
    the eleven spacings the two backends both land on, the ratio is 0.994-1.013x.

    The comparison is refused rather than annotated because there is no honest
    way to interpret the number. Reported per-backend spacing would have made the
    mismatch *visible* — `grade_field` already emitted it in `notes` — and it was
    visible, and the verdict was still wrong, which is ROADMAP.md section 7's
    point about a check that cannot fail wearing the costume of a measurement.

    Phase is spacing *and* offset: two grids of the same h with the solute half a
    cell apart in them are different staircases. Both are checked. Every backend
    here centres an odd-dimensioned grid on the molecule, so the offsets agree by
    construction today — which is exactly why the check is written from the
    geometry rather than trusted to stay that way.
    """
    grids = [run.potential for run in runs if run.potential is not None]
    if not grids:
        raise Incomparable("no backend produced a volumetric field")

    spacings = np.array([np.asarray(grid.spacing, dtype=float) for grid in grids])
    # Per axis **across grids**, not a global max-min. A global span folds the
    # grid's own anisotropy into the cross-backend spread, so two byte-identical
    # anisotropic lattices are refused — and the message then prints one `h` per
    # backend and contradicts itself. `apbs.grid.size_grid` returns anisotropic
    # spacing for any non-cubic solute (on `peptide-vdw`, [0.4672, 0.4393,
    # 0.4004]), so this fires the first time a grade is pointed at anything that
    # is not a single sphere.
    span = np.ptp(spacings, axis=0)
    if np.any(span > LATTICE_RTOL * spacings.mean(axis=0)):
        detail = ", ".join(
            f"{run.name} h = {np.round(np.asarray(run.potential.spacing, dtype=float), 6).tolist()}"
            for run in runs
            if run.potential is not None
        )
        raise Incomparable(
            f"cannot grade a field across differing lattices: {detail}. The near-field "
            "error depends more strongly on where the lattice falls relative to the "
            "solute than on which solver produced it — up to 21x across one backend at "
            "fixed physical radius — so a ratio taken across two spacings reports grid "
            "phase as accuracy. Solve every backend on one lattice, by choosing a "
            "padding they all round to the same spacing."
        )

    centre = np.asarray(centre, dtype=float).reshape(DIMENSIONS)
    offsets = np.array(
        [
            np.mod((centre - np.asarray(grid.origin, dtype=float)) / np.asarray(grid.spacing), 1.0)
            for grid in grids
        ]
    )
    # Wrapped, because an offset of 1 - eps and one of eps are the same phase.
    deltas = np.abs(offsets - offsets[0])
    deltas = np.minimum(deltas, 1.0 - deltas)
    if float(deltas.max()) > LATTICE_OFFSET_CELLS:
        raise Incomparable(
            "cannot grade a field across lattices that place the solute differently "
            f"within a cell: fractional offsets {offsets.tolist()}, differing by "
            f"{float(deltas.max()):.3g} of a cell. Same spacing, different staircase."
        )
    # The **coarsest** axis, because `sample_radii` has to clear the interface
    # cell on every axis and a mean would under-report it on an anisotropic grid.
    # Every grid here is the same grid by now, so this is a max over axes only.
    return float(spacings.max())


def check_samples_clear_the_box(
    runs: Sequence[BackendRun],
    centre: FloatArray,
    radius_a: float,
    spacing: float,
    cells_out: tuple[int, ...],
) -> None:
    """Refuse a grade whose outermost sample sits too near the box face.

    The sibling of `check_same_lattice`, and found the same way — by a review
    asking what *else* the comparison left free once the lattice was pinned.

    The boundary condition lives on the box face. A sample close to it reads that
    approximation rather than the solver, and the damage is one-sided: on
    `born-ion-vdw` at a margin of 1 A, **APBS** reads 0.637% at the outermost
    radius against 0.124% on the same lattice in a larger box, while DelPhi and
    debye are unmoved at 0.118 and 0.111%. An inflated *reference* raises the
    yardstick, and since the verdict is candidate/reference, it flatters the
    candidate — the exact direction this module exists to refuse.

    Worth being explicit that this was missed by a control that looked like it
    covered it: box-size independence was checked at the *innermost* sample
    (r = a + 2h), where padding moves the answer by 2%, and the contamination is
    at the *outermost* (r = a + 8h). A control has to be evaluated where the
    effect would be, not where the measurement is most convenient.
    """
    if not cells_out:
        return  # `sample_radii` refuses this, with a better message
    outermost = radius_a + max(cells_out) * spacing
    centre = np.asarray(centre, dtype=float).reshape(DIMENSIONS)
    for run in runs:
        if run.potential is None:
            continue
        grid = run.potential
        origin = np.asarray(grid.origin, dtype=float)
        far = origin + (np.asarray(grid.shape) - 1) * np.asarray(grid.spacing, dtype=float)
        margin = float(np.min(np.minimum(centre - origin, far - centre)) - outermost)
        if margin < MIN_BOX_MARGIN_RATIO * outermost:
            raise Incomparable(
                f"{run.name}'s box clears the outermost sample (r = {outermost:.4g} A) by "
                f"only {margin:.4g} A, under the {MIN_BOX_MARGIN_RATIO:g}x of that radius "
                "this needs. The boundary condition is applied on the box face, so a "
                "sample near it measures that approximation rather than the solver — and "
                "it inflates the *reference*, which flatters the candidate. Increase "
                "padding, keeping every backend on one lattice."
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

    **Every backend must have solved on one lattice**, which `check_same_lattice`
    enforces and which is what makes the ratio mean anything — grid phase moves
    the near-field error by more than these solvers differ from each other. The
    sample radii therefore come from that single shared spacing (its coarsest
    axis, so a sample clears the interface cell on every axis), and the box must
    clear the outermost sample by `MIN_BOX_MARGIN_RATIO` times its radius.

    An earlier version took the radii from the *coarsest grid* of several, which
    is what you do when the lattices are allowed to differ. They are not.
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
        # The stated reason is no longer that no closed form exists — M3 added
        # `analytic.screened_born_potential`, and the corpus grades salted fields
        # against it. What is missing here is the Stern radius: `FieldRun` carries
        # no `ion_radius`, so this function cannot say where the screening starts,
        # and a field graded at the wrong exclusion radius is the mismatched
        # reference the comment on `FieldRun` describes. Lifting this means adding
        # that field and cross-checking it against each run, which is a change to
        # a gate that has been wrong twice and was not worth folding into M3.
        raise Incomparable(
            "a field grade needs the ion-exclusion radius the run used, and `FieldRun` "
            f"does not carry one, so {candidate!r} cannot be graded at "
            f"{by_name[candidate].ionic_strength} M. The screened closed form exists "
            "(`sashimi.analytic.screened_born_potential`); the missing piece is the "
            "Stern radius, not the physics."
        )

    # One lattice for every backend, or no comparison. See `check_same_lattice`:
    # grid phase moves the near-field error by more than the solvers differ, so
    # this is what makes the ratio below mean anything at all.
    spacing = check_same_lattice(with_field, centre)
    check_samples_clear_the_box(with_field, centre, radius_a, spacing, cells_out)
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

    notes = (
        f"every backend solved on h = {spacing:.6g} A, a/h = {radius_a / spacing:.4g}",
        *(f"{run.name}: {run.accuracy_tier.value}" for run in with_field),
    )
    return FieldGrade(
        radii_a=tuple(radii),
        spacing_used_a=spacing,
        cells_across_radius=radius_a / spacing,
        errors=errors,
        worst_directions=directions,
        best_reference=tuple(best_names),
        candidate=candidate,
        ratios=tuple(ratios),
        factor=factor,
        notes=notes,
    )
