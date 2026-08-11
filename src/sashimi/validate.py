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
from enum import StrEnum
from typing import Any

import numpy as np

from sashimi.errors import InputError
from sashimi.protocol import (
    DIMENSIONS,
    BoundaryElementRequest,
    EnergyTerm,
    Equation,
    FiniteDifferenceRequest,
    FloatArray,
    GridSpec,
    PotentialGrid,
    PQRData,
    SolventModel,
    Solver,
    SolveRequest,
    SolveResult,
    SurfaceModel,
)

__all__ = [
    "N_PROBES",
    "PROBE_SEED",
    "Backend",
    "BackendRun",
    "Comparison",
    "Incomparable",
    "SolverFamily",
    "System",
    "compare_results",
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


class SolverFamily(StrEnum):
    """Which request family a backend takes.

    The `Solver` protocol is generic in its request type, so a type checker
    already refuses to hand a `BoundaryElementRequest` to an FD backend. That
    guarantee is static, and cross-family validation has to make the same
    decision at runtime — hence an explicit declaration rather than
    introspection, which cannot recover a type parameter.
    """

    FINITE_DIFFERENCE = "finite-difference"
    BOUNDARY_ELEMENT = "boundary-element"


@dataclass(frozen=True)
class System:
    """One physical system, expressible as either family's request.

    This is the seam ROADMAP.md section 2 designed `SolveRequest` for and never
    had to use until a boundary-element backend existed. Everything a solve
    needs regardless of family — structure, solvent, what to compute — lives on
    the base class; `GridSpec` and `mesh_density` are the family-specific parts,
    and this holds both so that one physical question can be put to solvers that
    cannot read each other's requests.

    `want_potential` defaults to False because the two families do not return
    comparable fields: a volume and a triangulated surface have no shared
    representation, so a cross-family comparison rests on energies alone.
    Same-family runs can still ask for potentials and get the pointwise
    comparison.
    """

    structure: PQRData
    # Frozen dataclasses, so a shared instance is safe; the protocol's own
    # request types default the same way.
    solvent: SolventModel = SolventModel()  # noqa: RUF009
    grid: GridSpec = GridSpec()  # noqa: RUF009
    mesh_density: float = 2.0
    want_energy: bool = True
    want_potential: bool = False

    def request_for(self, family: SolverFamily) -> SolveRequest:
        """The same physical question, in the dialect that family can read."""
        if family is SolverFamily.FINITE_DIFFERENCE:
            return FiniteDifferenceRequest(
                structure=self.structure,
                solvent=self.solvent,
                want_energy=self.want_energy,
                want_potential=self.want_potential,
                grid=self.grid,
            )
        return BoundaryElementRequest(
            structure=self.structure,
            solvent=self.solvent,
            want_energy=self.want_energy,
            want_potential=self.want_potential,
            mesh_density=self.mesh_density,
        )


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
    wall_seconds: float | None = None

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
            wall_seconds=result.provenance.wall_seconds,
        )


@dataclass
class Comparison:
    """What the backends said, and whether it constitutes agreement."""

    runs: list[BackendRun]
    energy_spread: float | None = None  # max pairwise relative difference
    energy_range_kj_mol: tuple[float, float] | None = None
    potential_rmsd_kt_e: float | None = None
    potential_max_abs_kt_e: float | None = None
    n_probes: int = 0
    agrees: bool = False
    tolerance: float = DEFAULT_ENERGY_TOLERANCE
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backends": [r.name for r in self.runs],
            "energies_kj_mol": {r.name: r.energy_kj_mol for r in self.runs},
            "energy_spread": self.energy_spread,
            "energy_range_kj_mol": list(self.energy_range_kj_mol)
            if self.energy_range_kj_mol
            else None,
            "potential_rmsd_kt_e": self.potential_rmsd_kt_e,
            "potential_max_abs_kt_e": self.potential_max_abs_kt_e,
            "n_probes": self.n_probes,
            "tolerance": self.tolerance,
            "agrees": self.agrees,
            "notes": self.notes,
        }

    def summary(self) -> str:
        if self.energy_spread is None:
            return "no energies to compare"
        lo, hi = self.energy_range_kj_mol or (float("nan"), float("nan"))
        verdict = "agree" if self.agrees else "DISAGREE"
        detail = ""
        if self.potential_rmsd_kt_e is not None:
            detail = (
                f", potential RMSD {self.potential_rmsd_kt_e:.4f} kT/e over {self.n_probes} pts"
            )
        return (
            f"{len(self.runs)} backends {verdict}: {self.energy_spread:.2%} spread "
            f"({lo:.3f} to {hi:.3f} kJ/mol, tolerance {self.tolerance:.0%}){detail}"
        )


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


def compare_results(
    runs: list[BackendRun],
    *,
    tolerance: float = DEFAULT_ENERGY_TOLERANCE,
    allow_mismatch: bool = False,
) -> Comparison:
    """Reduce N backend answers to a spread, refusing when that would mislead."""
    if len(runs) < 2:  # noqa: PLR2004 — a spread needs two things to spread between
        raise Incomparable(
            f"cross-validation needs at least two backends, got {len(runs)}. "
            "One backend trivially agrees with itself."
        )

    notes = check_comparable(runs, allow_mismatch=allow_mismatch)
    comparison = Comparison(runs=runs, tolerance=tolerance, notes=notes)

    energies = [r.energy_kj_mol for r in runs if r.energy_kj_mol is not None]
    if len(energies) >= 2:  # noqa: PLR2004
        lo, hi = min(energies), max(energies)
        scale = max(abs(v) for v in energies)
        comparison.energy_range_kj_mol = (lo, hi)
        comparison.energy_spread = (hi - lo) / scale if scale else 0.0
        comparison.agrees = comparison.energy_spread <= tolerance
    else:
        comparison.notes.append("fewer than two backends returned an energy")

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
    return compare_results(runs, tolerance=tolerance, allow_mismatch=allow_mismatch)


def validate_system(
    system: System,
    backends: Sequence[Backend],
    *,
    tolerance: float = DEFAULT_ENERGY_TOLERANCE,
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
    return compare_results(runs, tolerance=tolerance, allow_mismatch=allow_mismatch)


def with_surface_model(
    request: FiniteDifferenceRequest, model: SurfaceModel
) -> FiniteDifferenceRequest:
    """The same request on a different surface, for picking a shared model."""
    return dataclasses.replace(
        request, solvent=dataclasses.replace(request.solvent, surface_model=model)
    )
