"""The solver contract.

Everything here speaks physics — angstroms, molar ionic strength, dielectric
constants, kT/e, kJ/mol. Nothing here knows APBS exists, and
`tests/test_protocol_boundary.py` enforces that rather than trusting it.

The shape is driven by the FD/BEM split (ROADMAP.md §2). Finite-difference
solvers take a grid and return a volume; boundary-element solvers take a mesh
and return values on a surface. What both families share lives in
`SolveRequest`; what only one family has lives in its own request type, so a
request a backend cannot honor is unrepresentable rather than merely rejected.
That is why `GridSpec` and the choice of equation sit on
`FiniteDifferenceRequest` and not here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

DIMENSIONS = 3  # space, throughout

# Every array crossing this boundary is float64. Pinning the dtype in the type
# is what keeps `Any` from leaking out of numpy operations into the public API.
FloatArray = npt.NDArray[np.float64]

# Diagnostics are a deliberately open, JSON-shaped bag: what a backend can
# usefully report differs per backend, so the values are not a fixed union.
Diagnostics = dict[str, Any]

__all__ = [
    "DIMENSIONS",
    "AccuracyTier",
    "BoundaryElementRequest",
    "Diagnostics",
    "EnergyTerm",
    "Equation",
    "FiniteDifferenceRequest",
    "FloatArray",
    "GridSpec",
    "PQRData",
    "Potential",
    "PotentialGrid",
    "Provenance",
    "SolveRequest",
    "SolveResult",
    "SolventModel",
    "Solver",
    "SurfaceModel",
    "SurfacePotential",
]


class Equation(StrEnum):
    """Which Poisson-Boltzmann equation to solve.

    Lives on the finite-difference request only. BEM formulations are built on
    the linearized operator's Green function, so a nonlinear BEM request is not
    something to reject — it is something that cannot be written down.
    """

    LINEAR = "linear"
    NONLINEAR = "nonlinear"


class EnergyTerm(StrEnum):
    """Which energy a backend reports in `SolveResult.energy_kj_mol`.

    Not a formatting detail: two backends can solve the same equation on the
    same structure with the same surface and still return different quantities,
    and the difference is invisible in the number. APBS reports a difference
    between the solvated state and a uniform-dielectric, ion-free reference, so
    it carries the mobile-ion contribution; DelPhi reports the polarization term
    alone and does not move with ionic strength at all. At zero salt the two
    coincide, which is exactly what makes the gap easy to miss.

    Cross-solver comparison refuses to report a spread across differing terms
    for the same reason it refuses across differing surface models: the number
    would be a definitional difference misreported as a solver disagreement.
    ROADMAP.md section 14 states the rule as "same reported term", not "same
    equation".
    """

    POLAR_SOLVATION = "polar-solvation"  # solvated minus uniform-dielectric, ion-free
    REACTION_FIELD = "reaction-field"  # polarization only; excludes the mobile-ion term


class AccuracyTier(StrEnum):
    """How close to solving the Poisson-Boltzmann equation a backend gets.

    `EnergyTerm` answers "what quantity is this"; this answers "how was it
    obtained". Two backends that discretize the same equation differently
    should agree to within discretization noise — a few percent — and a wider
    gap means someone has a bug. An approximation that never solves the
    equation at all has no such obligation: it is expected to differ, by tens
    of percent, and saying so is the honest description rather than an excuse.

    Cross-solver comparison needs the distinction because it has exactly one
    verdict to give. Averaging an approximation into a spread with the solvers
    it approximates destroys both numbers: the disagreement is reported as if
    it were a defect, and the reference solvers' own agreement disappears into
    a tolerance wide enough to hide a real regression.
    """

    REFERENCE = "reference"  # a discretization of the PB equation
    APPROXIMATE = "approximate"  # an analytic approximation to it


class SurfaceModel(StrEnum):
    """How the dielectric boundary between solute and solvent is defined.

    Solver-neutral by design: physical descriptions, not any backend's keywords.
    Backends map them, and raise `UnsupportedRequest` for members they have no
    equivalent of.

    This is the single largest modelling confounder in the field — on this code,
    varying only this moves a dipeptide's solvation energy across 25.7%
    (ROADMAP.md §5). It therefore travels in provenance, and cross-solver
    comparison refuses to proceed across a mismatch.

    Spline-smoothed surfaces are deliberately absent: they exist to give smooth
    derivatives for force calculations, and using one for solvation energy is a
    misuse that accounts for most of that 25.7%. They stay reachable through
    backend-specific options, where the choice has to be made explicitly.
    """

    MOLECULAR = "molecular"  # solvent-excluded (Connolly) surface
    SMOOTHED_MOLECULAR = "smoothed-molecular"  # harmonically averaged
    VAN_DER_WAALS = "van-der-waals"  # union of atomic spheres, no probe
    GAUSSIAN = "gaussian"  # density-based smooth dielectric, no sharp boundary


@dataclass(frozen=True)
class PQRData:
    """A charged, radius-assigned structure. Units: angstroms and elementary charge."""

    coords: FloatArray  # (N, 3), A
    charges: FloatArray  # (N,), e
    radii: FloatArray  # (N,), A
    labels: tuple[str, ...] = ()  # optional per-atom "resName resSeq atomName"

    def __post_init__(self) -> None:
        n = len(self.charges)
        if self.coords.shape != (n, DIMENSIONS):
            raise ValueError(f"coords must be (N, 3) to match {n} charges, got {self.coords.shape}")
        if self.radii.shape != (n,):
            raise ValueError(f"radii must be (N,) to match {n} charges, got {self.radii.shape}")
        if n == 0:
            raise ValueError("PQRData needs at least one atom")

    @property
    def n_atoms(self) -> int:
        return len(self.charges)

    @property
    def total_charge(self) -> float:
        return float(self.charges.sum())

    def _bounds(self) -> tuple[FloatArray, FloatArray]:
        lo = np.asarray((self.coords - self.radii[:, None]).min(axis=0), dtype=np.float64)
        hi = np.asarray((self.coords + self.radii[:, None]).max(axis=0), dtype=np.float64)
        return lo, hi

    def extent(self) -> FloatArray:
        """Bounding-box side lengths including atomic radii, (3,) in A."""
        lo, hi = self._bounds()
        return hi - lo

    def center(self) -> FloatArray:
        """Geometric center of the radius-inflated bounding box, (3,) in A."""
        lo, hi = self._bounds()
        return (lo + hi) / 2.0


@dataclass(frozen=True)
class GridSpec:
    """Grid intent in physical terms. Finite-difference backends only.

    Deliberately has no `dime`: legal multigrid dimensions are an APBS
    implementation detail. A grid-flexible backend honors `resolution` and
    `padding` directly.
    """

    resolution: float = 0.5  # target spacing, A (fine grid)
    padding: float = 10.0  # min distance, molecule surface -> fine-grid edge, A
    max_points: int = 161**3  # memory guardrail; backend errors if unsatisfiable

    def __post_init__(self) -> None:
        if self.resolution <= 0:
            raise ValueError(f"resolution must be positive, got {self.resolution}")
        if self.padding < 0:
            raise ValueError(f"padding must be non-negative, got {self.padding}")
        if self.max_points <= 0:
            raise ValueError(f"max_points must be positive, got {self.max_points}")


@dataclass(frozen=True)
class SolventModel:
    solvent_dielectric: float = 78.54
    solute_dielectric: float = 2.0
    ionic_strength: float = 0.150  # M, 1:1 salt
    ion_radius: float = 2.0  # A
    temperature: float = 298.15  # K
    surface_model: SurfaceModel = SurfaceModel.SMOOTHED_MOLECULAR
    surface_radius: float = 1.4  # solvent probe, A

    def __post_init__(self) -> None:
        if self.solvent_dielectric <= 0 or self.solute_dielectric <= 0:
            raise ValueError("dielectric constants must be positive")
        if self.ionic_strength < 0:
            raise ValueError(f"ionic_strength must be non-negative, got {self.ionic_strength}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.surface_radius < 0:
            raise ValueError(f"surface_radius must be non-negative, got {self.surface_radius}")


# --- requests ---------------------------------------------------------------


@dataclass(frozen=True)
class SolveRequest:
    """What every solver family needs, and nothing more.

    `want_potential` is a request rather than a promise: a backend may decline
    to produce a field it cannot represent. `want_energy` is honored by every
    backend, because solvation energy is the one quantity all of them compute.
    """

    structure: PQRData
    solvent: SolventModel = SolventModel()
    want_energy: bool = True
    want_potential: bool = True

    def __post_init__(self) -> None:
        if not (self.want_energy or self.want_potential):
            raise ValueError("a request that wants neither energy nor potential does nothing")


@dataclass(frozen=True)
class FiniteDifferenceRequest(SolveRequest):
    """A request for a grid-based solver: APBS, DelPhi, PBSA, debye v1."""

    grid: GridSpec = GridSpec()
    equation: Equation = Equation.LINEAR


@dataclass(frozen=True)
class BoundaryElementRequest(SolveRequest):
    """A request for a surface-based solver: TABI-PB, PyGBe.

    No grid and no equation choice, by construction — see `Equation`.
    """

    mesh_density: float = 1.0  # triangles per A^2 of molecular surface

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mesh_density <= 0:
            raise ValueError(f"mesh_density must be positive, got {self.mesh_density}")


# --- results ----------------------------------------------------------------


@dataclass
class PotentialGrid:
    """A uniform volumetric scalar field of electrostatic potential, kT/e."""

    values: FloatArray  # (nx, ny, nz), kT/e
    origin: FloatArray  # (3,), A — position of values[0, 0, 0]
    spacing: FloatArray  # (3,), A — uniform per-axis

    def __post_init__(self) -> None:
        if self.values.ndim != DIMENSIONS:
            raise ValueError(f"values must be 3-D, got shape {self.values.shape}")
        self.origin = np.asarray(self.origin, dtype=float).reshape(DIMENSIONS)
        self.spacing = np.asarray(self.spacing, dtype=float).reshape(DIMENSIONS)
        if np.any(self.spacing <= 0):
            raise ValueError(f"spacing must be positive, got {self.spacing}")

    @property
    def shape(self) -> tuple[int, int, int]:
        nx, ny, nz = self.values.shape
        return nx, ny, nz

    def to_dx(self, path: str | os.PathLike[str]) -> None:
        """Write as OpenDX, for PyMOL/ChimeraX. Backend-independent by design."""
        # Imported here, not at module scope: dx.py imports PotentialGrid from
        # this module, so a top-level import would be circular.
        from sashimi.dx import write_dx  # noqa: PLC0415

        write_dx(path, self)

    def value_at(self, points: npt.ArrayLike) -> FloatArray:
        """Trilinearly interpolate at arbitrary coordinates, (M, 3) A -> (M,) kT/e.

        Points outside the grid yield NaN rather than a clamped edge value: a
        silently clamped potential reads as a real measurement.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if pts.shape[-1] != DIMENSIONS:
            raise ValueError(f"points must be (M, 3), got {pts.shape}")

        frac = (pts - self.origin) / self.spacing
        base = np.floor(frac).astype(int)
        w = frac - base

        n = np.array(self.values.shape)
        inside = np.all((base >= 0) & (base <= n - 2), axis=1)

        out = np.full(len(pts), np.nan)
        if not inside.any():
            return out

        b, t = base[inside], w[inside]
        acc = np.zeros(len(b))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    wt = (
                        (t[:, 0] if dx else 1 - t[:, 0])
                        * (t[:, 1] if dy else 1 - t[:, 1])
                        * (t[:, 2] if dz else 1 - t[:, 2])
                    )
                    acc += wt * self.values[b[:, 0] + dx, b[:, 1] + dy, b[:, 2] + dz]
        out[inside] = acc
        return out

    def stats(self) -> Diagnostics:
        v = self.values
        return {
            "kind": "volume",
            "shape": list(v.shape),
            "origin": self.origin.tolist(),
            "spacing": self.spacing.tolist(),
            "min": float(v.min()),
            "max": float(v.max()),
            "mean": float(v.mean()),
            "std": float(v.std()),
        }


@dataclass
class SurfacePotential:
    """Potential sampled on the dielectric interface, kT/e.

    What a boundary-element solver natively produces. No shipped backend emits
    one yet; it exists so the result type does not have to change when one does,
    and so `tests/test_bem_contract.py` can prove the protocol admits it.
    """

    vertices: FloatArray  # (M, 3), A
    values: FloatArray  # (M,), kT/e
    normals: FloatArray | None = None  # (M, 3), unit outward normals
    triangles: npt.NDArray[np.int64] | None = None  # (T, 3) vertex indices

    def __post_init__(self) -> None:
        m = len(self.values)
        if self.vertices.shape != (m, DIMENSIONS):
            raise ValueError(f"vertices must be (M, 3) to match {m} values")
        if self.normals is not None and self.normals.shape != (m, DIMENSIONS):
            raise ValueError(f"normals must be (M, 3) to match {m} values")
        if m == 0:
            raise ValueError("SurfacePotential needs at least one vertex")

    @property
    def n_vertices(self) -> int:
        return len(self.values)

    def stats(self) -> Diagnostics:
        v = self.values
        return {
            "kind": "surface",
            "n_vertices": self.n_vertices,
            "min": float(v.min()),
            "max": float(v.max()),
            "mean": float(v.mean()),
            "std": float(v.std()),
        }


Potential = PotentialGrid | SurfacePotential


@dataclass(frozen=True)
class Provenance:
    """Everything needed to know what produced a number, and to reproduce it.

    `resolved_parameters` is the parameter set the backend actually used — not
    what was asked for. It is what makes a relaxed grid, a mapped surface model
    or a defaulted tolerance visible after the fact, and it is what cross-solver
    comparison checks before reporting a spread.
    """

    backend: str  # "apbs-3.4.1" | "debye-x.y"
    binary_path: str | None = None
    binary_sha256: str | None = None
    resolved_parameters: Diagnostics = field(default_factory=dict)
    wall_seconds: float | None = None
    # What `SolveResult.energy_kj_mol` actually is. Optional only so that a
    # backend predating this field still constructs; `sashimi.validate` treats
    # an unstated term as uncomparable rather than assuming it matches.
    energy_term: EnergyTerm | None = None
    # How the number was obtained. Defaults rather than being optional, which is
    # the opposite of `energy_term` above and deliberate. An unstated energy term
    # is a silently wrong comparison, so it is refused; an unstated tier has one
    # correct answer for every backend that predates the field — all of them
    # discretize the equation — and the cost of a mis-defaulted approximation is
    # the behaviour that existed before this field: compared at the reference
    # tolerance and loudly reported as a disagreement. Loud, not silent.
    accuracy_tier: AccuracyTier = AccuracyTier.REFERENCE

    def summary(self) -> str:
        checksum = f" sha256:{self.binary_sha256[:12]}" if self.binary_sha256 else ""
        return f"{self.backend}{checksum}"


@dataclass
class SolveResult:
    """Energy is the universal currency; a field is not.

    Every solver family computes a solvation energy, so `energy_kj_mol` is
    populated whenever the request asked for it. `potential` is optional and
    polymorphic: a volume from an FD backend, a surface from a BEM backend, or
    nothing when it was not requested.
    """

    provenance: Provenance
    energy_kj_mol: float | None = None  # total polar solvation energy
    potential: Potential | None = None
    diagnostics: Diagnostics = field(default_factory=dict)

    @property
    def backend(self) -> str:
        """Shorthand for `provenance.backend`, which is what most callers want."""
        return self.provenance.backend

    def check_satisfies(self, request: SolveRequest) -> None:
        """Assert the backend delivered what was asked. Backends call this."""
        if request.want_energy and self.energy_kj_mol is None:
            raise ValueError("request asked for energy but the result carries none")
        if request.want_potential and self.potential is None:
            raise ValueError("request asked for potential but the result carries none")


# --- the contract -----------------------------------------------------------

RequestT_contra = TypeVar("RequestT_contra", bound=SolveRequest, contravariant=True)


class Solver(Protocol[RequestT_contra]):
    """A Poisson-Boltzmann solver.

    Generic in its request type, so `Solver[FiniteDifferenceRequest]` and
    `Solver[BoundaryElementRequest]` are distinct types and a checker rejects
    handing one family's request to the other's backend. Use `Solver[Any]` for
    heterogeneous collections such as a backend registry.
    """

    def solve(self, request: RequestT_contra) -> SolveResult: ...
