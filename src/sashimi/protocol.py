"""The solver contract.

Everything here speaks physics — angstroms, molar ionic strength, dielectric
constants, kT/e, kJ/mol. Nothing here knows APBS exists. A future native solver
(debye) implements `Solver` and slots in underneath without any client change,
so no APBS concept (dime, cglen, fglen, chgm, srfm) may leak into this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

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
    "Diagnostics",
    "FloatArray",
    "GridSpec",
    "PQRData",
    "PotentialGrid",
    "SolveResult",
    "SolventModel",
    "Solver",
]


@dataclass(frozen=True)
class PQRData:
    """A charged, radius-assigned structure. Units: angstroms and elementary charge."""

    coords: FloatArray  # (N, 3), A
    charges: FloatArray  # (N,), e
    radii: FloatArray  # (N,), A
    labels: tuple[str, ...] = ()  # optional per-atom "resName resSeq atomName"

    def __post_init__(self) -> None:
        n = len(self.charges)
        if self.coords.shape != (n, 3):
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
    """Grid intent in physical terms. Backends translate to their own constraints.

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
    surface_method: Literal["smol", "spl2", "mol"] = "smol"
    surface_radius: float = 1.4  # solvent probe, A

    def __post_init__(self) -> None:
        if self.solvent_dielectric <= 0 or self.solute_dielectric <= 0:
            raise ValueError("dielectric constants must be positive")
        if self.ionic_strength < 0:
            raise ValueError(f"ionic_strength must be non-negative, got {self.ionic_strength}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")


@dataclass
class PotentialGrid:
    """A uniform scalar field of electrostatic potential in kT/e."""

    values: FloatArray  # (nx, ny, nz), kT/e
    origin: FloatArray  # (3,), A — position of values[0, 0, 0]
    spacing: FloatArray  # (3,), A — uniform per-axis

    def __post_init__(self) -> None:
        if self.values.ndim != DIMENSIONS:
            raise ValueError(f"values must be 3-D, got shape {self.values.shape}")
        self.origin = np.asarray(self.origin, dtype=float).reshape(3)
        self.spacing = np.asarray(self.spacing, dtype=float).reshape(3)
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
            "shape": list(v.shape),
            "origin": self.origin.tolist(),
            "spacing": self.spacing.tolist(),
            "min": float(v.min()),
            "max": float(v.max()),
            "mean": float(v.mean()),
            "std": float(v.std()),
        }


@dataclass
class SolveResult:
    potential: PotentialGrid
    energy_kj_mol: float | None = None  # total polar solvation energy when requested
    backend: str = ""  # "apbs-3.4.1" | "debye-x.y" — provenance travels
    diagnostics: Diagnostics = field(default_factory=dict)


class Solver(Protocol):
    def solve_lpbe(
        self,
        pqr: PQRData,
        grid: GridSpec,
        # B008 is about mutable defaults; SolventModel is a frozen dataclass, and
        # spelling the default inline is what documents the physics at the call site.
        solvent: SolventModel = SolventModel(),  # noqa: B008
        *,
        compute_energy: bool = False,
    ) -> SolveResult: ...
