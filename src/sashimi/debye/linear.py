"""The discrete operator, and the multigrid that inverts it.

`-div(eps grad phi) + kappabar^2 phi = 4 pi l_B rho`, in finite-volume form on a
Cartesian grid: for each node, the sum of fluxes through its six faces plus the
Boltzmann term equals the charge it carries. Writing it as a flux balance rather
than as a differenced Laplacian is what makes the dielectric jump conservative —
the flux leaving one control volume is exactly the flux entering the next,
whatever eps does between them — and it makes the matrix symmetric positive
definite, which is what the solver below relies on twice over.

**Why multigrid rather than the conjugate gradients alone.** The condition
number of this operator grows like (L/h)^2, so plain CG needs iterations
proportional to L/h: several hundred on the 105^3 grid the Born gate case
resolves to, each one a pass over 1.2M points. A multigrid V-cycle costs about
three such passes and removes error at every wavelength at once, so it converges
in a number of cycles that does not grow with the grid. The measured rate on the
gate case is a residual reduction of ~0.06 per cycle.

**Why CG anyway, with the V-cycle as its preconditioner.** Geometric multigrid
with re-discretized coefficients is not guaranteed to be robust across a 78:1
dielectric jump; the textbook rate degrades when the coarse grid cannot see the
interface, and the failure is a stall rather than a wrong answer. Wrapping it in
CG costs one extra inner product per iteration and converts that stall into
slower-but-still-converging: CG only needs the preconditioner to be symmetric
positive definite, which a V(2,2) cycle with its post-smoothing sweeps run in
the reverse colour order is. That symmetry is not decoration — with the sweeps
in the same order both ways the preconditioner is nonsymmetric, CG's search
directions stop being conjugate, and the residual wanders instead of falling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sashimi.debye.dielectric import dielectric_faces, screening_nodes
from sashimi.debye.grid import DebyeGrid, grid_hierarchy
from sashimi.errors import ConvergenceFailure
from sashimi.protocol import Diagnostics, FloatArray, PQRData, SolventModel

__all__ = ["Level", "SolveReport", "build_levels", "solve_system"]

INNER = (slice(1, -1), slice(1, -1), slice(1, -1))


@dataclass
class Level:
    """One grid in the hierarchy, with its operator baked into face conductances.

    A conductance is `eps * area / distance` for the face between two nodes:
    the coefficient multiplying the potential difference across it. Holding
    those rather than the dielectric itself means the anisotropic case — a box
    whose axes rounded to different spacings — costs nothing extra at solve
    time, and the operator has one form rather than three.
    """

    grid: DebyeGrid
    cx: FloatArray  # (nx-1, ny, nz)
    cy: FloatArray  # (nx, ny-1, nz)
    cz: FloatArray  # (nx, ny, nz-1)
    screening: FloatArray  # kappabar^2 at nodes, 1/A^2
    diagonal: FloatArray = field(init=False)  # interior shape
    _red: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.diagonal = (
            self._cxl + self._cxr + self._cyl + self._cyr + self._czl + self._czr
        ) + self.screening[INNER] * self.cell_volume
        # Checkerboard over the interior. Global parity, so the colouring is
        # consistent between levels; it need not be, but a reader comparing two
        # levels' sweeps should not have to work that out.
        nx, ny, nz = (n - 2 for n in self.grid.shape)
        i, j, k = np.ogrid[1 : nx + 1, 1 : ny + 1, 1 : nz + 1]
        self._red = ((i + j + k) % 2) == 0

    # Face conductances as seen by an interior node: `_cxl` is the face on its
    # low-x side, `_cxr` the one on its high-x side. Views, not copies.
    @property
    def _cxl(self) -> FloatArray:
        return self.cx[:-1, 1:-1, 1:-1]

    @property
    def _cxr(self) -> FloatArray:
        return self.cx[1:, 1:-1, 1:-1]

    @property
    def _cyl(self) -> FloatArray:
        return self.cy[1:-1, :-1, 1:-1]

    @property
    def _cyr(self) -> FloatArray:
        return self.cy[1:-1, 1:, 1:-1]

    @property
    def _czl(self) -> FloatArray:
        return self.cz[1:-1, 1:-1, :-1]

    @property
    def _czr(self) -> FloatArray:
        return self.cz[1:-1, 1:-1, 1:]

    @property
    def cell_volume(self) -> float:
        hx, hy, hz = self.grid.spacing
        return hx * hy * hz

    def _neighbour_sum(self, x: FloatArray) -> FloatArray:
        """sum of c_face * phi_neighbour over the six faces, interior shape."""
        return (
            self._cxl * x[:-2, 1:-1, 1:-1]
            + self._cxr * x[2:, 1:-1, 1:-1]
            + self._cyl * x[1:-1, :-2, 1:-1]
            + self._cyr * x[1:-1, 2:, 1:-1]
            + self._czl * x[1:-1, 1:-1, :-2]
            + self._czr * x[1:-1, 1:-1, 2:]
        )

    def apply(self, x: FloatArray) -> FloatArray:
        """A x, for a full-shaped vector whose boundary is zero.

        Vectors carry their boundary ring even though it is always zero: it
        makes every array in the solver the same shape, so restriction,
        prolongation and the inner products need no special case at the edge.
        """
        out = np.zeros_like(x)
        out[INNER] = self.diagonal * x[INNER] - self._neighbour_sum(x)
        return out

    def residual(self, x: FloatArray, b: FloatArray) -> FloatArray:
        return b - self.apply(x)

    def smooth(self, x: FloatArray, b: FloatArray, sweeps: int, *, reverse: bool = False) -> None:
        """Red-black Gauss-Seidel, in place.

        `reverse` swaps the colour order, which is what makes a V-cycle with
        matching pre- and post-smoothing counts a symmetric operator and so a
        legal CG preconditioner.
        """
        colours = [~self._red, self._red] if reverse else [self._red, ~self._red]
        for _ in range(sweeps):
            for colour in colours:
                updated = (b[INNER] + self._neighbour_sum(x)) / self.diagonal
                x[INNER] = np.where(colour, updated, x[INNER])


def _restrict(fine: FloatArray) -> FloatArray:
    """Full weighting, fine -> coarse, one separable pass per axis.

    Exactly the transpose of `_prolong`, with no averaging factor, and the
    missing factor is the whole subtlety. Textbook full weighting carries a
    1/2, 1, 1/2 stencil scaled by a half per axis — an *average* — because the
    textbook operator is a differenced Laplacian, where the residual is a
    pointwise quantity. This operator is a finite-volume flux balance, so its
    residual is an integral over a control volume, and a coarse cell contains
    eight fine ones: the residuals must be *summed*, not averaged.

    Getting it wrong is not a wrong answer, which is what makes it worth a
    paragraph. With the extra 1/8 the coarse-grid correction is right in shape
    and an eighth of the size, so every V-cycle removes an eighth of the smooth
    error it should. Wrapped in CG that still converges, to the same energy, and
    the only symptom is the iteration count growing with the grid — 20, 33, 55
    cycles at 0.8, 0.46 and 0.25 A, where h-independence is the property
    multigrid exists to have. It was found by noticing that the *uniform*
    dielectric solve, a plain Poisson problem with nothing hard in it, was
    taking 53 cycles.
    """
    out = fine
    for axis in range(3):
        n = out.shape[axis]
        coarse_shape = list(out.shape)
        coarse_shape[axis] = (n - 1) // 2 + 1
        coarse = np.zeros(coarse_shape, dtype=np.float64)

        def sl(
            start: int | None, stop: int | None, step: int | None, ax: int = axis
        ) -> tuple[slice, slice, slice]:
            index: list[slice] = [slice(None)] * 3
            index[ax] = slice(start, stop, step)
            return index[0], index[1], index[2]

        interior: list[slice] = [slice(None)] * 3
        interior[axis] = slice(1, -1)
        coarse[tuple(interior)] = (
            0.5 * out[sl(1, -2, 2)] + out[sl(2, -1, 2)] + 0.5 * out[sl(3, None, 2)]
        )
        out = coarse
    return out


def _prolong(coarse: FloatArray) -> FloatArray:
    """Trilinear interpolation, coarse -> fine, one separable pass per axis."""
    out = coarse
    for axis in range(3):
        fine_shape = list(out.shape)
        fine_shape[axis] = 2 * (out.shape[axis] - 1) + 1
        fine = np.zeros(fine_shape, dtype=np.float64)

        def sl(
            start: int | None, stop: int | None, step: int | None, ax: int = axis
        ) -> tuple[slice, slice, slice]:
            index: list[slice] = [slice(None)] * 3
            index[ax] = slice(start, stop, step)
            return index[0], index[1], index[2]

        fine[sl(0, None, 2)] = out
        fine[sl(1, None, 2)] = 0.5 * (out[sl(0, -1, None)] + out[sl(1, None, None)])
        out = fine
    return out


def build_levels(grid: DebyeGrid, structure: PQRData, solvent: SolventModel) -> list[Level]:
    """The multigrid hierarchy, each level's operator built from the geometry.

    Re-discretized rather than algebraically coarsened: each level samples the
    same union of spheres on its own lattice. That is legitimate here because
    `coarsen` preserves the box exactly, so a coarse node sits on a fine node
    and both are asking about the same surface — and it is far cheaper than
    forming a Galerkin product matrix-free. The cost is that a coarse grid
    whose spacing exceeds an atomic radius stops seeing that atom at all, which
    is a known weakness of re-discretization and is why the outer iteration is
    CG rather than plain V-cycles.
    """
    levels = []
    for level_grid in grid_hierarchy(grid):
        eps_x, eps_y, eps_z = dielectric_faces(level_grid, structure, solvent)
        screening, _ = screening_nodes(level_grid, structure, solvent)
        hx, hy, hz = level_grid.spacing
        levels.append(
            Level(
                grid=level_grid,
                cx=eps_x * (hy * hz / hx),
                cy=eps_y * (hx * hz / hy),
                cz=eps_z * (hx * hy / hz),
                screening=screening,
            )
        )
    return levels


def _v_cycle(levels: list[Level], index: int, x: FloatArray, b: FloatArray, *, sweeps: int) -> None:
    level = levels[index]
    if index == len(levels) - 1:
        _coarse_solve(level, x, b)
        return

    level.smooth(x, b, sweeps)
    coarse_b = _restrict(level.residual(x, b))
    coarse_x = np.zeros_like(coarse_b)
    _v_cycle(levels, index + 1, coarse_x, coarse_b, sweeps=sweeps)
    x += _prolong(coarse_x)
    # Same count, reverse colour order: that is what makes the cycle its own
    # adjoint, and so a legal CG preconditioner. See `DebyeOptions`.
    level.smooth(x, b, sweeps, reverse=True)


def _coarse_solve(level: Level, x: FloatArray, b: FloatArray, *, iterations: int = 400) -> None:
    """Diagonally preconditioned CG on the coarsest grid, to convergence.

    The coarsest grid is a few thousand unknowns, so "solve it properly" costs
    less than one fine-grid smoothing sweep — and an inexactly solved coarse
    problem is the classic way a V-cycle's convergence rate silently halves.
    """
    r = level.residual(x, b)
    norm_b = float(np.linalg.norm(b))
    if norm_b == 0.0:
        return
    z = np.zeros_like(r)
    z[INNER] = r[INNER] / level.diagonal
    p = z.copy()
    rz = float(np.vdot(r, z))
    for _ in range(iterations):
        ap = level.apply(p)
        denominator = float(np.vdot(p, ap))
        if denominator <= 0.0:
            return
        alpha = rz / denominator
        x += alpha * p
        r -= alpha * ap
        if float(np.linalg.norm(r)) <= 1e-12 * norm_b:
            return
        z[INNER] = r[INNER] / level.diagonal
        rz_next = float(np.vdot(r, z))
        p = z + (rz_next / rz) * p
        rz = rz_next


@dataclass(frozen=True)
class SolveReport:
    """What the iteration did, for provenance rather than for the caller's eyes."""

    iterations: int
    relative_residual: float
    grid_shape: tuple[int, int, int]
    levels: int

    def as_diagnostics(self) -> Diagnostics:
        return {
            "iterations": self.iterations,
            "relative_residual": float(f"{self.relative_residual:.3e}"),
            "multigrid_levels": self.levels,
        }


def solve_system(
    levels: list[Level],
    b: FloatArray,
    *,
    tolerance: float,
    max_cycles: int,
    smoothing_sweeps: int,
) -> tuple[FloatArray, SolveReport]:
    """Solve A x = b with multigrid-preconditioned CG. Homogeneous Dirichlet.

    Raises `ConvergenceFailure` rather than returning the best effort so far.
    A potential that has not converged is not a worse answer, it is a different
    physical system's answer — ROADMAP.md section 4.2 puts this failure in the
    taxonomy for that reason.
    """
    fine = levels[0]
    x = np.zeros_like(b)
    r = fine.residual(x, b)
    norm_b = float(np.linalg.norm(b))
    if norm_b == 0.0:
        return x, SolveReport(0, 0.0, fine.grid.shape, len(levels))

    def precondition(rhs: FloatArray) -> FloatArray:
        z = np.zeros_like(rhs)
        _v_cycle(levels, 0, z, rhs, sweeps=smoothing_sweeps)
        return z

    z = precondition(r)
    p = z.copy()
    rz = float(np.vdot(r, z))
    relative = float(np.linalg.norm(r)) / norm_b

    for cycle in range(1, max_cycles + 1):
        ap = fine.apply(p)
        denominator = float(np.vdot(p, ap))
        if denominator <= 0.0 or not math.isfinite(denominator):
            raise ConvergenceFailure(
                "the discrete operator is not positive definite, which it is by "
                "construction unless a dielectric or a grid spacing is "
                f"non-positive (p.Ap = {denominator})"
            )
        alpha = rz / denominator
        x += alpha * p
        r -= alpha * ap
        relative = float(np.linalg.norm(r)) / norm_b
        if not math.isfinite(relative):
            raise ConvergenceFailure("the iteration diverged to a non-finite residual")
        if relative <= tolerance:
            return x, SolveReport(cycle, relative, fine.grid.shape, len(levels))
        z = precondition(r)
        rz_next = float(np.vdot(r, z))
        p = z + (rz_next / rz) * p
        rz = rz_next

    raise ConvergenceFailure(
        f"multigrid-preconditioned CG reached {relative:.2e} after {max_cycles} cycles, "
        f"short of {tolerance:.0e}, on a {'x'.join(str(n) for n in fine.grid.shape)} grid. "
        "Raise DebyeOptions.max_cycles, or loosen DebyeOptions.tolerance if the field "
        "is wanted at lower precision than the energy."
    )
