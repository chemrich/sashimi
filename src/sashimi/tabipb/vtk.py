"""VTK legacy POLYDATA -> SurfacePotential.

TABI-PB's CSV output is a single summary row; the per-vertex field lives in its
VTK file, which is why sashimi asks for that one. The subset needed here is
small and rigidly ordered — `POINTS`, `POLYGONS`, then `POINT_DATA` with two
`SCALARS` blocks — so a dependency on a VTK reader would cost far more than it
saves, the same reasoning `dx.py` records for OpenDX.

The second scalar block, `NormalPotential`, is the normal derivative of the
potential on the surface -- the other half of what a boundary-element solve
natively produces. **It is the *interior* derivative, `eps_p`'s side**, measured
rather than assumed: sweeping the solute dielectric on a four-atom Born sphere
moves the file's values as `1/eps_p` exactly and leaves them invariant to
`eps_s` to six figures. The two sides differ by `eps_s/eps_p`, so a reader that
takes this for the solvent-side gradient is wrong by 39.27 at the protocol's
defaults. `SurfacePotential.interior_normal_derivative` carries it under that
name for exactly that reason; it is also still handed back on `ParsedSurface`,
which is what `tests/test_tabipb.py` asserts against.

**Units here are TABI-PB's, not the protocol's.** The file carries the potential
in kJ/mol/e; the protocol boundary fixes kT/e (ROADMAP §4). This module is a
format reader and does not know the temperature, so it hands back what the file
says and `backend.to_kt_per_e` converts. Nothing between the two may treat a
`SurfacePotential` from here as satisfying the protocol's contract.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sashimi.errors import MalformedStructure
from sashimi.protocol import DIMENSIONS, FloatArray, SurfacePotential

__all__ = ["ParsedSurface", "parse_vtk", "read_vtk"]

_TRIANGLE_VERTICES = 3


@dataclass
class ParsedSurface:
    """What the VTK file holds: a mesh, a potential, and its normal derivative.

    In the file's own units — kJ/mol/e and kJ/mol/e/A — so the potential is not
    yet protocol-conformant. See the module docstring.
    """

    potential: SurfacePotential
    normal_derivative: FloatArray


def _read_block(lines: list[str], start: int, count: int, width: int) -> FloatArray:
    """Read `count` rows of `width` floats, tolerating blank lines.

    A non-numeric line ends the block rather than raising, because that is what
    it means in this format: sections run until the next keyword. Reaching one
    early therefore reports the block as short, which is the actionable
    diagnosis — a header promising more rows than the file contains.
    """
    values: list[float] = []
    needed = count * width
    for line in lines[start:]:
        if len(values) >= needed:
            break
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = [float(v) for v in stripped.split()]
        except ValueError:
            break  # the next section began
        values.extend(row)
    if len(values) < needed:
        raise MalformedStructure(f"VTK block truncated: wanted {needed} values, got {len(values)}")
    return np.array(values[:needed], dtype=float).reshape(count, width)


def _find(lines: list[str], keyword: str) -> tuple[int, list[str]]:
    for i, line in enumerate(lines):
        if line.strip().startswith(keyword):
            return i, line.split()
    raise MalformedStructure(f"VTK file has no {keyword} section")


def parse_vtk(text: str) -> ParsedSurface:
    lines = text.splitlines()

    points_at, points_header = _find(lines, "POINTS")
    n_points = int(points_header[1])
    vertices = _read_block(lines, points_at + 1, n_points, DIMENSIONS)

    triangles = _parse_polygons(lines)
    potential = _parse_scalars(lines, "Potential", n_points)
    normal = _parse_scalars(lines, "NormalPotential", n_points)

    return ParsedSurface(
        potential=SurfacePotential(
            vertices=vertices,
            values=potential,
            triangles=triangles,
            # In the file's units here, like `values` -- `backend.to_kt_per_e`
            # converts both on the way to the protocol boundary.
            interior_normal_derivative=normal,
        ),
        normal_derivative=normal,
    )


def _parse_polygons(lines: list[str]) -> np.ndarray | None:
    """Triangle connectivity, or None if the file carries none.

    Each row is `3 i j k` — the leading count is part of the legacy format and
    is verified rather than assumed, since a quad mesh would parse into
    nonsense otherwise.
    """
    try:
        at, header = _find(lines, "POLYGONS")
    except MalformedStructure:
        return None

    n_cells = int(header[1])
    rows = _read_block(lines, at + 1, n_cells, _TRIANGLE_VERTICES + 1)
    if not np.all(rows[:, 0] == _TRIANGLE_VERTICES):
        raise MalformedStructure("only triangular VTK meshes are supported")
    return np.asarray(rows[:, 1:], dtype=np.int64)


def _parse_scalars(lines: list[str], name: str, count: int) -> FloatArray:
    """One named SCALARS block. `LOOKUP_TABLE` follows the header, then values."""
    for i, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0] == "SCALARS" and tokens[1] == name:  # noqa: PLR2004
            start = i + 2 if lines[i + 1].strip().startswith("LOOKUP_TABLE") else i + 1
            return _read_block(lines, start, count, 1).reshape(count)
    raise MalformedStructure(f"VTK file has no SCALARS block named {name!r}")


def read_vtk(path: str | os.PathLike[str]) -> ParsedSurface:
    return parse_vtk(Path(path).read_text())
