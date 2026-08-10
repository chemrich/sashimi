"""Gaussian Cube -> PotentialGrid.

DelPhi's native `.phi` map is an unformatted Fortran binary whose record layout
depends on the compiler that built it; both flavours can write a Gaussian Cube
instead, which is text, self-describing and identical across builds. That makes
Cube the only volumetric format the two DelPhi flavours agree on, so it is the
one sashimi asks for.

The format is atomic units **by definition** — origin and axis vectors are in
Bohr — which is the single conversion this module exists to perform. A map read
as angstroms would be 1.89x too small in every dimension, and every potential in
it would be attributed to the wrong coordinate.

Data order matches DX and `PotentialGrid`: x slowest, z fastest, C order.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from sashimi.delphi.units import BOHR_TO_ANGSTROM
from sashimi.errors import MalformedStructure
from sashimi.protocol import DIMENSIONS, PotentialGrid

__all__ = ["parse_cube", "read_cube"]

_HEADER_LINES = 2  # two free-text comment lines before anything structured
_SKEW_TOLERANCE = 1e-9


def parse_cube(text: str) -> PotentialGrid:
    lines = text.splitlines()
    if len(lines) < _HEADER_LINES + 1 + DIMENSIONS:
        raise MalformedStructure("cube file is too short to contain a header")

    try:
        head = [line.split() for line in lines[_HEADER_LINES : _HEADER_LINES + 1 + DIMENSIONS]]
        n_atoms = int(head[0][0])
        origin = np.array([float(v) for v in head[0][1:4]], dtype=float)
        counts = tuple(int(row[0]) for row in head[1:])
        axes = np.array([[float(v) for v in row[1:4]] for row in head[1:]], dtype=float)
    except (IndexError, ValueError) as exc:
        raise MalformedStructure(f"malformed cube header: {exc}") from exc

    if any(n <= 0 for n in counts):
        # A negative count is the Gaussian convention for "voxel vectors are in
        # angstroms". DelPhi never writes it, and guessing would be worse than
        # refusing, since the error is a silent 1.89x scale factor.
        raise MalformedStructure(
            f"cube grid counts must be positive, got {list(counts)}; sashimi reads only "
            "the atomic-units form that DelPhi writes"
        )

    off_diagonal = axes - np.diag(np.diag(axes))
    if np.any(np.abs(off_diagonal) > _SKEW_TOLERANCE):
        raise MalformedStructure("non-axis-aligned cube grids are not supported")

    spacing = np.diag(axes) * BOHR_TO_ANGSTROM
    if np.any(spacing <= 0):
        raise MalformedStructure(f"cube voxel vectors must be positive, got {spacing}")

    # A negative atom count adds a dataset-index line after the atom block.
    data_start = _HEADER_LINES + 1 + DIMENSIONS + abs(n_atoms) + (1 if n_atoms < 0 else 0)

    expected = counts[0] * counts[1] * counts[2]
    values: list[float] = []
    for raw in lines[data_start:]:
        line = raw.strip()
        if not line:
            continue
        try:
            values.extend(float(v) for v in line.split())
        except ValueError as exc:
            raise MalformedStructure(f"non-numeric cube data: {line!r}") from exc
        if len(values) >= expected:
            break

    if len(values) < expected:
        raise MalformedStructure(f"cube truncated: expected {expected} values, found {len(values)}")

    grid = np.array(values[:expected], dtype=float).reshape(counts)  # C order
    return PotentialGrid(
        values=grid,
        origin=origin * BOHR_TO_ANGSTROM,
        spacing=spacing,
    )


def read_cube(path: str | os.PathLike[str]) -> PotentialGrid:
    return parse_cube(Path(path).read_text())
