"""OpenDX read/write.

Own reader/writer rather than gridData/MDAnalysis: the format is trivial and
owning it keeps the dependency tree at numpy. The writer exists so that any
backend's output can be exported for PyMOL/ChimeraX — it is not APBS-specific,
which is why it lives here rather than under `apbs/`.

Layout (verified against APBS 3.4.1 output): a header giving counts, origin and
three delta vectors, then the values in C order, three per line.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sashimi.errors import MalformedStructure
from sashimi.protocol import PotentialGrid

__all__ = ["parse_dx", "read_dx", "write_dx"]

_AXES = 3
_SKEW_TOLERANCE = 1e-9  # a delta matrix this close to diagonal is axis-aligned


@dataclass
class _Header:
    counts: tuple[int, int, int]
    origin: np.ndarray
    deltas: list[list[float]]
    n_items: int | None
    data_start: int


def _parse_header(lines: list[str]) -> _Header:
    counts: tuple[int, int, int] | None = None
    origin: np.ndarray | None = None
    deltas: list[list[float]] = []
    n_items: int | None = None
    data_start: int | None = None

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tok = line.split()
        if tok[0] == "origin":
            origin = np.array([float(v) for v in tok[1:4]])
        elif tok[0] == "delta":
            deltas.append([float(v) for v in tok[1:4]])
        elif tok[0] == "object" and "gridpositions" in line:
            counts = (int(tok[-3]), int(tok[-2]), int(tok[-1]))
        elif tok[0] == "object" and "class array" in line:
            if "items" in tok:
                n_items = int(tok[tok.index("items") + 1])
            data_start = i + 1
            break

    if counts is None or origin is None or len(deltas) < _AXES or data_start is None:
        raise MalformedStructure(
            "malformed DX header: missing counts, origin, deltas, or data marker"
        )
    return _Header(counts, origin, deltas, n_items, data_start)


def parse_dx(text: str) -> PotentialGrid:
    lines = text.splitlines()
    header = _parse_header(lines)
    counts, origin, data_start = header.counts, header.origin, header.data_start
    n_items = header.n_items

    d = np.array(header.deltas[:_AXES], dtype=float)
    off_diagonal = d - np.diag(np.diag(d))
    if np.any(np.abs(off_diagonal) > _SKEW_TOLERANCE):
        raise MalformedStructure("non-axis-aligned DX grids are not supported")
    spacing = np.diag(d)
    if np.any(spacing <= 0):
        raise MalformedStructure(f"DX deltas must be positive, got {spacing}")

    expected = counts[0] * counts[1] * counts[2]
    values: list[float] = []
    for raw in lines[data_start:]:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tok = line.split()
        if not _looks_numeric(tok[0]):
            break  # trailing attribute/component/object stanzas
        values.extend(float(v) for v in tok)
        if len(values) >= expected:
            break

    if len(values) < expected:
        raise MalformedStructure(f"DX truncated: expected {expected} values, found {len(values)}")
    if n_items is not None and n_items != expected:
        raise MalformedStructure(
            f"DX header inconsistent: items={n_items} but counts imply {expected}"
        )

    grid = np.array(values[:expected], dtype=float).reshape(counts)  # C order
    return PotentialGrid(values=grid, origin=origin, spacing=spacing)


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def read_dx(path: str | os.PathLike[str]) -> PotentialGrid:
    return parse_dx(Path(path).read_text())


def write_dx(path: str | os.PathLike[str], grid: PotentialGrid, *, comment: str = "") -> None:
    nx, ny, nz = grid.values.shape
    n = nx * ny * nz
    ox, oy, oz = grid.origin
    hx, hy, hz = grid.spacing

    head = [
        # ASCII only, deliberately. Compared byte for byte against a file APBS
        # 3.4.1 wrote, sashimi's DX differs in exactly one place — this comment
        # — and it was the only line in either file carrying a non-ASCII
        # character (an em dash). Every consumer of these maps is a third-party
        # viewer whose parser we do not control and cannot test here, so the
        # one thing worth guaranteeing is that our header is the header APBS
        # emits, which those parsers already read.
        f"# Data from sashimi{(': ' + comment) if comment else ''}",
        "#",
        "# POTENTIAL (kT/e)",
        "#",
        f"object 1 class gridpositions counts {nx} {ny} {nz}",
        f"origin {ox:.6e} {oy:.6e} {oz:.6e}",
        f"delta {hx:.6e} 0.000000e+00 0.000000e+00",
        f"delta 0.000000e+00 {hy:.6e} 0.000000e+00",
        f"delta 0.000000e+00 0.000000e+00 {hz:.6e}",
        f"object 2 class gridconnections counts {nx} {ny} {nz}",
        f"object 3 class array type double rank 0 items {n} data follows",
    ]
    tail = [
        'attribute "dep" string "positions"',
        'object "regular positions regular connections" class field',
        'component "positions" value 1',
        'component "connections" value 2',
        'component "data" value 3',
        "",
    ]

    flat = grid.values.reshape(-1)  # C order
    body = [" ".join(f"{v:.6e}" for v in flat[i : i + 3]) for i in range(0, n, 3)]
    Path(path).write_text("\n".join(head + body + tail))
