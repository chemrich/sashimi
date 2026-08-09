"""PQR read/write.

Own parser rather than a dependency: the format is trivial and owning it keeps
the dependency tree at numpy. PQR is whitespace-delimited in practice (unlike
PDB it is not column-fixed, because charges and radii overflow their columns),
so the fields are recovered positionally from the end of the line, which is the
only part of the layout every writer agrees on.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from sashimi.protocol import PQRData

__all__ = ["format_pqr", "parse_pqr", "read_pqr", "write_pqr"]

_RECORDS = ("ATOM", "HETATM")


def parse_pqr(text: str) -> PQRData:
    coords: list[tuple[float, float, float]] = []
    charges: list[float] = []
    radii: list[float] = []
    labels: list[str] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.startswith(_RECORDS):
            continue
        tok = raw.split()
        # <record> <serial> <atom> <res> [chain] <resSeq> x y z q r
        if len(tok) < 9:
            raise ValueError(f"line {lineno}: expected at least 9 fields, got {len(tok)}: {raw!r}")
        try:
            x, y, z, q, r = (float(v) for v in tok[-5:])
        except ValueError as exc:
            raise ValueError(f"line {lineno}: non-numeric coordinate/charge/radius: {raw!r}") from exc
        if r < 0:
            raise ValueError(f"line {lineno}: negative radius {r}")
        coords.append((x, y, z))
        charges.append(q)
        radii.append(r)
        atom_name, res_name = tok[2], tok[3]
        res_seq = tok[-6] if len(tok) >= 10 else ""
        labels.append(f"{res_name} {res_seq} {atom_name}".strip())

    if not coords:
        raise ValueError("no ATOM/HETATM records found")

    return PQRData(
        coords=np.array(coords, dtype=float),
        charges=np.array(charges, dtype=float),
        radii=np.array(radii, dtype=float),
        labels=tuple(labels),
    )


def read_pqr(path: str | os.PathLike[str]) -> PQRData:
    return parse_pqr(Path(path).read_text())


def format_pqr(pqr: PQRData) -> str:
    """Render a PQR APBS accepts.

    Atom and residue names come from `labels` when present; the fallback names
    are inert placeholders, since nothing downstream of the solver reads them —
    only coordinates, charges, and radii affect the calculation.
    """
    lines = []
    for i in range(pqr.n_atoms):
        label = pqr.labels[i] if i < len(pqr.labels) else ""
        parts = label.split()
        if len(parts) == 3:
            res_name, res_seq, atom_name = parts
        else:
            res_name, res_seq, atom_name = "UNK", str(i + 1), "X"
        x, y, z = pqr.coords[i]
        lines.append(
            f"ATOM  {i + 1:5d} {atom_name:>4s} {res_name:>3s} {res_seq:>5s}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f} {pqr.charges[i]:7.4f} {pqr.radii[i]:6.4f}"
        )
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_pqr(path: str | os.PathLike[str], pqr: PQRData) -> None:
    Path(path).write_text(format_pqr(pqr))
