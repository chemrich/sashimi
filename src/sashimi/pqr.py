"""PQR read/write.

Own parser rather than a dependency: the format is trivial and owning it keeps
the dependency tree at numpy. PQR is whitespace-delimited in practice (unlike
PDB it is not column-fixed, because charges and radii overflow their columns),
so the fields are recovered positionally from the end of the line, which is the
only part of the layout every writer agrees on.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np

from sashimi.errors import MalformedStructure
from sashimi.protocol import DIMENSIONS, PQRData

__all__ = ["format_pqr", "parse_pqr", "read_pqr", "write_pqr"]

_RECORDS = ("ATOM", "HETATM")

# <record> <serial> <atom> <res> [chain] <resSeq> x y z q r
_MIN_FIELDS = 9  # the same line without a chain ID
_FIELDS_WITH_RESSEQ = 10
_FIELDS_WITH_CHAIN = 11


def parse_pqr(text: str) -> PQRData:
    coords: list[tuple[float, float, float]] = []
    charges: list[float] = []
    radii: list[float] = []
    labels: list[str] = []
    chains: list[str] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.startswith(_RECORDS):
            continue
        tok = raw.split()
        if len(tok) < _MIN_FIELDS:
            raise MalformedStructure(
                f"line {lineno}: expected at least {_MIN_FIELDS} fields, got {len(tok)}: {raw!r}"
            )
        try:
            x, y, z, q, r = (float(v) for v in tok[-5:])
        except ValueError as exc:
            raise MalformedStructure(
                f"line {lineno}: non-numeric coordinate/charge/radius: {raw!r}"
            ) from exc
        if r < 0:
            raise MalformedStructure(f"line {lineno}: negative radius {r}")
        coords.append((x, y, z))
        charges.append(q)
        radii.append(r)
        atom_name, res_name = tok[2], tok[3]
        res_seq = tok[-6] if len(tok) >= _FIELDS_WITH_RESSEQ else ""
        labels.append(f"{res_name} {res_seq} {atom_name}".strip())
        # Same tail-relative reasoning as `res_seq`: the chain sits one field
        # further left, and only exists when the line carries one at all. It
        # stays out of the label because `format_pqr` splits that back into
        # exactly three names.
        chains.append(tok[-7] if len(tok) >= _FIELDS_WITH_CHAIN else "")

    if not coords:
        raise MalformedStructure("no ATOM/HETATM records found")

    return PQRData(
        coords=np.array(coords, dtype=float),
        charges=np.array(charges, dtype=float),
        radii=np.array(radii, dtype=float),
        labels=tuple(labels),
        # All-empty means the file had no chain column; carrying a tuple of
        # empty strings would make "absent" indistinguishable from "blank".
        chains=tuple(chains) if any(chains) else (),
    )


def read_pqr(path: str | os.PathLike[str]) -> PQRData:
    """Read a PQR, transparently decompressing a `.pqr.gz`.

    Compression is here rather than at the call sites because nothing
    downstream ever sees the file: both binary backends re-serialise from
    `PQRData` with `format_pqr`, so the on-disk form is this function's business
    alone.

    It exists for one fixture and says so. A protonated 1,200-residue protein is
    18,000 atoms and 1.25 MB of text, which is half the size of `tests/data`
    entire and over the 1 MB limit the pre-commit hook enforces; gzipped it is
    324 KB, smaller than the largest PQR already committed. The alternative was
    a corpus case whose structure had to be fetched, and a case that can be
    absent is a case that silently does not run — the trap
    `.github/workflows/ci.yml` and ROADMAP.md section 7 both already record.
    """
    location = Path(path)
    if location.suffix == ".gz":
        with gzip.open(location, "rt") as handle:
            return parse_pqr(handle.read())
    return parse_pqr(location.read_text())


def format_pqr(pqr: PQRData) -> str:
    """Render a PQR in **fixed columns**, which is not the same as readable.

    Atom and residue names come from `labels` when present; the fallback names
    are inert placeholders, since nothing downstream of the solver reads them —
    only coordinates, charges, and radii affect the calculation.

    The name fields are truncated to their column widths rather than allowed to
    overflow, and that is load-bearing. They used to be minimum widths, so a
    four-character residue name — `TARG` in the APBS example set, `MEOH` in
    another — pushed every field after it one column to the right. sashimi's own
    reader splits on whitespace and APBS's is lenient, so both round-tripped it
    without complaint. DelPhi reads fixed columns: it parsed acetate as two
    charged atoms carrying +80.84 e where the file says eight and -1, and
    returned -865,205 kJ/mol against APBS's -196.90. Silently, for a year, on
    any structure whose residue name did not happen to be three characters.

    Truncation loses a character of a name nothing reads. Overflow loses the
    charges and radii, which are the only things that matter.

    `chains` is deliberately **not** written, for the same reason. A chain ID
    occupies its own column between the residue name and the sequence number, so
    emitting one shifts every field after it — the exact failure above, on every
    structure that has chains rather than only four-character residue names. No
    backend reads the chain, so writing it could only cost.
    """
    lines = []
    for i in range(pqr.n_atoms):
        label = pqr.labels[i] if i < len(pqr.labels) else ""
        parts = label.split()
        if len(parts) == DIMENSIONS:  # res_name, res_seq, atom_name
            res_name, res_seq, atom_name = parts
        else:
            res_name, res_seq, atom_name = "UNK", str(i + 1), "X"
        x, y, z = pqr.coords[i]
        lines.append(
            # Widths are exact, not minimum: `.4s` truncates as well as pads.
            # Names shorter than their field render byte-identically to the
            # previous minimum-width form, which is why no recorded corpus
            # number moves.
            f"ATOM  {i + 1:5d} {atom_name:>4.4s} {res_name:<4.4s}{res_seq:>5s}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f} {pqr.charges[i]:7.4f} {pqr.radii[i]:6.4f}"
        )
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_pqr(path: str | os.PathLike[str], pqr: PQRData) -> None:
    Path(path).write_text(format_pqr(pqr))
