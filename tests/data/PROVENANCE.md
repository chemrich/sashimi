# Structures prepared here, rather than vendored

`apbs-examples/` holds PQR files copied unmodified from the APBS distribution,
and `apbs-examples/PROVENANCE.md` records them. This file covers the structures
this project prepared itself, which need more recording rather than less: a
vendored file arrives fixed, where a prepared one carries every choice made
during preparation into every energy recorded against it.

The reason is the one `apbs-examples/PROVENANCE.md` already gives — **a
downloaded file is a file that can change underneath a recorded number** — with
a second edge. pdb2pqr's output depends on its version, the force field, the pH
and the flags, and none of that is visible in the PQR it writes. A corpus
recording is only reproducible if those are written down somewhere, so they are
written down here.

| file | atoms | residues | net charge | source |
|---|---|---|---|---|
| `ala-gly.pqr` | 20 | 2 | 0.000 | Hand-built dipeptide; the smallest multi-atom case in the corpus |
| `1ao6.pqr.gz` | 18,242 | 1,156 | −30.000 | PDB [1AO6](https://www.rcsb.org/structure/1AO6), human serum albumin |

## `1ao6.pqr.gz`

Human serum albumin, 2.5 Å X-ray, two copies of the same 585-residue chain in
the asymmetric unit. It is the corpus's largest solute by more than a factor of
two, and it exists to grade the top of the size range protean actually works in
— ROADMAP.md §12 sets that at 250 to 1,200 residues, and before this the corpus
stopped at 8,279 atoms.

Prepared with:

```sh
curl -O https://files.rcsb.org/download/1AO6.pdb
pdb2pqr --ff=AMBER --drop-water --with-ph=7.0 1AO6.pdb 1ao6.pqr
gzip -9 1ao6.pqr
```

- **pdb2pqr 3.7.1**, which is the version this project pins in `pyproject.toml`.
- **AMBER radii**, matching what the roadmap's protein cases use.
- **pH 7.0**, stated rather than defaulted, because titration state is the
  largest chemical choice in preparing a structure and a default is not a record.
- **`--drop-water`**, discussed below.

**Gzipped, and `read_pqr` decompresses `.pqr.gz` transparently.** The text is
1.25 MB — half the size of `tests/data` entire, and over the 1 MB limit
`.pre-commit-config.yaml` enforces. At 318 KB it is smaller than the largest PQR
already committed. Nothing downstream sees the file: both binary backends
re-serialise from `PQRData` with `format_pqr`.

### The seven crystallographic waters, and why they are gone

1AO6 ships 7 ordered waters, 21 atoms. The first version of this fixture kept
them, and **no other structure in `tests/data` contains a single `HOH`** — so a
reader comparing cases would have had one fixture silently modelling something
none of the others do.

Kept as solute they are not neutral bookkeeping. Each becomes an isolated
low-dielectric island — a lone 1.66 Å oxygen sphere, its two hydrogens at zero
radius — sitting in bulk solvent with a full TIP3P charge distribution inside
it, ±0.834 e. That is a defensible model of a buried structural water and a poor
one of a surface water that the continuum should simply be.

**Measured before deciding, on the recorded lattice**: dropping them moves APBS
by **−3.24 kJ/mol (0.01%)** and DelPhi C++ by **−7.98 (0.02%)**. So the waters
were never the reason the three reference backends spread 10.4% on this case —
that is APBS focusing against debye solving the full box, three orders of
magnitude larger. The measurement is what says removing them is safe; the
consistency with every other fixture is why it is right.
