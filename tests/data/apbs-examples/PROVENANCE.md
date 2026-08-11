# Structures vendored from the APBS examples

These PQR files are copied unmodified from the `examples/` directory of the APBS
3.4.1 distribution. APBS is BSD-3-Clause; `LICENSE.md` here is its licence text,
retained as that licence requires. Individual examples came to APBS from the
authors named below, and are redistributed under the same terms.

They are checked in rather than fetched because the corpus starts from PQR by
design (ROADMAP.md §7): preparation is pdb2pqr's business and carries its own
version, so starting from a fixed PQR means a corpus diff implicates the solver
rather than the structure-prep pipeline. A downloaded file is a file that can
change underneath a recorded number.

| file | atoms | net charge | origin | why it is here |
|---|---|---|---|---|
| `methanol.pqr` | 3 | 0.000 | UHBD, via APBS `examples/solv` | Smallest neutral molecule in the corpus; one atom has a 0.2 Å radius |
| `methoxide.pqr` | 2 | −1.000 | UHBD, via APBS `examples/solv` | The methanol anion — a charged 2-atom solute |
| `acetic-acid.pqr` | 8 | 0.000 | UHBD, via APBS `examples/ionize` | Neutral acid; pairs with acetate |
| `acetate.pqr` | 8 | −1.000 | UHBD, via APBS `examples/ionize` | Its conjugate base, the ionization pair |
| `fas2.pqr` | 906 | +4.053 | APBS `examples/misc` | Smallest protein here, and the only non-integer net charge |
| `barstar.pqr` | 1403 | −5.000 | APBS `examples/pbsam-barn_bars` | The most negatively charged case in the corpus |
| `barnase.pqr` | 1730 | +2.000 | APBS `examples/pbsam-barn_bars` | Its binding partner; a charge-complementary pair |
| `2LZT-ASP66.pqr` | 1960 | +8.000 | APBS `examples/bem-pKa` | Hen lysozyme, the structure every finding in this project came from |
| `1a63.pqr` | 2065 | −1.000 | APBS `examples/bem` | Protein–RNA complex: nucleic acid, which nothing else here covers |
| `hca.pqr` | 2482 | +1.000 | APBS `examples/hca-bind` | Human carbonic anhydrase; the largest, and metalloprotein chemistry |

## A trap, recorded where it will be found

**The energies APBS's own READMEs publish for these structures are not the
quantity sashimi reports**, and the gap is large enough to look like a bug:

| | APBS README | sashimi |
|---|---|---|
| methanol | −36.2486 | −25.16 |
| methoxide | −390.4122 | −201.96 |

Nothing is wrong with either. The `solv` example computes its reference state
with `sdie 1.00` — the solute's interior dielectric against *vacuum* — while
sashimi's `EnergyTerm.POLAR_SOLVATION` is solvated minus a *uniform* dielectric,
which is APBS's own convention in its `born` example and the one the protocol
commits to. Change that single keyword in their input file and APBS returns
−25.2538 and −201.5878, which sashimi reproduces to **0.37% and 0.18%**.

So these published numbers cannot be used as corpus references without exposing
the reference state as a knob, which is the raw-input passthrough sashimi
deliberately does not have. They are recorded here so that anyone comparing
sashimi against the APBS documentation finds the explanation before filing a
bug. `chgm spl0` versus sashimi's `spl4` accounts for a further 0.7% on
methanol, and is the smaller half of a difference that is mostly definitional.
