# Changelog

Notable changes to `sashimi-electro`. Versions follow [semantic
versioning](https://semver.org/) in intent, but see the warning in
[README.md](README.md): this is an alpha, and interfaces change without a
deprecation period.

## Unreleased

Alpha, so read the warning above: these are breaking changes and there is no
deprecation period. Every number below was measured; where one replaces a
number this project previously recorded, both are given.

### Fixed

- **`residue_potentials` was sampling inside the solute it was meant to sample
  around.** Probes are placed outside each atom's own radius to escape its
  self-energy singularity, and were never tested against *neighbouring* atoms
  — so on a protein the majority of samples were interior points: **55.9% on
  fas2, 61.4% on 1a63, 63.2% on serum albumin**. Probes are now rejected
  against every other atom's radius plus a solvent probe radius, which puts
  them provably outside the solvent-excluded surface for any surface model
  built with that probe or smaller. This is the third and last of the "at the
  atoms" family; `potential_extrema` was fixed in phase 5 and
  `potential_in_sphere` in 0.1.0's follow-up.

  **It moves every number this tool returns**, and by a lot: on
  `fas2-molecular` the median |value| goes 3.09 → 0.67 kT/e and debye's
  disagreement with APBS goes 0.4478 → 0.0126 kT/e per residue. The
  disagreements shrink further than the values do — noise-to-signal 0.177 →
  0.020 — so the axis got cleaner rather than quieter.

- **`sashimi_potential_extrema` reported `solute_masked: true` for a structure
  that masked nothing.** The badge was gated on `pqr_path` being supplied
  rather than on the mask having excluded a point, so a PQR from another
  structure, frame or unit convention produced unmasked extrema — the
  self-energy singularities — labelled as masked. It now reports
  `n_points_excluded_as_solute` and says so loudly when that is zero. The same
  mismatch was fixed in `sashimi_potential_in_sphere` before this; this was its
  sibling.

### Changed

- **`residue_potentials` samples 26 directions per atom, not 6.** Once probes
  inside neighbours are rejected, six axis directions leave **20 of 130
  residues on 1a63 and 23 of 110 on barnase** with nothing to sample at all;
  twenty-six leaves 14 and 17, and lifts the median surviving probes per
  residue from 5–7 to 26.
- **`ResiduePotential` lost `n_sampled` and gained probe counts** —
  `n_probes`, `n_probes_used`, `n_probes_excluded_as_solute`,
  `n_probes_outside_grid`, which sum to `n_probes`. `n_sampled` was removed
  rather than redefined so that a reader of the old field gets an
  `AttributeError` instead of a number that quietly means something else.
- **A residue with no usable probe keeps its row**, with `value` of `None`,
  no `mean_kT_e` key at all, and a `note` saying which of the two nothings it
  is — buried in the solute, or off the map. It used to vanish from a list
  callers read as a ranking.
- **`sashimi_residue_potentials`' response counts over the whole structure**,
  in keys that say so (`n_probes_all_residues` and friends), so `top` truncates
  the rows and nothing else. `under_sampled` is gone; `partly_off_map` and
  `unsampled` replace it and mean different things.
- **`sashimi.analysis.solute_mask` is public**, and `potential_extrema` accepts
  the mask it returns via `exclude_mask`. A caller that must report how much
  was excluded has to hold the mask anyway, and searching both signs used to
  build it twice.
- **`SolveResult` energies moved** where `ion_radius >= surface_radius`, which
  is every shipped case (#96). The ion-exclusion region is an exact union
  test rather than a lattice dilation, whose reach was quantised by what the
  lattice happened to admit: across nineteen spacings between 0.82 and 1.00 Å
  on the Born ion, the dilated path landed outside 2% on twelve of them and
  the union path on none.
- **`sashimi_compare_maps` no longer returns `max_abs_diff_kT_e` on the
  sampled path** (#104). A maximum over samples is a lower bound at any sample
  count, so it carries its own key there and the exact one is *absent* — a
  real number or a `KeyError`, never a quiet underestimate.

### Added

- **`SurfacePotential.interior_normal_derivative` and
  `exterior_normal_derivative`** (#95). A boundary-element answer has two
  halves and `sashimi.tabipb` had always parsed the second and dropped it at
  the protocol boundary. Named for the side each is on, because the bare name
  is ambiguous at a dielectric interface.
- **An exterior field evaluator that shares no construction with the lattice**
  (#98), and `sashimi.analytic.kirkwood_potential`'s referee behind it.

## 0.1.0 — 2026-08-27

First release. Phase 5 of [ROADMAP.md](ROADMAP.md).

### What is in it

- **The protocol** (`sashimi.protocol`) — a solver-neutral contract for
  Poisson–Boltzmann electrostatics. The request type is per solver family, so a
  request a backend cannot honour is unrepresentable rather than merely
  rejected: `FiniteDifferenceRequest` carries a grid,
  `BoundaryElementRequest` carries a mesh density and no grid at all.
  `tests/test_protocol_boundary.py` holds the seam closed.
- **Five backends across three solver families**, all behind that one
  interface — APBS 3.4.1, DelPhi in two flavours (C++ and pyDelPhi), TABI-PB,
  a Generalized Born tier, and `debye`.
- **`debye`**, a clean-room finite-difference solver in pure Python. It needs
  no binary at all, which makes `pip install sashimi-electro` a working
  install rather than a wrapper waiting for one. `sashimi-electro[fast]` adds
  a compiled surface kernel — 3.3–4.2x on a whole solve, bit-identical
  energies.
- **An MCP server** with nine tools: `sashimi_prepare_structure`,
  `sashimi_solve`, `sashimi_potential_at`, `sashimi_compare_maps`,
  `sashimi_potential_extrema`, `sashimi_potential_in_sphere`,
  `sashimi_residue_potentials`, `sashimi_capabilities` and
  `sashimi_validate_inputs`. The derived queries answer in bytes rather than
  megabytes, which is the point — an agent cannot use a 12 MB grid.
- **A CLI** — `sashimi corpus build|verify`, `sashimi validate`,
  `sashimi bench`.
- **A golden corpus** of 100 cases, 37 of them graded against a closed form
  (Born ion, Kirkwood sphere) rather than against a recording.

### Known limits, stated rather than discovered

- **APBS is a separate install.** It is a compiled binary, so no Python
  installer can provide it. Platform wheels are phase 6 and are not built.
- Whole phases of ROADMAP.md are unbuilt, and §4.1 records where the shipped
  protocol knowingly diverges from the target.
- The measurements throughout ROADMAP.md are research notes reviewed by nobody
  but their author.
