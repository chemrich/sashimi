# Changelog

Notable changes to `sashimi-electro`. Versions follow [semantic
versioning](https://semver.org/) in intent, but see the warning in
[README.md](README.md): this is an alpha, and interfaces change without a
deprecation period.

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
