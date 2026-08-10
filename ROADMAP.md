# sashimi — Roadmap & Architecture Plan

> Thin, protocol-first wrapper around Poisson–Boltzmann electrostatics solvers,
> exposed as an MCP server. APBS is the first backend; the protocol is designed
> to outlive it. `debye` is the eventual clean-room solver implementation that
> slots in behind the same interface.

Status: planning / early implementation
Last updated: 2026-08-10

---

## 1. Thesis

Every serious PB solver (APBS, DelPhi, TABI-PB, MIBPB, PBSA, …) is an academic
C/Fortran code with its own input dialect, output format, packaging story, and
sharp edges. Nobody has made them interchangeable. sashimi's bet:

1. Define a small, stable protocol — `PQRData`, `GridSpec`, `PotentialGrid`,
   `SolveResult` — that is **solver-shaped, not APBS-shaped**.
2. Drive solvers as subprocesses behind that protocol (transport-agnostic:
   the same types serve MCP, CLI, and library use).
3. Make installation trivial (`pip install sashimi` eventually vendors the
   solver binary in platform wheels).
4. Use the multi-solver landscape as a *feature*: cross-solver validation is
   both our test harness and a user-facing capability.
5. When the protocol is proven, implement `debye` — a clean-room FD solver —
   validated against the ecosystem it abstracts.

## 2. Protocol design principles

The FD/BEM split is the acid test for protocol generality:

- **FD solvers** (APBS mg-auto, DelPhi, PBSA): Eulerian 3D grid in, volumetric
  potential out. `GridSpec` is meaningful as a *request* parameter.
- **BEM solvers** (TABI-PB, PyGBe, AFMPB): surface mesh, not a volume grid;
  native output is surface potential + normal derivative on the dielectric
  interface. No natural `GridSpec`.

Design consequences:

- `GridSpec` is an **FD-specific request detail**, not a top-level protocol
  requirement. Model it as part of a solver-family-specific request payload.
- Potential output should be representable as **either** a volumetric
  `PotentialGrid` **or** surface samples. Consider
  `SolveResult.potential: PotentialGrid | SurfacePotential | None` with
  energies always present (solvation energy is the universal currency —
  every solver produces it).
- `SolveResult` carries **provenance**: solver name, exact version, binary
  checksum, full resolved parameter set, wall time. This makes results
  reproducible and makes cross-solver comparison first-class.
- `PQRData` is the universal input (positions, charges, radii). Keep it
  solver-neutral; per-solver quirks (dielectric models, surface definitions,
  boundary conditions) live in a typed `SolverOptions` per backend with
  sane cross-solver defaults.
- Errors are structured: distinguish *input errors* (bad PQR, impossible
  grid), *solver errors* (non-zero exit, convergence failure — parse stderr),
  and *environment errors* (binary missing, wrong arch).

## 3. Backend strategy

### Phase order

| Phase | Backend | Why | Integration mode |
|-------|---------|-----|------------------|
| 1 | **APBS** (FD, mg-auto) | Community default, broadest features, conda-forge packaged | subprocess |
| 2 | **DelPhi** | FD sibling; Gaussian dielectric, focusing workflows; cheap triangulation partner for cross-validation | subprocess |
| 3 | **TABI-PB** | BEM; forces the protocol to handle surface potentials; already integrated into APBS but available standalone | subprocess |
| 3b | **PyGBe** | BEM, Python-native → **in-process** backend; stress-tests transport-agnosticism of the protocol types | import |
| 4 | **GB tier** (Amber GB / Bluues) | Fast approximation for high-throughput triage → PB refinement of top hits | subprocess |
| — | **MIBPB** | Not a production backend; the *accuracy referee* (~0.4% relative error on binding tests, rigorous interface treatment) | validation harness only |

### Cross-solver validation as a feature

`sashimi validate`: run the same system through N backends, report solvation
energy spread. Disagreement beyond discretization noise flags input-generation
bugs (ours) or parameter sensitivity (the user's problem, surfaced honestly).
Solvation energies are notoriously sensitive to surface definitions and grid
parameters — making the spread visible is genuinely useful, and it doubles as
our integration test suite.

## 4. APBS specifics (Phase 1)

### What we drive

Core finite-difference solver only (`mg-auto` / `mg-manual`). BEM
(TABI), geoflow, PBAM/PBSAM components are out of scope for the wrapper —
and disabled in any build we own (`-DENABLE_BEM=OFF -DENABLE_GEOFLOW=OFF
-DENABLE_PBAM=OFF`), which sidesteps the crustiest dependencies.

### Binary availability (as of Aug 2026)

- conda-forge `apbs` 3.4.1: `linux-64`, `osx-64`, `osx-arm64`, `win-64`.
- **No `linux-aarch64` build exists** (conda-forge or upstream GitHub releases).
- APBS releases essentially never; 3.4.1 has been current for years. Pinning
  is safe and maintenance burden of owned builds is low.

### Consequences for dev/test environments

- **Mac (Apple Silicon), local dev**: native `osx-arm64` APBS via
  `micromamba install -c conda-forge apbs`. Fast iteration.
- **Linux arm64 containers**: no prebuilt APBS → do **not** test the
  subprocess layer here. Run protocol-layer tests only.
- **Linux amd64 (the artifact users actually run)**:
  - Day-to-day: OrbStack containers with `--platform linux/amd64` (Rosetta).
    APBS is a plain CPU-bound FD solver; Rosetta handles it fine.
  - Validation tier: Ubuntu VM on the Proxmox box (real amd64 hardware).
    Required for anything timing-sensitive — Rosetta timings are not
    representative, which matters for timeout handling and any timing
    reported in `SolveResult`.
  - CI: GitHub Actions `ubuntu-latest` (amd64); conda-forge APBS is a
    one-liner. Full integration suite on every push.

### Test partitioning

- Protocol layer (PQRData/GridSpec/PotentialGrid/SolveResult, serialization,
  MCP plumbing) is pure Python → test natively on every arch, no solver
  required.
- Subprocess integration tests gated behind `@pytest.mark.apbs` (and later
  `@pytest.mark.delphi`, etc.). Only run where the real binary exists.
- `debye` inverts this: pure implementation, whole suite runs natively
  anywhere. That portability is itself a tested differentiator.

## 5. Distribution roadmap

### v1: conda-forge dependency

Ship sashimi on PyPI; document `micromamba install -c conda-forge apbs` (or
detect an `apbs` on PATH). Lowest effort, unblocks users today.

### v2: vendored platform wheels ("pip install sashimi just works")

Repackage the APBS binary inside platform wheels — the `cmake`/`ruff`/`ninja`
pattern: binary vendored in the wheel, resolved via an internal path, never
PATH-dependent. Collapses install from conda-env choreography to one command.
For an MCP server that agents spin up, removing the conda prerequisite is a
major friction win. Wheel packaging is identical regardless of whose binary is
inside — nothing in v1 forecloses this.

### Owned builds (enables v2 + fills the aarch64 gap)

Build matrix in GitHub Actions: `linux-64`, `linux-aarch64`, `osx-arm64`,
optionally `win-64`. Crib from the conda-forge feedstock's recipe — it encodes
every patch and flag that makes APBS build on each platform; read it before
the upstream docs. Trimmed configure (core FD only), mostly-static linking for
a single portable, checksummable file. Contributing the `linux-aarch64` build
upstream (feedstock migration or GitHub release) makes us the fix, not a fork.

### Licensing obligations for redistribution

- APBS 3.x itself: **BSD-3-Clause**.
- Linked Holst-group stack: **MALOC is LGPL-2.1 in later versions; Fedora
  packages it GPL-2.0-or-later; FETK components historically GPL** — terms
  depend on the exact FETK version vendored. Older APBS carried a special GPL
  exception for aggregation.
- Redistribution is permitted either way. Obligations: ship license texts,
  provide corresponding source for copyleft components (a pinned build repo
  satisfies this). If any linked component is GPL rather than LGPL, the
  combined *binary* is GPL-distributable — fine for open-source sashimi;
  only relevant if the binary were embedded in something proprietary.
- sashimi invokes APBS as a **subprocess**, so copyleft stops at the binary.
  sashimi's own license is unconstrained.
- Action item before v2 ships: read the actual license files in the exact
  FETK/MALOC versions our build vendors. (Not legal advice; verify.)

## 6. debye — clean-room solver roadmap

### Ordering

1. **FD solver first.** Linearized PB on a Cartesian grid; mirror the APBS
   mg-auto contract so it's a drop-in behind the existing backend interface.
2. **BEM as the eventual second engine.** The method family that most rewards
   a clean modern implementation — incumbents are Fortran/C academic codes
   with exactly the packaging problems sashimi routes around.

### Validation ladder

1. **Analytic ground truth**: Born ion, Kirkwood dielectric sphere. Exact
   answers; catches sign/units/BC bugs immediately.
2. **APBS agreement**: same PQR + GridSpec through both; converge under grid
   refinement.
3. **Referee tier**: MIBPB (rigorous interface treatment) arbitrates when
   debye and APBS disagree; DelPhi/PBSA as additional triangulation.
4. **Portability as a test**: full suite passes natively on osx-arm64,
   linux-aarch64, linux-64 — the thing no incumbent can do.

APBS (pinned, checksummed, via owned builds) is debye's fixed reference
implementation. Trimmed reproducible builds exist partly for this.

## 7. Infrastructure notes

### Remote access to the amd64 validation VM

- **Tailscale** on the Ubuntu VM (or LXC subnet router on Proxmox):
  SSO login (YubiKey as second factor at the IdP), zero ports forwarded on
  the UDM Pro. SSH + VS Code Remote / Claude Code rather than a desktop —
  no RDP surface at all.
- Alternatives considered: UniFi WireGuard (no true second factor),
  Guacamole + TOTP or behind Cloudflare Access with a hardware-key policy
  (most capable, most setup). Tailscale chosen for effort/benefit.

### Environments summary

| Environment | Arch | APBS source | Role |
|---|---|---|---|
| Mac local | osx-arm64 | conda-forge native | dev loop, protocol tests |
| OrbStack container | linux/amd64 (Rosetta) | conda-forge | subprocess integration tests |
| Proxmox Ubuntu VM | linux-64 native | conda-forge / owned build | timing-sensitive validation, benchmarks |
| GitHub Actions | linux-64 | conda-forge one-liner | full suite per push |
| (future) arm64 Linux | linux-aarch64 | owned build | once we ship it |

## 8. Phased plan

### Phase 0 — Protocol hardening (now)
- [ ] Finalize `PQRData`, `SolveResult` as solver-neutral; demote `GridSpec`
      to FD-family request detail
- [ ] `PotentialGrid | SurfacePotential` output union
- [ ] Structured error taxonomy (input / solver / environment)
- [ ] Provenance fields in `SolveResult` (solver, version, checksum, params,
      wall time)
- [ ] Protocol test suite — pure Python, runs everywhere

### Phase 1 — APBS backend, shippable
- [ ] Subprocess driver: input-file generation, invocation, stdout/stderr/DX
      parsing, exit-code handling
- [ ] `@pytest.mark.apbs` integration suite; amd64 container + CI wiring
- [ ] MCP server surface (solve, validate-inputs, capabilities)
- [ ] PyPI release; conda-forge APBS documented as prerequisite
- [ ] Proxmox VM stood up + Tailscale for benchmarks/timeouts work

### Phase 2 — Distribution
- [ ] Owned APBS build matrix (trimmed, mostly-static; feedstock-derived)
- [ ] linux-aarch64 build; offer upstream
- [ ] License-file audit of vendored FETK/MALOC versions
- [ ] Platform wheels with vendored binary; `pip install sashimi` end-to-end

### Phase 3 — Multi-backend
- [ ] DelPhi backend + cross-validation harness (`sashimi validate`)
- [ ] TABI-PB backend → forces SurfacePotential path to be real
- [ ] PyGBe in-process backend → proves transport-agnosticism
- [ ] Optional GB tier for triage→refine workflows

### Phase 4 — debye
- [ ] FD solver vs analytic ladder (Born, Kirkwood)
- [ ] APBS agreement under grid refinement; MIBPB referee harness
- [ ] Drop-in behind backend interface; portability suite green on all arches
- [ ] BEM engine (later)

## 9. Open questions

- Nonlinear PB in the protocol: APBS/DelPhi support it, BEM mostly doesn't.
  Capability flags per backend, or restrict v1 protocol to linearized PB?
- Surface definition (SES vs Gaussian vs spline) is the biggest cross-solver
  comparability confounder — how much do we normalize vs expose?
- DX-format `PotentialGrid` payloads are large; MCP transport strategy
  (inline vs file-reference vs downsampled) for volumetric results.
- Does `sashimi validate` belong in core or as a separate tool that consumes
  the protocol?
