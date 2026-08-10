# sashimi — Roadmap and Implementation Plan

> Thin, protocol-first wrapper around Poisson–Boltzmann electrostatics solvers,
> exposed as an MCP server. APBS is the first backend; the protocol is designed
> to outlive it. `debye` is the eventual clean-room solver that slots in behind
> the same interface.

Status: phases 0–3 shipped; protocol hardening next.
Last updated: 2026-08-10

This is the single planning document. It supersedes the earlier split between
ROADMAP.md (intent) and PLAN.md (APBS implementation), which had two
incompatible phase numberings and had begun to disagree with each other and
with the code. Where this document and the code disagree, **this document is
the intent and the code is the debt** — §4 records exactly where that gap is.

---

## 1. Thesis, goals, non-goals

Every serious PB solver (APBS, DelPhi, TABI-PB, MIBPB, PBSA, …) is an academic
C/Fortran code with its own input dialect, output format, packaging story, and
sharp edges. Nobody has made them interchangeable. sashimi's bet:

1. Define a small, stable protocol — `PQRData`, `SolveResult`, and friends —
   that is **solver-shaped, not APBS-shaped**.
2. Drive solvers as subprocesses behind that protocol, transport-agnostically:
   the same types serve MCP, CLI, and library use.
3. Make installation trivial (eventually `pip install sashimi`, with the solver
   binary vendored in platform wheels).
4. Treat the multi-solver landscape as a *feature*: cross-solver validation is
   both the test harness and a user-facing capability.
5. When the protocol is proven, implement `debye` — a clean-room FD solver —
   validated against the ecosystem it abstracts.

The near-term product is narrower: a clean Python interface and FastMCP server
computing electrostatic potential maps from structures, using a frozen APBS
3.4.1 as backend, so that mcpymol and protean never touch APBS input files,
temp directories, or OpenDX parsing directly.

**Non-goals.** sashimi does not compile, vendor, or patch APBS *source*. It
does not expose the FEM, geoflow, BEM, PBAM, or PBSAM solvers — only the
`mg-auto` finite-difference path, which covers visualization-grade and standard
solvation-energy work. It does not attempt nonlinear PBE in v1 (the API leaves
room; LPBE is the contract debye will initially honor). It is not a general
APBS input-file generator, and it deliberately exposes no raw-input passthrough:
anyone who needs `fe-manual` needs APBS, not sashimi.

## 2. Architecture

Three layers, strictly separated so the bottom one is replaceable:

```
┌─────────────────────────────────────────────┐
│  sashimi.mcp        FastMCP server (stdio)   │   ← what mcpymol/protean connect to
├─────────────────────────────────────────────┤
│  sashimi.protocol   Types + Solver protocol  │   ← the debye contract; zero APBS knowledge
├─────────────────────────────────────────────┤
│  sashimi.apbs       Subprocess backend       │   ← input gen, execution, DX parsing
│  sashimi.prep       pdb2pqr wrapper          │
└─────────────────────────────────────────────┘
```

The protocol layer is the load-bearing decision. Everything above it speaks in
physical terms (angstroms, molar ionic strength, dielectric constants);
everything below speaks APBS (dime, cglen, fglen, chgm spl4). debye replaces
only the bottom-left box.

### The FD/BEM split is the acid test

- **FD solvers** (APBS mg-auto, DelPhi, PBSA): Eulerian 3-D grid in, volumetric
  potential out. A grid specification is meaningful as a *request* parameter.
- **BEM solvers** (TABI-PB, PyGBe, AFMPB): surface mesh, not a volume grid;
  native output is surface potential and normal derivative on the dielectric
  interface. There is no natural grid specification at all.

Any protocol decision that assumes a volumetric grid forecloses half the
landscape. That is the single most important constraint on §4.

## 3. Package layout

```
sashimi/
├── pyproject.toml            # uv-managed; runtime deps: numpy, pdb2pqr, pydantic, fastmcp
├── uv.lock                   # the single lockfile; APBS is a system prerequisite
├── src/sashimi/
│   ├── protocol.py           # PQRData, GridSpec, PotentialGrid, SolveResult, Solver
│   ├── errors.py             # backend-neutral failure modes
│   ├── pqr.py                # PQR read/write (own parser — trivial format, no deps)
│   ├── dx.py                 # OpenDX read/write ↔ numpy + metadata
│   ├── prep.py               # pdb2pqr subprocess wrapper (PDB → PQRData + warnings)
│   ├── corpus.py             # golden-corpus manifest, build, verify
│   ├── cli.py                # `sashimi corpus build|verify`
│   ├── apbs/
│   │   ├── backend.py        # ApbsSolver implements Solver
│   │   ├── input.py          # grid + solvent → mg-auto input file (f-strings, no jinja)
│   │   ├── grid.py           # physical grid intent → legal dime/cglen/fglen (psize logic)
│   │   ├── run.py            # subprocess, tmpdir, timeout, stdout parsing, error mapping
│   │   └── discover.py       # binary discovery: $SASHIMI_APBS_PATH → which() → conda env
│   └── mcp/server.py         # FastMCP tools
├── tests/
│   ├── test_dx.py, test_pqr.py, test_grid.py, test_prep.py   # no binary required
│   ├── test_mcp.py           # drives a real fastmcp.Client session
│   ├── test_solver.py        # @pytest.mark.apbs — requires binary
│   ├── test_analytic.py      # born ion vs closed form
│   ├── test_corpus.py        # golden corpus + fault injection
│   ├── data/                 # PQR/PDB fixtures
│   └── corpus/               # checked-in golden summaries (JSON, not raw grids)
└── .github/workflows/ci.yml  # micromamba installs apbs; uv does the rest
```

## 4. The protocol — the debye contract

Units are fixed at this boundary (Å, kT/e, kJ/mol) so debye never inherits
APBS's unit conventions by accident.

```python
@dataclass(frozen=True)
class GridSpec:
    """Grid intent in physical terms. Backends translate to their own constraints."""
    resolution: float = 0.5        # target spacing, Å (fine grid)
    padding: float = 10.0          # min distance, molecule surface → fine-grid edge, Å
    max_points: int = 161**3       # memory guardrail; backend errors if unsatisfiable

@dataclass(frozen=True)
class SolventModel:
    solvent_dielectric: float = 78.54
    solute_dielectric: float = 2.0
    ionic_strength: float = 0.150   # M, 1:1 salt
    ion_radius: float = 2.0         # Å
    temperature: float = 298.15     # K
    surface_method: Literal["smol", "spl2", "mol"] = "smol"
    surface_radius: float = 1.4     # solvent probe, Å

@dataclass
class SolveResult:
    potential: PotentialGrid
    energy_kj_mol: float | None     # total polar solvation energy when requested
    backend: str                    # "apbs-3.4.1" | "debye-x.y" — provenance travels
    diagnostics: dict               # grid actually used, wall time, resolved path
```

`GridSpec` deliberately has no `dime`: legal PMG dimensions (n = c·2^(l+1)+1,
i.e. 33, 65, 97, 129, 161 …) are an APBS implementation detail computed in
`apbs/grid.py`, which reimplements the essentials of the classic `psize.py`
sizing — coarse grid for the focusing boundary, fine grid at extent + 2·padding,
smallest legal dime meeting the resolution target, capped by `max_points`.
`diagnostics` reports the grid actually used, so callers can detect when the
resolution target was relaxed.

### 4.1 Resolved in phase 4

The protocol shipped in phases 1-3 was APBS-shaped in five specific ways. All
five are now paid; the table is kept because the *reasons* remain the design
rationale, and because a future backend will test them again.

| # | Was | Now |
|---|---|---|
| 1 | `GridSpec` was the second positional parameter of `solve_lpbe` | `FiniteDifferenceRequest.grid`; `BoundaryElementRequest` has no such field |
| 2 | `potential` required, `energy_kj_mol` optional | `potential: PotentialGrid \| SurfacePotential \| None`; both are requested via `want_energy` / `want_potential` and `check_satisfies` enforces delivery |
| 3 | Provenance was a backend string | `Provenance` with binary path, **sha256**, resolved parameters and wall time |
| 4 | Errors split backend-neutral vs APBS-specific | `InputError` / `SolverError` / `BackendUnavailable`, by who can act |
| 5 | `surface_method: Literal["smol", "spl2", "mol"]` | `SurfaceModel` enum; APBS keywords live in `sashimi.apbs.options` |

Item 2 deviates from the original wording in one respect worth flagging: energy
is *requested*, not unconditionally computed. Making it mandatory would force
the APBS reference calculation on every solve — roughly double the work — for a
caller who only wants a map to colour a surface with. `want_energy` defaults to
true, so the common path is unchanged; `check_satisfies` is what makes the
request binding.

`tests/test_bem_contract.py` is the standing check: a real `Solver`
implementation returning surface potentials, exercised through the same types
APBS uses, in the binary-free tier.

### 4.2 Error taxonomy

Current, in `errors.py` with APBS-specific subclasses under `sashimi.apbs`:
`SashimiError` → `SolverNotFound` (`ApbsNotFound`), `GridTooLarge`,
`ConvergenceFailure`, `SolverCrash` (`ApbsCrash`); `PreparationFailed` for
pdb2pqr. Every message carries the actionable next step — `GridTooLarge` states
the achievable resolution, `ApbsNotFound` carries the install one-liner.

Target taxonomy distinguishes **input errors** (bad PQR, impossible grid),
**solver errors** (non-zero exit, convergence failure, parsed from stderr), and
**environment errors** (binary missing, wrong architecture). These are the three
things a caller can act on differently.

## 5. APBS backend specifics

**Input generation** targets one template: `mg-auto`, `lpbe`, `bcfl sdh`,
`chgm spl4`, dielectric and ion declarations per `SolventModel`, `write pot dx`.

Energy needs **two** `elec` blocks. A single block reports total electrostatic
energy, dominated by point-charge self-energy — not what anyone means by
solvation energy. The polar solvation energy is the difference between the
solvated state and a uniform-dielectric, ion-free reference, which is what APBS
prints as `Global net ELEC energy`.

**Execution** (`run.py`): each solve runs in a fresh `TemporaryDirectory` —
APBS writes `io.mc` and its DX output into the working directory, so anything
less leaks files into the caller's cwd and lets concurrent solves collide.
Success is verified **structurally, not by exit code**, because APBS exits 0 on
several failures: the DX file must exist and parse, and stdout is scanned for
error signatures. Failures raise typed exceptions carrying the tail of stdout.

Two build-dependent behaviours, both found by cross-platform CI:

- MPI-enabled builds (Debian's) name the output `potential-PE0.dx` — the
  processing-element rank — where serial builds (conda-forge, Homebrew) write
  `potential.dx`. `find_potential` accepts either.
- Every invocation writes `io.mc` to cwd, `--version` included, so the version
  probe runs inside a temp dir too.

**Surface definition** is the dominant modelling confounder, measured on this
code at 0.4 Å resolution: holding structure, radii and grid fixed and varying
only `srfm` moves the ALA-GLY solvation energy across a **25.7%** range
(`mol` −212.13, `smol` −209.66, `spl2` −268.84 kJ/mol); even the Born ion, a
single sphere with no reentrant surface, spans 8.3%. For scale, the radii set
spans 16% (AMBER −209.66, PARSE −243.40, CHARMM −210.36) and the probe radius
5%. The corpus gates numerical drift at 0.01%, so a legal parameter change moves
the answer 2,500× the tolerance we treat as "the solver moved".

`spl2` is most of that spread and is a trap: spline surfaces exist to give
smooth derivatives for *force* calculations, and using one for solvation energy
is a known misuse. Per §14's decision it moves behind APBS-specific options
rather than sitting as a peer of `mol` and `smol` in the protocol.

**DX I/O** (`dx.py`): own reader/writer rather than a gridData/MDAnalysis
dependency — the format is trivial (header with counts/origin/deltas, then
3 floats per line, C order) and owning it keeps the dependency tree small. The
writer exists so any backend's output can be exported for PyMOL/ChimeraX.

**Binary discovery**: `$SASHIMI_APBS_PATH` wins, then `shutil.which("apbs")`,
then an active conda environment as a courtesy. APBS is compiled, so no Python
installer can provide it and `which` is the normal answer. `SolveResult.backend`
records the resolved path alongside the version, so which binary produced a
result is always recoverable.

**Structure prep** (`prep.py`): subprocess wrapper around pdb2pqr 3.7.1 — its
Python API is not a stability contract, and process isolation means a pdb2pqr
hang cannot take down the MCP server. Returns `PQRData` plus a structured
summary of what was rebuilt (missing heavy atoms, debumped residues), which the
MCP layer surfaces rather than buries: an agent should know three sidechains
were rebuilt before trusting the energies.

### Binary availability (as of Aug 2026)

- conda-forge `apbs` 3.4.1: `linux-64`, `osx-64`, `osx-arm64`, `win-64`.
- **No `linux-aarch64` build exists**, conda-forge or upstream.
- Debian/Ubuntu 24.04+ ships 3.4.1 (`apt install apbs`); Ubuntu 22.04 has 3.0.0.
- macOS Homebrew has it only in the third-party `brewsci/bio` tap, which recent
  Homebrew refuses to load without an explicit trust step. conda-forge is
  preferred for provenance, and is what CI tests.
- APBS releases essentially never; 3.4.1 has been current for years. Pinning is
  safe and the maintenance burden of owned builds is low.

## 6. FastMCP server

Tools are prefixed `sashimi_`, take flat physically-named parameters (agents
handle those better than nested config objects), and return structured content
plus a short human-readable summary.

`sashimi_prepare_structure(pdb_path, forcefield, ph, output_pqr)` — runs
pdb2pqr; the returned warnings summary is the important part.

`sashimi_solve(pqr_path, resolution, padding, ionic_strength, solute_dielectric,
solvent_dielectric, compute_energy, output_dx)` — the workhorse. Returns grid
stats, the DX path, energy when requested, backend provenance, and
`resolution_relaxed` so a guardrail-relaxed grid is visible rather than silent.

`sashimi_potential_at(dx_path, points)` — trilinear interpolation of a saved
map. Out-of-grid points return null, never a clamped edge value, because a
clamped number reads as a real measurement.

`sashimi_compare_maps(dx_a, dx_b)` — RMSD, max absolute difference, correlation.
Mismatched grids are refused rather than silently resampled. Useful for
mutant-versus-wildtype now; doubles as solver-versus-solver validation later.

Deliberately absent: a PDB-fetching tool (mcpymol owns structure acquisition)
and raw APBS-input passthrough (defeats the abstraction). A test asserts no APBS
vocabulary appears anywhere in the tool surface.

Transport is stdio. Solves at default resolution on a ~300-residue protein run
in seconds; no async job queue in v1, but a 300 s timeout with a clear error
keeps a pathological grid from wedging the server.

**Volumetric delivery.** Inlining a grid is arithmetically impossible, not a
tradeoff: 97³ is 12.3 MB of DX (~3.1M tokens), 161³ — the `max_points` cap — is
56.4 MB. Downsampling to a 25k-token budget means a 19³ grid at ~3 Å spacing,
which is below the resolution where a molecular surface means anything. Tools
therefore return a **path**, and per §14 that path becomes content-addressed
from the resolved-parameter hash (so re-solving cannot silently overwrite), with
a documented cleanup contract and an explicit local-filesystem assumption. MCP
Resources (`sashimi://maps/<id>`) are the correct answer for a non-local
transport and are deferred until one exists.

The complementary half is **derived queries** — an agent cannot use a grid, only
answers about one. `sashimi_potential_at` is the first of that family; extrema
with coordinates, per-residue averages and potential over a selection follow in
phase 5. That is where the product value is, rather than in moving bytes.

Note: `sashimi_solve` exposes no `max_points`, so `GridTooLarge` is unreachable
from the MCP surface — the guardrail can only relax a request, never reject it.
That is the right default for an agent-facing tool, and `resolution_relaxed`
keeps the relaxation visible.

## 7. Testing and the golden corpus

**Three tiers.** Pure unit tests (PQR parsing, DX round-trip, dime legality,
psize math, structure prep, the whole MCP surface bar solving) run anywhere with
no binary. Analytic tests solve the Born ion — a single +1 charge, 3 Å sphere —
against the closed form. Integration tests exercise the real subprocess.

**Analytic calibration**, both facts learned the hard way:

- The APBS `examples/born` README states an analytic solvation energy of
  −230.62 kJ/mol; the closed-form expression with current CODATA constants gives
  **−228.61**. That 0.87% spread straddles a tight gate — measured against the
  CODATA value `smol` errs 0.176% and `mol` 0.509%; against APBS's, 0.70% and
  0.37%. So each reference fails a different variant. sashimi computes its own
  value from named constants and asserts at **1%**.
- **Probe only well outside the dielectric boundary.** At exactly r = a = 3 Å
  the potential is off by 71% because the point sits on the smoothed
  discontinuity, and the grid centre is the point-charge singularity. From
  1.25 a outward, agreement is 0.5–1.1%.

**The golden corpus is a first-class deliverable**, not a test artifact.
`sashimi corpus build` runs a fixed manifest (structure + grid + solvent, seeds
pinned) and writes per case a JSON summary: grid geometry, energy, potential
min/max/mean/std, and the potential at 50 deterministically-placed probe points.
Summaries are checked in; raw grids are reproducible on demand.
`sashimi corpus verify --backend X` re-runs the manifest against any `Solver`
and diffs within stated tolerances.

Cases start from checked-in PQR, never PDB: preparation is pdb2pqr's business
and carries its own version, so starting from PQR means a corpus diff implicates
the solver rather than the prep pipeline.

`verify_case` compares a fresh solve against a *reference dict*, and does not
care where that dict came from — feeding it `build_case` output from a second
backend is cross-solver validation with no new engine. Phase 4 makes that
explicit by giving the reference a type (`RecordedReference` vs
`BackendReference`), which is what turns §8's `sashimi validate` into a thin
addition rather than a parallel implementation.

Day one this is the regression net for sashimi and for the system APBS that no
lockfile pins any more. The day debye exists, `corpus verify --backend debye` is
its acceptance test, with APBS ground truth baked in and no APBS installation
required.

**Test partitioning by architecture.** Protocol-layer tests are pure Python and
run natively everywhere. Subprocess integration tests are gated behind
`@pytest.mark.apbs` (later `@pytest.mark.delphi`, …) and only run where the real
binary exists — which matters because there is no `linux-aarch64` APBS. debye
inverts this: pure implementation, whole suite runs natively anywhere, and that
portability is itself a tested differentiator.

**CI**: GitHub Actions on `ubuntu-latest` and `macos-latest`, `uv sync --frozen`
against the committed lockfile, with APBS installed from conda-forge via
micromamba — whose only job is fetching that one binary. A single `ci-ok` gate
job fronts the matrix so the required status-check name survives matrix changes.

## 8. Backend strategy beyond APBS

| Phase | Backend | Why | Integration |
|-------|---------|-----|-------------|
| 1 | **APBS** (FD, mg-auto) | Community default, broadest features, conda-forge packaged | subprocess |
| 2 | **DelPhi** | FD sibling; Gaussian dielectric, focusing workflows; cheap triangulation partner | subprocess |
| 3 | **TABI-PB** | BEM; forces the protocol to handle surface potentials | subprocess |
| 3b | **PyGBe** | BEM, Python-native → **in-process**; stress-tests transport-agnosticism | import |
| 4 | **GB tier** (Amber GB / Bluues) | Fast approximation for high-throughput triage → PB refinement | subprocess |
| — | **MIBPB** | Not production; the *accuracy referee* (~0.4% relative error, rigorous interface treatment) | validation harness |

**Cross-solver validation as a feature.** `sashimi validate`: run one system
through N backends, report the solvation-energy spread. Disagreement beyond
discretization noise flags input-generation bugs (ours) or parameter sensitivity
(the user's problem, surfaced honestly). Solvation energies are notoriously
sensitive to surface definitions and grid parameters; making the spread visible
is genuinely useful, and it doubles as the integration suite.

Relationship to `sashimi corpus`: the corpus verifies one backend against
*recorded* numbers; `validate` compares N backends against *each other* live.
The corpus is the regression net, `validate` is the product feature — but they
are **one engine with two reference kinds**, and `validate` lives in core for
that reason (§14). Additional backends ship as optional extras
(`sashimi[delphi]`) so core stays lean and no dependency cycle forms.

`validate` **refuses to report a spread across mismatched surface models**
unless explicitly overridden. Given the 25.7% measured in §5, a spread computed
across differing surface definitions would not be a solver disagreement at all —
it would be a modelling difference misreported as one, which is worse than no
number. This is why resolved parameters must reach provenance (§4.1 item 3).

## 9. Distribution and packaging

**v1 — conda-forge dependency.** Ship sashimi on PyPI; document
`micromamba install -c conda-forge apbs`, or detect an `apbs` on PATH. Lowest
effort, unblocks users today.

**v2 — vendored platform wheels.** Repackage the APBS binary inside platform
wheels, the `cmake`/`ruff`/`ninja` pattern: binary vendored, resolved via an
internal path, never PATH-dependent. Collapses install from conda-env
choreography to one command. For an MCP server that agents spin up, removing the
conda prerequisite is a major friction win. Nothing in v1 forecloses it.

**Owned builds** (enables v2, fills the aarch64 gap). Build matrix in Actions:
`linux-64`, `linux-aarch64`, `osx-arm64`, optionally `win-64`. Crib from the
conda-forge feedstock recipe — it encodes every patch and flag that makes APBS
build on each platform; read it before the upstream docs. Trimmed configure
(core FD only: `-DENABLE_BEM=OFF -DENABLE_GEOFLOW=OFF -DENABLE_PBAM=OFF`),
mostly-static linking, one portable checksummable file. Contributing the
`linux-aarch64` build upstream makes us the fix, not a fork.

### Licensing obligations for redistribution

- APBS 3.x itself: **BSD-3-Clause**.
- Linked Holst-group stack: **MALOC is LGPL-2.1 in later versions; Fedora
  packages it GPL-2.0-or-later; FETK components historically GPL** — terms
  depend on the exact FETK version vendored. Older APBS carried a special GPL
  exception for aggregation.
- Redistribution is permitted either way. Obligations: ship license texts and
  provide corresponding source for copyleft components (a pinned build repo
  satisfies this). If a linked component is GPL rather than LGPL, the combined
  *binary* is GPL-distributable — fine for open-source sashimi, relevant only if
  embedded in something proprietary.
- sashimi invokes APBS as a **subprocess**, so copyleft stops at the binary;
  sashimi's own license is unconstrained.
- Action item before v2 ships: read the actual license files in the exact
  FETK/MALOC versions the build vendors. (Not legal advice; verify.)

## 10. debye — clean-room solver

**Ordering.** FD solver first: linearized PB on a Cartesian grid, mirroring the
mg-auto contract so it drops in behind the existing backend interface. BEM as
the eventual second engine — the family that most rewards a clean modern
implementation, since the incumbents are Fortran/C academic codes with exactly
the packaging problems sashimi routes around.

**Validation ladder.**

1. **Analytic ground truth**: Born ion, Kirkwood dielectric sphere. Catches
   sign, unit and boundary-condition bugs immediately.
2. **APBS agreement**: same input through both; must converge under refinement.
3. **Referee tier**: MIBPB arbitrates when debye and APBS disagree;
   DelPhi/PBSA for additional triangulation.
4. **Portability as a test**: full suite passes natively on osx-arm64,
   linux-aarch64 and linux-64 — the thing no incumbent can do.

**Handoff.** debye ships as a separate repo implementing `sashimi.protocol`'s
solver interface. Per §14 the types **graduate to a shared `pb-protocol`
package when debye starts**, not before: `{protocol, dx, pqr, errors}` is
already a closed set depending only on numpy, and `tests/test_protocol_boundary.py`
holds it that way so the extraction stays mechanical. Extracting earlier would
mean versioning a package with one consumer through phase 4's churn, and
designing its API by guessing at debye's needs — the mistake §4.1 documents.

The decisive argument is not leanness but cycles: if the types stay in sashimi,
debye depends on sashimi while `sashimi[debye]` depends on debye. Note also that
`Solver` is a `typing.Protocol`, so *interface* conformance is structural and
free; the concrete dataclasses are what require sharing. Two things to settle at
extraction time: the PyPI name, and that protocol changes become semver events
with a migration window rather than internal refactors. Its acceptance
gate is `sashimi corpus verify --backend debye` within tolerances. The MCP
server grows a `backend` parameter defaulting to auto-selection, and
`SolveResult.backend` provenance means every downstream artifact already records
which solver produced it. Nothing in mcpymol or protean changes.

APBS — pinned, checksummed, via owned builds — is debye's fixed reference. The
trimmed reproducible builds exist partly for this.

## 11. Environments and infrastructure

| Environment | Arch | APBS source | Role |
|---|---|---|---|
| Mac local | osx-arm64 | conda-forge native | dev loop, protocol tests |
| OrbStack container | linux/amd64 (Rosetta) | conda-forge | subprocess integration tests |
| Proxmox Ubuntu VM | linux-64 native | conda-forge / owned build | timing-sensitive validation, benchmarks |
| GitHub Actions | linux-64, osx-arm64 | conda-forge via micromamba | full suite per push |
| (future) arm64 Linux | linux-aarch64 | owned build | once we ship it |

Rosetta handles APBS fine — it is a plain CPU-bound FD solver — but Rosetta
timings are not representative, which matters for timeout handling and any
timing reported in `SolveResult`. That is what the Proxmox VM is for.

**Remote access to the amd64 validation VM.** Tailscale on the Ubuntu VM (or an
LXC subnet router on Proxmox): SSO login with a hardware key as second factor at
the IdP, zero ports forwarded on the UDM Pro, SSH + VS Code Remote / Claude Code
rather than a desktop, so no RDP surface at all. Alternatives considered: UniFi
WireGuard (no true second factor), Guacamole + TOTP or behind Cloudflare Access
with a hardware-key policy (most capable, most setup). Tailscale chosen on
effort/benefit.

## 12. Phases

One sequence. Phases 0–3 are the delivered APBS wrapper; 4 onward is the
roadmap that outlives it.

**Phase 0 — Spike. ✅** uv/conda env with APBS 3.4.1 and pdb2pqr 3.7.1; the
`examples/born` case reproduces published energies on osx-arm64 to seven
significant figures. Confirmed the DX contract §5 assumes: 65³, uniform
0.1875 Å spacing, C order, 3 floats per line, kT/e. Retired the project's
stated main environmental risk — an `osx-arm64` APBS build exists.

**Phase 1 — Core library. ✅** `protocol.py`, `pqr.py`, `dx.py`, `errors.py`,
`apbs/`. Exit criterion met: the Born ion returns −230.03 / −228.87 / −228.56
kJ/mol at 0.41 / 0.20 / 0.16 Å spacing against a closed form of −228.61 —
0.62% → 0.11% → 0.02%, converging monotonically, which is what rules out a
systematic unit error.

**Phase 2 — MCP server. ✅** The four tools of §6, error mapping, and a
`sashimi-mcp` stdio entry point. Validated programmatically rather than by MCP
Inspector: the tests drive a real `fastmcp.Client`, so schema generation,
argument validation and error translation are all exercised. Exit criterion met
end to end — PDB → PQR → 97³ map at 0.31 Å, −209.660 kJ/mol → sampled at
coordinates → compared against a 1 M-salt solve (RMSD 0.093 kT/e, correlation
0.99989). *Registration alongside mcpymol is still outstanding.*

**Phase 3 — Golden corpus. ✅** Five-case manifest with `sashimi corpus
build|verify`, summaries checked in. Exit criterion demonstrated four ways: a
`Solver` wrapper scaling energy by 4.184 (kJ↔kcal), potential by 25.693
(kT/e↔mV), spacing by 10 (Å↔nm), or flipping the energy sign is caught, with the
moved field named. Tolerances tested from both sides.

**Phase 4 — Protocol hardening. ✅** All five items of §4.1, plus the equation
in the FD-family request payload (representable; `ApbsSolver` refuses to solve
it rather than returning untested numbers), a `SurfaceModel` enum with APBS
keywords moved to `sashimi.apbs.options`, and binary checksums in provenance.

Exit criterion met: `sashimi.bem_stub.StubBemSolver` implements
`Solver[BoundaryElementRequest]`, returns a `SurfacePotential` through the
shared `SolveResult`, declines a volumetric field, and refuses a Gaussian
dielectric as an `InputError` — with no APBS-shaped concession anywhere. The
strict `xfail` on `surface_method` started passing and its marker is gone.

Evidence the rewrite changed no physics: all five corpus energies are
bit-identical across it, to the last recorded digit. The only diffs in the
recorded summaries are `surface_method: "smol"` becoming
`surface_model: "smoothed-molecular"` and the new `resolved_parameters` block.

Content-addressed DX filenames and the corpus reference type landed in phase
5 rather than here, as additive follow-ups.

**Phase 5 — Ship it. ◐ in progress.**

Done: MIT licence; content-addressed DX filenames, so re-solving with different
parameters can no longer silently overwrite an earlier map and an identical
request reuses its file; the artifact contract (`sashimi.artifacts`) stating in
one place that maps are never deleted and paths are local; and the corpus
`Reference` protocol — `RecordedReference` answers "has this backend changed?",
`BackendReference` answers "do these two agree right now?", through the same
`verify_case`. Cross-solver validation now works end to end and only lacks its
CLI.

Also done: the **derived-query tools** §6 argues are the real agent-facing
value. `sashimi.analysis` is pure and binary-free; three MCP tools sit on it —
`sashimi_potential_extrema` (strongest patches, with coordinates),
`sashimi_potential_in_sphere` (statistics over a pocket) and
`sashimi_residue_potentials` (per-residue means, most negative first).

One finding worth carrying forward: the strongest values in any map are the
point-charge self-energy singularities at the atom centres — ±500 kT/e on a
dipeptide — so an unfiltered extrema search reliably answers "at the atoms",
which is true and useless. Passing the structure masks the solute interior and
the same query returns −1.82 kT/e at a solvent-side position instead. The tool
says in its response when the solute was not masked, because an agent cannot
otherwise tell the difference between a binding site and an atom.

Also done: the **capabilities and validate-inputs surface**.
`sashimi_capabilities` reports backends and their state, portable surface
models, units, grid defaults and what is deliberately unsupported — and reports
a *missing* backend rather than raising, since the one tool that can explain an
absent APBS must not be the one tool that cannot run.
`sashimi_validate_inputs` dry-runs a request in single-digit milliseconds: grid
sizing and surface mapping are pure arithmetic, so the dime, point count,
estimated map size and any blocking problem are all knowable before a solve that
would take a minute and write 56 MB. A relaxed resolution warns rather than
blocks, because the solve would still run and would say so. A test asserts the
prediction matches what `sashimi_solve` actually does — a forecast that can
disagree with the event is worse than none.

Remaining: the PyPI release itself. The distribution name is settled —
**`sashimi-electro`**, since plain `sashimi` belongs to an unrelated dormant
library — so the install name and the import name differ and the README says
so. Still outstanding: `authors`, `classifiers` and `urls` in the manifest,
registration alongside mcpymol, and the Proxmox VM plus Tailscale for benchmark
and timeout work.

**Phase 6 — Distribution.** Owned APBS build matrix (trimmed, mostly static,
feedstock-derived); `linux-aarch64` build offered upstream; license-file audit
of the vendored FETK/MALOC versions; platform wheels so `pip install sashimi`
works end to end.

**Phase 7 — Multi-backend.** DelPhi backend and the `sashimi validate`
cross-validation harness; TABI-PB, which forces the surface-potential path to be
real; PyGBe in-process, which proves transport-agnosticism; optional GB tier for
triage→refine workflows.

**Phase 8 — debye.** The validation ladder of §10; drop-in behind the backend
interface; portability suite green on all architectures; BEM engine later.

**Phase 9 — Integration, ongoing.** mcpymol grows a convenience chaining
`sashimi_solve` → load DX → surface coloring; protean consumes `SolveResult`.
Sashimi itself should go quiet after this — a wrapper that needs constant
attention has failed at its one job.

### Numbering history

The two superseded documents numbered phases differently. For reading old
commits and comments:

| This document | old PLAN.md | old ROADMAP.md |
|---|---|---|
| 0–3 (delivered) | 0–3 | part of Phase 1 |
| 4 Protocol hardening | — | Phase 0 |
| 5 Ship it | — | Phase 1 (remainder) |
| 6 Distribution | — | Phase 2 |
| 7 Multi-backend | — | Phase 3 |
| 8 debye | Phase 10 (handoff) | Phase 4 |
| 9 Integration | Phase 4 | — |

## 13. Risks and mitigations

The conda-forge APBS package going unbuildable on future platforms is the
long-tail risk; the recipe is small and forkable, and the corpus makes a
re-validated fork cheap to trust. The acute version — no `osx-arm64` build — was
retired in phase 0.

APBS's silent-failure modes (exit 0 on error) are handled by structural output
verification rather than exit codes. pdb2pqr's opinionated rebuilding is
surfaced, not hidden. Grid memory blowups are capped by `max_points` with an
error stating the achievable resolution. The abandoned upstream `apbs` PyPI
package sharing the import name is avoided by never depending on it.

New since the split: APBS is no longer version-pinned by a lockfile, because
uv cannot install a compiled binary. The corpus carries that weight — it asserts
the version and re-checks the numbers, so drift fails loudly in kJ/mol.

## 14. Decisions and remaining questions

The five questions this section used to pose were resolved on 2026-08-10. Each
is recorded where it takes effect — §4.1, §5, §6, §7, §8, §10, §12 — and
summarised here.

| # | Question | Decision |
|---|---|---|
| 1 | Nonlinear PB in the protocol? | **Representable, not implemented.** The equation is a field of the FD-family request payload, so BEM backends cannot receive one. No `npbe` code until a case needs it |
| 2 | Normalize or expose surface definition? | **Curated solver-neutral enum**, native values behind per-backend options, resolved values in provenance, and `validate` refuses to compare across mismatched surface models |
| 3 | Volumetric payloads over MCP? | **Never inline.** Content-addressed paths now with a cleanup contract; MCP Resources when a non-local transport exists; derived queries pulled forward to phase 5 |
| 4 | Does `validate` belong in core? | **Core, unified with the corpus** — one engine, two reference kinds. Backends as optional extras |
| 5 | Does the protocol graduate to `pb-protocol`? | **At debye, not before.** The seam is a closed set today and is guarded by a test so extraction stays mechanical |

### Still open

- **Is `max_points = 161³` the right guardrail?** It is sized for memory, but it
  also permits a 56 MB artifact per solve. Now that maps are content-addressed
  and accumulate, disk and hand-around-ability may be the binding constraint.
- **Does `validate` become an MCP tool?** An agent asking "is this energy
  trustworthy?" is a good capability, but it needs N backends installed, so it is
  CLI-first regardless. Revisit in phase 7.
- **NPBE-versus-LPBE energy comparability.** The total-energy integral differs
  between the two equations, so the corpus and `validate` must compare like with
  like once nonlinear is implemented. Needs a concrete rule before phase 7.
- **`pb-protocol` naming and semver.** The PyPI name should be checked early
  since it appears in every downstream dependency list, and graduation turns
  protocol changes into major-version events with migration windows.
- **Surface-model mapping table.** Which solver-neutral enum members exist, and
  how each backend maps them, is phase 4 design work — DelPhi's Gaussian
  dielectric has no APBS equivalent, so the enum cannot simply be APBS's set
  renamed.
