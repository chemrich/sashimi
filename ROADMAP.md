# sashimi — Roadmap and Implementation Plan

> Thin, protocol-first wrapper around Poisson–Boltzmann electrostatics solvers,
> exposed as an MCP server. APBS is the first backend; the protocol is designed
> to outlive it. `debye` is the eventual clean-room solver that slots in behind
> the same interface.

Status: phases 0–4 shipped, 5 bar the PyPI release; 6 (distribution) not
started; 7 (multi-backend) in progress — DelPhi, `sashimi validate` and the
TABI-PB boundary-element backend have landed.
Last updated: 2026-08-11

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
does not expose *APBS's* FEM, geoflow, BEM, PBAM or PBSAM solvers — only its
`mg-auto` finite-difference path, which covers visualization-grade and standard
solvation-energy work. (Boundary-element solving itself is no longer a non-goal:
phase 7 added TABI-PB as its own backend, which is a different thing from
exposing APBS's bundled copy.) It does not attempt nonlinear PBE in v1 (the API leaves
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

### Two reference kinds, and the one the corpus was missing

A recorded summary answers **"has this backend changed?"** to 1e-4. It cannot
answer **"is it right?"** — a backend wrong from the first build reproduces its
own wrong number forever and every check passes. Four of the original five cases
had no independent check at all, and the fifth's closed form lived in a test
rather than in the corpus.

`AnalyticReference` closes that. Where the geometry has a closed form, the case
carries it, and `verify_case` checks the fresh solve against the *physics* as
well as against the recording — loose, because the gap is discretization, with a
per-case `rtol` measured from what APBS 3.4.1 actually does rather than chosen.
`tests/test_corpus_manifest.py` asserts the same thing over the checked-in
summaries without a solver, so it runs everywhere.

The sweep matters more than any single case. Radius (1, 2, 3, 4, 6 Å), charge
(−1, +1, +2) and solute dielectric (1, 2, 4) turn one agreeing number into a
functional form that has to agree: a missing factor of two passes one case and
fails eight. Two of them are pure invariants — solvation goes as q², so −1e must
reproduce +1e exactly and +2e must be exactly 4×, which no other case in the
corpus can catch because every other case is positively charged.

The 1 Å ion at 0.5 Å spacing earns its place by being *bad*: two grid points
across the ion and 5.1% from exact, falling to 3.2% at 0.25 Å. A corpus that
only contains cases the solver handles well does not record where it stops
working.

**Where a closed form is declined.** The salted Born ion has no analytic
reference, deliberately. The Debye-Hückel screening term depends on an
ion-exclusion convention the backends do not share: APBS's ionic contribution is
−0.688 kJ/mol and DelPhi's is −0.496, both reporting `polar-solvation`, and
DelPhi's is resolution-independent where APBS's carries grid noise. That is a
different convention beneath the same declared quantity — not the `EnergyTerm`
gap of §12 — and pinning either as "the" closed form would encode one code's
choice as physics.

### Tiers, because the corpus is meant to grow

`CaseTier` splits the manifest by wall time, cumulatively: `fast` (seconds,
every push), `standard` (~2 minutes, every push), `full` (tens of minutes,
nightly or on demand). The split is cost, not importance.

The target is **50 cases**. The axis that matters is what each is checked
against, not the count — 50 self-recorded cases would be 50 change-detectors and
a 50-line diff every time the physics legitimately moves. The intended mix:

- **Analytic** — exact, no solver needed to state the answer. 11 cases today.
- **Third-party anchored** — APBS ships its own `examples/` with per-version
  reference values *and* independent UHBD numbers (methanol: −36.2486 APBS
  against −35.595 UHBD). BSD-3-Clause, so vendorable with attribution. ~1%
  anchors rather than 1e-4 references, because those inputs specify explicit
  `cglen/fglen/dime` that `GridSpec` does not express.
- **Cross-solver** — `BackendReference`, already built, over more structures.
- **Self-recorded** — the original kind. Kept small and fast.

73 example structures span 1 to 16,090 atoms. Cost is bounded by the existing
`max_points` guardrail rather than by atom count: a 16,090-atom solute relaxes
to the same 161³ grid as a 2,000-atom one, so the largest case is 14.5 s against
lysozyme's 6.8 s, not hours.

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
| 2 | **DelPhi** ✅ | FD sibling; Gaussian dielectric, focusing workflows; cheap triangulation partner | subprocess, two flavours |
| 3 | **TABI-PB** ✅ | BEM; forces the protocol to handle surface potentials | subprocess + NanoShaper |
| 3b | ~~**PyGBe**~~ ❌ | Dropped: builds only on Python ≤3.11, so it cannot be imported into a 3.13 process at all. §12 records the measurement | — |
| 4 | **GB tier** ✅ | Fast approximation for high-throughput triage → PB refinement — **and**, being pure numpy, the in-process proof PyGBe was meant to supply | none: in process |
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
that reason (§14).

**Not every backend is a reference.** The GB tier approximates the equation the
others discretize, so a spread that averaged it in would report the method
working as designed as a defect, and would need a tolerance wide enough to hide
a real regression in the solvers it was averaged with. `AccuracyTier` in
provenance partitions them: the spread describes the reference tier, each
approximation is reported as its deviation from what that tier agreed on. §12
records what this cost and what it caught.

Additional backends were to ship as optional extras (`sashimi[delphi]`) so core
stays lean. **Phase 7 retired that plan**: neither DelPhi flavour is on PyPI —
the C++ build is a tarball needing a compiler, and pyDelPhi is a git checkout —
so the extra cannot exist. Discovery via `$SASHIMI_DELPHI_PATH` and documented
install steps is the substitute, and it costs nothing, since a solver binary
was never going to arrive through a Python installer anyway. Core stays lean by
construction rather than by packaging.

**Shipped in phase 7** as `sashimi.validate` and `sashimi validate`. See §12 for
what it does and what the energy-term rule turned out to require.

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

| Environment | Arch | APBS source | Role | Status |
|---|---|---|---|---|
| Mac local | osx-arm64 | conda-forge native | dev loop, protocol tests | in use |
| GitHub Actions | linux-64, osx-arm64 | conda-forge via micromamba | **full suite per push, both platforms** | in use |
| OrbStack container | linux/amd64 (Rosetta) | conda-forge | local linux reproduction when CI is too slow a loop | optional |
| Proxmox Ubuntu VM | linux-64 native | conda-forge / owned build | stable timings for benchmarking | **deferred to phase 8** |
| (future) arm64 Linux | linux-aarch64 | owned build | the platform gap of §9 | phase 6 |

### Why the benchmark VM is deferred

This section originally justified a native amd64 VM on the grounds that Rosetta
timings are unrepresentative, "which matters for timeout handling and any timing
reported in `SolveResult`". That was written when the linux-64 test environment
was a Rosetta container and CI was an afterthought. It no longer holds:

- **CI is native amd64.** `ubuntu-latest` runs the full suite, APBS-marked tests
  included, on every push. "Validated on real amd64 hardware" is already true and
  automatic.
- **Nothing consumes a timing.** `Provenance.wall_seconds` is recorded and read
  by no test and no branch. It is informational metadata; an accurate number
  would make it prettier, not the code more correct.
- **The timeout needs no calibration.** Measured: the `max_points` cap of 161³
  solves in 9.9 s, and 225³ — a grid the guardrail refuses to produce — in
  33.9 s. Against a 300 s default that is 30× and 9× headroom. Rosetta being
  2–3× slower does not threaten a 30× margin, so there is no question here that
  measurement on any machine fails to answer.

What the VM would genuinely buy is *stable* timings, which CI cannot give at any
architecture because its runners are shared and noisy. Nothing needs that until
**debye makes a performance claim against APBS (phase 8)** — that is the trigger
to revisit, and the honest one: the first time a decision depends on a timing
number.

Worth being clear about what it would *not* buy: the `linux-aarch64` gap is the
real platform debt (§9), and an x86 VM does nothing for it. That needs arm64
Linux — GitHub's arm64 runners, or different hardware.

**Remote access, when the VM exists.** Tailscale on the Ubuntu VM (or an LXC
subnet router on Proxmox): SSO login with a hardware key as second factor at the
IdP, zero ports forwarded on the UDM Pro, SSH + VS Code Remote / Claude Code
rather than a desktop, so no RDP surface at all. Alternatives considered: UniFi
WireGuard (no true second factor), Guacamole + TOTP or behind Cloudflare Access
with a hardware-key policy (most capable, most setup). Tailscale chosen on
effort/benefit. Deferred with the VM — it exists only to reach it.

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
`sashimi-mcp` stdio entry point. The exit criterion was finally satisfied
*visually* in phase 5, not merely programmatically: hen lysozyme (1AKI) through
prepare → solve → PyMOL, which loaded the DX (`2,679,201 data points`) and
rendered a predominantly positive surface consistent with its net charge of
+8 e. Until then nothing had confirmed PyMOL could read our writer's output. Validated programmatically rather than by MCP
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

Also done: packaging metadata (`authors`, `classifiers`, `urls`, `keywords`) and
registration as an MCP server. And the first real end-to-end run, which earned
its place immediately — see below.

Remaining: the PyPI release itself. The distribution name is settled —
**`sashimi-electro`**, since plain `sashimi` belongs to an unrelated dormant
library — so the install name and the import name differ and the README says
so. Still outstanding: `authors`, `classifiers` and `urls` in the manifest, and
nothing else: the manifest is complete and the server is registered.

**What the first real protein found.** Everything worked, and the numbers were
meaningful — pdb2pqr gave hen lysozyme its textbook net charge of +8 e with zero
warnings, and the most negative residues included Asp52, one of its two
catalytic residues. But `sashimi_potential_extrema` took **64 seconds**, three
times longer than the 19-second solve it was analysing. `_solute_mask` evaluated
every atom against the whole grid: 1,960 atoms against 2.7M points. Restricting
each atom to its own bounding box is **50x faster** (1.28 s) for a bit-identical
result, checked against the naive implementation as an oracle over random
geometry, with a scale guard so the complexity cannot regress.

No amount of dipeptide testing would have surfaced that. It is the argument for
doing integration earlier rather than last.

The Proxmox VM and Tailscale have moved to phase 8. §11 records why: CI is
already native amd64 and runs the full suite, nothing consumes
`Provenance.wall_seconds`, and the 300 s timeout has 30× headroom over the
largest grid the guardrail permits. The VM's remaining value is stable timings,
which nothing needs until debye makes a performance claim.

**Phase 6 — Distribution.** Owned APBS build matrix (trimmed, mostly static,
feedstock-derived); `linux-aarch64` build offered upstream; license-file audit
of the vendored FETK/MALOC versions; platform wheels so `pip install sashimi`
works end to end.

**Phase 7 — Multi-backend. ✅** Four backends, three solver families, one
unchanged protocol.

Done: the **DelPhi backend** (`sashimi.delphi`), covering both flavours — the
C++ reference build and pyDelPhi, the same lab's Python/numba reimplementation.
One input generator drives both; `DelphiFlavour` names the places they differ.

The headline result is that **the protocol needed no change at all**. A cubic
grid instead of a multigrid lattice, a Gaussian-cube map in Bohr, energies in
kT, and a temperature parameter in Celsius all absorbed below `Solver`, and the
only edit above `sashimi.delphi` was registering it. §2's claim for the protocol
layer has now been tested by something other than the backend it was designed
around.

**What the second backend found.** Every one of these is a silent wrong answer
rather than an error, and none was reachable from the APBS side:

- **DelPhi's `pqr` reader is not pdb2pqr's PQR.** Its plain reader takes the
  radius from columns 62-68, its `pqr4` reader from 63-69, and pdb2pqr — like
  sashimi's own writer — emits the latter. Read as `pqr`, an ALA-GLY radius of
  1.824 Å parses as **4.0 Å** and the solve proceeds on wrong-sized atoms. The
  backend pins `in(modpdb4, format=pqr)`; a binary-free test encodes the column
  arithmetic, including a C `atof` model, because Python's `float()` raises
  where `atof` silently truncates.
- **`temper` is in degrees Celsius** in the C++ build — its parser ends
  `fTemper -= dAbsoluteZero` — and in kelvin in pyDelPhi. Writing 298.15 to the
  C++ program runs the solve at 571.3 K and reports −48.13 kT where the correct
  answer is −92.22. Neither program complains, because 571 K is legal.
- **DelPhi 8.6 defaults to `linit=0`, `maxc=0.0` and never terminates.** A Born
  ion passed 1.39 million iterations with residuals at 3e-16 — machine epsilon —
  because the convergence test compares against a threshold of zero. sashimi
  always writes both values.
- **DelPhi's "solvation energy" is not APBS's.** Both DelPhi flavours report the
  polarization term alone, which does not move with salt at all (−92.22 kT from
  the C++ build at both 0 M and 0.5 M, while its own aggregate including the
  ionic term moves to −92.56). APBS's difference-of-blocks carries the mobile-ion
  contribution by construction. The gap is definitional, not numerical, and
  diagnostics name the term. *This bullet first claimed three definitions rather
  than two, on a measurement comparing pyDelPhi's Gaussian path against the C++
  build's molecular one; on the same surface model the flavours agree exactly.*

**The surface-model mapping table** (§14's last open design question) is
resolved:

| `SurfaceModel` | APBS | DelPhi C++ | pyDelPhi |
|---|---|---|---|
| `MOLECULAR` | `srfm mol` | `prbrad > 0` | `surfmethod vdw`, `prbrad > 0` |
| `SMOOTHED_MOLECULAR` | `srfm smol` | no | no |
| `VAN_DER_WAALS` | `mol`, `srad 0` | `prbrad 0` | no (upstream crash) |
| `GAUSSIAN` | no | `gaussian 1` | `surfmethod gaussian` |

The pyDelPhi row cost a wrong answer before it was right, and the mistake is
worth recording. `surfmethod=vdw` was read as naming the *surface*; it names the
**construction** — roll a probe over van der Waals spheres, the algorithm its
`vdwms` module is named for. With a probe it is the molecular surface, and it
reproduces the C++ build's `molecular` result exactly: −84.33 kT on ALA-GLY and
−92.22 kT on the Born ion, identical on matched grids. Its probe-dependence
(−84.33 at 1.4 Å against −86.88 at 0.5 Å), first read as proof it was *not* the
molecular surface, is that surface behaving correctly.

`VAN_DER_WAALS` is the one model pyDelPhi genuinely cannot deliver: `prbrad=0`,
the natural limit of its own method, aborts it with a numba `TypingError` in
0.2.0 and 0.3.0 alike. That is an upstream bug rather than a modelling
difference, so the model is declined rather than mapped onto something adjacent.
`SMOOTHED_MOLECULAR` — sashimi's *default* — is APBS-only, so a DelPhi solve at
defaults raises `UnsupportedRequest` rather than substituting `MOLECULAR` and
moving the answer by 2,000× the corpus tolerance.

`GAUSSIAN` has no APBS counterpart and no closed form, and **no equivalent
request has been established across the two DelPhi flavours**: matching `sigma`
and `srfcut` still leaves them at −152.43 and −38.90 kT on the same grid, and
pyDelPhi's answer does not move with either, so those are not the corresponding
knobs. That is an unfinished comparison, not evidence the models differ, and
`capabilities` marks the model unvalidated rather than asserting either.

`describe_capabilities` reports the models the installed backends genuinely
share — `molecular` for every pair shipped here — and returns an empty list when
fewer than two backends are installed, since one backend trivially shares
everything with itself.

**Cross-validation, where it is legitimate.** APBS against DelPhi C++ on the
models they share: 2.30% on the Born ion and 2.31% on ALA-GLY (`molecular`),
and **2.44% on hen lysozyme**, 1,960 atoms, at 0.5 Å. On the Born ion, where a
closed form exists, DelPhi lands on **−228.609 kJ/mol against −228.611** while
APBS is 2.36% off at the same nominal resolution.

**Cross-validation in CI.** `tests/test_cross_validation.py` runs the spread
check on every push, gated on `comparable_surface_models()` being non-empty, so
it exercises itself only where a comparison is legitimate. CI builds the C++
DelPhi from the Clemson tarball on the Linux leg — measured at 40 s under
emulation on Ubuntu 24.04 with g++ 13.3, ~13 s natively, needing only boost
headers — and runs pyDelPhi on macOS, so both flavours are covered per push and
**both legs run the comparison**, since both share `molecular` with APBS. The
Linux-built binary
reproduces the Born ion to the last printed digit of the macOS arm64 build
(−92.22 kT), which is the evidence that a build from that tarball is a
reproducible artifact rather than a local accident.

One thing learned here changed the plan: **backends cannot ship as extras** the
way §8 assumed. Neither DelPhi flavour is on PyPI, so `sashimi[delphi]` cannot
exist; discovery via `$SASHIMI_DELPHI_PATH` plus documentation is the honest
substitute.

### `sashimi validate` ✅

`sashimi.validate` plus a `sashimi validate` subcommand: run one system through
N backends, report the spread, and refuse when the number would mislead.

**Most of the module is refusals**, which is the point. A spread is a solver
disagreement only if surface model, equation and *reported energy term* were all
held fixed, and none of those differences is visible in the number itself.

The energy term is the one this phase discovered and the one §14 asked for.
`EnergyTerm` is now a protocol-level enum carried in `Provenance`, because
"what quantity is this" is a question every backend must answer and no
comparison can be trusted without it.

**The better fix was to remove the mismatch rather than tolerate it.** DelPhi's
headline "corrected reaction field energy" is the polarization term alone, but
the C++ build can be asked for APBS's quantity, and now is. The request has to
say `energy(s,c,ion)`, which is not guessable: `s` gives the reaction field,
adding `ion` changes nothing because the ion-atmosphere terms
(`fEnergy_SolvToChgIn/Out`) are computed *inside the Coulombic routine*, so `c`
must be requested too and subtracted back off the aggregate. DelPhi has a
dedicated line that computes exactly this, and it is commented out in 8.5.0.

That the reconstructed term is the right one is evidenced three ways rather than
assumed: it is zero at zero salt, it grows monotonically with salt (−0.20 kT at
0.15 M, −0.34 at 0.5 M on a Born ion), and adding it makes the gap to APBS
**salt-independent** — 2.30 / 2.59 / 2.70% across 0 / 0.15 / 0.5 M becomes
2.30 / 2.38 / 2.34%. A missing term produces that signature; a coincidence does
not.

So APBS and the C++ DelPhi now both report `polar-solvation`, and **all five
corpus cases compare** where three did before. Hen lysozyme at physiological
salt — previously refused outright — agrees to **1.97%**.

The comparability rule survives for the case it cannot fix: pyDelPhi's results
CSV has no ion-atmosphere column, so it stays on `reaction-field`. The rule is
**same term, or terms that provably coincide under this request** — the two
differ by exactly the mobile-ion contribution, so at zero ionic strength they
are the same quantity and the comparison proceeds with a note, and at nonzero
salt it refuses. That refinement was itself a correction: the first version was
"same term" and refused all five corpus cases, which is correct and useless.

**The `verify_case` blocker is resolved** rather than worked around. That
function compares grid shape first and bails, which is right for "has this
backend changed" and impossible for two backends whose legal grids differ by
construction (APBS's 32c+1 dime against DelPhi's any-odd cubic gsize).
`validate` never touches grid indices: energies are grid-independent scalars,
and potentials are compared by interpolating both maps at the same *physical*
coordinates inside the box they share. `corpus` is untouched.

Measured on the molecular surface: **1.70% on hen lysozyme** at zero salt and
**1.97% at 0.15 M** (1,960 atoms, APBS 129×161×129 against DelPhi 133³,
potential RMSD 4.29 kT/e over 200 shared points); 2.30% on the Born ion at
0.5 Å and **0.62% at 0.25 Å** — the two codes converging on each other as the
grid refines, which is the behaviour that makes the agreement meaningful rather
than coincidental.

### TABI-PB ✅ — the protocol's acid test, passed

`sashimi.tabipb` implements `Solver[BoundaryElementRequest]` against TABI-PB 3.0
(treecode-accelerated boundary integral, BSD-3-Clause, University of Michigan),
which triangulates with NanoShaper and integrates over the surface.

**The protocol needed no change.** §2 calls the FD/BEM split "the single most
important constraint": a protocol assuming a volumetric grid forecloses half the
landscape. Phase 4 built `BoundaryElementRequest` and `SurfacePotential` for a
solver that did not exist and guarded them with `bem_stub`. This is the same
types carrying a real solver's real answer — 1,034 vertices and 2,064 triangles
on ALA-GLY, through the same `SolveResult` APBS uses.

**Three solvers, two families, one number.** ALA-GLY, molecular surface, zero
salt: APBS −213.70, DelPhi −209.25, TABI-PB −211.39 kJ/mol — a **2.08% spread**,
with the boundary-element answer falling between the two finite-difference ones.
TABI-PB's energy converges monotonically with mesh density (−221.57 → −216.84 →
−212.96 → −211.39 kJ/mol at `sdens` 1.5 → 4.0), the BEM analogue of grid
refinement, and it reports kJ/mol directly — the only backend needing no unit
conversion.

**What a BEM solver costs, stated rather than hidden.** It cannot answer
`sashimi_potential_at`: there is no volume to interpolate. It cannot solve the
Born ion at all, because NanoShaper will not triangulate fewer than four atoms —
so the one case in this project with a closed-form answer is the one case this
backend cannot be calibrated against. And `SMOOTHED_MOLECULAR` and `GAUSSIAN`
are refused as *grid concepts* rather than missing features: they describe how a
dielectric varies across a volume, which a surface solver does not have.

Two traps, both silent: `tabipb` invokes `NanoShaper` **by bare name through a
shell**, so an installed-but-not-on-PATH mesher dies with `command not found`
inside an uncaught C++ exception — `run.py` therefore builds the subprocess PATH
itself. And a `mesh_density` below 1.5 aborts the same way, which matters
because `BoundaryElementRequest.mesh_density` **defaults to 1.0** — so the most
obvious first call fails unintelligibly. The backend names the cause; whether
the protocol default should move is recorded in §14.

### Cross-family validation ✅

`validate_system` completes the picture: `System` holds one physical question
and produces whichever request family a backend speaks, so a grid solver and a
surface solver can be asked the same thing without either reading the other's
dialect. That is the seam `SolveRequest` was split for in phase 4, used for the
first time, and it was the small addition predicted rather than a redesign.

Measured across **two solver families**: ALA-GLY at physiological salt gives
APBS −214.20, DelPhi −209.42, TABI-PB −217.36 kJ/mol, a **3.65% spread**; the
low-dielectric case 3.58%. Energies only — a volume and a triangulated surface
have no shared representation, and inventing one would be the category error the
comparability checks exist to refuse.

Crossing families widens who can participate without relaxing anything: the
surface-model, equation and energy-term checks all still apply, and `equation`
is read as linear for a boundary-element run because a nonlinear one is
unrepresentable there rather than rejected.

The corpus stays finite-difference by construction — every case records grid
geometry — so `sashimi corpus --backend tabipb` refuses and points at `validate`.

### Accuracy tiers — what `validate` needed before a GB backend could exist

`AccuracyTier` in `Provenance`, the same shape of addition as `EnergyTerm` and
for the level above it. `EnergyTerm` answers "what quantity is this";
`AccuracyTier` answers "how was it obtained": a discretization of the equation,
or an approximation to it. The distinction is invisible in the number and
decides what a disagreement means.

The forcing case is that `validate` has one tolerance and one verdict. A
Generalized Born answer sits 10–30% from a PB one *by construction*, so on the
existing machinery every GB comparison would report DISAGREE — true, and
useless. The fix is a partition rather than a wider band: the headline spread
stays a statement about the reference tier at 10%, and each approximation is
reported separately as its distance from what that tier agreed on. Widening one
tolerance to fit both would have cost the reference number too, since a real
APBS/DelPhi regression fits comfortably inside 30%.

It defaults where `EnergyTerm` is optional-and-refused, which is deliberate: an
unstated energy term is a silently wrong comparison, while an unstated tier has
one right answer for every backend that predates the field, and a mis-defaulted
approximation merely gets the old behaviour — compared at 10% and loudly called
a disagreement. Refuse the silent failure; default the loud one.

**A latent bug surfaced while planning for it.**
`comparable_surface_models()` promised "two or more installed backends" and
computed the intersection across *all* of them. Harmless while three backends
shared `molecular`, and wrong in the direction that hurts: one backend
supporting a disjoint set empties the result for everybody, and an empty list
here stops `sashimi validate` and skips the whole cross-validation tier. Adding
a backend must not be able to switch off the comparisons between the others.

*The example that motivated the fix was itself wrong, and is worth recording as
such: the GB tier was expected to support only `van-der-waals`, and measurement
later showed it belongs on `molecular` — see below. The bug was real
independently of it.* Counting backends per model, as the docstring always
said, revealed a comparison that was legitimate all along and had never run:
APBS against DelPhi on `van-der-waals`, which TABI-PB cannot mesh and which the
intersection therefore hid. Now exercised on every push, at 2.30% on the Born
ion, 3.93% on ALA-GLY and 3.66% on the low-dielectric peptide.

### Generalized Born ✅ — the approximation, and the first in-process backend

`sashimi.gb` implements `Solver[SolveRequest]`: HCT pairwise descreening, OBC
effective radii, Still's equation, Debye-Hückel screening. Roughly 300 lines of
numpy and **no binary at all** — no discovery, no environment variable, no
install step, no CI build. It closes both of this phase's remaining items at
once, and neither the way §8 planned.

**A third solver family, and the protocol needed one enum member.** Generalized
Born needs neither a grid nor a mesh, so it takes the base `SolveRequest` and
`System.request_for` grew an `ANALYTIC` arm. Because `Solver` is contravariant
in its request type, `Solver[SolveRequest]` is already assignable everywhere an
FD solver is expected — the registry needed no signature change. Two backends
in two phases have now been added without the protocol moving: §2's claim is
holding up under a third kind of solver, not just a second.

**What being in-process actually costs**, now measured rather than predicted:
`Provenance.binary_path` and `binary_sha256` are `None` for the first time on a
real backend, and nothing downstream cared — `validate` already tolerated it,
having only ever been exercised that way by stubs. `timeout` is the one that
does not survive: a function call cannot be interrupted the way
`subprocess.run(timeout=)` can. The substitute is that the cost is knowable
before it is paid — O(N²), no iteration, no convergence criterion, so it cannot
fail to terminate — and chunking bounds the memory that size costs. **And this
tier cannot silently skip**, which is the failure this project has shipped
twice: there is no binary to be missing.

**Two mistakes, both caught by running it against real structures, and neither
visible from the physics tests.** The suite was green for both.

- **The surface model is `molecular`, not `van-der-waals`.** Descreening
  integrates over van der Waals spheres, so the intuitive declaration is
  `van-der-waals` — and that is the construction, not the surface. The OBC
  rescaling exists precisely to carry the union of spheres onto the
  solvent-excluded volume, and the parameters were fit to reproduce
  Poisson-Boltzmann on the molecular surface. On hen lysozyme: **4.72%** against
  APBS on `molecular`, **31.35%** on `van-der-waals`. This is the pyDelPhi
  `surfmethod=vdw` lesson a second time, in a place nothing connected it to the
  first: *the name of the construction is not the name of the surface.*
- **pdb2pqr's radii are not Generalized Born radii.** They are Lennard-Jones
  parameters, and AMBER gives hydroxyl and sulfhydryl hydrogens a radius of
  exactly **0** — their volume is subsumed into the heavy atom. A grid solver
  spreads that charge over grid points and never notices; a method that divides
  by the radius gets an infinite self-energy. Hen lysozyme has twenty such
  atoms carrying **+8.34 e** between them. This is why Amber ships the mbondi
  sets, and the difference is a third of the answer: **4.72%** with mbondi
  against **35.43%** with pdb2pqr's. mbondi is therefore the default, the
  substitution is counted into diagnostics, and `GbRadii.AS_GIVEN` turns it off.

**Measured against the reference tier**, molecular surface, after both fixes:

| case | APBS | DelPhi | GB | reference spread | GB deviation |
|---|---|---|---|---|---|
| born-ion-coarse | −234.00 | −228.61 | −235.68 | 2.30% | 1.89% |
| born-ion-fine | −230.03 | −228.61 | −235.68 | 0.62% | 2.77% |
| born-ion-salt | −234.69 | −229.11 | −236.62 | 2.38% | 2.04% |
| peptide-default | −214.20 | −209.42 | −226.84 | 2.23% | 7.10% |
| peptide-low-dielectric | −434.42 | −425.09 | −458.77 | 2.15% | 6.75% |
| **hen lysozyme** | −3976.6 | −3898.3 | −3879.1 | 1.97% | **1.48%** |

1,960 atoms in **0.196 s** against APBS's 6.79 s — 35× — which is the triage
claim §8 made for this tier, met. `DEFAULT_APPROXIMATION_TOLERANCE` is set from
these numbers at **15%**: roughly twice the worst of them, and comfortably tight
enough to have caught both mistakes above, which is the more useful direction.

**The one calibration no other backend here gets.** For a single sphere the
interpolating denominator collapses to the effective radius, so Generalized Born
*is* the Born formula rather than an approximation to it — matched to 4e-16.
TABI-PB cannot be pinned this way at all, since NanoShaper will not triangulate
fewer than four atoms. The descreening integral is checked separately against
direct numerical quadrature, because it was derived here rather than quoted and
a misremembered sign in it produces entirely plausible numbers.

### PyGBe — dropped, with the measurement

§8 row 3b wanted PyGBe to prove the protocol was transport-agnostic by being
imported rather than executed. It cannot, and the reason is not a matter of
effort:

- `pip install` fails during metadata generation on **Python 3.12+** — its
  vendored `versioneer.py` calls `configparser.SafeConfigParser()`, removed in
  3.12. sashimi is `requires-python = ">=3.13"`, so PyGBe can never be imported
  into sashimi's own interpreter.
- On 3.11 the build then fails on osx-arm64: its Cython extensions hardcode
  `-msse3` and `-fopenmp`.

A package that builds only on ≤3.11 cannot be imported by a 3.13 process, so the
in-process claim is unavailable at any price, and subprocessing it would prove
nothing `sashimi.tabipb` has not already proven. `sashimi.gb` supplies the
in-process proof instead, and supplies it better: no transport at all rather
than a different one. The same conclusion as "backends cannot ship as extras",
reached the same way — by trying it.

**Phase 7 is complete.** Four backends across three solver families — finite
difference (APBS, DelPhi ×2 flavours), boundary element (TABI-PB), analytic
(GB) — all exercised in CI, and the protocol absorbed every one of them with
two enum members and no new types.

**Phase 8 — debye.** The validation ladder of §10; drop-in behind the backend
interface; portability suite green on all architectures; BEM engine later.

This is where benchmark infrastructure finally earns its place. A performance
claim against APBS is the first decision in the project that depends on a timing
number, and CI cannot supply one — its runners are shared and noisy at every
architecture. The Proxmox VM and Tailscale (§11) belong here, not in phase 5.

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

- **Is `max_points = 161³` the right guardrail?** It is sized for memory, but
  memory turns out not to be the constraint: APBS peaks at 122 MB high-water on
  a 161³ grid. The real costs are wall time and disk — 161³ is 9.9 s and 56 MB,
  225³ is 33.9 s and 154 MB. The current cap happens to sit in a defensible
  place for both, so this is now a question of whether to *document* that
  reasoning rather than whether to change the number.
- **Does `validate` become an MCP tool?** Still open, and now concrete: the CLI
  and `sashimi.validate` both exist, so exposing it is a thin wrapper. The
  argument against remains that it needs two backends installed and most
  installations will have one — but `sashimi_capabilities` already reports
  exactly that, so the tool could answer honestly rather than failing. The real
  question is what an agent does with a spread it cannot act on.
- ~~**NPBE-versus-LPBE energy comparability.**~~ **Resolved in phase 7**, and
  more generally than posed. The problem was never specific to the equation:
  backends disagree about what "solvation energy" means at *fixed* equation, so
  the rule is "same reported term", not "same equation". `EnergyTerm` carries it
  in provenance and `sashimi.validate` checks it — with the refinement that
  terms differing only by the mobile-ion contribution are comparable where that
  contribution is zero. Nonlinear, when it arrives, adds an `EnergyTerm` member
  rather than a new kind of check.
- **`pb-protocol` naming and semver.** The PyPI name should be checked early
  since it appears in every downstream dependency list, and graduation turns
  protocol changes into major-version events with migration windows.
- ~~**Surface-model mapping table.**~~ **Resolved in phase 7** against two real
  DelPhi implementations; the table and its consequences are recorded there and
  in `sashimi.delphi.options`. The guess that motivated it was right: the enum
  is not APBS's set renamed. Every backend shipped here shares `MOLECULAR`;
  `SMOOTHED_MOLECULAR` is APBS-only and `GAUSSIAN` DelPhi-only, and the latter
  is not yet comparable even between the two DelPhi flavours.
- **Does `BoundaryElementRequest.mesh_density` default too low?** It defaults to
  1.0, and TABI-PB — the only BEM backend — aborts below 1.5 on a dipeptide. A
  protocol default that no shipped backend can honor is a smell, but the right
  value is solute-dependent and backend-specific, so raising it would be
  trading one arbitrary number for another. The backend names the cause today.
- **Does `SMOOTHED_MOLECULAR` belong as the default?** Phase 7 made it
  load-bearing in a way it was not before: sashimi's default surface model is
  supported by exactly one backend, so every DelPhi solve at defaults raises
  rather than running. Refusing is right — substituting would move the answer
  by 2,000× the corpus tolerance — but a default that no second backend can
  honor is worth revisiting now that there is a second backend to weigh it
  against.
