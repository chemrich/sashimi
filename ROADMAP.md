# sashimi — Roadmap and Implementation Plan

> Thin, protocol-first wrapper around Poisson–Boltzmann electrostatics solvers,
> exposed as an MCP server. APBS is the first backend; the protocol is designed
> to outlive it. `debye` is the eventual clean-room solver that slots in behind
> the same interface.

Status: phases 0–4, 5 and 7 shipped; 6 (distribution) not started; 8 (debye)
in progress — four backends across three solver families
(APBS, DelPhi in two flavours, TABI-PB, Generalized Born) and `sashimi validate`
have landed.
Last updated: 2026-08-26 (phase 8: M0–M5, M8, M8a and M9 met; M6 recorded
rather than gated and M7 parked; §12 carries the ladder)

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
3. Make installation trivial (eventually `pip install sashimi-electro`, with
   the solver binary vendored in platform wheels).
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
│   ├── cli.py                # `sashimi corpus build|verify`, `validate`
│   ├── backends.py           # registry: name, family, construction, self-description
│   ├── capabilities.py       # what this installation can answer
│   ├── validate.py           # cross-solver spread and accuracy tiers
│   ├── analytic.py           # Born, Kirkwood and screened closed forms
│   ├── analysis.py, field.py, invariants.py, artifacts.py, bench.py
│   ├── constants.py, bem_stub.py
│   ├── apbs/
│   │   ├── backend.py        # ApbsSolver implements Solver
│   │   ├── input.py          # grid + solvent → mg-auto input file (f-strings, no jinja)
│   │   ├── grid.py           # physical grid intent → legal dime/cglen/fglen (psize logic)
│   │   ├── options.py        # SurfaceModel → APBS srfm/srad keywords
│   │   ├── run.py            # subprocess, tmpdir, timeout, stdout parsing, error mapping
│   │   └── discover.py       # binary discovery: $SASHIMI_APBS_PATH → which() → conda env
│   ├── debye/                # the clean-room finite-difference solver (§10)
│   ├── delphi/               # DelPhi C++ and pyDelPhi backends
│   ├── gb/                   # Generalized Born, in-process, no binary
│   ├── tabipb/               # boundary-element backend (§4.1's acid test)
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

Current, in `errors.py`, with backend-specific subclasses under each backend
package: `SashimiError` → `InputError` (`MalformedStructure`, `GridTooLarge`,
`UnsupportedRequest`), `SolverError` (`ConvergenceFailure`, `SolverCrash` /
`ApbsCrash`, `PreparationFailed` for pdb2pqr) and `BackendUnavailable`
(`ApbsNotFound` and its siblings). Every message carries the actionable next
step — `GridTooLarge` states the achievable resolution, `ApbsNotFound` carries
the install one-liner.

The split distinguishes **input errors** (bad PQR, impossible grid), **solver
errors** (non-zero exit, convergence failure, parsed from stderr), and
**environment errors** (binary missing, wrong architecture). These are the three
things a caller can act on differently, and phase 4 delivered them — §4.1
item 4. The earlier `SolverNotFound` name is gone.

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
installer can provide it and `which` is the normal answer. `Provenance` records the resolved
path and its sha256 alongside the version label — `binary_path`,
`binary_sha256`, with `SolveResult.backend` carrying the label alone — so which
binary produced a result is always recoverable.

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

`sashimi_solve(pqr_path, resolution, padding, mesh_density, ionic_strength,
solute_dielectric, solvent_dielectric, compute_energy, surface_model, backend,
prefer, output_dx)` — the
workhorse. Returns energy when requested, backend provenance, and — for the
backends that fill a volume — grid stats, the DX path and `resolution_relaxed`
so a guardrail-relaxed grid is visible rather than silent.

**`mesh_density` is the boundary-element cost knob**, the analogue of
`resolution` for a solver that meshes a surface instead of filling a volume, and
**a parameter the chosen backend cannot use is refused rather than ignored**. No
backend has both: `resolution` and `padding` describe a grid, `mesh_density` a
triangulation. Silently accepting one that does nothing is how a caller comes to
believe it made a 450-second mesh cheaper by halving a resolution the mesher
never reads — and quietly-wrong parameters are this project's most expensive
recurring failure (§12: DelPhi reading `temper` as Celsius, DelPhi reading a
different PQR radius column, Generalized Born handed the wrong radius dialect).
Each produced a plausible number from an input that was not what the caller
thought it was.

**`backend` selects the solver**, and **the response shape follows what that
solver returned** rather than a schema it must satisfy — the same rule the
corpus summaries follow (§7). A finite-difference solve carries a map; a
boundary-element one carries surface statistics and no `dx_path`, because there
is no volume to write; an analytic one carries an energy and nothing else,
because it computed nothing else. Asking `gb` for a solve with
`compute_energy=false` is refused rather than answered with an empty result: it
computes one number, so not wanting it is a request for nothing.

Backends come from `sashimi.backends`, a registry holding each one's name,
request family, how to construct it and how it describes itself. It exists
because that knowledge was previously in three places — the CLI's factories,
`capabilities`' report functions, and nothing at all for a library consumer —
which could disagree in a way nothing noticed: a backend `capabilities` reports
as available but `--backend` cannot name is a report about a solver the caller
then cannot run. debye registers there in one line.

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
reference on its **energy**, deliberately, and M3 sharpened the reason into two
that are worth keeping apart.

The first is the one this paragraph originally gave: the Debye-Hückel screening
term depends on an ion-exclusion convention the backends do not share. APBS's
ionic contribution is −0.688 kJ/mol and DelPhi's is −0.496, both reporting
`polar-solvation`, and DelPhi's is resolution-independent where APBS's carries
grid noise. That is a different convention beneath the same declared quantity —
not the `EnergyTerm` gap of §12. **M3 narrowed it: the disagreement is DelPhi's
alone.** APBS and debye, sharing no source, both land within 0.2% of the
linearized expression, so this is not a coin flip between two conventions. And
APBS's "grid noise" turned out to belong to the *probe-based surfaces* — on
`van-der-waals` it reproduces the closed form to 0.03% at every spacing, against
9-13% on the other two, which is why `born-ion-salt` being `smoothed-molecular`
is load-bearing rather than incidental.

The second reason survives all of that and is now the operative one: **the ionic
term is 0.3% of the total where discretization is 1.6%**, so an energy check
against a closed form cannot see the salt whatever the convention. Every
mutation of debye's screening tried at M3 — including deleting the Boltzmann
term outright — leaves the total inside the band APBS itself needs. What a
salted case *can* carry is an `AnalyticField`, added at M3, and a gate on
`G(I) − G(0)` across a pair of recordings.

### Charge placement, and the axis nothing covered

Two additions worth naming, because both closed gaps that had been invisible.

**Kirkwood's series** — a charge off-centre in a sphere — is the second closed
form in the project. The Born ion is symmetric in every way a solver could be
wrong about direction, so it cannot catch a mistake in *where* a charge is;
every multipole above the monopole only exists once the charge moves. Expressing
it needs nothing new: one uncharged sphere for the boundary plus a zero-radius
atom carrying the charge, which is only possible because a PQR separates charge
from radius. Verified against APBS at 0.25 Å: 0.097%, 0.473%, 0.114% and 7.678%
at d/a of 0.3, 0.5, 0.7 and 0.9 — the last being 0.3 Å from the boundary, kept
precisely because it records where charge placement stops being resolvable.

**Every case was `smoothed-molecular` until now**, which is APBS's alone. Two
consequences, both bad and neither noticed until the manifest was surveyed: the
single largest modelling choice in the calculation — 25.7% on a dipeptide (§5) —
was untested, so a backend that ignored the surface model entirely would have
passed the whole corpus; and *no corpus case could ever be verified against
another backend*, which quietly undercut the corpus's stated job as debye's
acceptance gate. Eight cases now run on `molecular` or `van-der-waals`. One of
them is an invariant no other case can express: for a lone sphere the two
surfaces coincide, because a probe cannot carve a re-entrant surface out of one
atom, so `born-ion-molecular` and `born-ion-vdw` must agree *exactly* — and do.

### Tiers, because the corpus is meant to grow

`CaseTier` splits the manifest by wall time, cumulatively. Membership comes from
*measured* per-case cost, which matters because intuition had already drifted:
the 0.25 Å cases look small because their solutes are, and a Kirkwood sphere at
that spacing is 5.2 s against a Born ion's 0.47 s at 0.5 Å — so 52 seconds of
work had accumulated in a tier whose contract said "seconds".

| tier | cases | cumulative | who runs it |
|---|---|---|---|
| `fast` | 40 | 25 s | `pytest`, so the local edit-test loop stays a loop |
| `standard` | 75 | 186 s | a dedicated CI step per push |
| `full` | 100 | 443 s | `sashimi corpus verify --tier full`, on demand |

(APBS 3.4.1, osx-arm64, idle machine, 2026-08-26. **The `full` figure is the
only one that moves**: re-measured while 44 agents were running it read 1,708 s,
a 3.9× swing, where `fast` and `standard` moved by under 5% — 24 s and 195 s on
the same loaded run. The large cases are memory-bandwidth bound and the small
ones are not, so §11's 1.9× contention effect is a floor for this tier rather
than a bound. Quote the idle numbers and do not compare a loaded run against
them.)

Those are APBS's costs, so they are a statement about one backend. A
boundary-element solver's cost is its mesh rather than its atom count, which is
why the TABI-PB tier names what it re-verifies instead of reading a tier from
the manifest — see "Widening the shared set" below.

The standard tier is its own CI step rather than a test, for the same reason
CLAUDE.md treats a corpus diff as a real result change: it should read as a
corpus failure, not as one failure among four hundred.

The target was **50 cases**, and the corpus passed it — 100 now, after the
widening below and the closed-form and salt arms of §12. The axis that mattered
was what each case is checked against, not the count: 100 self-recorded cases
would be 100 change-detectors and a 100-line diff every time the physics
legitimately moved. What they are made of, by what can contradict them:

| kind | cases | checked against |
|---|---|---|
| Closed form | 39 | Born and Kirkwood, to a measured per-case tolerance — 37 on the energy, and `born-ion-vdw-salt` and `born-ion-vdw-high-salt` on the field alone, where the screened form is a correction the energy cannot see |
| Real structures | 58 | recorded APBS, the invariants below, and — on the shared surface — a second and third backend |
| Neither | 3 | recordings alone: `born-ion-salt` against APBS only, `born-ion-molecular-salt` and `born-ion-molecular-high-salt` also against DelPhi and debye, because two codes' ion conventions differ by 39% and pinning either would encode a choice as physics |

Twenty-one structures from 2 to 18,242 atoms: methanol and methoxide, an acetic
acid / acetate ionization pair, a lone aspartate residue, an ALA-GLY dipeptide,
a 906-atom protein with a non-integer net charge, barnase and barstar, three
lysozyme charge states, a 2,065-atom protein (`protein-1a63`), an FKBP
apo/holo pair, carbonic
anhydrase with and without its ligand, a 260-atom solute carrying +21.69 e, an
actin monomer, acetylcholinesterase, and serum albumin at 18,242 atoms.

*One of those names was wrong and the recordings carried it. `protein-rna` is
`apbs-examples/1a63.pqr`, whose residues are ALA through VAL and nothing else —
there is no nucleic acid in it, so the corpus has never covered a protein-RNA
complex and the claim that it did was inherited from the filename. The case is
sound as a 2,065-atom protein and its numbers stand; only the label was false.
**Renamed to `protein-1a63` on 2026-08-28**, moving nine recordings, three
manifest entries, four test files and `PROVENANCE.md`. No energy moved: the
rename edits `name` and `description` in each recording and nothing else, and
the script asserted every `energy_kj_mol` unchanged as it went.*

**What the corpus can now assert that no single case could.** Solvation goes as
q², so the ±1e Born pair must agree exactly and the +2e case must be exactly 4×.
A probe cannot carve a re-entrant surface out of one atom, so the molecular and
van der Waals Born cases must agree to the last digit. Two lysozymes at the same
+9e must *not* agree, because what separates them is the charge distribution
rather than its total — which is the entire reason to solve the equation instead
of using Born. Refining the grid must move an answer toward the closed form, on
a real protein and not only on a sphere. And FKBP with and without DMSO differ
by 0.26% of either, which is what a binding energy is: a few kJ/mol extracted
from two numbers three orders of magnitude larger, and the reason energies are
held to 1e-4 rather than to something comfortable.

Deliberately absent: `achbp` at 16,090 atoms, whose PQR is 1,068 KB against the
repository's 1,024 KB large-file guard. Weakening a guard for one convenience
case is a bad trade — though it is no longer the only route: `serum-albumin`'s
PQR is committed gzipped as `tests/data/1ao6.pqr.gz` at 318 KB, which is what
`achbp` would need. Acetylcholinesterase at 8,279 atoms already exercises what
it would have — the `max_points` cap relaxing 0.5 Å to 0.60/0.54/0.49 Å, which
is why 8,279 atoms costs 15 s rather than an hour. The largest case is now
`serum-albumin` at 18,242 atoms, which asks for 1.0 Å and so never reaches the
cap.

**The third-party anchored tier was planned and then abandoned, on measurement.**
APBS ships `examples/` with per-version reference values and independent UHBD
numbers, which looked like the richest source of external truth available. It is
not usable: those examples compute their reference state with `sdie 1.00` — the
solute's interior dielectric against *vacuum* — where sashimi's
`EnergyTerm.POLAR_SOLVATION` is solvated minus a *uniform* dielectric, APBS's own
convention in its `born` example and the one the protocol commits to. The gap is
not subtle:

| | APBS README | sashimi |
|---|---|---|
| methanol | −36.2486 | −25.16 |
| methoxide | −390.4122 | −201.96 |

Change that one keyword in their input and APBS returns −25.2538 and −201.5878,
which sashimi reproduces to **0.37% and 0.18%**. So nothing is wrong with either
code, the published numbers answer a different question, and using them as
references would mean exposing the reference state as a knob — the raw-input
passthrough §5 deliberately refuses. UHBD's numbers follow the same convention,
so they go too. Recorded in `tests/data/apbs-examples/PROVENANCE.md`, where
someone comparing sashimi against the APBS docs will find it before filing a bug.

### The corpus stops being finite-difference by construction

`Case` recorded grid geometry and `request()` built one request type, so a
curated set of fifty physically meaningful systems was usable by exactly one
backend and `corpus --backend gb` refused outright. `Case.system()` is the fix,
and it is the seam phase 7 already built for cross-family validation: a case is
a physical question, and which dialect it is asked in is the backend's business.

**`System` and `SolverFamily` moved into `protocol.py`.** They were in
`sashimi.validate`, which made the regression net depend on the product feature
built on top of it. A statement about request types is protocol vocabulary, and
an extracted `pb-protocol` (§10) would need it for the same reason the corpus
does. `validate` re-exports them, so nothing downstream moved.

**A summary's shape follows what the backend returned**, rather than a schema it
must satisfy. A volumetric answer records geometry, statistics and pinned
probes; a boundary-element answer records its vertex count and surface
statistics *and no probes*, because vertices are the mesher's choice and move
when it is rebuilt; an analytic answer records the energy and nothing else,
because it computed nothing else. Comparing a recording of one shape against a
solve of another is refused rather than partially attempted — that is a backend
swap, not a drift.

What this buys, in order of value:

- **A regression net on our own code.** Generalized Born is a few hundred lines
  of numpy that will keep changing, and it had unit tests against closed forms
  but no recorded answers on real structures. Five at first and thirty-one now, in
  `tests/corpus/gb/`, and because there is no binary they are verified by
  `pytest` on every machine — the one part of the corpus that cannot skip.
- **A golden for a backend CI compiles from source.** TABI-PB's mesher version
  is part of a result's identity, and `tests/corpus/tabipb/` pins six.
- One fewer hand-assembled `System`: `sashimi validate` and the GB reference
  tests both went through their own copy of that construction, which stops
  matching the moment `Case` grows a field — `mesh_density` already did.

**Coverage was thin, and the reason was the surface model.** Generalized Born
answers only on `molecular`, so it took 5 of 50 cases; TABI-PB needs four atoms
as well, and `acetate-molecular` at eight atoms does not finish inside its own
600 s timeout (measured twice), so it took 1. Recording that was better than
recording nothing, and the way to widen it was more molecular-surface cases
rather than more machinery — which is what the next section did.

### What the flavours agree to, and what a recording can hold

Building `tests/corpus/delphi/` from the C++ build raised a question the
project could not answer by inspection: a recording is only useful if CI can
verify it, and CI verifies on whichever DelPhi its runner has. Two unknowns,
both measured on 2026-08-12 by having the Linux leg — which carries both
flavours since macOS came off the per-push path — check the same nineteen
recordings.

**A C++ recording is portable across architectures.** All 19 cases recorded on
osx-arm64 reproduce against a linux-64 build of the same Clemson tarball, at
full corpus tolerance: 1e-4 on energies, 1e-3 on potential statistics, and all
fifty interpolated probes per case. That is a stronger statement than the "to
the last printed digit" this document previously rested on, and it is direct
evidence for §10's fourth validation rung — portability as a test — and for the
standing claim that a build from that tarball is a reproducible artifact rather
than a local accident.

**pyDelPhi cannot verify them**, failing 15 of 19: energies 0.047% to 0.426%
out, the Born ion's potential minimum 2.5%, grid origins differing in their last
digits. That is not a wrong answer — 0.4% is far tighter than the 2.3% between
DelPhi and APBS, and it is the ordinary distance between two implementations of
one iterative method. It is ~43× the 1e-4 a *recording* is held to — the
whole-corpus bound is 1.257% and ~125×, measured 2026-08-20 — which
is a different question from whether a backend is correct.

So the corpus holds one C++ recording set, the re-solve test gates on the
flavour exactly as the charge-echo guard does, and pyDelPhi keeps the
behavioural tier it already had. The alternative — a per-backend tolerance loose
enough to admit both — would have to be near 0.5%, fifty times the corpus's, and
wide enough to hide the regressions the corpus exists to catch. The flavours
remain interchangeable as *backends*; they are not interchangeable as *sources
of a recorded number*.

Split by measured cost like every other tier here: 27 cases per push and the
remaining 31 on demand — the on-demand set is derived as the complement of the
per-push list, so a recording added later joins it by existing rather than
falling out of both. DelPhi's cost is its cubic grid, which follows
the bounding box rather than the solute — `fas2-molecular` is 906 atoms and
takes 11.5 s where `lysozyme-molecular` is 1,960 and takes 5.8 s.

### Widening the shared set

Fourteen molecular-surface cases, no new machinery and no new structures: each
one is a sibling of a case the corpus already had on `smoothed-molecular`, so it
carries an axis that was APBS-only into the set every backend can answer. The
salt and temperature sweeps, the dielectric arm of the analytic sweep, both
halves of an ionization pair, a binding pair, a nucleic acid, and the sign of
the charge at protein scale.

| tier | cases | before | after |
|---|---|---|---|
| Reference, finite difference (APBS) | 100 | 50 | 64 |
| Reference, finite difference (DelPhi, C++) | 58 | 0 | 19 |
| Reference, boundary element (TABI-PB) | 6 | 1 | 6 |
| Approximate, analytic (Generalized Born) | 31 | 5 | 19 |
| Reference, finite difference (debye) | 58 | — | — |

**The two reference-tier families agree to 1.0–1.6% on every case they share.**
TABI-PB meshes a surface where APBS fills a volume, so they have no
discretization in common and no way for an error in charge handling, units or
boundary conditions to cancel between them. That band is the corpus's strongest
correctness statement that needs no closed form, and it is what makes the
approximate tier's 0.7–28% legible as a property of the method rather than as
corpus noise. Both are checked from the recorded files, with no binary
installed, in `tests/test_corpus_manifest.py` and `tests/test_corpus_gb.py`.

**What the widening found, which is the reason to do it at all:**

- **A binding difference is not a difference of approximations.** FKBP with and
  without DMSO sits 2.6% and 3.2% from the reference tier — inside anything
  anyone would call agreement — and the *difference* of those two numbers has
  the wrong sign: APBS pays **+6.25 kJ/mol** to bury the ligand where
  Generalized Born is handed **−8.32**. A binding energy is that difference, so
  a small absolute error is not evidence the tier can be used for the quantity
  most callers want. Handing GB the structure's own radii recovers the sign
  (+21.6) and costs 50% on the absolute energy, so there is no setting that is
  right for both. `AccuracyTier` already refuses to average this tier into a
  spread; this is the sharper statement of why.
- **A boundary-element solver's cost is its mesh, not its atom count.**
  `fas2-molecular` is 906 atoms and meshes and solves in **48 s** at 21,850
  vertices; `ion-protein-complex-molecular` is 260 atoms — a third as many — and
  takes **450 s** at 68,054, because it is a united-atom structure whose large
  radii produce a much bigger surface. A tier assignment derived from APBS cost
  says nothing about this backend, so `tests/test_tabipb_solver.py` names what
  `pytest` re-verifies per push rather than filtering the manifest by tier.
- **Three small solutes, three unrelated failures.** `acetate-molecular` at 8
  atoms runs past the 600 s timeout; `aspartate-residue-molecular` at 12 aborts
  immediately on `stoul: no conversion` *after* NanoShaper reports the surface
  built; `born-ion-molecular` at 1 and `methanol-molecular` at 3 are below the
  mesher's four-atom floor. "Too small for BEM" is three different bugs wearing
  one description.
- **The Born case at eps_p = 2 pins the method's systematic.** Generalized Born
  is 3.093% from the closed form there and 3.093% at eps_p = 1 — the same number
  to four digits, which is OBC2's offset on a lone sphere and says the
  dielectric factor itself is not where its error is.

**What the first ten structures found immediately.** Generalized Born's
deviation from APBS is 1.6–4.5% on proteins whose PQR carries AMBER
Lennard-Jones radii, and 13–28% on methanol, 2LZT lysozyme and carbonic
anhydrase. The cause is not the method: those three arrive with PARSE-like radii,
and GB substitutes mbondi, so the two solvers are handed measurably different
solutes. Handing GB the structure's own radii reverses it — and reverses it the
other way on AMBER-like input, where `AS_GIVEN` reaches 55% and can return a
*positive* solvation energy. Neither setting is universally right; mbondi is
right for what `sashimi_prepare_structure` produces, which is why it stays the
default. Six corpus cases could not have shown this and twenty-four did.

### The PQR a fixed-column reader sees

Recording the DelPhi tier found a defect in `sashimi.pqr.format_pqr` that had
been shipping since phase 4, in the module set §10 calls the most stable thing
in the project.

It wrote **minimum-width** fields, so a four-character residue name — `TARG` in
the APBS example set, `MEOH` in another — pushed every field after it one column
right. sashimi's own reader splits on whitespace and APBS's is lenient, so both
round-tripped it perfectly. **DelPhi reads fixed columns.** It parsed acetate as
two charged atoms carrying +80.84 e where the file says seven and -1, and
returned **-865,205 kJ/mol against APBS's -196.90** — and the *identical* value
for acetic acid, which is a different molecule. Two structures, one answer, to
six decimals.

Invisible for a year because DelPhi had only ever been run on the Born ion and
ALA-GLY: residues `ION`, `ALA`, `GLY`, all three characters. Three of the
nineteen shared corpus cases were affected the moment the tier was recorded.

Two fixes, and the second matters more:

- The fields are exact widths now, truncating names rather than overflowing.
  Names that fit render byte-identically, which is why all 64 recorded APBS
  cases reproduce unchanged.
- **The backend checks DelPhi's own echo of what it read** — net charge and
  charged-atom count — against the structure, and refuses rather than solving.
  DelPhi printed a warning the whole time and nothing read it. This is the
  structural-output verification §13 already applies to APBS, which likewise
  exits 0 on failure.

The lesson worth keeping: every test of the writer round-tripped it through a
reader that splits on whitespace, so none of them could see a column. Round
-tripping was necessary and never sufficient, and the only thing that found this
was handing the file to a stricter consumer.

**Test partitioning by architecture.** Protocol-layer tests are pure Python and
run natively everywhere. Subprocess integration tests are gated behind
`@pytest.mark.apbs` (later `@pytest.mark.delphi`, …) and only run where the real
binary exists — which matters because there is no `linux-aarch64` APBS. debye
inverts this: pure implementation, whole suite runs natively anywhere, and that
portability is itself a tested differentiator.

**CI**: GitHub Actions, `uv sync --frozen` against the committed lockfile, with
APBS installed from conda-forge via micromamba — whose only job is fetching that
one binary. A single `ci-ok` gate job fronts the matrix so the required
status-check name survives matrix changes, which is what makes the matrix free
to change.

Three Linux legs run per push, and they cover different things:

| leg | carries | what it is for |
|---|---|---|
| `ubuntu-latest, full` | APBS, DelPhi (both flavours), TABI-PB | every backend, the corpus, cross-validation |
| `ubuntu-latest, apbs-only` | APBS | **the README's own recommended install**, and nothing else |
| `ubuntu-latest, none` | nothing | a fresh checkout, and the configuration `sashimi.gb` exists for |
| `macos-latest, full` | APBS, pyDelPhi | osx-arm64 as a first-class platform — main, weekly, on demand |

**A marker selects and deselects; it does not skip.** `tests/conftest.py` makes
it skip, once, for every marked test present and future — because fixing the
instances one at a time was treating a rule that was never enforced. Four bugs
of that exact shape surfaced on 2026-08-12 alone. Until it existed, a bare
checkout failed 56 tests and CI hid it by deselecting the markers instead, so
"sashimi works with nothing installed" — the property `sashimi.gb` provides and
protean's fallback path depends on — was never tested. It is now **945 passed and 199 skipped** of a 1,144-test suite — CI's own
`none` leg, 2026-08-26 — and that leg holds it there.

What stops a skip-on-absence rule from hiding real breakage is not the hook: it
is the "Verify the <backend> tier actually ran" steps, which assert each tier
ran wherever its binary exists. A skip is only safe when something else insists
the tests are not always skipped.

The `apbs-only` leg exists because a test that needs a binary without saying so
passes wherever that binary happens to be installed, and every leg used to carry
at least two backends. It found two such bugs the day it was added, one of them
five tests failing on the documented install since phase 7 —
`comparable_surface_models()` counts Generalized Born, which is always
available, so "fewer than two backends installed" stopped being the thing it
tested. **A marker selects and deselects; it does not skip**, and
`tests.helpers.installed_or_skip` is now the one guard that does.

macOS moved off the per-push path on 2026-08-12 for cost: GitHub bills arm64
macOS at 10x per minute, and at ~6 minutes a run it was roughly 90% of this
project's CI spend — enough to exhaust the account's Actions budget mid-day.
What it uniquely proves is that conda-forge still ships a working osx-arm64
APBS 3.4.1, which is a fact about a third party that changes rarely, and every
commit is exercised on osx-arm64 locally before it is pushed. It still runs on
main after a merge, weekly against a moving conda-forge, and on demand.

### What a survey of the corpus found, and what it cost to learn

**2026-08-26/27.** A survey of the corpus against its own purpose proposed eight
additions. Six were attacked adversarially and **one survived**. The kills are
worth more than the list would have been, because each carries a measurement
that would be expensive to rediscover — and the survey's own strongest finding
was that **the binding constraint is referees, not structures.**

Every count below was re-measured against `MANIFEST` rather than carried over
from the survey.

#### The analytic tier has never refereed a difference

**All 37 closed-form energy cases carry exactly one charge.** Twelve of them
have two *atoms*, which is not the same thing: those are Kirkwood, where atom 0
is the uncharged dielectric sphere (`q=0, r=3.0`) and atom 1 is the point charge.
The net charge is never zero either — 33 cases at +1e, two at −1e, two at +2e.

So the tier that grades against a closed form has never refereed a
charge–charge interaction, a net-neutral solute, or a **difference of two
energies** — while the corpus grades differences constantly: binding, ionization,
titration, protonation. It can check none of them.

#### 27 of 100 cases are graded against nothing but their own APBS recording

`smoothed-molecular` is APBS-only across the entire registry:

| backend | surface models |
|---|---|
| APBS | molecular, smoothed-molecular, van-der-waals |
| DelPhi C++ | gaussian, molecular, van-der-waals |
| pyDelPhi | gaussian, molecular |
| TABI-PB | molecular |
| Generalized Born | molecular |
| debye | molecular, van-der-waals |

61 cases carry no closed form at all. 34 of those can at least be put to a second
backend. **The remaining 27 cannot, and every one of them is
`smoothed-molecular`** — so the only thing they are checked against is what APBS
said last time.

That set is not a remainder. It holds nearly every *relationship* the corpus
advertises: the salt sweep (`born-ion-salt`, `peptide-no-salt`,
`peptide-default`, `peptide-high-salt`), the temperature pair (`peptide-cold` at
277 K against `peptide-default` at 298.15 K), both ionization pairs
(`acetic-acid`/`acetate`, `methanol`/`methoxide`), the lysozyme trio
(`lysozyme`, `lysozyme-protonated`, `lysozyme-deleted-residue`), the binding
pair (`fkbp-apo`/`fkbp-dmso`), the metalloprotein pair
(`carbonic-anhydrase`/`hca-complex`) and the convergence pair
(`fas2`/`fas2-fine`).

**A recorded relationship whose only witness is the backend that produced it is
a regression test, not a check.** That is the referee gap, stated in cases
rather than in atoms.

#### The corpus is aimed away from its consumer

protean reads a **1.0 Å potential field** on 250–1,200-residue assemblies, and
never an energy. The corpus:

| resolution | cases |
|---|---|
| 0.25 Å | 26 |
| 0.35 Å | 1 |
| 0.5 Å | 70 |
| 0.6 Å | 1 |
| **1.0 Å** | **2** |

97 of 100 cases sit at 0.5 Å or finer; two sit where the consumer actually
reads. Every cross-backend check is on the energy, and 20 cases carry a
closed-form *field* reference against 37 for the energy. The survey measured
debye's `fas2` energy at 11.03% out at 1.0 Å against 2.82% at 0.5 Å, **with the
backend ordering inverted** — a regime nothing in the corpus currently sees.

#### Knobs that are not knobs

Single-valued across all 100 cases: `ion_radius` 2.0, `surface_radius` 1.4,
`mesh_density` 2.0, `max_points` 4,173,281, `compute_energy` True, and
`Equation.LINEAR`. `SurfaceModel.GAUSSIAN` has **zero** cases, though both
DelPhi flavours can build one. `padding` takes two values and `temperature`
three.

#### The one case that looked like coverage and was not

**`protein-rna` contained no RNA.** Its source is
`tests/data/apbs-examples/1a63.pqr`: 2,065 ATOM records, eighteen distinct
residue names, **all of them standard amino acids**, and not one phosphorus atom.
Phosphorus appears in exactly two atoms corpus-wide, both in actin's ADP.

The case name, the case description ("the only nucleic acid in the corpus, so
the only case where phosphate backbone charges are exercised at all") and
`PROVENANCE.md` ("Protein–RNA complex: nucleic acid, which nothing else here
covers") all assert the opposite. The upstream PDB entry may well be a
protein–RNA complex; the file APBS's example set vendors is the protein alone.

This was the corpus's most misleading entry, because it is the one that made a
real gap look closed — and it did exactly that during this survey, before the
file was read. **Corrected 2026-08-28: the case is `protein-1a63`**, and the
count was nine recordings rather than eight, `tests/corpus/tabipb/` having
gained one in between. No energy moved.

#### The five kills, each with its measurement

- **Multi-charge Kirkwood** — the best idea of the eight, and the first
  closed-form referee for a *difference*. The closed form is correct and
  reproduces the shipped single-charge function to 1e-16. **The knob was never
  swept.** At 0.5 Å, padding 8/10/11/12 Å gives +3.26 / −6.77 / −13.58 / +26.73
  kJ/mol — two of them with the wrong sign. The celebrated 0.028% agreement was
  a padding-10 coincidence.
  It left a real finding behind: `E(+1,−1) + E(+1,+1) − 4·E(single) = 0`
  identically for any solver linear in the charges. debye violates it by
  **0.0000**, DelPhi by 0.004%, **APBS by 2.7–13 kJ/mol** — larger than the
  interaction it was meant to referee.
- **1.0 Å siblings refereed by TABI-PB** — §12 already says in bold that its
  recording is one mesh density, "a rung and not a limit".
- **A Stern-radius sweep** — premise measured false. Ignoring `ion_radius` does
  not pass at the existing point; it fails by 18%.
- **An hca binding pair** — the referee sits *inside* APBS's own grid noise:
  487 / 472 / 486 / 473 kJ/mol at 1.0 / 0.7 / 0.5 / 0.4 Å, a 3.27% spread and
  non-monotone.
- **A lysozyme pKa cycle** — leg A needs no new files, and the two new files buy
  only leg B, where the backends disagree by 7–17%.

Rejected on principle rather than on measurement: the charged rod, whose
finite-length error (2.2–6.2%) exceeds what it would grade; **Gouy–Chapman and
Manning condensation, which solve the NONLINEAR equation** and cannot referee a
solver that is linear in the charges — all 100 cases are `Equation.LINEAR`; two
dielectric spheres, which is a convergent linear system rather than a formula; a
membrane, which `SolventModel` cannot express; and experimental pKa values,
which test the model rather than the solver.

#### What survived

The **1d30 triple** — B-DNA dodecamer, a minor-groove binder, and their complex,
from APBS 3.4.1's own `examples/bem-binding-energy`, 87 KB, the same BSD-3 tree
the corpus already vendors from. It closes the chemistry gap `protein-1a63`
only appeared to close — the duplex carries residues DA/DC/DC5/DG/DG3/DT and **22
phosphorus atoms**, against two in the whole corpus today — and its ΔΔG is a
rigid-body binding difference: the
decomposition is exact — verified from the files: 796 = 758 + 38 atoms and
−20.000 = −22.000 + 2.000 e — and APBS's own ΔΔG moves 0.18% across 0.5/0.4/0.3 Å
while the energies it is built from move 0.46–1.93%, so the 1.0% cross-family
spread is five times the noise.

**Not taken yet, and the reason is this section.** Three more cases at 0.5 Å
graded on the energy add coverage in the regime that already has 97 of 100
cases, on the observable the consumer never reads. The chemistry argument is
real; the priority argument is not. If it is taken, it should carry a 1.0 Å
sibling.

## 8. Backend strategy beyond APBS

| Phase | Backend | Why | Integration |
|-------|---------|-----|-------------|
| 1 | **APBS** (FD, mg-auto) | Community default, broadest features, conda-forge packaged | subprocess |
| 2 | **DelPhi** ✅ | FD sibling; Gaussian dielectric, focusing workflows; cheap triangulation partner | subprocess, two flavours |
| 3 | **TABI-PB** ✅ | BEM; forces the protocol to handle surface potentials | subprocess + NanoShaper |
| 3b | ~~**PyGBe**~~ ❌ | Dropped: builds only on Python ≤3.11, so it cannot be imported into a 3.13 process at all. §12 records the measurement | — |
| 4 | **GB tier** ✅ | Fast approximation for high-throughput triage → PB refinement — **and**, being pure numpy, the in-process proof PyGBe was meant to supply | none: in process |
| — | **MIBPB** | Not production; the *accuracy referee* (~0.4% relative error, rigorous interface treatment) | validation harness |

*One caution on that last row, added 2026-08-14.* §12's M1b was briefly read as
independent evidence for prioritising MIBPB — debye supposedly falling behind
DelPhi near an under-resolved boundary — and that reading was an artifact of grid
phase, corrected in §12. MIBPB's case rests on what it was always resting on: the
O(1) error *at* the interface that no staircase-dielectric code can avoid. That
case is unchanged and still good. It has not been independently confirmed.

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

**Handoff. Superseded 2026-08-13: debye starts in-repo as `sashimi.debye`**, at
Charlie's direction, and the extraction below happens when something other than
sashimi wants the types. The cycle argument two paragraphs down is still correct
and is simply not yet load-bearing — it bites at packaging time, and there are no
external consumers. Everything else here stands, including the ladder above and
the boundary test, which matters *more* under this decision: debye is the first
module that would want to reach into `sashimi.apbs` for a shortcut. §12's "run-up
to debye" carries the current plan; what follows is the original, kept because
the extraction is deferred rather than cancelled.

debye was to ship as a separate repo implementing `sashimi.protocol`'s
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
| GitHub Actions | linux-64 | conda-forge via micromamba | **full suite per push**, three legs: every backend, APBS alone, and nothing installed | in use |
| GitHub Actions | osx-arm64 | conda-forge via micromamba | platform proof: main, weekly, on demand — 10x billing, see §7 | in use |
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
0.99989).

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

**Released as `sashimi-electro` 0.1.0.** The distribution name is settled since
plain `sashimi` belongs to an unrelated dormant library, so the install name and
the import name differ and the README says so. `.github/workflows/release.yml`
builds and publishes on a `v*` tag through PyPI Trusted Publishing, refusing to
run when the tag and `pyproject.toml` disagree — PyPI will not let a version
number be reused, so a mismatch is not fixable after the fact.

This is §9's **v1**: ship on PyPI, document the conda-forge APBS. What makes v1
worth more than it was when §9 was written is `debye` — a `pip install
sashimi-electro` is a *working solver*, pure Python, rather than a wrapper
idling until a binary arrives. v2, the vendored platform wheels that collapse
the install to one command, is phase 6 and is not foreclosed by anything here.

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
of the vendored FETK/MALOC versions; platform wheels so
`pip install sashimi-electro` works end to end.

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
  build's molecular one; on the same surface model the flavours agree to within
  0.4%, which at printed precision looked exact.*

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
reproduces the C++ build's `molecular` result: −84.33 kT on ALA-GLY and
−92.22 kT on the Born ion, identical on matched grids **to the two decimals
DelPhi prints**, which is as far as that comparison could see. Recording the
corpus measured it properly and it is not exact — see "What the flavours agree
to" below. Its probe-dependence
(−84.33 at 1.4 Å against −86.88 at 0.5 Å), first read as proof it was *not* the
molecular surface, is that surface behaving correctly.

`VAN_DER_WAALS` is the one model pyDelPhi genuinely cannot deliver: `prbrad=0`,
the natural limit of its own method, aborts it with a numba `TypingError` in
0.2.0 and 0.3.0 alike. That is an upstream bug rather than a modelling
difference, so the model is declined rather than mapped onto something adjacent.
`SMOOTHED_MOLECULAR` is APBS-only, so a DelPhi solve that asks for it raises
`UnsupportedRequest` rather than substituting `MOLECULAR` and moving the answer
by 2,000× the corpus tolerance. It was sashimi's *default* until 2026-08-13,
which made that refusal the reply to every defaulted DelPhi solve; the default
is now `MOLECULAR`, the one model on this table every backend can answer.

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
headers — and installs pyDelPhi beside it on that same `full` leg, where a
dedicated step re-runs the DelPhi tier against it. So **both flavours run the
comparison per push, on Linux**, since both share `molecular` with APBS; the
macOS leg carries pyDelPhi alone and runs on main, weekly and on demand. The
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

*That ladder is ALA-GLY's, and it has since been read as licence to treat the
checked-in TABI-PB numbers as converged references. They are single rungs:
`tests/corpus/tabipb/fas2-molecular.json` records `sdens 2.0` and no ladder
exists for it. A backend whose energy converges **with** mesh density is one
whose single-density recording is a rung — see "The referee gap" below, and do
not gate against it.*

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

The corpus was still finite-difference by construction at this point — `Case`
recorded grid geometry and `request()` built one request type — so
`sashimi corpus --backend tabipb` refused and pointed at `validate`.
`Case.system()` lifted that later: `tests/corpus/tabipb/` now pins six
recordings, and the cases TABI-PB cannot mesh are reported `n/a` case by case.

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

### The run-up to debye, and debye itself

Planned 2026-08-13, in this order. The two items before debye are small and both
remove something that would otherwise have to be worked around inside it.

**1. The `molecular` default ✅** — landed 2026-08-13. §14 resolved it; this
lands it. **42 of 64 corpus cases inherited `smoothed-molecular` from
`SolventModel`'s dataclass default and 22 named a model explicitly**, so
flipping the default silently rewrites what those 42 are asking. The order was
therefore not optional: all 42 were made to name `SMOOTHED_MOLECULAR`
explicitly, all 64 verified bit-identical, and only then the default moved,
along with the MCP tool defaults, `cli._pick_surface_model`, the README and the
capabilities text.

Bit-identity was checked at two levels, because the cheap one is the one that
actually pins the question: a canonical dump of what all 64 cases *ask* —
solvent model, grid, tier, analytic reference, and a hash of the coordinates,
charges and radii each resolves to — is unchanged across both steps, and all 64
re-solved `ok` against APBS after each. The default flip moving nothing is the
whole return on doing the explicitness step first.

The cost of the switch, recomputed from the recordings rather than quoted:
**0.80% on ALA-GLY** (−212.496 → −214.196 kJ/mol) and **2.35% on hen lysozyme**
(−4885.721 → −5000.598). At defaults on 1AKI, APBS now returns −3976.571 kJ/mol
and `gb` −3879.083 — 2.45% apart, and the second of those is a call that
refused outright before this change.

Two guards came with it, both checked against the configuration where they
should fire. `test_every_case_names_its_surface_model_rather_than_inheriting_one`
reads `corpus.py`'s AST, because at runtime an inherited value and a stated one
are the same value — the drift it catches is invisible from the objects.
`test_every_backend_can_answer_the_default_surface_model` reads the surface sets
from the modules that own them — both DelPhi flavours included — so it runs on a
bare machine and cannot skip; reverting the default makes it name
`['delphi/delphicpp', 'delphi/pydelphi', 'gb', 'tabipb']`, which is the
measurement of the problem the flip fixes.

That second guard was wrong once first, in this project's usual shape. It read
`backends.reports()`, whose `surface_models` is empty for an *undiscoverable*
DelPhi — the flavour decides the set, so an absent binary means an unknown one —
and the docstring claimed it therefore ran anywhere. It passed locally, where
all four backends are installed, and failed CI's `none` and `apbs-only` legs.
The claim "this needs no binary" was itself untested; the three-profile matrix
is what caught it, one push after being built. *Exit criterion, met:*
`SolventModel()` is molecular, the corpus is bit-identical across the change,
and a defaulted `sashimi_solve` reaches every backend rather than one.

**2. Cross-flavour agreement — deferred 2026-08-13, at Charlie's direction.**
**The C++ flavour is the touchstone until further notice**, and pyDelPhi is not
on the path to debye. Nothing below depends on it, and the reasoning that
produced it is kept because it will be right again when the flavour comes back:
a `delphi`-marked test solving three or four cheap shared cases through both
flavours and asserting agreement within **0.5%** — the measured band is
0.047–0.426% — gated on both being discoverable, so it runs on CI's Linux leg
and skips elsewhere.

Two practical consequences of naming C++ the touchstone. It is the flavour with
`VAN_DER_WAALS` (`prbrad 0`), which is the surface debye climbs first, so the
one backend that can grade M1 at all is the one now designated. And it is the
sharper of the two against the closed forms by a wide margin — see the table
below — so "graded against DelPhi C++" is a stronger statement than "graded
against a second opinion".

*This reverses an earlier recommendation, and the reasoning matters more than the
conclusion.* A `tests/corpus/pydelphi/` was the obvious move: the pyDelPhi path
is genuinely distinct code we own — its own parameter file, kelvin instead of
Celsius, its own surface mapping and CSV parsing — covered today only by
behavioural bands that catch a factor of two and miss five percent. But a
recorded exact number from a third-party program **cannot distinguish a fix from
a regression**: when upstream legitimately improves, the corpus goes red and
nothing in it can judge the change. The corpus escapes that for APBS by carrying
closed forms and cross-backend agreement, which is exactly what pyDelPhi
recordings would lack. The relationship *between the flavours* is the invariant
worth holding, it catches any drift in our input generation more directly, and
it costs four cases instead of nineteen files.

**3. debye.** Both decisions are now taken.

- **In-repo `sashimi.debye`. Decided 2026-08-13 by Charlie, overriding §10 and
  §14 Q5**, which chose a separate repo with `pb-protocol` extracted at that
  moment. The cycle they were avoiding — debye depends on sashimi while
  `sashimi[debye]` depends on debye — only bites at *packaging* time, and there
  are no external consumers yet, while the corpus, the registry and the protocol
  all live here. Extract when something other than sashimi wants it.
  `tests/test_protocol_boundary.py` is what keeps that extraction mechanical, and
  it becomes more load-bearing under this decision rather than less: debye is the
  first module that would *want* to reach into `sashimi.apbs` for a shortcut, and
  the layering test is the thing that says no.
- **Climb the ladder on van der Waals**, and build the solvent-excluded surface
  before the corpus gate. The analytic rungs — Born, Kirkwood — need only a union
  of spheres; the SES construction is the hardest single piece and sits on the
  critical path, so knowing that at the start is worth more than discovering it
  at M4.

### Closing the closed-form gap, before M1 ✅

**Landed 2026-08-13.** The corpus is 79 cases, 37 of them answerable by a
sharp-boundary solver (17 in the fast tier), with 31 closed forms and 8 field
checks. Recordings: APBS 79, DelPhi C++ 35, GB 29, TABI-PB 6. What the work
found that the plan did not anticipate:

- **A closed form cannot judge an approximation, and it had been trying to.**
  Tightening the sharp-boundary tolerances to their measured values turned the
  GB tier red — correctly. `AccuracyTier.APPROXIMATE` names a backend that does
  not discretize the equation, so measured discretization error is not a bar it
  can be held to: GB is 14.6% from Born on a 1 A sphere and that is the method.
  It had only been passing because the old tolerances were loose enough to
  swallow it. `_verify_analytic` now skips the approximate tier and says why;
  what grades GB is its recorded deviation from the reference tier.
- **The "two structures, one answer" guard fired on legitimate physics.** The
  sharp ladder introduced three coincidences it could not distinguish from the
  fixed-column PQR bug it was written for: q² makes −1e and +1e identical, a
  lone sphere has the same molecular and van der Waals boundary, and DelPhi's
  corrected reaction field on a sphere does not move with the grid. It now
  groups by what the energy is *allowed* to depend on — geometry, |charge|,
  dielectric and salt — and still catches acetate carrying acetic acid's answer.
- **GB's exclusion is by property and the obvious property was the wrong one.**
  "GB substitutes radii" excludes every real protein, where substituting mbondi
  radii is the method working. The rule that holds is *synthetic* geometry whose
  radii GB would replace: the radii are the model there, and GB answers about a
  different molecule.

The field axis is 8 cases across radii 1–6 Å on both sharp surfaces, sampled at
`a + k·h` with k ∈ {2, 4, 8}. Worst measured: APBS 0.83% and DelPhi 0.74% on the
fine pair, rising to 4.7% on the coarse 2 Å sphere — the tolerances are twice
the worst observed, per case, as the manifest's convention requires. A solver
that integrates to the exact Born energy and hands back a uniformly wrong field
is now caught, which is the failure nothing in the corpus could see before.

*What the plan said, kept for the record:*


**Measured 2026-08-13, and it is why this step exists.** Of the corpus's 18
closed-form cases, **15 are on `smoothed-molecular`** — APBS's harmonic
averaging, which debye has no reason to implement and which sashimi has just
moved its default away from. Only `born-ion-molecular`, `born-ion-molecular-eps2`
and `born-ion-vdw` sit on a boundary debye builds early, and **all four Kirkwood
cases are `smoothed-molecular`**, so M2 as written cannot be met by a solver that
does not implement APBS's smoothing. Cases a sharp-boundary solver could verify
against today: **fast 8 of 25, standard 13 of 49, full 22 of 64.**

The closed form itself is surface-independent for a sphere — the boundary is the
same sphere either way — so the physics transfers. What does not transfer is the
per-case `rtol`, every one of which was measured against APBS at `smol`.

**What the sharp boundary actually costs, measured.** Born ion, *q* = +1e,
*a* = 3 Å, ε_p = 1, exact −228.6108 kJ/mol; padding 10 Å; achieved spacing in
parentheses, because the `max_points` guardrail relaxes a 0.125 Å request to
0.1625 Å and a convergence pair built from *requested* spacing would be measuring
nothing:

| requested | APBS `molecular` | APBS `van-der-waals` | DelPhi C++ |
|---|---|---|---|
| 0.5 Å (0.406) | 2.3572% | 2.3572% | 0.0006% |
| 0.25 Å (0.203) | 0.6212% | 0.7866% | 0.0006% |
| 0.125 Å (0.163, capped) | 0.3606% | 0.4802% | 0.0006% |
| 0.125 Å (0.116, cap raised) | 0.3020% | — | — |

Two things fall out of that table, and both change the plan.

**APBS on a sharp boundary is roughly four times worse than on `smol`** — 2.36%
against 0.62% for the same ion at the same nominal spacing — because `smol`'s
harmonic averaging *is* APBS's discretization-error reduction. DelPhi C++ is
0.0006% and does not move with the grid, which is the corrected reaction field
being a different quantity from a grid self-energy rather than a better one.
Confirmed against its resolved parameters: `gsize` 53 → 105 → 159 and `scale`
2.0 → 4.0 → 6.08, so the resolution is genuinely reaching it and the answer
genuinely does not depend on it.

**One `rtol` per case is shared by every backend that runs it**, and
`_verify_analytic` applies it without asking which backend produced the answer.
So the tolerance is set by the *worst* backend on that case: `born-ion-molecular`
carries `rtol=0.05` because APBS needs it, which means **a debye that is 4% wrong
would pass a milestone whose stated criterion is 1%.** That is a check that
cannot fail, in the exact sense §7 keeps finding. Two fixes, and this plan takes
both:

- Put the gate cases at **0.25 Å**, where APBS is 0.62–0.90% rather than
  2.4–4.8%, so the shared tolerance can be tight enough to mean something.
- Give `AnalyticReference` an optional **per-backend tolerance**, so the corpus
  tolerance stays "does any backend disagree with physics grossly" while debye is
  held to the number its milestone names. Additive to a frozen dataclass, and
  recorded in the summary next to `rtol` so a reader sees which one applied.

**Kirkwood, measured on the sharp boundary.** *a* = 3 Å, ε_p = 1:

| case | exact | APBS 0.5 / 0.25 / 0.125 | DelPhi 0.5 / 0.25 / 0.125 |
|---|---|---|---|
| d/a = 0.5 | −304.2867 | 4.73% / 0.90% / 0.84% | 0.15% / 0.21% / 0.14% |
| d/a = 0.9 | −1193.9551 | 6.40% / 9.85% / 9.05% | 26.67% / 4.29% / 7.50% |

**d/a = 0.9 is not a usable gate and must not become one.** Both codes get
*worse* under refinement, non-monotonically, and disagree with each other by up
to 20.3 points at 0.5 Å — the charge sits 0.3 Å inside the boundary and the near-interface
self-energy is what the grid cannot resolve. It is worth *recording*, in the way
`born-ion-r1-coarse` records 5.1% and says so; it is not worth gating a new
solver on a number no existing solver reproduces. M2's rungs are d/a ∈
{0.3, 0.5, 0.7}.

**Which backends can record these at all.** TABI-PB **cannot**: NanoShaper needs
at least four atoms and the sphere geometries are one and two. GB can run the
Born cases (3.09% from exact, inside its documented approximate band) and **must
never record the Kirkwood ones** — it returns −873.54 against −304.29, **187%
out**, because Kirkwood's charge-bearing atom has *zero radius* and anything that
divides by radius is undefined there. That is the third appearance of "pdb2pqr's
radii are not GB radii", and the corpus should encode the exclusion rather than
rediscover it.

### The axis this was missing, which matters more than another sphere

**Every closed-form check in the corpus is on the energy** — one integrated
scalar. The potential *field* is compared only against itself: `_verify_probes`
diffs a recording against a fresh solve, so a backend wrong in the field from its
first build stays wrong and passes. The one test that does check a field against
physics, `test_potential_outside_the_ion_matches_closed_form`, stays at r ≥ 1.25a
by construction and blames the smoothed surface for a "~70%" divergence at the
boundary.

That attribution is wrong, and the correction matters to the consumer this
project exists for. APBS at 0.25 Å against the Born closed form for φ:

| r/a | 1.00 | 1.05 | 1.10 | 1.25 | 1.50 | 2.00 |
|---|---|---|---|---|---|---|
| `smoothed-molecular` | 85.6% | 3.36% | 2.21% | 0.88% | 0.32% | 0.13% |
| `molecular` | 108.4% | **0.34%** | 0.67% | 0.87% | 0.52% | 0.20% |
| `van-der-waals` | 104.1% | 0.52% | 0.90% | 1.04% | 0.59% | 0.23% |

**Sampling exactly on the dielectric boundary is ~100% wrong for every surface
model, and that is not a solver defect.** φ is continuous there but ∂φ/∂n is not
— ε∂φ/∂n is the conserved quantity, so at ε_s/ε_p ≈ 78.5 the gradient jumps by
nearly two orders of magnitude, and trilinear interpolation across that kink is
O(1) wrong by construction. It is an ill-posed question, not a target for debye
to hit. This is the problem the interface-method literature exists for — the
Immersed Interface Method and Matched Interface and Boundary — and **MIBPB, which
§10 already names as the referee tier, is the PB solver built on it**. Those two
paragraphs were written years apart in this document and are about the same
thing.

**One grid cell out, the sharp boundary is ten times better than the smoothed
one** — 0.34% against 3.36% — so the default this phase just moved improves
precisely the quantity protean and mcpymol display, which nobody had measured.

And the field is where the touchstone stops being decisive. Worst error over
r/a ∈ [1.05, 2.0] at 0.25 Å:

| | `molecular` | `van-der-waals` |
|---|---|---|
| APBS | 0.87% | **1.04%** |
| DelPhi C++ | 0.75% | 0.75% |

Near-peers, where on the *energy* DelPhi is four thousand times sharper: its
advantage is the corrected reaction field, not the grid potential. So a field
gate is not a restatement of an energy gate — it is an independent axis, and the
one debye's actual purpose lives on.

**Read the `van-der-waals` column before setting a bar there**, because M1
climbs that surface and APBS does not clear 1% on it. The bar is still 1%:
DelPhi manages 0.75% on both surfaces, so it is achievable rather than
aspirational, and it sits deliberately above one of the two reference solvers on
the surface M1 uses. That is a statement about interface handling — the thing
§10's referee tier exists for — and it should be stated rather than quietly
widened to 1.25% so that everything passes.

*Flagged 2026-08-14, not re-measured:* the APBS-versus-DelPhi field numbers in
this table were each taken on that backend's own lattice, which is the comparison
M1b's correction shows can swing by 5–21× on grid phase alone. The 1% bar is
unaffected — it is a per-backend tolerance, not a ratio — but the 0.87/0.75 and
1.04/0.75 *gaps* between the two codes are not evidence of anything until they
are retaken on a shared lattice. Do not build an argument on them.

**The cases and checks to add**, all on sharp boundaries, all naming their
surface model explicitly per the manifest rule:

| arm | cases | why |
|---|---|---|
| Born radius | *a* ∈ {1, 2, 4, 6} on `molecular` | the functional form has to agree, not one point |
| Born charge | −1e, +2e on `molecular` | q² scaling, and the sign path |
| Born dielectric | ε_p = 4 on `molecular` (ε_p = 2 exists) | the 1/ε_p − 1/ε_s arm |
| Born convergence | `molecular` and `van-der-waals` at 0.25 Å | M1's *monotonic* claim needs a pair per surface |
| Kirkwood | d/a ∈ {0.3, 0.5, 0.7} on `molecular` at 0.25 Å | M2's rungs |
| Kirkwood, recorded not gated | d/a = 0.9 on `molecular` | documents where it gives up |
| Salt | *I* ∈ {0.15, 0.5} on `molecular` | M3, and no closed form, for the reason `born-ion-salt` states |
| **Field, against the closed form** | Born φ at r/a ∈ {1.05, 1.1, 1.25, 1.5, 2.0}, `molecular` and `van-der-waals` | the axis above; the quantity the consumer reads |

Fifteen cases plus the field check — one fewer than the first draft of this
plan, which also spent a case on a `van-der-waals` Kirkwood. That budget buys the
field axis instead, on the evidence in the table above: another sphere geometry
re-measures what the existing rungs already measure, where the field is
unmeasured entirely.

**The sampling rule has to be part of the corpus, not left to each caller, and
it has to be in grid cells rather than in fractions of *a*.** "At the surface" is
not a well-posed grid question. The obvious rule — sample at 1.05a — is wrong,
and wrong for the radii this plan adds: the margin it leaves is 0.05a, which has
to beat the spacing, and at small *a* it does not. Computed with `size_grid` at
padding 10 Å and 0.25 Å:

| a | dime | h (Å) | cell containing 1.05a | straddles r = a? |
|---|---|---|---|---|
| 1 | 97 | 0.2292 | [0.9167, 1.1458] | **yes — the sample is inside it** |
| 2 | 97 | 0.2500 | [2.0000, 2.2500] | **corner exactly on the interface** |
| 3 | 129 | 0.2031 | [3.0469, 3.2500] | no, by 0.047 Å |
| 4 | 129 | 0.2188 | [4.1562, 4.3750] | no |
| 6 | 129 | 0.2500 | [6.2500, 6.5000] | no |

The Born radius arm is *a* ∈ {1, 2, 4, 6}. At *a* = 1 the sample sits inside the
straddling cell — the O(1)-wrong configuration described three paragraphs above —
and at *a* = 2 a stencil corner lands on the boundary. The field numbers in this
section were all taken at *a* = 3, the one radius where 1.05a happens to clear,
and it clears by 0.23 h of grid alignment rather than by construction.

**So the rule is r = a + k·h on the achieved spacing, k ≥ 2**, which puts the
whole interpolation cell outside the interface for every radius. It follows that
the gate numbers above are measured at the wrong positions for the rule and get
**re-measured at r = a + k·h when the cases are built** — they are quoted here as
evidence that a field check is worth having, not as the tolerances themselves.
A rule chosen per test is how two checks come to disagree about what they
measured; a rule chosen per *radius* is how one silently measures nothing.

Cost, from the pilot: a sphere is ~0.4 s of APBS at 0.5 Å and ~3.5 s at 0.25 Å,
plus ~0.3 s of DelPhi. Nine cases sit at 0.5 Å (the radius, charge, dielectric
and salt arms) and six at 0.25 Å (the two convergence cases and the four
Kirkwood), so the addition is **~25 s of APBS and ~5 s of DelPhi** — tiers
assigned from measured cost, as §7 requires, which should put the nine in `fast`
and the six in `standard`. The field check re-reads a map a case already solved,
so it costs nothing beyond what is already paid.

**One existing test has to be revisited, and it is a genuine finding rather than
a chore.** `test_a_lone_sphere_has_the_same_molecular_and_van_der_waals_boundary`
asserts `born-ion-molecular` and `born-ion-vdw` agree *exactly*, on the sound
argument that a probe cannot carve a re-entrant surface out of one sphere. That
is true of the physics and true of the recordings at 0.5 Å — and **false at
0.25 Å**, where APBS gives 0.6212% and 0.7866% for the same boundary, because
`srad 0` and `srad 1.4` build different dielectric maps before they describe the
same surface. The exact-equality assertion is a coarse-grid coincidence. It
should keep its exact form at 0.5 Å and gain a fine pair that states the
numerical truth instead of extending the coincidence.

**The guard this step owes.** The gap was invisible because nothing asserted the
closed-form set spans more than one surface model — 15 of 18 on a single
APBS-only boundary passed every check the corpus has. A test that the Born *and*
Kirkwood families are each represented on a boundary every backend can build is
what would have caught it, and it is what stops the next widening drifting back.

*Exit criterion:* the Born and Kirkwood families each have closed-form cases on
`molecular` and `van-der-waals` with per-case tolerances measured on this
hardware; **the corpus checks a potential against a closed form and not only an
energy**, on the `a + k·h` sampling rule above; APBS and DelPhi C++ have recorded
every one; `AnalyticReference` can hold a per-backend tolerance and does for the
M1/M2 gate cases; the two code changes below are made rather than described; and
the number of cases a sharp-boundary solver can be verified against goes from
**22 of 64 to 37 of 79**, with the reachable fast tier from 8 to 17.

**Two code changes this needs, which "documented" does not cover.**

*GB's case set is derived, not curated.* `tests/test_corpus_gb.py` builds it as
every `MANIFEST` case whose surface model is `MOLECULAR`, so the planned
Kirkwood-on-molecular cases **enrol GB automatically**, and
`test_every_shared_case_has_a_recorded_deviation` — which asserts the recorded
deviations and the derived case set are the same list — fails until the predicate
gains an exclusion. Worse than the 187% suggests: GB reports
`n_radii_substituted: 2` on a Kirkwood structure, meaning it replaced *both*
radii from its name-keyed table and answered about a different molecule
altogether — the 3 Å dielectric sphere became a 1.62 Å atom and the zero-radius
charge a 0.79 Å one. The Born cases are untouched (`0/1`, radius 3.0 → 2.91 by
the mbondi offset), so those recordings stand. **The exclusion must therefore be
by property, not by name**: a case whose radii *are* the geometry cannot be
answered by a backend that substitutes radii, and `n_radii_substituted == 0` is
the assertion that says so for every case present and future. Fourth appearance
of "assume every new consumer wants a different dialect of the same file".

*"Recorded, not gated" has no mechanism today.* `_verify_analytic` returns early
only when `case.analytic is None`, so **any** `AnalyticReference` is gated, and
the only way to record d/a = 0.9 without gating it is an `rtol` slack enough to
absorb 9.85% and 26.67% — which is `kirkwood-09`'s existing `rtol=0.12`, and is
exactly the check that cannot fail this section condemns four paragraphs earlier.
`AnalyticReference` needs an explicit record-only flag so the summary carries the
closed form and the deviation while `verify_case` declines to judge it, and the
reason travels with the case rather than living in a tolerance nobody can read as
deliberate.

### M1 — the solver exists ✅

**Landed 2026-08-13.** `sashimi.debye` is ~900 lines of numpy across seven
modules, no binary, no compiled extension, nothing to install. **The Born ion is
0.853% from the closed form at 0.25 Å** against M1's 1% bar, and the error falls
at every refinement: **3.836% / 1.576% / 0.853% / 0.479%** at 1.0 / 0.5 / 0.25 /
0.125 Å. Both gate cases carry a `debye` entry in `per_backend_rtol` —
`born-ion-vdw` at 0.032, twice its measurement, and `born-ion-vdw-fine` at
**0.01, which is the milestone's number and not twice anything**, because a
milestone tolerance derived from what the solver already does is a milestone that
cannot be failed.

What it discretizes: a finite-volume flux balance with the dielectric on the
faces between nodes, cloud-in-cell charge assignment, multiple Debye-Hückel on
the box face, and the Boltzmann term as a diagonal — written and tested at M1
though every case that grades it is at zero salt, because a solver that only
works without salt is not a Poisson-Boltzmann solver and would not say so.
The energy is two solves differenced, solvated minus uniform-dielectric, which
is APBS's construction and is why the reported term is `POLAR_SOLVATION`. The
linear system is multigrid-preconditioned CG: V(2,2), red-black Gauss-Seidel,
and each level's coefficients **re-discretized from the geometry** rather than
coarsened algebraically, which `coarsen` licenses by preserving the box exactly.

**debye and DelPhi C++ produce the same field, and that is the strongest result
here.** Same Born case, same 105³ grid, sampled along four directions at three
radii — every pair agrees to within 0.002 percentage points (+0.738% against
+0.736% on axis, −1.889% against −1.890% on the body diagonal). Two
implementations sharing no code, one of them a C++ program from Clemson, landing
on the same numbers is a much stronger statement than either being within 1% of
a closed form. Note what it is *not*: on the **energy** DelPhi is 0.0006% where
debye is 0.853%, because DelPhi reports a corrected reaction field rather than a
grid self-energy difference. debye matches DelPhi's discretization and APBS's
energy construction, and the two facts are independent.

**Three findings, in descending order of how much they change the plan.**

**1. The corpus's field check samples +x only, and which direction is worst
depends on the backend.** Measured on `born-ion-vdw-fine` at k = 2:

| direction | APBS (h=0.203) | DelPhi C++ (h=0.25) | debye (h=0.25) |
|---|---|---|---|
| +x, +y, +z | **+1.019%** | +0.736% | +0.738% |
| ⟨110⟩ | −1.049% | −0.936% | −0.935% |
| ⟨111⟩ | −0.674% | **−1.890%** | **−1.889%** |
| worst | 1.049% | 1.890% | 1.889% |

`_analytic_field_summary` builds its sample points as `centre + r·x̂`. For APBS
that happens to be the worst direction and the recorded 1.019% is honest. For
DelPhi it is the *best* one, and the recorded 0.736% understates its true worst
by a factor of 2.6. This matters beyond bookkeeping: **M1b's 1% bar is justified
in this document by "DelPhi C++ manages 0.75%, so the bar is achievable", and
that 0.75% is an axis-only number.** No shipped solver clears 1% at k = 2 in the
worst direction. The staircase error of a sphere on a Cartesian grid is not
spherically symmetric, and a single ray cannot see that. The fix belongs to the
corpus rather than to debye — sample the ⟨111⟩ and ⟨110⟩ directions too and
re-measure all eight field tolerances for APBS and DelPhi — and it has to land
**before M1b**, because grading a new solver on the most favourable ray is the
shape §7 keeps finding. It is a corpus result change, so it gets its own step.

**2. A restriction operator off by a constant still restricts, and every energy
was right while it did.** debye's first working version had the textbook
full-weighting stencil, which carries a half per axis because the textbook
operator is a differenced Laplacian whose residual is pointwise. A finite-volume
residual is an integral over a control volume and eight fine cells make one
coarse cell, so the restriction must be exactly the transpose of prolongation.
With the extra 1/8 every V-cycle applied an eighth of the coarse correction it
should. **CG on top still converged, to energies identical in all four recorded
decimals**; the only symptom was the cycle count climbing with the grid — 20, 33,
55 against 8, 8, 9 — and a solve 6.8× slower. It was found by noticing that the
*uniform-dielectric* solve, a plain Poisson problem, was taking 53 cycles. Two
guards now hold it, and reinstating the bug reddens exactly those two while all
thirty answer-comparison tests stay green: an exact-transpose identity, and an
assertion that the iteration count does not grow with refinement. **The general
form is worth carrying: when a defect is invisible in the answer, the assertion
has to be on the mechanism, and "it converged to the right number" is not one.**

**3. `test_protocol_boundary.py` did not hold the line §10 and §12 say it holds.**
Both sections state it is what stops debye reaching into `sashimi.apbs` for a
shortcut. Every test in it was parametrized over the four protocol modules, so
the claim was about a module nothing checked — true the moment debye existed,
and false until then in a way no run could reveal. It now covers every
`sashimi/debye/*.py` for both rules, imports and APBS vocabulary.

**Measured on real structures, because spheres are not where the bugs are.**
ALA-GLY on `peptide-vdw` (20 atoms, 0.15 M salt, ε_p = 2, a non-cubic grid, and
two atoms of radius 0.6 Å) is **0.409% from APBS**; barnase (1,730 atoms) is
**1.10% from APBS** in 8.7 s. Both inside the 1.0–1.6% the reference families
differ by. Speed at the gate case is **1.8 s against APBS's 3.5 s** on a 105³
grid, which is not a performance claim — that is M7 — but does say the pure-Python
solver is not in a different class.

**Not registered in `sashimi.backends`, deliberately — until M5.** Registry
integration waited on `test_every_backend_can_answer_the_default_surface_model`:
every registered backend must answer `molecular`, which debye could not until
M4. Registering it earlier would have made a defaulted `sashimi_solve` fail on
an unremarkable request, and narrowing the default to accommodate a half-built
backend is exactly the argument that guard exists to force. *M4 gave debye the
molecular surface and M5 registered it the same day; the guard released it
rather than being edited around, which is the outcome it was written for.*

**One caveat to carry into M3.** debye takes κ from `sashimi.analytic`'s
`debye_length_a`, which is also what the screened Born closed form uses. That is
the right kind of reuse — one place for the physics — but it means a corpus check
of debye's screening against that closed form would share a definition and be
partly circular. Cross-backend agreement with APBS, whose κ is computed inside
its own C, is the check that is not. *Acted on: M3's gate has both halves, at the
same bar, for exactly this reason — and the caveat was worth writing down, since
the closed-form half alone reads 0.10% and would have looked like enough.*

### M1a — the field check sees more than one ray ✅

**Landed 2026-08-14.** `_analytic_field_summary` sampled `centre + r·x̂` and
nothing else. For a spherically symmetric problem that reads as an
arbitrary-but-harmless choice, and it is not: **the solution is spherically
symmetric and the error is not.** A sphere on a Cartesian grid is a staircase,
and a staircase has the grid's cubic symmetry, so the discretization error
varies over the three direction classes — ⟨100⟩, ⟨110⟩, ⟨111⟩.

The check now samples eight directions covering all three classes, two of them
sign-flipped. Every field recording was re-measured for both reference
backends. Worst relative error, +x alone against all eight:

| case | APBS +x | APBS all | worst dir | DelPhi +x | DelPhi all | worst dir |
|---|---|---|---|---|---|---|
| `born-ion-molecular` | 3.186% | 3.186% | −x | 0.150% | **0.789%** | ⟨011⟩ |
| `born-ion-vdw` | 3.186% | 3.186% | −x | 0.150% | **0.789%** | ⟨011⟩ |
| `born-ion-molecular-r1` | 1.165% | 1.165% | +y | 1.727% | 1.727% | −x |
| `born-ion-molecular-r2` | 4.744% | 4.744% | −x | 1.556% | **2.521%** | ⟨111⟩ |
| `born-ion-molecular-r4` | 1.412% | 1.412% | −x | 1.126% | 1.239% | ⟨111⟩ |
| `born-ion-molecular-r6` | 0.899% | **1.531%** | ⟨111⟩ | 0.722% | **1.902%** | ⟨111⟩ |
| `born-ion-molecular-fine` | 0.827% | **1.107%** | ⟨111⟩ | 0.736% | **1.891%** | ⟨111⟩ |
| `born-ion-vdw-fine` | 1.019% | 1.050% | ⟨011⟩ | 0.736% | **1.891%** | ⟨111⟩ |

**Which ray is worst depends on the backend, which is why one ray could not be
right for both.** APBS is worst along the axes on every coarse case; DelPhi C++
is worst along the body diagonal on five of eight. So +x recorded APBS's true
worst case and understated DelPhi's by up to 5.3× — `born-ion-molecular`, where
DelPhi's recorded field error was 0.150% and is 0.789%.

Three tolerances rose because of it: `born-ion-molecular-r6` from 0.020 to
0.038, and both fine cases from 0.017/0.021 to 0.038. **The old 0.020 on `r6`
would now fail**, which is the honest form of this change — it was passing
because the check was not looking where the error was. Four tolerances tightened
slightly, since 2× a measurement that did not move is a marginally smaller
number than the one recorded before. `born-ion-molecular`, `born-ion-vdw` and
`born-ion-molecular-r2` gained or kept a `delphicpp` entry, now at the values
the diagonals actually show.

**Only the `analytic_field` block moved.** Sixteen recordings changed, and a
key-by-key diff against `HEAD` confirms `energy_kj_mol`, `geometry`, `probes`
and `resolved_parameters` are byte-identical in all sixteen — this is a change
to what the corpus *looks at*, not to what any solver computes. All 79 cases
still reproduce against APBS, all 35 against DelPhi C++, all 29 against GB and
all 6 against TABI-PB.

**The grid-centring invariant fell out of it and is now asserted.** For a lone
sphere the atom sits at the centre of its own bounding box and every backend
builds an odd-sized grid, so the atom lands exactly on a node and the four
⟨100⟩ samples must agree. They do, to under 0.001 percentage points. A grid
centred half a cell off would break that while leaving the *worst* error — and
so every tolerance — untouched, so nothing else here would have seen it. The
test solves with debye rather than reading a file, which means it runs on the
bare CI leg where the recordings' own backends do not exist.

*Two notes for whoever reads this next.* The one-ray rule is the **second**
parameter this sampling rule turned out to be silently conditioned on: M0 caught
it on *radius* — `1.05a` was verified at a = 3 Å and fails at 1 and 2 Å — and
fixing that instance felt like closing the class, so nobody asked what else the
rule assumed. And `sashimi corpus build --backend delphi` does **not** imply
`tests/corpus/delphi/`; without `--directory` it overwrites the APBS recordings
with DelPhi's numbers, which is what happened here and was caught only because
the recordings were diffed key-by-key before being trusted.

### M1b — grading debye against the incumbents, and what it found

**The bar, decided 2026-08-14: debye within 2× the best reference-tier solver
installed, at each sample.** Not a round number, and the reason is the one §7
keeps rediscovering. debye reproduces DelPhi C++'s discretization to three
decimal places, so any bar of the form "no worse than the *worst* incumbent" is
one it satisfies by construction rather than by merit — on the fine case it
would pass by 0.002 percentage points. Against the *best* incumbent it is a real
measurement: agreeing with DelPhi does not put a solver within a factor of APBS.

**A precondition nobody had noticed: cross-backend field numbers were not
comparable at all.** The corpus samples `a + k·h` on each backend's *own*
achieved spacing, which is correct there — every sample must clear *its*
interface cell — and means a coarser grid is sampled further from the boundary,
where the error is smaller. On `born-ion-vdw` that put DelPhi's samples at
r = 4.0 Å and APBS's at r = 3.81 Å. Comparing those reads a sampling difference
as an accuracy difference. `sashimi.validate.grade_field` therefore takes its
radii from the **coarsest** grid in the comparison, so every backend clears its
own interface cell and none is handed an easier question. *This was half the
precondition and was taken for the whole of it — see the correction below.*

**Two cases were added to make the gate mean anything.** M1b rested on the two
van der Waals field cases, both at a = 3 Å, so `born-ion-vdw-r1` and
`born-ion-vdw-r6` now carry the extremes of the radius arm onto the surface debye
can build. The corpus was 81 cases here; M2 took it to 85.

**The first measurement was wrong, and the correction is below.** It reported
debye at 5.24× the best reference on `born-ion-vdw` and 8.64× on
`born-ion-vdw-r1`, concluded that debye's near-interface handling fell behind
DelPhi's wherever the sphere was under-resolved, and pointed the next milestone
at §10's referee tier and the Matched Interface and Boundary literature. It
landed as two passing tests and two **strict** xfails.

### M1b, corrected — the gap was grid phase, and M1b is met

**Equal sample radii were necessary and not sufficient.** The comparison put
every backend at the same physical radii but left each on the lattice its own
grid sizing chose: it graded debye at a/h = 6.46 against DelPhi at a/h = 6.00.
That is not a small residual difference. Holding the sample radius fixed at
r = 4 Å on the 3 Å sphere and varying **only** the spacing across 0.43–0.50 Å,
the worst-direction error swings

| backend | span over grid phase, `born-ion-vdw` | `born-ion-vdw-r1` |
|---|---|---|
| APBS | 0.585 – 3.915% (6.7×) | 0.357 – 7.516% (**21.1×**) |
| DelPhi C++ | 0.763 – 3.837% (5.0×) | 1.586 – 9.931% (6.3×) |
| debye | 0.773 – 4.101% (5.3×) | 1.214 – 9.912% (8.2×) |

The error collapses wherever a/h nears an integer. The discretized cavity is a
staircase and its shape changes **discretely** as face centres cross the sphere,
so at low a/h a single face flipping is a large fraction of the boundary. Every
finite-difference backend here does this: it is a property of hard midpoint
dielectric assignment, not of debye. DelPhi's apparent 0.789% at a/h = 6 was
DelPhi landing on h = 0.5 exactly, where a = 3 Å is six cells and the sample at
r = 4 Å is a grid node.

**At the spacings two backends both land on, debye is at parity with both:**

| | `born-ion-vdw` | `born-ion-vdw-r1` |
|---|---|---|
| debye vs DelPhi C++ | 0.994–1.013× over 11 shared spacings | 1.000–1.062× over 13 |
| debye vs APBS | 0.871–1.116× over 16 shared spacings | 1.037–1.617× over 18 |

**Regraded on one lattice, M1b is met on all four cases** — the two xfails turned
red, which is what strict xfails are for, though not for the reason they were
written:

| case | a/h | h | ratios | as first measured |
|---|---|---|---|---|
| `born-ion-vdw-r6` | 12 | 0.50 Å | 1.00 / 1.01 / 1.01× | 1.01 / 1.02 / 1.03× |
| `born-ion-vdw-fine` | 12 | 0.25 Å | 1.00 / 1.01 / 1.00× | 1.77 / 1.00 / 1.01× |
| `born-ion-vdw` | 6 | 0.50 Å | **1.00 / 1.05 / 1.01×** | 5.24 / 4.81 / 3.74× |
| `born-ion-vdw-r1` | 2 | 0.50 Å | **1.69 / 1.19 / 1.04×** | 8.64 / 3.61 / 1.55× |

Even `fine`, which passed, was being compared across lattices — APBS at
h = 0.2031 Å against debye's 0.25 Å — and its 1.77× was the same artifact
passing rather than failing.

**Three controls, because the claim is that a recorded measurement was wrong.**
Reading grid **nodes** directly, with no interpolation anywhere, shows the same
swing (−0.019% at h = 0.5 → +4.089% at h = 0.473), so it is the solver's field
and not `sashimi.field`'s sampling. The error is anisotropic across the cubic
direction classes — positive on ⟨100⟩, negative on ⟨110⟩ and ⟨111⟩ — which is the
staircase's signature and not a monopole defect. And at fixed h = 0.5 Å, changing
the padding from 3 to 13 Å moves debye's error only 0.7735% → 0.7916%, so pinning
a padding to obtain a common lattice is not what produces the result.

**`grade_field` now refuses a cross-lattice comparison outright**
(`sashimi.validate.check_same_lattice`, which checks spacing *and* the solute's
fractional offset within a cell — same h with the solute half a cell over is a
different staircase). Refused rather than annotated: the per-backend spacings
were already reported in `notes`, they were visible, and the verdict was wrong
anyway. `FieldGrade` now carries `cells_across_radius`, because a field verdict
that does not state its a/h is not reproducible.

**What this costs §10's referee tier.** Nothing yet, but the argument for it is
now unmade. MIBPB may still be worth having, and interface methods are still the
literature for the O(1) error *at* the boundary that M0 recorded — but "DelPhi
holds up where debye does not" was the evidence for prioritising it, and that
evidence has evaporated. Do not treat the third arrival at interface methods as
independent confirmation; it was the same measurement read twice.

**The real finding underneath, which is new and unrecorded elsewhere: every
shipped FD solver's near field swings 5–21× with grid phase.** That is far larger
than any difference between the solvers, and it means **a field tolerance is only
meaningful with a/h pinned**. The corpus's twenty closed-form field cases each
measured their tolerance at whatever lattice that case's `GridSpec` happens to
produce, so those tolerances silently inherit their case's phase — they are not
wrong, but they are not transferable, and a case whose padding or resolution ever
changes should expect its field tolerance to move by much more than the edit
looks like it should cost. Flagged here rather than re-measured; the recordings
are untouched by this milestone.

**And a second free variable, found by review after the lattice was pinned: the
box.** Pinning a padding to obtain a common lattice chose, for three of the four
cases, a box whose face was too near the *outermost* sample — and the boundary
condition lives on that face. Measured at fixed lattice, worst-direction error at
r = a + 8h against the margin in units of that radius:

| margin / r_out | APBS | DelPhi C++ | debye |
|---|---|---|---|
| 0.14 | **0.637%** | 0.118% | 0.111% |
| 0.60 | 0.301–0.413% | 0.233–0.370% | 0.234–0.380% |
| ≥ 1.29 | 0.119–0.396% | converged | converged |

**Only APBS moves** — consistent with mg-auto's focusing carrying its coarse
grid's boundary treatment inward — and APBS is a *reference*. An inflated
reference raises the yardstick, and the verdict is candidate ÷ reference, so the
contamination flatters debye. Same direction as the other two.

`check_samples_clear_the_box` now refuses it, and the paddings moved to the
smallest common lattice that clears the margin. The ratios in the table above are
the corrected ones; the contaminated paddings read 1.01 / 1.10 / **0.94**× on
`born-ion-vdw`, and that 0.94× — debye apparently *beating* the best reference —
was the contamination showing.

**Worth recording how this was missed, because the control looked sufficient.**
Box-size independence *was* checked, at the innermost sample (r = a + 2h), where
padding moves the answer by 2%. The contamination is at the outermost (r = a +
8h), where it moves it by 5.4×. **A control has to be evaluated where the effect
would be, not where the measurement is most convenient** — which is §7's "ask
what the rule is conditioned on, then evaluate at the extremes", arrived at from
yet another direction.

*This is §7's class again, and two new members of it: a gate that pinned one
variable and left a stronger one free — twice.* Item 9 in that list was a verdict
that moved with the *installed set*; these are verdicts that moved with the
*lattice each backend happened to choose* and with the *box that lattice implied*.
All three moved in the incumbents' favour, and that is not coincidence: the
incumbents' defaults are what the cases were built around, so an uncontrolled
variable tends to sit where it suits them. The lesson that generalises: when a
comparison controls for a variable, ask what else varies that the controlled
variable was standing in for — and when you find one, **enumerate the rest rather
than pinning the one that just bit**.

### M1c — the dielectric spike, as planned

> **This section is the plan; the spike then ran and the answer was no.** Read
> "M1c — the spike ran" below for the result and for what happened to M4a. Kept
> because the reasoning it records is what a future attempt would otherwise
> reconstruct — and because the argument below is a good one that measurement
> overturned, which is the point.

**Split decided 2026-08-14 by Charlie: spike now, implement after M4.**

The phase oscillation is a property of **hard midpoint dielectric assignment** —
ε is ε_p or ε_s at each face centre, so the discretized cavity changes shape in
steps as face centres cross the surface. A dielectric that varied *continuously*
with h would not, and from M1 until M1c `sashimi/debye/dielectric.py` named
volume-fraction averaging as "the obvious place to look". This is the first
measurement that makes a case for paying for it, and it is also the reason not to
pay for all of it yet: the case is a strong physical argument, not a result.

**M1c is half a day and buys a number.** The cheapest scheme that varies
continuously is a smoothed Heaviside on signed distance — for a union of spheres
the distance is a small change to the window loop `inside_union_of_spheres`
already runs, then ε = ε_p + (ε_s − ε_p)·H(d/w) with w ≈ h. Roughly 40 lines and
no new data structures. Rerun the phase sweep from the section above and read off
whether debye's 5.3× drops. **The exit criterion is the number, not an
improvement** — "it does not help" is a result that closes a question which has
been sitting open in a docstring since M1.

**Why the full implementation is two to three days, not half of one.** Real
area-fraction averaging needs the fraction of a face lying inside a *union* of
spheres, which has no closed form, so it is quadrature: an n×n subsample per
face, union-correct because each subsample tests against every nearby sphere. Then
arithmetic versus harmonic averaging is a **measurement** rather than a choice,
which doubles the experiment count. The implementation trap is cost, and it is a
trap this repo has already fallen into once: `_solute_mask` took 64 s on a
1,960-atom protein by evaluating every atom against the whole grid (§7), and a
naive per-face quadrature walks straight back into it.

**The risk to watch, because it is quiet.** Born energy goes as 1/a, so a 1%
shift in the *effective* cavity radius is a 1% shift in energy — and M1's
headline is 0.853% against a 1% bar. Any smoothing not symmetric about the true
surface moves the effective radius and can spend that whole margin without
anything looking wrong. M1c should therefore report the **energy** alongside the
oscillation, not just the field.

**Why it waits for M4.** M4 rewrites what counts as solute — a probe rolled over
the union of spheres — so an averaging scheme built now against a union of
spheres is partly rework. That risk is mostly avoidable by construction, and the
avoidance is the design constraint: **write the averaging against an
inside/distance oracle rather than against spheres**, so M4 swaps the oracle and
keeps the averaging. Building it once, after the surface debye ships exists, is
what the ordering buys.

**What is being given up by waiting**, stated so it is a decision and not a
drift: the measurement harness is warm now, the M1b gate is phase-fair as of this
milestone and is exactly the instrument this change needs grading against, and
debye has **no corpus recordings** — so today the change moves none, where after
M5 the same change moves every debye case and needs a `BACKEND_VERSION` bump.
Against that, nothing is blocked: M1b is met, and the oscillation hits the
incumbents equally, so debye is not disadvantaged by it. It is the one place
debye could be *better* than APBS and DelPhi rather than at parity — every other
rung, M2 through M4, is debye catching up to what they already do.

### M1c — the spike ran, and the answer is no

**Result, 2026-08-14. A smoothed dielectric is not worth it, M4a is dropped, and
the face-centre sample stays.** The exit criterion was a number either way; this
is the "either way", and it saves the two to three days M4a was scoped at.

The spike put a smoothed Heaviside on a signed-distance field —
`min_i(|x - c_i| - r_i)`, saturated outside a band of `w` cells — behind a
default-off option, and measured both axes M1c required.

**Axis 1, the phase oscillation it was aimed at. It improves, but far less than
the summary statistic says.** Worst-direction error at r = 4 Å on `born-ion-vdw`,
swept over paddings 3–11 Å. *The shipped row reads 4.138% where M1b's table above
reads 4.101% for the same case and quantity, and the reason is the sweep range:
M1b swept paddings **2–9 Å** and peaked at h = 0.46354, this swept **3–11 Å** and
picked up padding 10 → h = 0.46429, just past the peak. Neither range contains
the other, both are debye at its worst over what was swept, and the two maxima
are 0.9% apart on the shoulder of the same peak.*

| w (cells) | blend | best h | worst h | swing |
|---|---|---|---|---|
| 0 (shipped) | — | 0.773% | **4.138%** | 5.35× |
| 0.5 | arithmetic | 1.681% | 3.221% | 1.92× |
| 0.5 | harmonic | 1.737% | 4.044% | 2.33× |
| 1.0 | harmonic | 1.975% | **3.085%** | 1.56× |

**The swing ratio falls mostly because the floor rises.** The quantity a consumer
actually sees is the worst case, and the best any variant does there is
4.138% → 3.085% — 25%, against a metric that reads as a 3.4× improvement.
Harmonic at w = 0.5 barely moves it at all (4.044%) while still reporting a 2.33×
swing. *A summary statistic that improves while the thing it summarises does not*
is §7's class wearing yet another costume, and it nearly bought a three-day build
on its own.

**Axis 2, the energy — where the arithmetic blend dies and the harmonic one
looked like a triumph.** Arithmetically blended, `0.853% → 3.545%` at 0.25 Å,
through M1's 1% bar at every width tried. Harmonically blended — the textbook
mean for flux normal to a layered interface — signed error against Born:

| h | hard (w=0) | harmonic w=1 |
|---|---|---|
| 1.000 | −3.836% | −1.209% |
| 0.500 | −1.576% | −0.352% |
| 0.250 | −0.853% | **−0.107%** |
| 0.200 | −0.820% | **−0.064%** |

Eight times better at M1's gate, monotonic, no sign flips, consistent across the
whole ladder. On the strength of that it should ship.

**The first draft of this section said it should not, on real-structure evidence
that does not hold up. `/code-review` caught it; the retraction is below, and it
is the most useful thing here.**

What was measured on real structures was the deviation from **APBS** — ALA-GLY
(20 atoms) −0.409% → +3.565% and barnase (1,730 atoms) −1.102% → +5.153% at
w = 1, both at the corpus request of 0.5 Å with 10 Å padding. Note that neither
backend is even on one lattice there, let alone on each other's: debye achieves
[0.467, 0.439, 0.458] on ALA-GLY where APBS achieves [0.467, 0.439, 0.400], and
the grids are anisotropic. M1b's `check_same_lattice` refuses exactly this
comparison for the field; nothing enforces it for an energy, and an energy is far
less lattice-sensitive — but it is one more reason these numbers cannot carry the
weight the first draft put on them. That was read as "harmonic is worse on real
geometry", and **it does not support that reading, because APBS is not an
independent reference for this question.** `SurfaceModel.VAN_DER_WAALS` maps to
APBS's `srfm mol` with `srad 0` (`apbs/options.py:19-26`) — the *same* hard
midpoint dielectric assignment debye's default uses, as this document's own
`dielectric.py` docstring has said since M1. So "hard agrees with APBS, harmonic
does not" is in part agreement about a **shared discretization bias**, and a
verdict whose reference shares the incumbent's bias is exactly the class M1b was
corrected for one milestone earlier.

**The self-contained test, which needs no external reference.** Solve ALA-GLY down
a refinement ladder under each scheme and compare where they are heading and how
fast they get there. Energies in kJ/mol, tabulated against the **achieved**
max-axis spacing — the quantity `check_same_lattice` returns — with
`max_points` raised from its 161³ default so the fine end is a refinement point:

| scheme | 0.4672 | 0.3905 | 0.2929 | 0.2492 | 0.1967 | 0.1699 | 0.1495 | 0.1289 | fitted L | A |
|---|---|---|---|---|---|---|---|---|---|---|
| hard | −227.82 | −224.20 | −222.39 | −220.74 | −220.61 | −220.27 | −220.02 | −219.20 | −216.84 | **−19.7** |
| harmonic w=0.5 | −218.16 | −217.62 | −216.92 | −216.81 | −216.75 | −216.83 | −216.86 | −216.86 | −217.09 | 1.67 |
| harmonic w=1.0 | −218.80 | −217.97 | −216.93 | −216.70 | −216.41 | −216.42 | −216.44 | −216.46 | −216.54 | 0.70 |

*The first version of this table was wrong twice and a review caught both: it
regressed on the **requested** resolution rather than the achieved spacing —
`size_grid` turns a 0.5 Å request into `[0.4672, 0.4393, 0.4576]` — and its two
finest points were identical not through "lattice quantisation" but because
`GridSpec.max_points` defaults to 4,173,281 = 161³ and `grid.py`'s step-down loop
clamped both to it. Regressing on the wrong abscissa against a clamp artefact is
how it read a stalled gap.*

**Corrected, the three schemes agree.** The extrapolated limits sit **0.25%
apart** (0.549 kJ/mol), and the gap is clean `O(h)` — gap/h is flat at −16 to −21
across the whole ladder — so they do share a continuum limit, exactly as "the
band is `w·h`, so it vanishes" predicts. The earlier "stalls near 3.5 kJ" was the
clamp: the two finest points were one grid, so the gap could not move.

**And harmonic is far nearer that shared limit at every usable spacing.** Against
L ≈ −216.8: at h = 0.25 hard is 1.8% out where harmonic is 0.00–0.05%; at
h = 0.13 hard is still **1.1%** out where harmonic is 0.03–0.16%. The fitted
slopes say the same — hard is at −19.7 and still moving fast at the finest point,
harmonic at 0.70–1.67 and essentially arrived by h ≈ 0.29. Hard's fit also
carries 10–20× harmonic's residual, which is itself the signature of a scheme not
yet in its asymptotic regime.

**So the honest position on real geometry is: the admissible evidence points
toward harmonic, not away.** A reference-free convergence study says it reaches
the common answer several times sooner, and the one exact reference (Born) says
it is 8× better. The APBS comparison that said otherwise is disqualified. Nothing
measured here shows harmonic to be worse on a union of spheres.

**What would settle it properly**, for whoever picks this up: a reference that
discretizes no volumetric dielectric — TABI-PB is the boundary-element backend
already in the tree, and a Kirkwood case brings a closed form to an off-centre
charge. *Note the finer grids above needed only `GridSpec(max_points=...)`; an
earlier draft of this section blamed the `n = m·8 + 1` lattice, which is wrong —
the lattice produces those spacings happily, and the cap is one keyword away.*

*The lesson I drew first — "grade a dielectric change on a real structure before
the closed form" — was the right instinct and the wrong conclusion, because the
real-structure grade I reached for was against a solver making the same choice
under test. **Check what your reference is made of before you let it overturn a
closed form.***

**What this does to M4a: dropped, and on one ground only.**

**Axis 1 is the whole of it.** M4a existed to damp the grid-phase oscillation,
and the most favourable variant moved the worst-case near-field error only
4.138% → 3.085%. That number is graded against the exact Born potential, so it is
admissible in a way the real-structure energies are not. M4a was scoped at two to
three days to buy about a quarter in the quantity a consumer sees, on an axis
where debye is **already at parity with both incumbents** (M1b, above). Charlie's
framing made M4a conditional on M1c; a 25% improvement in the target quantity
does not clear that bar.

**Everything else here argues the other way, and none of it is a reason to keep
M4a as written.** Do not read "M4a dropped" as "smoothed dielectrics are bad":

- Area-fraction averaging — M4a's actual proposal — was **never tested**. The
  tempting shortcut, "a `w = 0.5` band has one cell of support, the same as an
  area-fraction average, and that failed", is wrong twice over: the real-geometry
  failure it appeals to has been retracted above, and the supports are not
  equivalent anyway — for an interface parallel to a face the inside-area
  fraction jumps 0→1 with *zero* support along the normal, where a band smooths
  over `w·h`.
- The energy result points the opposite way and is **left open deliberately**:
  harmonic blending made debye's Born energy 8× more accurate (0.853% → 0.107% at
  0.25 Å), monotonic, no sign flips, across the whole ladder. That is not what
  M4a was for and it is not blocked by dropping M4a — see "the open lead" below.

Numbers a future attempt has to beat: **3.085%** worst-case near field, and
**−0.107%** Born energy at 0.25 Å.

*The knob landed on 2026-08-22 as `DebyeOptions.dielectric_smoothing`, but it is
not this variant — the shipped ramp is linear in the union's own signed gap and
harmonic-only — so those numbers still need a recipe rather than a knob setting.
A generator for it belongs in `studies/`, which is exempt from the `print` rule
`tests/` carries.* Replace
`sashimi.debye.linear.dielectric_faces` (patch it **there**, not on
`sashimi.debye.dielectric`, which `linear` imported from directly) with one that
computes, per face-centre axis, `d = min_i(|x − c_i| − r_i)` saturated at `+band`
with `band = w·h` — **an approximate signed distance, and the approximation is
worth keeping in view**: it is the exact exterior distance to each sphere
separately, but near a concave junction between two spheres it is not the
distance to their union. That caveat was written down in the first draft, then
deleted along with the APBS-based claim it happened to sit next to; it never
depended on APBS and remains a live candidate explanation for any residual
hard-versus-harmonic difference on real geometry — a solvent fraction
`f = clip(½(1 + t + sin(πt)/π), 0, 1)` at `t = d/band` — antisymmetric, so the
half-way dielectric sits on the surface — and then either
`ε = ε_p + f(ε_s − ε_p)` (arithmetic) or `1/ε = (1−f)/ε_p + f/ε_s` (harmonic).
Grade with `solute_dielectric=1.0, ionic_strength=0.0`: the Born expression is
unscreened, and taking `SolventModel`'s defaults instead reads as a 48% solver
error, which is the "closed form describing different physics" trap
`validate.grade_field` refuses on the field axis.

### The open lead M1c turned up: harmonic averaging and debye's energy

> **This lead was taken.** Read Q0/Q1 and M8a below for what it became, and
> "The referee gap" and the ramp sections for where it ended up. The knob is
> `DebyeOptions.dielectric_smoothing`, shipped 2026-08-22 and off by default.
> The section is kept because the reasoning in it — an exact reference, a
> reference-free ladder, and a disqualified APBS comparison — is the reasoning
> the later work turned out to be right about.

~~Filed rather than pursued, because it is a different question from the one M1c
was asked and it should not be smuggled in under a dropped milestone.~~
**Pursued, at M8.**

Harmonically blending the dielectric over a band made the Born solvation energy
**8× more accurate** — the one measurement in this milestone with an exact
reference and no methodological objection against it. The reference-free
refinement study above then agreed independently: all three schemes extrapolate
to within 0.25% of one common limit, and harmonic arrives there several times
sooner (fitted slope 0.70–1.67 against hard's −19.7, and still 1.1% out at
h = 0.13 Å where harmonic is inside 0.16%). Two independent lines now point the
same way, which is more than the retracted claim ever had against them.

**Before anyone acts on this**, the reference problem has to be solved first —
TABI-PB, a Kirkwood closed form, or a genuinely converged grid. Then the field
axis has to be re-checked, because harmonic at w = 0.5 barely improves the worst
near-field error (4.044% against 4.138%) even while it transforms the energy, and
debye's consumer reads the *field*. An energy-only win is worth having and is not
worth regressing M1b for.

~~**The knob is not landed.** A default-off parameter that a measurement rejected
is the same shape as the `relaxation` knob `debye/options.py` refuses to carry.~~

**Landed 2026-08-22 as `DebyeOptions.dielectric_smoothing`, off by default**, and
the analogy to `relaxation` does not hold: `relaxation` changed no answer, where
this changes every answer it is switched on for. What keeps it a knob is the
**field axis** — which the paragraph above correctly names as the precondition,
and which was measured on 2026-08-25 and did not go the ramp's way — together
with coverage of the molecular surface, whose entire evidence is two tests on
one 20-atom dipeptide. Not the energy verdict, which reversed the other way.

### M2 — the off-centre charge, and the four cases that had to exist first

**Met 2026-08-14. debye reads 1.047 / 1.254 / 1.328% against the Kirkwood series
at d/a = 0.3 / 0.5 / 0.7, against a 1.5% bar.**

**The blocker, and it is M0's own lesson one surface along.** Every Kirkwood rung
in the corpus was on `smoothed-molecular` or `molecular`; debye's
`SUPPORTED_SURFACES` is `van-der-waals` alone. So M2's exit criterion named cases
debye refuses **by name** — it was unreachable by construction, exactly as M2 was
unreachable before M0 for the same reason at one surface less. M0 had considered
a van der Waals Kirkwood and dropped it deliberately, on the reasoning that
"another sphere geometry re-measures what the existing rungs already measure".
That is true of APBS and DelPhi, which build both surfaces, and false of the one
solver M2 exists to grade. **The class worth carrying: a case added for coverage
of the incumbents is not automatically coverage of the candidate.**

`kirkwood-vdw-{03,05,07}` are added gated and `kirkwood-vdw-09` recorded-not-
gated, reusing the existing PQRs. The corpus is **85 cases**; APBS records all
85, DelPhi C++ 41.

**The relabel is not an identity, which is worth knowing before assuming it.** For
the *Born* geometry it is — `born-ion-molecular` and `born-ion-vdw` record
−233.9996297277 to the last digit, because the solvent-excluded surface of an
isolated convex sphere is that sphere. Add Kirkwood's zero-radius charge atom and
APBS's two surfaces separate by ~0.24% (at d/a = 0.3, −253.191 against −253.792)
while DelPhi's stay bit-identical. So the new cases carry their own measured
tolerances rather than inheriting the molecular twins'.

| | APBS | DelPhi C++ | debye | shared rtol | debye's bar |
|---|---|---|---|---|---|
| d/a = 0.3 | 1.083% | 0.097% | **1.047%** | 2.2% | 1.5% |
| d/a = 0.5 | 1.239% | 0.205% | **1.254%** | 2.5% | 1.5% |
| d/a = 0.7 | 3.896% | 0.416% | **1.328%** | 8% | 1.5% |
| d/a = 0.9 | 9.854% | 4.288% | 8.280% | ungated | — |

**The bar is 1.5% flat, decided by Charlie over the shared tolerance**, and the
reason is §7's. debye reproduces APBS's discretization, and APBS is what sets the
shared number, so grading debye there is a bar it meets by construction. 1.5% is
fixed independently of what debye does — it would have failed had debye read 1.6%
at d/a = 0.7 — and it is **stricter than APBS manages on that rung**, which is
what makes M2 a claim rather than a formality. `test_debye_m2.py` asserts that
relation directly, so the bar cannot quietly become the loose one.

**The finding, recorded and deliberately not gated: at d/a ≥ 0.5 on a sharp
boundary, nothing converges monotonically.** M1 required the Born error to fall
at every refinement step. Across 1.0 / 0.5 / 0.35 / 0.25 / 0.2 Å at d/a = 0.7:

| backend | 1.0 | 0.5 | 0.35 | 0.25 | 0.2 |
|---|---|---|---|---|---|
| APBS | −38.311 | −2.136 | −2.713 | −3.896 | −0.381 |
| DelPhi C++ | −4.718 | −0.493 | −1.093 | −0.416 | −0.371 |
| debye | +0.726 | −28.171 | −6.211 | −1.328 | −1.893 |

**None of the three is monotonic**, so requiring it would be the mirror image of
a check that cannot fail — a check that cannot *pass*. §12 already made this call
at d/a = 0.9; the measurement extends it to 0.5 and 0.7, which nobody had run.
The single-resolution numbers M2 gates on are therefore accuracy claims and not
convergence claims, and the distinction is stated rather than blurred. It also
means debye's tidy 1.328% at d/a = 0.7 is partly the resolution it was asked for:
it reads −1.893% at 0.2 Å. The control that made "record, do not gate" the right
call rather than the convenient one is the two incumbent rows above — running
them is what turned "debye has a problem here" into "this geometry does".

### M3 — salt screening, and the gate that had to be a difference

**Met 2026-08-15. debye's ionic contribution reads +0.10% and +0.14% against the
screened Born closed form at 0.15 M and 0.5 M, and 0.13% and 0.22% against APBS,
against a 2% bar decided by Charlie.**

**Salt was already running and had never been graded.** `peptide-vdw` takes
`SolventModel`'s 0.150 M default, so M1's "0.409% from APBS on ALA-GLY" was a
salted solve — the Boltzmann term has been exercised since M1 and no case
measured it. `screening_nodes` was written at M1 rather than at M3 precisely so
this milestone would not be a rewrite, and it was not: **M3 changed no solver
code at all.** What it added is cases, a closed form, and a gate.

**The arm, on the van der Waals sphere (a = 3 Å, q = +1e, ε_p = 1), as
G(I) − G(0) in kJ/mol:**

| I (M) | closed form | debye | APBS | DelPhi C++ |
|---|---|---|---|---|
| 0.05 | −0.4753 | −0.4756 | −0.4761 | −0.2479 |
| 0.15 | −0.6880 | **−0.6887** | −0.6878 | −0.4958 |
| 0.50 | −0.9507 | **−0.9521** | −0.9501 | −0.8428 |
| 1.00 | −1.0997 | −1.1015 | −1.0990 | −1.0412 |

Identical to four digits at 0.25 Å. So §7's "two codes disagree 39% and pinning
either would encode a convention as physics" needs one refinement, and only one:
**the disagreement is DelPhi's alone.** Two codes sharing no source land on the
linearized Debye–Hückel expression to better than 0.2%, which is evidence about
which convention the analytic model describes rather than a preference between
two. The decision for `born-ion-salt` is unchanged — it is `smoothed-molecular`,
where see below.

**Why the gate is a difference between two recordings and not a number.** The
ionic contribution is 0.3% of the solvation energy where discretization is 1.6%.
Four mutations of debye's screening, each judged against the shipped
convention's closed form:

| mutation | ΔΔG error | **total** vs closed form |
|---|---|---|
| pristine | +0.10% | −1.571% |
| Boltzmann term deleted outright | −28.8% | −1.484% |
| Stern layer at the probe (1.4 Å) | +4.86% | −1.586% |
| no Stern layer at all | +18.2% | −1.626% |
| κ where κ² belongs | +63.4% | −1.761% |

**Every mutation leaves the total inside a band APBS alone needs 2.4% for.** An
`AnalyticReference` on a salted case would therefore pass a solver with no
mobile-ion term at all — the purest check that cannot fail the corpus could have
acquired, caught before being built rather than after. The salted cases carry an
`analytic_field` and no `analytic`, and `test_a_total_energy_check_could_not_see_the_salt`
holds that reasoning where a comment would rot.

**The bar is 2% on both halves, and the second half is why there are two.**
debye takes κ from `sashimi.analytic.debye_length_a`, which is also what the
closed form uses — the circularity M1 flagged in advance. APBS computes κ inside
its own C, so the cross-backend half is the one that closes it. 2% is twenty
times what debye reads and still sits below the *subtlest* mutation above, the
probe-radius Stern layer at 4.86%; `test_the_bar_rejects_a_stern_layer_at_the_probe_radius`
runs that mutation rather than citing it, which is the guards file's "land the
mutation with the assertion" applied at last.

**Two controls, because M1b's correction says to run them before believing a
number.** Padding 5 → 30 Å moves ΔΔG by 0.13% in total, so the gate is not
grading the box — worth checking, because deleting the interior Boltzmann term
still leaves 89.5% of the answer at padding 5 and 33.7% at padding 30, the
Debye–Hückel Dirichlet data on the box face supplying the rest. The two paths
sum correctly at every padding. And no shared lattice is pinned, unlike M1b:
ΔΔG moves under 1% across 0.5 / 0.35 / 0.25 / 0.2 Å, where the near field moves
5–21× on grid phase alone. The ionic term is simply not that quantity.

**Four cases, and it is the third milestone to pay for the same class.** The
corpus's salt arm was `born-ion-molecular-salt` and `-high-salt`, on a surface
debye refuses by name — exactly as M0 found for the closed forms and M2 for
Kirkwood. `born-ion-vdw-salt` and `born-ion-vdw-high-salt` are their siblings,
and `peptide-vdw-no-salt` / `-high-salt` complete a real-structure arm around the
`peptide-vdw` that was already salted. The corpus is **89 cases** at M3 — M4 takes it to 98; APBS records
all 89, DelPhi C++ 46.

**The closed form is new, and it is the third in the project.**
`screened_born_potential` is the salted sphere's *potential*: Poisson between the
dielectric boundary and the Stern radius `a + r_ion`, linearized PB beyond it,
matched in φ and ε ∂φ/∂n. `AnalyticField.exact_at` previously refused a salted
case by name; it now describes one, and reduces to `born_potential` exactly at
zero salt, so all ten pre-existing field recordings are byte-identical. Making
the reference salt-aware rather than guarding against salt is the guards file's
first lesson — an illegal state made unrepresentable — and it removes a refusal
that was reachable only by writing a case the manifest then had to remember not
to write.

Two things fell out of building it:

- **`exact_at` was defaulting the temperature rather than reading it**, the same
  defect as the `solvent_dielectric` one it was written to fix, one parameter
  along. Invisible: all ten field cases sit at 298.15 K, so the recordings do not
  move. It is worth naming because `peptide-cold` exists precisely to catch a
  *solver* reading a temperature in the wrong unit, and a reference that ignored
  temperature was the mirror image of that trap.
- **A sample may land on the Stern radius**, where one on the dielectric boundary
  is O(1) wrong for every solver — ε is equal on both sides, so φ and φ′ are
  continuous and only φ″ jumps. `born-ion-vdw-salt` puts DelPhi's **second**
  sample exactly there (r = 5.0 Å on its h = 0.5 lattice). Measured, for scale:
  matching φ alone and dropping the flux condition puts a **63.6%** step at that
  radius.

**The field, and the finding that the relative numbers hide.** debye's worst
error two cells out reads 4.47% at zero salt and 7.61% at 0.5 M, which looks like
screening being harder to resolve. It is not. In **absolute** terms, on one
lattice at one radius, it is 0.0812, 0.0805 and 0.0798 kT/e — the same
discretization error, and the screening adds none of its own. What changed is
the potential being divided by. The agreement loosens with distance (−6.6% and
−14.3% eight cells out, on errors a tenth the size), so the claim is gated only
where the error lives. *Watch for this shape: it is M1c's "a summary statistic
improving while the quantity it summarises does not", running the other way.*

**A docstring claim that was surface-specific and read as general.**
`screened_born_solvation_energy` said the ionic term "carries grid noise of order
10% even where the energies themselves have converged", from APBS reading
−0.688 / −0.777 / −0.694 across 0.5 / 0.25 / 0.125 Å. Measured on all three
surfaces at 0.15 M against an exact −0.6880:

| surface | 0.5 Å | 0.25 Å | 0.125 Å | swing |
|---|---|---|---|---|
| `molecular` | −0.6878 | −0.7766 | −0.7155 | 12.9% |
| `smoothed-molecular` | −0.6880 | −0.7087 | −0.7520 | 9.3% |
| `van-der-waals` | −0.6878 | −0.6878 | −0.6879 | **0.015%** |

The noise is a property of the **probe-based surfaces**, not of the quantity —
and the surface debye builds is the clean one. Offered as a suggestion rather
than a measurement: with a probe the dielectric boundary and the ion-exclusion
boundary are two differently-constructed surfaces whose discretizations move
apart with h, where at `srad 0` both are bare staircases that scale together.
This is also the honest reading of why the corpus declined a closed form for
`born-ion-salt`: that case is `smoothed-molecular`, where APBS swings 9%.

**Recorded and deliberately not gated: the ionic term stops being a monopole.**
Walking from the sphere to ALA-GLY one variable at a time, debye against APBS at
0.15 M:

| system | debye/APBS |
|---|---|
| 1 sphere (ε_p = 1, ε_p = 2, r = 0.6 Å) | 1.001 / 1.001 / 1.000 |
| 2 spheres +1/+1, 20 Å apart, then 4 Å apart | 1.014 / 1.004 |
| acetate, net −1 | 1.002 |
| **2 spheres +1/−1, net zero** | **0.900** |
| **ALA-GLY, net zero** | **0.922** |

**Every net-charged solute agrees to 1.4%; both net-neutral ones disagree by
8–10%**, stable across 0.5 / 0.35 / 0.25 / 0.2 Å, so it is a convention
difference and not grid noise. Once the monopole vanishes the ionic term is
dipole screening, an order of magnitude smaller (−0.196 against acetate's
−0.742), and the three reference codes spread over 22% — DelPhi −0.174, debye
−0.196, APBS −0.212 — with no closed form in reach. So M3 gates the monopole and
records this, the same call M2 made for its non-monotonic rungs, and the control
that made it the honest option rather than the convenient one is the sphere row:
debye is not adrift, the geometry class is. Barnase, net +2 across 1,730 atoms,
sits between at 0.966 — which is what "monopole dominance" rather than "sign of
the charge" predicts.

### M4 — the solvent-excluded surface, and the criterion that was wrong

**Met 2026-08-15.** `sashimi/debye/surface.py` builds the classical reduced
surface analytically: a probe resting on one atom (radial projection), wedged
between two (the rim where their accessible spheres meet), or seated against
three (trilateration). Each family produces an *actual* legal probe centre
checked against the neighbours, so each can only say "solvent" correctly, and
together they are exhaustive — the nearest point of the accessible set to any
node is on one of the three. **No sample count and no tuning constant.** Two
wrong constructions came first and both are recorded in the module docstring,
because both look reasonable: dilating the set of legal *grid nodes* inflates
the 3 Å Born ion's effective radius 3.0097 → 3.0717 and doubles ALA-GLY's probe
worth, and sampling candidate centres over each accessible sphere makes the
sample count the answer (+3.19% → −1.40% across 32 → 1024 samples, no plateau,
crossing the reference at 256).

**The stated criterion was wrong, and this is the retraction.** "Inside the 2.3%
band APBS and DelPhi already occupy" traced to a passing remark about pyDelPhi
in §7 rather than to a measurement; measured across the 33 shared `molecular`
cases the band is **0.41% to 5.74%**. Charlie's replacement, 2026-08-15: gate
**the probe's worth**, `(E_molecular − E_vdw)/|E_vdw|` — the one quantity the
surface alone decides — with debye **no further from APBS than DelPhi C++ is**.
Relational, so it carries no constant and cannot be met by drifting toward
either incumbent. That shape matters here more than usual: on the Kirkwood
geometry, where the exact answer is that the two surfaces coincide, **APBS
separates them by 0.09–0.33% while DelPhi C++ stays bit-identical** — APBS
over-fills its own SES, so "closer to APBS" is not "more correct" and a bar of
the form "within x% of APBS" would grade debye against a known bias.

**Nine cases had to exist first — the fourth milestone in a row to find its
criterion unstateable.** Every closed-form case in the corpus is blind to this
by construction (a lone convex sphere's SES *is* that sphere), so a solver
answering `molecular` by returning its van der Waals number passes all eighteen
exactly. ALA-GLY was the only multi-atom structure carrying both surfaces, and
the probe is worth 4% there against 17–36% on a protein — so the peptide was not
standing in for protein scale, it was a different question. Corpus 89 → 98.

**Measured on every structure to hand, one grid spec, all three reference-tier
backends:**

| structure | atoms | DelPhi C++ | debye | APBS | dby−APBS | dlp−APBS |
|---|---|---|---|---|---|---|
| fas2 | 906 | +16.331 | +17.647 | +18.502 | **0.855** | 2.171 |
| barstar | 1,403 | +18.096 | +19.693 | +20.500 | **0.807** | 2.404 |
| fkbp-apo | 1,663 | +24.731 | +26.942 | +27.479 | **0.537** | 2.748 |
| fkbp-dmso | 1,673 | +25.149 | +27.388 | +27.874 | **0.486** | 2.725 |
| barnase | 1,730 | +26.567 | +28.037 | +29.458 | **1.421** | 2.891 |
| lysozyme-asp66 | 1,960 | +33.211 | +35.323 | +35.761 | **0.438** | 2.550 |
| lysozyme-ash66 | 1,961 | +31.668 | +33.819 | +34.117 | **0.298** | 2.449 |
| protein-1a63 | 2,065 | +16.977 | +17.784 | +18.430 | **0.646** | 1.453 |
| hca | 2,482 | +24.611 | +26.082 | +26.787 | **0.705** | 2.176 |
| hca-complex | 2,500 | +25.502 | +27.077 | +27.859 | **0.782** | 2.357 |
| actin-monomer | 5,877 | +27.425 | +29.330 | +30.345 | **1.015** | 2.920 |
| mache | 8,279 | +27.583 | +30.243 | +31.124 | **0.881** | 3.541 |

**12 of 12, and debye is strictly between the incumbents every time**, 2–8×
closer to APBS than DelPhi is — across an apo/holo pair, a ligand complex and a
protonation-state pair, so it is not a single-structure
coincidence. Above ~5,000 atoms `max_points` relaxes the lattice and the three
codes relax differently, so those two rows compare across lattices.

**The probe's worth climbs 4–8× from peptide to protein in all three codes** —
buried volume grows faster than surface — so debye's +17.65% on fas2 is physics
rather than a defect. It was nearly filed as one.

**And it is converged at protein scale, which is the control the small molecules
fail.** fas2 down 0.7 / 0.6 / 0.5 / 0.4 / 0.35 Å with `max_points` raised so no
rung is silently relaxed: APBS 19.075 → 18.159 (swing 0.916), DelPhi 16.046 →
16.697 (0.651), **debye 17.925 → 17.439 (0.486, the tightest of the three)**, and
debye sits between the incumbents at *every* rung. The 2.2-point APBS/DelPhi gap
is a real disagreement between the incumbents, not lattice noise.

**Not below ~900 atoms, and that is measured.** At 0.5 Å the same comparison
*fails* on acetic-acid, acetate and `ion-protein-complex`. It is a resolution
artifact and the incumbent rows are what prove it: APBS reads **identically at
0.7 and 0.5 Å** (its `dime` steps in 32s, so both relax to one lattice) and then
moves acetic-acid by 2 points at 0.35; debye's 0.5 Å row is an outlier against
its own 0.35 and 0.25 values; by 0.25 Å all three agree. The probe's worth is a
*difference* of two energies, so where the difference is ~0.5% —
`ion-protein-complex`, 260 atoms and almost no re-entrant volume — it sits under
the 0.5–0.9 point lattice swing rather than one to two orders above it. Gating
that grades the lattice. Recorded, as M2 did for Kirkwood's ninth rung and M3
for the net-neutral solute.

**The construction was unusable and is not any more.** As first written, fas2
spent 71 s per lattice and barnase never finished. It is now **34.00 s for a fas2
molecular solve against 1160.23 s**, and barnase completes in 79.80 s — with
every dielectric mask bit-identical and every energy identical to the last digit,
verified by hashing the mask on all three staggered lattices before any change.
Five exact changes, none a tolerance: the seats are geometry rather than
discretization and are built once for all lattices (52.2 → 1.2 s); 59% of rims
are swallowed whole by a third atom and a circle's farthest point from a sphere
centre is a closed form, so the prune is exact; the same closed form read at the
*nearest* point gives each rim the only atoms that can reject a centre on it;
`_legal` became one broadcast instead of a numpy call per neighbour per feature;
and the homogeneous reference state, whose dielectric map is constant by
construction, no longer builds a surface it throws away. `ReducedSurface` is what
carries the geometry across the three staggered lattices and every multigrid
level. **This is not M7** — it is making a construction usable at all, where M7
is the benchmark claim against the incumbents.

### M5 — the registry, and what a partial-coverage backend needed from it

**Met 2026-08-15.** Registering debye was two lines — a `BackendEntry` and a
report — and `--backend`, `sashimi_solve` and `sashimi_capabilities` all reached
it with no edit, which is §2's claim about the registry cashed rather than
asserted. What took the work was that **debye is the first backend that refuses
part of the corpus on principle rather than through absence**, and the tooling
had no vocabulary for that.

**Three supporting changes, each fixing something that was already wrong.**

- **`corpus build --backend delphi` was filing DelPhi's answers as APBS's.**
  `summary_path` ignored `--backend` entirely and defaulted to the corpus root,
  so the per-backend layout lived only in the heads of the people typing
  `--directory`. It failed safe by accident — printing `skip (exists)` wherever
  APBS had already recorded — and would have written a wrong file on any case
  APBS had not. `corpus_dir_for` derives it now, which is why adding a fifth
  backend needed no new incantation.
- **A refusal crashed the build.** The loop had no `except`, so recording a
  partial-coverage backend meant naming its cases by hand and the first refusal
  killed the run. That is why the GB and TABI-PB tiers were recorded case by
  case. A refusal is a result: it is reported as `n/a` and counted.
- **A refusal counted as a discrepancy.** `verify` reported every missing
  recording as a MISS, so a backend that declines two surface models by design
  was permanently red and the stated exit criterion was unreachable. It now
  asks the backend's own published `BackendReport` — the same `surface_models`
  every caller sees, not a second copy — and falls through to MISS when the
  backend is unavailable, since an undiscoverable DelPhi reports no models at
  all and calling those refusals would turn "not installed" into "supports
  nothing".

**One measured tolerance, and it is the first that *widens*.**
`born-ion-molecular-r4`: APBS relaxes the request to 0.4375 Å where debye solves
the 0.5 Å it was asked for, and 1.504% at a/h = 8 sits exactly on M1's ladder
(1.576% at a/h = 6). DelPhi's 0.007% is not a third opinion — it reports a
*corrected reaction field*, and on the same 0.5 Å lattice its field error is
1.239% against debye's 1.236%. **The discretizations agree; the energy
definitions do not.** The `born-ion-*-r1` field bars widen for the other
recorded reason: at a/h = 2 the three codes land on three lattices (0.344 /
0.500 / 0.458 Å) and near-field error swings 5–21× with phase, which §12 already
flags as inherited by every corpus field tolerance. The phase-fair measurement
is M1b's, on one shared lattice, and it is unchanged.

That widening broke `test_the_debye_tolerances_are_actually_reaching_debye`,
which asserted `rtol_for(label) < rtol` — and the assertion was subtly wrong all
along. The property being guarded is that the key *matches*, not that it
tightens; it only looked like the latter because every per-backend tolerance so
far happened to tighten. It now compares against the declared value, and asserts
separately that the milestone bars are tight.

**What it buys immediately.** With nothing installed at all, `gb` and debye now
share `molecular`, so `comparable_surface_models()` is no longer empty on a bare
machine: cross-validation works with no APBS, no DelPhi and no TABI-PB. They are
not two of a kind — debye discretizes the equation and `gb` approximates it — so
`validate` reports a reference answer with a deviation beside it rather than a
spread, which is exactly what `AccuracyTier` was built to keep separate.

| | milestone | exit criterion |
|---|---|---|
| M0 ✅ | **The closed-form gap closed** | the section above — sharp-boundary Born and Kirkwood cases exist to be graded against, the field is checked against a closed form, and the GB-exclusion and record-only changes are made rather than described |
| M1 ✅ | LPBE on a Cartesian grid, vdW surface | **met**: 0.853% at 0.25 Å against a 1% per-backend tolerance, falling monotonically 3.836 → 1.576 → 0.853 → 0.479% under refinement |
| M1a ✅ | **The field check has to see more than one ray** | done — the section below: eight directions across the three cubic symmetry classes, all sixteen field recordings re-measured, and the tolerances that moved moved because the diagonal is worse than the axis |
| M1b ✅ | **The field, graded against the incumbents** | debye within **2× the best reference-tier solver installed**, at radii common to every backend **and on one shared lattice**. Decided 2026-08-14 by Charlie, over a round number: debye reproduces DelPhi C++'s discretization to three decimals, so "no worse than the worst incumbent" is a bar it meets by construction — a check that cannot fail. **Met on all four cases: 1.01 / 1.01 / 1.05 / 1.69× worst at a/h = 12, 12, 6, 2.** The first measurement reported 5.24× and 8.64× on the two under-resolved cases; that was grid phase, not interface handling, and the section above is the correction |
| M1c ✅ | **The dielectric spike** | **ran; M4a is dropped.** A smoothed dielectric moves the *worst* near-field error only 4.138% → 3.085% — the swing ratio flatters it — which does not justify M4a's two to three days on an axis where debye is already at parity. Separately it made the Born energy **8× better**, left open rather than acted on: the real-structure check that appeared to contradict it used APBS, which shares the hard assignment under test, and a reference-free convergence study since points the other way. ~~The knob is not landed~~ **The knob landed at M8 (2026-08-22) as `DebyeOptions.dielectric_smoothing`, off by default; the prose above this table struck the same sentence and this row was missed** |
| M2 ✅ | Off-centre charge | **met**: 1.047 / 1.254 / 1.328% against the Kirkwood series at d/a = 0.3 / 0.5 / 0.7, against a **1.5%** bar set independently of debye and stricter than APBS manages at the hardest rung. Needed four new `van-der-waals` Kirkwood cases first — every existing rung was on a surface debye refuses. **Not d/a = 0.9**, which no shipped solver reproduces; and at d/a ≥ 0.5 *nothing* converges monotonically, so M2 gates accuracy and records convergence |
| M3 ✅ | Salt screening | **met**: debye's ionic contribution `G(I) − G(0)` is +0.10% / +0.14% from the screened Born closed form at 0.15 / 0.5 M and 0.13% / 0.22% from APBS, against a **2%** bar on both halves — two halves because debye shares κ with the closed form and APBS does not. Gated as a *difference* rather than a total, because measurement showed a total-energy check cannot see the salt at all: four mutations of the screening, including deleting the Boltzmann term, all leave the total inside the 2.4% band APBS needs. Needed two new `van-der-waals` salt cases first — the third milestone to find its criterion named cases debye refuses by name. **Not the net-neutral solute**, where the monopole vanishes and the three reference codes spread 22% with no closed form: recorded, as M2 did for its non-monotonic rungs |
| M3a ✅ | **The ion-exclusion region's geometry** | **met 2026-08-28.** M4 made the Stern layer the solvent-excluded volume dilated by `ion_radius`, computed as a lattice dilation. That construction quantises its reach: `ball_offsets` keeps displacements no longer than `ion_radius`, so the *effective* reach is the largest one the lattice admits, and the accuracy is decided by commensurability rather than resolution. Measured across **nineteen achieved spacings between 0.82 and 1.00 A on the Born ion, the dilated path lands outside 2% on twelve and the union path on none** — 0.8929 A and 0.9000 A, eight parts in a thousand apart, read −0.920% and −3.660%, because 2.0 A is representable on one lattice and not the other. It also degraded to the identity once `ion_radius < min(spacing)`, and `build_levels` re-discretizes per level, so **the coarse levels carried no Stern layer at all.** *The fix is not a new construction but a theorem:* for `r ≥ probe`, `dilate(SES, r) = dilate(vdw, r)` exactly — any point of `SES \ vdw` lies within `probe` of the union, else a probe would fit there and it would be solvent, and a 1-Lipschitz step closes it — **so above the probe the exact union test is correct for `molecular` too**, and every shipped case sits at 2.0 against 1.4. *The hypothesis is used once and is sharp, which is what makes the guard a guard:* below the probe the union leaves **13.5% of fas2's solute nodes carrying bulk screening inside the dielectric body**, so `ion_radius < surface_radius` keeps the dilation — falling back rather than raising, because `ion_radius = 0` is the standard no-Stern-layer request and `dilate` serves it exactly. *Gated on a difference, as M3 was and for M3's reason:* on a lone sphere the two surface models describe one region, so `molecular` must read what `van-der-waals` reads. It now does **to the last digit**, where it read 0.73% and 1.07% against 0.10% and 0.14% before — a **0.25%** bar, which fails the old construction by 4.3× and passes the new one with 1.8× to spare, where M3's own 2% would have graded nothing. **M3's bar never covered this branch**: `test_debye_m3.py` applies it to `born-ion-vdw-*`, and `SurfaceModel.MOLECULAR` reached `dielectric.py` in #42 where the bar landed in #41. *Two claims of `screening_nodes`'s docstring died here: that the constructions "coincide exactly on van-der-waals" (true in the continuum, false for the lattice dilation on 18.8% of `ala-gly`'s excluded nodes at 0.87 A), and that `tests/test_debye_m4.py` asserts it — that file never calls `screening_nodes`.* **16 salted `molecular` recordings move; the 35 salted cases on other models cannot, because `smoothed-molecular` and `van-der-waals` already took the exact test.** §12 carries the sweep |
| M4 ✅ | Solvent-excluded surface | **met**: the probe's worth, `(E_molecular − E_vdw)/\|E_vdw\|`, with debye **no further from APBS than DelPhi C++ is** — relational, so it carries no constant and cannot be met by drifting toward either incumbent. **Passes on all twelve real structures measured, 906 → 8,279 atoms**, and debye lands *strictly between* the two incumbents on every one, 2–8× closer to APBS. The original criterion here — "inside the 2.3% band APBS and DelPhi already occupy" — was **wrong and is retracted**: that 2.3% traced to a passing remark about pyDelPhi in §7, not to a measurement, and across the 33 shared `molecular` cases the band is 0.41%–5.74%. Needed nine new cases first, the fourth milestone in a row to find its criterion unstateable by the corpus it had — every closed-form case is blind to the probe, because a lone convex sphere's SES *is* that sphere. **Not below ~900 atoms**, where the probe is worth less than the lattice swing around it: recorded, as M2 and M3 each did |
| ~~M4a~~ | ~~**Fractional-volume dielectric**~~ | **dropped by M1c, on cost/benefit rather than infeasibility.** True area-fraction averaging was *not* tested and neither of M1c's failure mechanisms would apply to it. What carries is that the goal is worth less than scoped: the most favourable variant moved the worst-case near-field error only 4.138% → 3.085%, where debye is already at parity with both incumbents. Numbers to beat if revived: 3.085% field, −0.107% Born energy |
| M5 ✅ | Registry integration | **met**: `sashimi corpus verify --backend debye --tier fast` passes, and so does `--tier standard`. debye is in `sashimi.backends`, so `--backend`, `sashimi_solve` and `sashimi_capabilities` all reach it — that was two lines, which is §2's claim about the registry cashed. It records 23 of the 40 fast cases and 39 of 75 standard, refusing the rest **by design**: `smoothed-molecular` is APBS's harmonic averaging and `gaussian` is DelPhi's. Getting there needed three supporting changes and one measured tolerance, below |
| M6 | **Potential field out** | a DX map protean's viewer loads, *and* residue potentials on a real protein inside the cross-backend band — loadable is not the same as right, and M1b is the sphere-scale half of this claim — **the protean-replacement milestone**. **The second half is met by measurement and recorded rather than gated**, decided 2026-08-17 by Charlie: debye sits inside the band, but the band is the same width as each solver's own grid noise, so a gate there would have a 0.001 margin and would go red for reasons unrelated to debye. ~~Revisit when fractional-volume dielectric averaging damps the oscillation.~~ **The lever exists now: M8/M8a shipped the sub-cell ramp and #75/#77 made it affordable, so "revisit when the oscillation is damped" has a mechanism rather than an intention. What stands in the way is that the field axis M6 lives on is where the ramp's case is weakest, and the molecular-surface evidence is two tests on one dipeptide.** See the section below |
| M7 | Performance claim | the §11 benchmark-VM question, revisited only here. **Groundwork done 2026-08-17**: debye is *geometry*-bound, not solver-bound (86% surface classification against 11% linear solve), so the dielectric lever was the wrong one; and wall clock is not a usable instrument here — identical code varies 1.9x on load. **Planned 2026-08-18** against a direct measurement of the rim loop rather than the profile: the `decided` early-out that forces the loop to be sequential prunes only **16%**, and the batched answer must come out **bit-identical**, so the rewrite is safer than its size suggests. **Landed 2026-08-18: 1.209× on the solve**, CPU time, minimum of 3, interleaved, energies bit-identical — against a **0.993× control on identical code**. The ~4.4× projected from a microbenchmark was wrong and is retracted: it timed arithmetic on contiguous arrays where the stage is a gather, and batching every stage measured **1.000×**. What batching actually buys is the coarse multigrid levels, 3–21×; the finest level had 2% per-call overhead to recover and got 1.4× *worse* when its legality test was batched. **Parked 2026-08-18**, and the measurement that parks it is that debye had been graded at 0.5 Å where protean asks for 1.0 Å — fas2 is 7.7 s, not 21.7 s, so the gap was about three times narrower than charted. **At protean's 1.0 Å the batched query is worth 1.966× / 1.902× on fas2 / barnase**, bit-identical — where it reads 1.209× at 0.5 Å. Threading inverts the other way: 2.28× at 0.5 Å, **1.06× at 1.0 Å**, and is dropped. Each lever is worth about what the other was worth depending only on the resolution it is graded at, because coarsening moves work out of large-array numpy and into per-call overhead. See the sections below |
| M8 ✅ | **The interface, graded without a reference** | **met for `van-der-waals` 2026-08-22, and the default surface is not that.** Two halves. The *instrument*: a pose spread sees only the phase-dependent half of the discretization error, so `grade_refinement` adds the other by Richardson over h, h/2, h/4 — validated against the Born closed form at **0.08–0.48%**, with a `converging` guard that refuses a ladder reading −172.6, −177.0, −171.7 rather than fitting it. *Both halves of that sentence were corrected 2026-08-24: the extrapolation does **not** beat its finest rung every time — three of sixteen Born windows do not — and the guard was a half-guard that admitted a 269.688% error until `MIN_SHRINKAGE` and a sign test were added.* The *scheme*: a hard face-centre dielectric replaced by a solute fraction ramped across **one cell** from a signed distance — a *bound* rather than a distance on the interior until the M8a row's 2026-08-25 repair — and blended harmonically, **4.8–7.8× against the Born closed form at `w = 0.5` (3.4–52.5× at `w = 0.25`) and 3.6–5.6× on pose dispersion**, with the ramp at 0.5 Å closer to converged than the hard assignment at 0.25 Å. Shipped off by default and bit-identical when off. Three claims of this document died in the process: the fitted convergence order is a property of the ladder and not the method, `posed`'s translation moves nothing because the box follows the solute, and averaging the indicator over a band of *whole* cells is worse than not averaging at all. **The precondition M1c set — the field axis — was measured 2026-08-25 and the ramp does not win it**: on `ala-gly`, refereed at **4×** the coarse spacing, the potential 2-3 Å outside the surface is **2.4-13× further** from the referee at `w ≥ 0.75`, while the energy on the same fixture and lattice is **4.9-8.8× closer**. On a Born sphere with an exact reference the two summaries disagree — worst-direction improves 12-23%, shell RMS is flat with a doubled floor — so there is no clean gain there either. *At `w = 0.5` the verdict is bounded rather than settled, and `w = 0.25` and `fas2` are not settled at all: a referee that shares a construction with a candidate is nearest that candidate, which is the shared-bias trap caught for the first time **inside one solver**, between two settings of one knob. The axis is bounded rather than discharged, and the bound is the deliverable. **Its premise is no longer a theorem: both schemes converge to the exact Born field, measured 2026-08-26, so the comparison is bounded and not meaningless.*** The two axes disagree and debye's consumer reads the field, so **the default stays off for a measured reason rather than a coverage one** |
| M8a ✅ | **The solvent-excluded distance** | **met 2026-08-22**, and it moved the number M8 could not: the ramp raises the energy's convergence order to **2.31–2.48**, from **1.009** on molecular and from a hard van der Waals ladder that the repaired `converging` **refuses to fit at all** — an interface treatment ceasing to bound the accuracy. *Re-measured 2026-08-24: the recorded "0.32" baseline came from a ladder whose corrections shrink by 1.2031x, under `MIN_SHRINKAGE`, so it never described a fit and is withdrawn; the conclusion survives, the left-hand number does not.* `ReducedSurface.signed_gap` is `probe - dist(x, A)`, out of the same three families that decide `inside`, and `sign(gap)` reproduces `inside` node for node. Both ramp widths work on both surfaces. *Their limits agreeing to 0.14% and 0.16% was offered as evidence they are real rather than fitted and is **struck**: the width is in cells, so the band vanishes with h and the limits must agree whatever the ramp does.* *An earlier draft reported the best width as surface-dependent; that was a one-sided distance of its own making and is withdrawn below.* *And the branch that repair left alone was not a distance either: `min_i(|x - c_i| - r_i)` is an upper bound inside a union of spheres, so the `van-der-waals` ramp read a bound as a depth on **19.0%** of its interior band faces until 2026-08-25. Repaired by taking the same three families with the probe removed — a union of spheres is the solvent-excluded surface of a zero probe — which moves `fas2` by **+44.79 kJ/mol at 1.0 Å and +10.06 at 0.5 Å**, 19.8% and 7.0% of the ramp's own offset, and makes the ramp **cheaper** (2.62× → 1.28× of hard at 0.5 Å) because the clamp it needs is what lets the bound be windowed. The molecular branch had the same defect in its search *reach* — `probe` where the consumer reads to `probe + band` — worth 0.012–0.024% and free to fix. The Born ion is bit-identical across both, which is why neither was ever seen.* Pose dispersion alone would have ratified the bug, which is what Q0's other half is for |
| M9 ✅ | **A boundary that does not cost `O(nodes × atoms)`** | *Shipped 2026-08-23 — #68 the decision, #70 the implementation.* The exact multi-atom Debye-Hückel sum evaluated on a **strided sub-lattice of the box face** and interpolated up, in place of one sum per face node; pitch in ångströms, capped at 0.6× the solute's clearance. Serum albumin's boundary goes **12.94 s → 0.24 s** and the whole solve 30.05 → 17.94 s. *Named "focusing" until the step (c) review priced a coarse pre-solve at 2.0–2.5 s against the strided face's 0.24 s, at no better accuracy and for a second grid hierarchy — the milestone kept its purpose and lost its mechanism.* **Exit criterion retired 2026-08-23, and it is worth reading as a record rather than a bar.** *Met:* total CPU strictly lower than the `mdh` baseline on **every** rung of the 906→18,242 ladder, 1.18–1.74× and widening with size; and, on the 3–4 Å shell against the exact sum on the same box and lattice, **r = 1.000000, sign 100.00%, magnitude within 0.04% and energy within 0.052%** on fas2, 1a63 and serum albumin, against bars of 0.999 / 99% / 1.02× / 0.5%. *Struck, each measured — and the first one for a corrected reason:* **atoms^≤1.05**, not because it is unreachable but because **it measures the ladder as much as the solver**. Two of the nine rungs use different input conventions — `2LZT-ASP66` has a mean radius of 1.031 Å against ~1.55 elsewhere, and `hca` is 17.8% hydrogen against ~49% — and dropping them to leave seven homogeneous rungs moves the fit from 1.075 to **1.048 ± 0.026** for `sdh` and 1.084 to **1.057 ± 0.023** for the shipped solver, halving the standard error and reproducing across two independent runs. *That composition effect is 0.027, against M9's own improvement of 0.113 in the same units — a quarter of the signal, coming from the choice of structures rather than from the solver.* So the honest reading is that the floor sits **at** the bar rather than above it and the shipped solver is one standard error over — marginal, not impossible. **An earlier version of this row said the bar sat below the floor and was wrong**; it was fitted across a ladder that mixes radius sets. **fas2 within 0.5% of its TABI-PB recording**, because the whole span from the exact boundary to `sdh` is 1.090 pp where the bar needs 2.375, so it graded discretization; and **`residue_potentials` as the observable**, because `sdh` passes it on all three structures at r ≥ 0.99998 — residue pooling averages away the near-surface `l ≥ 1` error that is the whole defect. **The three struck clauses share one cause, and it is the lesson of this milestone:** each graded something other than the quantity M9 changes — the wrong axis, the wrong referee, the wrong observable — while the *thresholds* were right every time. **A gate is a threshold and an observable, and the observable is the half that failed three times.** The Born ion and `ala-gly` pose dispersion remain regression anchors and were never bars: one atom makes `sdh` and `mdh` bit-identical, and a net-neutral dipeptide makes an `sdh` boundary identically zero. §12 carries the measurements and the forensics |

**What debye inherits that did not exist before 2026-08-13:** 64 corpus cases,
18 of them with closed forms; three independent reference backends to be graded
against rather than one; an approximate tier whose deviation is documented per
case; and a suite that passes on a machine with no binaries at all — which is
the only environment in which debye's central claim can even be stated.

**Phase 9 — Integration, ongoing.** mcpymol grows a convenience chaining
`sashimi_solve` → load DX → surface coloring; protean consumes `SolveResult`.
Sashimi itself should go quiet after this — a wrapper that needs constant
attention has failed at its one job. *The measurement that decides how far the
replacement goes — whether debye supersedes protean's screened-Coulomb default
or only its APBS path — is the last section of this phase, below.*

### `sashimi validate` defaults to the fast tier

**Decided 2026-08-16 by Charlie, and taken as its own change rather than folded
into M6.** `validate` with no arguments now compares the **fast** tier — 40 of
the 98 cases — where it previously ran the whole manifest.

**The measurement that forced it, and the part that is easy to get wrong.**
`--tier full` runs **over 40 minutes without finishing** on a fully-installed
machine. Registering debye at M5 is the obvious suspect and is *not* the cause:
the four-backend set (apbs + delphi + tabipb + gb) was already past 40 minutes
before debye was registered.

*State that carefully, because the first draft of this entry did not.* It said
the four-backend measurement was taken "on `--tier standard`", which the next
bullet contradicts — pre-M5 `validate` had no `--tier` at all, and
`git show a985ec8~1:src/sashimi/cli.py` shows `_select(MANIFEST, args.case)`.
The run it describes was necessarily over the whole manifest, and the exact
invocation was not recorded at the time. What survives the correction is the
attribution, which is what the paragraph is for: the cost predates debye.

**The 40 minutes is still a floor from a run nobody let finish, so here is the
cost from below instead, which is cheap to re-take.** A *single* 906-atom case
through five backends — `sashimi validate --case fas2-molecular` — is
**105.2 s**, against 36 s for all 40 cases of the fast tier put together. `full`
adds 58 cases to that tier, among them proteins up to 8,279 atoms. **The tier
boundary is not 98/40 = 2.5× of anything; it is where the proteins start**, and
one of them costs three fast tiers. That is the whole argument, and it does not
depend on the unfinished run.

Three things compound, none of them new and none of them one backend's:

- `validate` never respected tiers at all — it iterated `MANIFEST`
  unconditionally, and `--tier` only exists because M5 added it;
- it **overrides every case's surface model** to a shared one, so it asks each
  backend all 98 cases rather than the ones that backend recorded, `mache`
  included;
- per-case cost is the *slowest* backend's, which is TABI-PB meshing every
  structure as much as it is debye solving in-process.

The new default is **36.0 / 36.6 / 37.8 s** across three runs with all five
backends installed. A review of this change measured **64.6 s** for the same
command on the same machine under concurrent load, which is the useful caveat:
the figure is load-sensitive to roughly a factor of two, and what carries the
decision is the three orders of magnitude between it and `--tier full`, not the
second digit. `--tier full` is one flag away and unchanged.

**Why `fast` and not a middle tier.** `validate` re-solves everything and has no
recordings to fall back on, which is the difference from `corpus verify`:
exhaustive protein-scale verification is the *corpus's* job, where the answers
are recorded and a diff means something. `validate` was never a substitute for
that, and a default nobody can wait out is a trap rather than a default — it
teaches callers that the tool does not work. The guard
(`test_validate_defaults_to_the_cheapest_tier`) asserts the default is the
smallest tier the corpus offers rather than naming `fast`, so a new cheaper tier
moves it and widening it again means deleting the test on purpose.

**The regression the first cut shipped, because the shape recurs.** Narrowing
the default tier also narrowed `--case`: `sashimi validate --case
lysozyme-molecular` had worked, and started exiting 1 for all 58 cases outside
`fast`. **A cost default and a reach limit are different things, and one flag
was doing both.** `_select` now takes `tier_bounds_names=False` for `validate`,
whose tier is purely a cost knob. Caught in review, not by the suite — the
original guard read `parse_args(["validate"]).tier` and never called
`_validate`, so it would also have stayed green against the pre-M5 bug of
ignoring the tier entirely. Two behavioural tests replace that reasoning: each
reddens on its own mutation, checked.

Related, and the same review: the out-of-tier hint said "pass `--tier full`"
whatever tier the case was in, so reaching one `standard` case billed the whole
98-case manifest. It names the case's own tier now. That message became common
precisely because the default dropped.

**Four things this change does not fix, all pre-existing, all measured.** They
are recorded here rather than folded in, because each changes what `validate`
*means* rather than what it costs, and the first is worth its own small PR.
**Items 1-3 were taken in that PR — see the section below; item 4 stands.**

1. **One backend refusing a case discards every other backend's answer.**
   `validate_system` raises, `_validate` catches `SashimiError` and drops the
   case whole. That is why the default compares only **13 of its 40 cases**: 27
   are TABI-PB declining a structure with fewer than four atoms — the Born ion
   and every variant — after APBS and DelPhi have already solved it. `sashimi
   validate --case methanol` shows it at its worst: four backends can answer, one
   cannot mesh three atoms, and the run prints "Nothing was comparable".
   `backends.get(name).check(system)` already answers "would this backend
   decline" for free and `_refuses` already uses it, so dropping the *backend*
   per case — skipping only if fewer than `MIN_BACKENDS` remain — would restore
   the cases and cut the wasted solves.
2. **The surface override collapses distinct cases into identical systems.** The
   40 fast cases reduce to 18 distinct `System` values once `surface_model` is
   overridden; of the 14 compared, 7 are distinct. `born-ion-coarse`,
   `born-ion-molecular` and `born-ion-vdw` are one system solved three times.
   Deduplicating helps `--tier full` far more than narrowing the default does.
3. **Rows are labelled with a case name whose defining parameter was
   overridden.** A `-vdw` row reports a molecular-surface number. The review
   illustrated this with `born-ion-vdw` against `born-ion-molecular`, and that
   is the one pair where it is harmless — M2 measured them identical to the last
   digit (−233.9996297277) because an isolated convex sphere's SES *is* that
   sphere, so the override changes nothing there. **`peptide-vdw` against
   `peptide-molecular` is the real case**: M4 measured the probe worth +5.72% on
   ALA-GLY, and `validate` prints both names against the same molecular number,
   hiding exactly the quantity M4 gates on. The header discloses the surface
   once; the rows underneath keep contradicting it.
4. **The tier is defined by APBS wall time and `validate` pays the slowest
   backend's.** `CaseTier`'s own docstring says the cost "is APBS's, which makes
   it a statement about this backend and no other", which is why
   `test_tabipb_solver.py` refuses to read a tier. So a case added to `fast`
   because APBS solves it in 0.4 s can put minutes of TABI-PB meshing into the
   default, and nothing catches it.

And `peptide-low-solvent-dielectric` still disagrees — the three
finite-difference backends land within 1.8% while TABI-PB reads 19.4% away and
`gb` 46.6% — so the default run exits non-zero on a fully-installed machine
exactly as it did before this change. Now documented in the README, since the
old default never finished and nobody reached the verdict.

### M6 — the map is loadable, and the residue axis is measured rather than gated

**2026-08-17.** M6 has two halves and they turned out to need different
treatment, which is the milestone's main result rather than an inconvenience.

#### The DX half: gated, and the circularity removed first

`sashimi.dx`, `sashimi.analysis.residue_potentials` and debye's
`want_potential` path all existed before M6 started, so this half was never a
construction. It was an *evidence* problem: the round-trip test proves our
reader accepts our writer, which says nothing about a viewer we do not control.

The non-circular version: a real **APBS 3.4.1-written `.dx`** was intercepted
before its temp directory was cleaned, read with our reader — validating the
reader against an independent producer — and that same grid written back out
with our writer. Diffed with whitespace normalised, **exactly one line of the
first eleven differed**, the comment, and ours was the only file of the two
carrying a non-ASCII byte (an em dash). Fixed and guarded in
`test_written_dx_is_pure_ascii_even_with_a_comment`; the pre-existing header
test never passed a `comment`, the only path that can introduce one.

So the claim is not "our round-trip passes" but **"our file is structurally the
file those viewers already read"**. *Not claimed: that Mol* loads it.* That
needs a viewer in the loop, which this repo cannot provide, and saying so is
the difference between M6 being met and M6 being asserted.

#### The residue half: debye is inside the band, and the band is noise

The stated criterion — "residue potentials on a real protein inside the
cross-backend band" — is **met by measurement**. It is *not* turned into a gate,
decided 2026-08-17 by Charlie on the recommendation below. All numbers on
`fas2-molecular`, 906 atoms, 63 residues:

| comparison | median | max |
|---|---|---|
| APBS against *itself*, padding 8→11 Å | 0.49 kT/e | 2.12 |
| debye against *itself*, padding 8→11 Å | 0.66 kT/e | 3.18 |
| **debye against APBS**, common lattice | **0.32 kT/e** | 3.73 |
| debye against APBS at h ≈ 0.5 / 0.4 / 0.35 | 0.43 / 0.49 / 0.43 | — |

| rank comparison | Spearman | top-10 overlap |
|---|---|---|
| APBS against itself, padding 8→11 Å | ≥ 0.9916 | 9/10 |
| debye against itself, padding 8→11 Å | ≥ 0.9783 | 8/10 |
| debye against APBS | 0.9794–0.9842 | 7–8/10 |

**Every row says the same thing: the disagreement between solvers is the size
of each solver's own grid noise.** debye is not failing — the quantity is
dominated by discretization, not by which solver computes it. Two consequences
that decided the treatment:

- **The obvious relational bar has the wrong comparator.** "As well as APBS
  agrees with itself" *fails* (0.9794 < 0.9916). The comparator has to be the
  noisier participant, since a cross-solver difference cannot be expected to
  beat the noise of a solver inside it. That bar — debye against APBS ≥ debye
  against itself — passes by **0.0011**. A gate with that margin goes red for
  reasons unrelated to debye, which is §7's "check that cannot fail" inverted:
  a check that cannot *pass* reliably teaches as little as one that cannot fail.
- **Top-N is not gateable at all.** It is unstable *within* one backend — APBS
  drops to 9/10 against itself across a box change, debye to 8/10. An earlier
  draft of this gate proposed "the same top-10 set"; it would have been flaky by
  construction.

So M6 gates the DX half, records this one, and `tests/test_debye_m6.py` pins the
relationship the way M3 pinned its neutral solute — **if debye ever becomes
clearly better than the noise, the pin notices and says to revisit rather than
absorbing it silently.** That is what makes "revisit after fractional-volume
dielectric averaging" a mechanism rather than an intention.

#### Three findings that outlive M6

- **A three-way pinned lattice is impossible on a real protein, structurally.**
  DelPhi's grid is *isotropic* — one `scale`, 0.498 Å on every axis — where APBS
  and debye derive per-axis spacing from the bounding box. Over paddings
  5.0–25.0 Å: APBS/debye share one lattice (padding 25), DelPhi shares none with
  either. **M1b's common-lattice recipe worked because Born ions are spheres**
  and give cubic boxes; it does not transfer to a protein, and any future
  cross-backend field gate has to know that before it is written.
- **The control that separates phase from box cannot be run here.** It needs one
  spacing reachable from two different paddings; searching the whole
  `(padding, resolution)` space there are **zero**, for either backend. M1b
  separated them on a sphere and that separation is simply unavailable at
  protein scale.
- **`max_points` clamping is both the trap and the only route to a common
  lattice.** APBS at resolution 0.35 and 0.30 returns *identical* residue
  potentials — dime (161,161,161) = 4,173,281 = `max_points`, the same grid
  solved twice — which was one sentence away from being reported as "APBS has
  converged", the same trap PR #38 hit and recorded. The tell is an implausibly
  perfect zero; the check is to print the resolved shape beside every row of a
  convergence study. The same clamp is what puts debye and APBS on a
  bit-identical lattice — spacing, origin and shape all equal — at resolution
  0.3, which the padding scan says never otherwise happens.

#### The timings, recorded here so M7 starts from a measurement

Same protein, same settings, this machine: **APBS 6.0 s against debye 39.1 s at
h ≈ 0.5, and 16.9 s against 85.2 s at h ≈ 0.3** — roughly 5–6×. debye is the
backend with no binary, so it is the one protean would ship by default: **on the
machines that most need sashimi, sashimi is at its slowest.** M6's accuracy
numbers say the residue axis is limited by discretization rather than by the
solver, so the lead worth taking is the one that damps the oscillation *and*
would let a coarser grid do — fractional-volume dielectric averaging, which
`sashimi/debye/dielectric.py` names in its own docstring and which M4a dropped
on cost/benefit rather than on infeasibility. M6 is the first evidence it would
pay off on a quantity a user actually looks at.

*Written before M8. The lever was taken, and not in the shape named here: a
volume fraction over a band of whole cells is **worse than the hard assignment**
(Q1 below), where a solute fraction ramped across a single cell from the signed
distance is several times better. The mechanism is the harmonic mean on band
faces rather than the averaging.*

### M7 groundwork — where the time goes, and how to measure it at all

**2026-08-17.** Two findings, both of which changed the plan they were meant to
support. Neither is the optimisation; both had to come first.

#### debye is geometry-bound, and the lever we chose was the wrong one

The plan going in was fractional-volume dielectric averaging, on a chain: better
dielectric → less oscillation → a coarser grid suffices → faster. **The profile
breaks the chain at step three.** `fas2-molecular`, 68.9 s under `cProfile`:

| | cumulative | share |
|---|---|---|
| `build_levels` | 59.6 s | **86%** |
| ↳ `surface.inside` | 59.2 s | 86% |
| ↳ `_toroidally_reachable` | 45.6 s | **66%** |
| `solve_system` — the actual multigrid solve | 7.3 s | **11%** |

**Two thirds of a protein solve is classifying points against the toroidal
patches of the solvent-excluded surface; a ninth is the linear algebra.**
Fractional-volume averaging replaces a binary inside/outside test with a
fractional volume per face — *more* geometric work per query point. It would
make the dominant 86% more expensive to improve the 11%. It remains a real
accuracy lever, with M4a's recorded numbers to beat, and it belongs to M6's
follow-up where its evidence points. It is not a performance lever.

**The per-level split, which bounds the obvious fix and suggests a better one:**

| level | points | time | µs / 1000 points |
|---|---|---|---|
| 0 — finest, decides the answer | 1,514,073 | 41.2 s | 27 |
| 1 | 194,285 | 7.8 s | 40 |
| 2 | 25,575 | 5.0 s | 194 |
| 3 | 3,536 | 2.8 s | **778** |

`build_levels` re-discretizes the geometry at every multigrid level, and only the
finest decides the answer — the rest are preconditioner. So "cheapen the coarse
levels" is worth **27% of the geometry cost, about 23% of runtime**: real, and
bounded. The more useful reading is the last column: cost is wildly sublinear in
points, the coarsest level being **29× worse per point**. That is a large fixed
cost per *call*, and the profile names it — 162,740 calls to `_legal` and
182,080 to `near`, one per rim. **The problem is the number of small numpy
calls, not the arithmetic in any one of them**, so the change that follows is
batching the rim loop into hundreds of large calls rather than hundreds of
thousands of small ones. That is M7's remaining work.

*One thing landed here, and its size is the point.* `_Bins._block` walked its
bin range with a Python triple loop and a dict keyed by tuples, 197,050 times.
It is now one `searchsorted` over encoded bin codes with the variable-length
ranges flattened by `repeat`. **6.5% of CPU time, energies bit-identical.** It
was also written before the profile had been read one level deeper: `_block` is
8% of `tottime`, so 6.5% is close to all there was, and the 66% was never in
reach from there.

#### Wall clock cannot measure this, and a VM would not fix it

**§11 deferred the benchmark VM until "debye makes a performance claim against
APBS", so this is where that question comes due. The answer is that the VM is
not what was missing — the instrument was.**

Measured on this machine, interleaved, alternating between `main` and the branch
on the same case: `main` 61.8 / 79.1 / 42.5 s, branch 52.2 / 44.6 / 60.5 s.
**Identical code varies 1.9× run to run**, which is larger than anything M7
would claim, and the ranges overlap so completely that the comparison says
nothing. Earlier in the same session the same `sashimi validate` invocation read
36.0 s for one run and 64.6 s for another under review load.

That instability is *contention*, not architecture, and **a VM on the same host
contends the same way** — so the hardware §11 proposed would not have bought the
stability it was proposed for. What does:

- **CPU time rather than wall clock.** Other processes take wall-clock time away
  from a run; they do not change how many CPU-seconds the work costs.
- **Minimum of N rather than a mean.** The minimum is the least-contaminated
  sample; a mean averages in the contamination.

On that instrument the same comparison is clean and repeatable: `main`
**44.96 s** CPU against the branch's **42.04 s**, energies identical to the last
digit — the 6.5% above. The same pair on wall clock read as a *40% regression*
an hour earlier. **So M7 claims CPU-time ratios, measured back to back on one
machine, and §11's VM is retired on evidence rather than on preference.**

### M7 — the plan for the rim loop, measured before it is written

**2026-08-18.** The groundwork above named batching the rim loop as M7's
remaining work. This section is what that loop turns out to look like when it is
measured directly rather than read off a profile, which changed one thing that
matters and confirmed the rest.

#### The profiled ratio holds, and the profiled absolute does not

`_toroidally_reachable` measured with `time.process_time()` and no profiler
attached is **78.5% of `build_levels`** — against radial at 12.5% and vertex at
5.5% — where the section above reports 66% of the total run and `build_levels`
at 86%, so 77% of it. The two agree.

The absolutes do not: `build_levels` is **22.89 s** unprofiled against the
**59.6 s** `cProfile` recorded. A 2.6× inflation is what a deterministic profiler
does to code that makes millions of small calls, which is precisely the code
under diagnosis here. **The ratios in the section above are safe to plan from;
the seconds are not safe to quote**, and M7's claim has to come off the
instrument rather than off a profile.

#### The loop's shape, per `inside()` call

fas2, 906 atoms, h ≈ 0.5, the staggered-x lattice at each multigrid level:

| level | nodes | rims | `near` calls | (rim, node) pairs | `_legal` (node, blocker) pairs | CPU |
|---|---|---|---|---|---|---|
| 0 | 27,991 | 11,380 | 11,380 | 4,849,371 | 69,304,243 | 2.61 s |
| 1 | 3,476 | 11,380 | 11,380 | 601,043 | 8,596,374 | 0.82 s |
| 2 | 427 | 11,380 | 10,652 | 76,235 | 1,105,742 | 0.59 s |
| 3 | 51 | 11,380 | 3,801 | 9,717 | 148,968 | 0.42 s |

The rims are geometry, so there are 11,380 of them at every level — 11,380
`near` calls per `inside()` and sixteen `inside()` calls per solve gives the
182,080 the profile counted, so the accounting closes exactly. **Level 3 spends
0.42 s to decide fifty-one nodes**, which is the per-call floor with the
arithmetic removed and the clearest statement of what is wrong.

#### The dependency that forces the loop to be a loop is worth 16%

`decided` is why the rim loop cannot simply be flattened: each rim skips the
nodes an earlier rim has already claimed, so the iterations are not independent.
**Measured, that pruning removes 16% of the work** — live nodes are 84% of found
nodes at every level, 84 / 84 / 84 / 87. A batched pass gives up a sixth of the
pairs and buys every one of the call overheads.

**And the batched answer must come out bit-identical, not merely close.**
`decided` feeds a boolean OR and nothing else, so a node claimed by *any* rim is
the node claimed by *the first* rim; no float depends on which. That is the same
kind of gate M4 had, on a change in the same file, and it is the reason this
rewrite is safer than its size suggests.

#### The projection above was wrong, and how it was wrong is the finding

**The estimate.** The two expansions done as a handful of large calls instead of
tens of thousands of small ones: the projection stage, 4.85M pairs, 0.10 s; the
`_legal` stage, 69.3M pairs in 8M-pair chunks, 0.61 s. Against level 0's
measured 2.61 s that reads as 0.71 s of arithmetic inside a 2.61 s loop, and
projected to **4.4× on the family, ~2.1× on the solve**.

**The measurement, once it was built.** Batching every stage came out at
**1.000×** on the instrument, energies bit-identical. Per level, against the
loop:

| level | looped | batched |
|---|---|---|
| 0 | 2.61 s | **3.75 s** |
| 1 | 0.82 s | 0.49 s |
| 2 | 0.59 s | 0.08 s |
| 3 | 0.42 s | 0.02 s |

**Batching wins by 3× to 21× on the coarse levels and loses by 1.4× on the fine
one, and the fine one is 59% of the family.** The two cancel almost exactly.

**Why the estimate was wrong: it timed arithmetic on contiguous arrays, where
the real stage is a gather.** `_legal` broadcasts one point cloud against one
atom set, so it fetches 35 atom coordinates and keeps the whole product in
cache. The batched form has to fetch a coordinate *per pair*, and there are 69.3
million of those on the finest level. Stage-by-stage at level 0, batched: the
gathers alone are 1.43 s of a 3.75 s pass, against 0.81 s for the arithmetic
they feed. The benchmark that predicted 0.61 s had the arrays already in place.

**And the per-call overhead the milestone was aimed at is not where the profile
implied.** Timed directly inside the loop at level 0 — `near` 0.98 s, projection
0.34 s, `_legal` 1.27 s — the residue is **0.05 s, 2%**. There was never 4× to
recover there. At level 3 the same split is `near` 0.32 s of 0.44 s, **73%**,
which is the fixed cost per call the groundwork correctly identified; it is just
that the level it dominates holds fifty-one nodes.

*The general form, worth carrying past M7:* "cost is sublinear in points, so the
cost is per call" was sound, and "therefore batching pays" did not follow. The
per-call cost was concentrated in the levels that are cheap for the same reason
they have few points, and the level that carries the runtime was already
spending its time on arithmetic.

#### What landed: the query batched, the legality left alone

Batched: the ball query, `_Bins.near_many`, which replaces 11,380 `near` calls
per lattice with one per `PAIR_BATCH` pairs, and the projection that follows it.
Not batched: `_legal`, which keeps its per-rim broadcast for the measurement
above. `near_many` returns its pairs grouped by query, so splitting them per rim
is a `searchsorted` rather than a sort.

**1.209× on the solve, CPU time, minimum of 3, interleaved — energies
bit-identical at −2078.7814508266547 kJ/mol.** `_toroidally_reachable` goes
17.96 s → 13.63 s and `build_levels` 22.89 s → 18.51 s on fas2.

Two controls make that ratio readable. Identical code on both sides of the same
instrument reads **0.993×** at protein scale, so 1.209× is twenty times the
noise floor. And on the run that produced it the samples themselves spread only
1.00×–1.01×, against the 1.36× spread the control saw under load — with the
minima agreeing to 0.7% across both, which is the whole case for minimum-of-N.

**`PAIR_BATCH` is a memory bound and `tests/test_debye_m7.py` sweeps it** at
one, five hundred, fifty thousand and 10^12 *pairs*, asserting the mask does not
move by a single node. It is counted in pairs rather than rims because a rim
count bounds nothing — the pairs a rim expands to scale with node density. The
extremes have to land on opposite sides of the batching boundary and the test
asserts *that* too — a sweep that never crossed it would be unanimous for the
wrong reason.

The tests were checked by mutation rather than by inspection, and one of the
three mutations survived the first draft: relaxing the batched distance filter
from `<=` to `<` passed everything, because random points land on a query radius
with probability zero. A fixture with a point exactly on the radius is what
closes it — the same hole, in the same shape, as the twenty instances the
`guards that guard nothing` history already records.

#### What is left of M7

The claim is **1.209×**, not the 2.1× projected above, and debye is still
roughly 4–5× slower than APBS rather than 5–6×. What the measurements say about
where the rest would come from, in the order the evidence supports:

1. **`_legal` at level 0 is now the largest single stage** — 1.27 s of a 2.57 s
   pass, 69.3M (node, blocker) tests deciding 14,377 nodes. 1.99M of the pairs
   survive the `close` test but only ~71 per node are needed to find one legal
   rim, and a node that is *never* legal must exhaust all of them. A node-major
   pass that stops at the first legal rim is answer-preserving by the same
   boolean-or argument, and is the one remaining lever with a measured size
   behind it. **Unmeasured, and not to be claimed until it is.**
2. **The query is a ball where the test is a torus.** `near` returns 4.85M nodes
   at level 0 and the `close` test discards 59% of them.
3. **Coarse-level re-discretization**, which batching has now taken from 1.83 s
   to 0.59 s — most of what the groundwork bounded at ~23% is already collected.

### M7 is parked, and the two measurements that park it

**2026-08-18.** M7 stops here at **1.209×**, with the instrument and the batched
query landed and the remaining levers written down rather than taken. Two
measurements decided that, and the first is more important than anything else in
this section.

#### debye has been graded at a resolution its consumer does not use

protean — the reason debye exists, §10 — asks for **1.0 Å spacing and 10 Å of
padding**, and every performance number in this roadmap was taken at 0.5 Å.
Regraded at protean's own defaults, on this machine:

| structure | atoms | debye, CPU |
|---|---|---|
| fas2 | 906 | 7.7 s |
| barnase | 1,730 | 21.0 s |
| actin-monomer | 5,877 | 90 s |

fas2 is **7.7 s, not the 21.7 s** the 0.5 Å figure reports — so the gap M7 was
chartered to close was measured about three times too wide. Nothing was wrong
with the earlier numbers; they answer a question nobody was asking.

**Two things fall out of the same run.** `want_energy=False` saves nothing
(7.74 s against 8.03 s), because the reference state sets the two dielectrics
equal and `dielectric_faces` returns constants without touching the geometry —
so **protean gets a solvation energy for free with any potential it asks for**,
which is a quantity it has no way to produce today. And the cost is
**superlinear in atoms**: 6.5× the atoms costs 11.7× the time, which is the
scaling that actually threatens the use case, not the constant factor M7 chased.

#### Threads work at 0.5 Å and do not at 1.0 Å

The geometry is per-node independent and numpy releases the GIL over most of
it, so the twelve lattices across the multigrid hierarchy can be built
concurrently. Masks came back **identical at every width**, which is the only
reason this got as far as a table. At **0.5 Å**, fas2, wall clock:

| workers | wall | speed-up | CPU cost |
|---|---|---|---|
| 2 | 20.16 s | 1.47× | 1.20× |
| 3 | 18.63 s | 1.59× | 1.45× |
| 4 | **12.97 s** | **2.28×** | 1.41× |
| 8 | 13.49 s | 2.19× | 1.38× |

**At 1.0 Å — the resolution protean asks for — the same change is worth
1.06×.** Whole solves, energies bit-identical at every worker count:

| structure | serial | 3 workers | speed-up | CPU cost |
|---|---|---|---|---|
| fas2 | 5.84 s | 5.54 s | **1.06×** | 1.19× |
| barnase | 12.57 s | 11.67 s | **1.08×** | 1.16× |

Not Amdahl — geometry is still **92%** of the solve at 1.0 Å. It is array size:
the undecided shell holds a quarter as many nodes, so the rim loop's arrays are
four times smaller and a far larger share of the time is spent in code that
holds the GIL rather than in ufuncs that release it. **Threading is dropped**,
at 1.06× for 16–19% more CPU.

*A methodology note, because the first run of the 0.5 Å experiment was wrong.*
It built a `ThreadPoolExecutor` per repeat without closing it, so the wide runs
were measured inside a process carrying dozens of idle threads and read 1.36×
where the corrected run reads 2.28×. Wall-clock experiments accumulate state
that CPU-time experiments do not.

#### Both levers inverted at the consumer's resolution, in opposite directions

This is the finding to carry, and it is not about either optimisation:

| lever | at 0.5 Å | at 1.0 Å (protean's) | verdict |
|---|---|---|---|
| batched rim query | 1.209× | **1.966× / 1.902×** | **kept** |
| threading the lattices | 2.28× | **1.06× / 1.08×** | **dropped** |

**Each one is worth roughly what the other was worth, depending only on which
resolution it is graded at.** Both readings are correct measurements of
different questions, and the milestone had been asking the wrong one throughout
— §12's whole performance thread was graded at 0.5 Å because that is what the
corpus and the accuracy milestones use, and the accuracy work had every reason
to be there.

The mechanism is the same in both cases and is worth stating once: **coarsening
the grid shifts work out of large-array numpy and into per-call and per-iteration
overhead.** Batching removes that overhead, so it gains. Threading needs large
arrays to have any GIL to release, so it loses. A performance change here must
name the resolution it is claimed at, and the resolution that counts is the one
the caller uses.

M7's result at protean's settings is therefore **1.9–2.0×**, bit-identical,
measured on fas2 and barnase at minimum-of-5 and minimum-of-3 with spreads under
1.04×.

#### The batch bound was a constant nobody had measured

`/code-review high` on PR #52 found it, and it is the third instance in this
milestone of a number that was asserted rather than measured — so it is recorded
next to the other two rather than quietly fixed.

The batch was bounded at **2,000 rims**, with a comment calling that "~850,000
pairs, tens of megabytes". Both halves were wrong. A rim count bounds nothing,
because the pairs a rim expands to scale with node density; and the surviving
pairs are not the working set, since `near_many` expands every bin its query box
touches *before* the radius test thins it. Measured with `tracemalloc` over one
`inside()` on fas2:

| | peak, batched | peak, unbatched |
|---|---|---|
| 0.5 Å | **580 MB** | 13 MB |
| 1.0 Å | **348 MB** | 3 MB |

**A hundred times the memory for twice the speed**, on a structure of 906 atoms
and a grid `GridSpec.max_points` would allow four times over. Nothing in the
suite would have caught it, because every test asserted the *answer* and none
asserted the cost.

Bounded in pairs instead — `PAIR_BATCH`, 50,000 — the peak is flat in resolution.
And the sweep that set it says the original number was not a trade-off at all:

| pairs | 0.5 Å | 1.0 Å |
|---|---|---|
| 20,000 | 23 MB, 18.60 s | 29 MB, 6.97 s |
| 50,000 | 41 MB, 18.48 s | 37 MB, 6.86 s |
| 2,000,000 | 592 MB, 18.6 s | 354 MB, 7.02 s |

**CPU varies under 1% while the peak moves twentyfold.** The whole speed-up is
present at the bottom of the range, so the 2,000-rim bound bought nothing it
cost anything for. M7's number after the fix is **2.017×** on fas2 at 1.0 Å,
minimum of 5, energies bit-identical.

*The general form, and it is the same one M7 keeps producing:* a constant
introduced to bound a resource must be measured against that resource. The
`PAIR_BATCH` sweep in `test_debye_m7.py` was real and caught a real mutation, but
it swept the constant against the **answer** — which it could not change — and
never against the **memory**, which was the only thing it existed to control. A
guard aimed at the wrong axis is the same guard that guards nothing.

#### What M7 leaves on the table, in evidence order

1. **`_legal` at level 0**, the largest single stage at 0.5 Å — a node-major
   pass stopping at the first legal rim is answer-preserving by the same
   boolean-or argument. Unmeasured, and **to be graded at 1.0 Å first.**
2. **The ball query where the test is a torus** — 59% of what `near` returns is
   discarded by the `close` test.
3. **The superlinear scaling in atom count**, which none of the above addresses
   and which is the only one that matters above ~2,000 atoms.
4. **Threading**, measured and dropped, revisitable only if debye is ever asked
   for 0.5 Å work at protein scale.

**What debye is actually competing with, on fas2 at protean's defaults:**
coulombic **1.04 s** with no binary and no Poisson-Boltzmann in it — protean's
own label is "not a Poisson-Boltzmann solution; magnitudes are indicative only"
— and APBS **0.95 s** with a binary. debye is now **5.9 s** and is the only one
of the three that is both. **That, and not a constant factor, is the case for
shipping it.**

### The size range protean actually uses, and a compiled-kernel spike

**2026-08-18.** Charlie set the working range at **250 to 1,200 residues**,
which is the first time this roadmap has had one. Two things follow, and the
first is that most of the performance work above was graded on a structure below
the floor: **fas2 is 61 residues.**

#### Measured across the range, at 1.0 Å

| backend | scaling | 61 aa | 130 aa | 375 aa | 540 aa | 1,156 aa |
|---|---|---|---|---|---|---|
| APBS | atoms^0.96 | 0.7 s | 2.4 s | 3.8 s | 5.5 s | **14.2 s** |
| DelPhi C++ | atoms^1.60 | 0.4 s | 1.4 s | 5.8 s | 10.2 s | **53.8 s** |
| debye | atoms^1.13 | 5.9 s | 15.0 s | 52.8 s | 74.0 s | **159.1 s** |
| protean `coulombic` | atoms^1.87 | 0.7 s | 3.4 s | 20.7 s | 26.6 s | **201.4 s** |

**The ordering inverts inside the working range, and that is the headline.**
`coulombic` is protean's no-binary default and is 8.7× faster than debye at
61 residues; by 1,156 it is **slower**, because it is O(points × atoms) and
debye is near-linear. Measured crossover: **~1,290 residues**, with debye ahead
of it at the top of the range already. So above roughly 1,200 residues debye is
both faster than the default protean ships *and* the only one of the two solving
the Poisson-Boltzmann equation.

**DelPhi C++ is the other surprise.** It is the fastest backend at peptide scale
and scales at atoms^1.60, so by 1,156 residues it is 3.8× slower than APBS and
only 3.0× faster than our pure-numpy solver. **Two compiled incumbents differ by
3.8× from each other**, which is nearly the 3.0× between the slower of them and
debye — so on this problem the implementation and the algorithm dominate the
language. APBS's near-linear exponent is focusing; debye has none, by design.
*Withdrawn 2026-08-22 — see "M9 — a boundary that does not cost
`O(nodes × atoms)`". Focusing costs an extra solve and APBS solves 1.65× more points
than debye at 1,156 residues; it is not where the exponent comes from. debye's
own exponent is its `O(nodes × atoms)` boundary sum, measured at atoms^1.45.*

#### The numba spike: 7×, and where that leaves Rust

Ported `_toroidally_reachable` — bin walk, projection and blocker test — to a
single `njit` kernel and graded it against the shipped numpy version. **Masks
bit-identical at every level, on both structures tested.**

| | level 0 | total, 4–5 levels |
|---|---|---|
| actin-monomer, 375 aa | 8.2× | **6.8×** |
| serum-albumin, 1,156 aa | 9.5× | **7.0×** |

**`parallel=True` bought nothing** — 7.0× against 6.8× single-threaded — so this
is compiled code, not parallelism, and it is the second measurement in this
milestone saying the parallel axis is not the one to push here.

Geometry is 92% of the solve at 1.0 Å, so **porting all three families projects
to ~4.7× overall: 1,156 residues from 159 s to ~34 s**, between DelPhi and APBS,
and beating `coulombic` by 6× at the top of the range. That is the number a port
has to be worth; it is a projection and the last four in this section were wrong,
so it is to be re-measured per family and not assumed.

*It was wrong too, and by the largest margin of the five — the port measured
**1.73×**, 86 s rather than 34 s. The families were never 92%: the profile that
said so charged three one-time builders to the families that first touched them.
"The port is finished, and it says the port was the wrong target" below has the
exclusive re-measurement.*

**What the spike did not settle was what to ship**, and the answer taken on
2026-08-19 is: **both, with the caller choosing.** `sashimi-electro[fast]` is an
optional extra carrying numba; `sashimi.debye.kernel` holds the compiled rim
loop and `surface.py` dispatches to it when it is importable.

The shape matters more than the choice. **The numpy path stays the reference** —
it defines the answer, it is what the corpus is recorded against, and it is what
two of CI's three legs run. The kernel is required to be *bit*-identical, never
merely close, for the same reason the batching was: `decided` feeds a boolean or,
so nothing downstream can depend on which rim won.

**145 MB is the reason it is not a dependency.** numba plus llvmlite is several
times the rest of the install, against a package whose whole proposition is that
it needs nothing fetched by hand. A caller solving one peptide should not pay it;
a caller doing protein electrostatics should. So the cost is stated wherever the
decision gets made — the README says it in the install section, and
`sashimi_capabilities` reports `acceleration.compiled_surface_kernel` with the
size and the measured worth beside it. `SASHIMI_NO_NUMBA=true` turns it off.

**Two verification steps, because an unexercised second implementation is the
trap this repo keeps hitting.** CI installs the extra on `full` only and asserts
the compiled path is live there; the other two legs assert it is *not*, so the
fallback is genuinely the thing most installs run rather than an untested branch.
That is the DelPhi lesson — a skipped tier and a passing tier look identical —
applied one implementation over.

*One test earned its place immediately.* With the extra installed,
`test_the_structure_actually_exercises_the_rim_loop` failed: it counts
`near_many` calls, and the kernel never makes any. The M7 tests are tests of the
*reference* implementation — `PAIR_BATCH`, the batch sweep, `near_many` — so they
now pin `SASHIMI_NO_NUMBA`. The failure was the check doing exactly its job.

Rust via PyO3 remains the better long-term vehicle on size alone, and this
does not foreclose it: the dispatch seam is where a second accelerator would
land, and the bit-identity test is the gate it would have to pass.

### Both DelPhi flavours, baselined — and the instrument extended to reach them

**2026-08-20.** §12's item 2 deferred cross-flavour DelPhi agreement on
2026-08-13, making the C++ build the touchstone on a 19-case spot check. This is
that comparison over the whole corpus and the whole size range, with both
flavours installed locally for the first time. **Every number was measured
twice**, in separate runs at load averages 4.7 and 5.9, and every number is
reproducible with `sashimi bench --backend`.

#### The instrument reached one backend of four

`time.process_time()` excludes child processes by definition. That was correct
for `sashimi bench` as written — it only ever solved with debye, in-process — but
APBS, DelPhi and TABI-PB are all subprocesses, so **the tool could not time
three of the five backends at all**, and the first pass at this baseline fell
back to wall clock without saying so. On wall clock the same APBS case read
**13.8 s and 32.9 s** in two runs.

`resource.getrusage(RUSAGE_CHILDREN)` is the missing half, and `bench.cpu_seconds`
now adds it. Two preconditions, since the counter is process-wide: a child
contributes only once **reaped**, and any other child reaped inside the window
is counted too — so it is not safe across concurrent subprocess work. Both hold
here. This does not contradict `_remote_sample`, which has the *other* checkout
time itself: we can patch our own tree and not APBS.

`sashimi bench --backend {apbs,delphi,debye,gb,tabipb}` is the result, so the
tables below are re-runnable rather than the output of a script that lived
nowhere — which is the failure `bench.py`'s own docstring says it exists to
prevent, and which the first draft of this section committed.

Spreads fell from **1.4–2.7× on wall clock to 1.00–1.07× on CPU**, and the two
runs agree to 0.8% (C++), 2.4% (pyDelPhi) and 2.2% (debye). **CPU time is a large
improvement here, not an immunity** — memory and cache contention still inflate
it, and the two runs differ by 8.6% on APBS's 0.7 s rung, which is one of the
endpoints anchoring its exponent. Treat exponents below as ±0.05, which is
comfortably inside the gap the argument uses but not inside every gap.

#### The ladder — CPU seconds, `molecular` at 1.0 Å

Mean of two runs, each minimum-of-three, on this machine.

| backend | CPU scaling | 61 aa | 130 aa | 375 aa | 540 aa | 1,156 aa |
|---|---|---|---|---|---|---|
| APBS | atoms^0.93 | 0.7 | 2.3 | 3.7 | 5.1 | **12.4** |
| DelPhi C++ | atoms^1.56 | 0.4 | 1.4 | 5.7 | 9.1 | **43.2** |
| pyDelPhi | atoms^0.87 | 2.0 | 3.7 | 8.0 | 10.6 | **27.3** |
| debye, pure numpy | atoms^1.11 | 5.9 | 14.5 | 50.3 | 66.5 | **147.3** |
| debye + `[fast]` | atoms^1.13 | 3.7 | 9.4 | 34.1 | 41.7 | **99.0** |

Reproduce a row with `sashimi bench --backend <name> --structure <pqr>
--resolution 1.0 --repeats 3`. Two of the five need an environment variable as
well, because `--backend` names a *registry entry* and these rows are variants
of one: the DelPhi flavour follows `SASHIMI_DELPHI_PATH` (unset for the C++
build on PATH, set to `.pydelphi/bin/pydelphi-static` for pyDelPhi), and the
debye rows follow `SASHIMI_NO_NUMBA` and whether `[fast]` is installed.

**Both debye rows are given because the extra is opt-in**, so the first is what a
default install does and the second is what someone who read the README gets.
The compiled kernel is worth **1.47–1.60× across the ladder** — the toroidal
family only, which is why it is not the ~4.7× a full port projects.

**pyDelPhi overtakes the compiled C++ build at ~9,800 atoms, about 620
residues**, and is **1.58× more efficient** at the top: 27.3 s against 43.2 s.
Same program, Python with numba against C++, winning on CPU rather than by
spending cores — its wall is only 1.11× under its CPU there, so it is barely
threaded.

**DelPhi C++ is the only backend measured whose exponent exceeds 1.2.** APBS
0.93, pyDelPhi 0.87, debye 1.11–1.13 — so the compiled incumbent is the one with
the scaling problem, which is the strongest form yet of this section's claim that
implementation dominates language here. It also settles the vehicle question left
open when the kernel shipped: **a numba solver beating C++ at protein scale is
why numba is a legitimate implementation and not a placeholder for Rust.**

*Where debye stands at 1,156 residues:* **3.4× DelPhi C++ and 11.8× APBS on the
default install, 2.3× and 8.0× with `[fast]`.** Closing on the incumbent that
scales worst; not on APBS.

#### Cross-flavour agreement, over the whole corpus

The flavours overlap on **exactly the 35 `molecular` cases**, and the arithmetic
is clean: 100 total, 35 shared, 42 `smoothed-molecular` — APBS's own harmonic
averaging, which **neither** DelPhi flavour implements — and 23 `van-der-waals`,
which is the one genuine asymmetry, since pyDelPhi's `prbrad 0` crashes inside
numba. Nothing failed; every exclusion is a refusal.

| | deviation |
|---|---|
| median | **0.014%** |
| mean | 0.135% |
| max | **1.257%** — `ion-protein-complex-molecular` |
| next | 0.450% `serum-albumin`, 0.426% `lysozyme-molecular` |

**The worst case triples the 0.426% bound the 19-case spot check found, and it is
not a size effect.** `ion-protein-complex` is 260 atoms. It is *not* the most
charged solute in the corpus — `serum-albumin` is, at −30 e against +21.69 —
but it is by far the most charged **per atom**: 0.0834 e/atom against 0.0016, a
factor of **51**. Concentrated charge on few large spheres, in a united-atom
structure with no hydrogens, is where two implementations of one iterative method
diverge most. The largest solute, at seventy times the atoms, sits at 0.450%.

This does not change the decision recorded in §7: **the flavours remain
interchangeable as backends and not as sources of a recorded number.** But two
things it says are now stale and are corrected with it. The bound is **1.257%
over 35 cases**, not 0.426% over 19. And that bound is **~125× the 1e-4 a
recording is held to — a bit over two orders of magnitude, not the "4,000×" §7
and `tests/test_delphi_solver.py` both carried**, which came from dividing a
percentage by a fraction.

**The deferred cross-flavour test in item 2 above would fail as designed.** It
specifies "three or four cheap shared cases … asserting agreement within 0.5%",
and `ion-protein-complex-molecular` is a 260-atom shared case at 1.257% — exactly
what "cheap" would select. Any such gate needs a per-case bound, or to exclude
the high-charge-density case deliberately and say so.

*Suite state, identical across both runs:* C++ **47 passed**; pyDelPhi **14
passed, 33 skipped** — 27 corpus recordings it cannot verify, 5 M1b cases needing
a van der Waals boundary, 1 guard reading a line only the C++ build prints. Each
skip is a documented flavour limitation. **Separately and pre-existing: 23 of the
58 C++ recordings are named by no per-push or on-demand list** — 21
`van-der-waals`, `barnase-molecular`, and `serum-albumin`, which is the top rung
of the ladder above. `corpus verify --backend delphi --tier full` reaches them;
no `delphi`-marked test does. ~~That is the same dead-weight failure
`tests/test_delphi_solver.py` records having made once at M0, and it is open.~~

**Closed 2026-08-24 (#73), and the fix is the shape rather than the list.**
`DELPHI_ON_DEMAND` is now the *complement* of the hand-kept per-push tuple over
the recorded directory, so the split is total by construction and a recording
added later joins it by existing rather than falling out of both lists and
looking like neither. `test_every_delphi_recording_is_either_re_solved_or_named`
asserts the partition; the assertion that can actually fail is
`set(DELPHI_PER_PUSH) <= recorded`, which reddens when the fast tuple names a
case with no recording — the disjointness and the union hold by construction and
are there to say so rather than to catch anything.

### Quality where there is no reference: two invariants, across every backend

**2026-08-20.** The baseline above measures speed carefully and grades quality
by distance from APBS, which assumes the answer. The corpus cannot do better
above a peptide, and the reason is worth stating as a number: **all 37
closed-form energies and all 12 closed-form fields are one- or two-atom
solutes** — Born ions and Kirkwood spheres, because those are the geometries a
closed form exists for. **Thirty-two cases sit above 500 atoms and not one has
any ground truth.** So the 10.4% spread at 1,156 residues says the solvers
disagree and gives no way to say which is closer to right.

What closes that is not a better reference. It is the identities the answer must
satisfy whatever the answer is. `sashimi.invariants` adds two, and
`tests/test_invariants.py` runs both across every registered backend.

#### Charge scaling: an exact identity, on any solute

The linearized equation is linear in the charge and the dielectric map does not
depend on it, so scaling every partial charge by `lam` scales the energy by
exactly `lam**2`. Not an approximation, and not family-specific — a
boundary-element method and a Generalized Born radius both obey it.

Measured at `lam = 2`: **APBS 0 to 2.4e-13, pyDelPhi 0 to 6.2e-8, TABI-PB
2.5e-10, debye and `gb` exactly 0** — and **DelPhi C++ 3.0e-5**, which is not
solver error but *printed precision*, since it reports two decimals in kT. The
gate is 1e-4, three times the worst.

**This is the check that would have caught the bug §7 records costing a year.**
`format_pqr` wrote minimum-width fields, a four-character residue name shifted
every column, and DelPhi solved on charges that were not in the file — returning
−865,205 kJ/mol for acetate against APBS's −197. Any mis-assignment of charge
breaks the square, on every structure, with no reference needed.

#### Rigid-motion invariance: a discretization error bar at protein scale

Solvation energy is a property of the solute, so translating or rotating it
cannot change the answer. On a fixed lattice it does, and **that spread is the
backend's discretization error** — available at any size, with no closed form.

fas2 at 1.0 Å, twelve poses (proper rotations about the centroid plus
sub-spacing translations, so grid phase moves and the boundary condition does
not):

| backend | range | **std** | split-half |
|---|---|---|---|
| `gb` | 0.0000% | **0.0000%** | — |
| TABI-PB | 0.167% | **0.052%** | 0.028 / 0.072 |
| DelPhi C++ | 1.304% | **0.410%** | 0.285 / 0.535 |
| pyDelPhi | 1.326% | **0.412%** | 0.279 / 0.542 |
| APBS | 2.601% | **0.764%** | 0.880 / 0.531 |
| debye | 4.220% | **1.416%** | 1.135 / 1.561 |

**`gb` is the control, and it is what makes the metric trustworthy.** An
analytic method has no lattice to fall out of phase with, so its spread must be
zero — measured 2.6e-16, one ulp from rotating coordinates, against 0.05–1.4%
for everything that discretizes. Twelve orders of magnitude between the control
and the signal.

**Gate on the standard deviation, not the range.** The range is the range of a
small sample and moves with the draw: debye read **3.01% and 0.60%** on two
different five-pose draws of the same structure, which is why the first version
of this measurement was reported wrong. Twelve poses split into halves agree to
within a factor of 1.4.

**Two independent confirmations that this measures the method.** The DelPhi
flavours land at 0.410% and 0.412% — two implementations of one algorithm,
agreeing on their own discretization error to three digits. And TABI-PB is
eightfold better than the best finite-difference backend, which is what a
boundary-element method should be: it has a surface mesh but no volumetric
lattice.

**debye is 1.8× the worst reference-tier backend and 3.9× the best** — on
`ala-gly` at 0.5 Å, where the corrected sub-cell shift gives APBS 0.515%, DelPhi
0.236% and debye 0.915%. *That is a real quality gap, and it is larger than the
speed gap M7 closed.*

**It is gated as a recorded value, not relationally, and that reverses what M1b
and M4 chose.** A relational bar reads "no worse than N× the worst reference-tier
backend installed", and here that bar **moves with the machine**: with APBS
present the worst is 0.515% and debye passes at 3×; with only DelPhi the worst is
0.236% and the same unchanged debye goes red on a contributor's checkout. A
verdict that depends on which binaries someone happens to have is not a gate. The
cross-backend comparison stays a measurement, recorded above.

**The first thing these checks found was a backend defect, not a debye one.**
TABI-PB solves `ala-gly` as it sits in the corpus and **aborts on every rotated
pose of it** — `terminating due to uncaught exception`, exit −6, from the mesher
rather than from sashimi. It is marked `xfail(strict=True)` in
`tests/test_invariants.py`, so a TABI-PB release that fixes it turns the suite
red and the marker comes out, rather than the xfail outliving the defect. Worth
recording twice over: the first draft of that test caught `SashimiError` and
reported the crash as "tabipb is unavailable here", so a real orientation
dependence in a shipped backend showed up as a green skip. **Only
`BackendUnavailable` may skip.**

*A trap from writing these, recorded because it cost an hour.* Mutation-testing
`posed` by changing `index == 0` to `index >= 0` produced a file of **identical
length**, and the restore landed within the same mtime second — so Python's
`.pyc` invalidation, which keys on (mtime, size), served stale bytecode and the
"restored" tree kept failing. `inspect.getsource` reads the *file* and showed
the correct source throughout, which is what made it look impossible. Clear
`__pycache__` when a same-length mutation is reverted.

### Decision: APBS is the fast option, pyDelPhi the stable one, and the caller picks

**2026-08-21, Charlie's call.** The cross-flavour baseline made pyDelPhi look
like a candidate to replace debye outright — faster than it, better discretized,
and needing no compiled binary, which is debye's whole charter. It is not, and
the reasons are worth recording because two of them are not about solver
quality at all.

#### The licensing question, and the assumption it rests on

pyDelPhi is **AGPL-3.0-or-later** with no linking or additional-permission
exception; sashimi and protean are both MIT. **DelPhi C++ carries no licence at
all** — no `LICENSE`, nothing in its README, no headers in its source — which is
a weaker footing than AGPL for anything redistributed, and it is the flavour
`tests/corpus/delphi/` is recorded against. APBS is BSD-3-Clause. Neither DelPhi
is on PyPI; pyDelPhi installs from a pinned git SHA into its own virtualenv.

**The decision is taken on the stated assumption that protean is installed and
run locally by its user**, who obtains any solver binary themselves. On that
model sashimi conveys nothing: pyDelPhi is not a declared dependency, cannot be
one, is never imported — verified, zero import sites — and is driven as a
subprocess exactly as APBS is. `src/sashimi/delphi/__init__.py` already recorded
that boundary as deliberate.

**Written down because it is load-bearing.** If protean is ever containerised,
bundled or hosted, this changes and the question has to be asked again. Two
items were flagged for counsel rather than settled here: the corpus commits
energies produced by the licence-less C++ build into an MIT repository, and
"local only" needs to stay true.

#### Why pyDelPhi does not replace debye

**It needs a manual install step and debye does not.** That was always debye's
actual charter — not "best solver" but "works with nothing extra" — and pyDelPhi
does not have it. And if a user is doing a manual install anyway, APBS is the
better thing to install: **twice as fast, BSD, on conda-forge, and it answers
100 of the corpus's 100 cases where pyDelPhi answers 35.** pyDelPhi has no
`van-der-waals` (`prbrad 0` crashes inside numba) and no `smoothed-molecular`.

*One platform where that argument fails, and it is the one §9 already names.*
conda-forge ships APBS 3.4.1 for `linux-64`, `osx-64`, `osx-arm64` and `win-64`
— **not `linux-aarch64`**. There, pyDelPhi and debye are the only options.

#### What pyDelPhi is better at, measured

**A common lattice is unreachable between these two**, so the M1b technique does
not apply: APBS resolves an *anisotropic* per-axis spacing and pyDelPhi a cubic
one, and for a non-cubic solute they cannot coincide on all three axes at any
padding. What is comparable is the trend against the effective spacing each
actually solved on — and it is one-sided:

| structure | asked | APBS eff. h | APBS | pyDelPhi eff. h | pyDelPhi |
|---|---|---|---|---|---|
| fas2 | 0.7 Å | 0.564 | 0.420% | 0.692 | **0.241%** |
| fas2 | 1.0 Å | 0.846 | 0.706% | 0.982 | **0.447%** |
| fas2 | 1.4 Å | 0.846 | 0.699% | 1.383 | **0.524%** |
| hca | 0.7 Å | 0.557 | 0.425% | 0.699 | **0.226%** |
| hca | 1.0 Å | 0.742 | 0.635% | 0.986 | **0.151%** |
| hca | 1.4 Å | 1.113 | 1.068% | 1.373 | **0.659%** |

**pyDelPhi is more pose-stable in all six, by 1.3× to 4.2×, and in every one it
is solving on the *coarser* grid** — the harder condition, since a finer lattice
should discretize better. The first draft of this comparison was a single
measurement at one resolution with an unexamined confound; it survived being
measured properly, which is not something this milestone can say of every claim
it has made.

#### What shipped

**The flavours are addressable.** `delphi` still means "whichever build is
installed" and is what `tests/corpus/delphi/` is recorded against; `delphi-cpp`
and `pydelphi` pin it. A registry entry that cannot say which executable it
means cannot be chosen on purpose, and the two are not interchangeable —
`delphi-cpp` answers `van-der-waals`, `pydelphi` needs no compiler.

**`Preference` — `fast` / `stable` / `portable` — resolves against what is
installed *and* what the request needs.** The surface-awareness is not a detail:
`stable` on a `van-der-waals` request falls through to `delphi-cpp`, because
handing two thirds of requests to a backend that refuses them would be worse
than not having the preference. `sashimi_capabilities` reports the whole
resolution table for the machine it is on, and the result carries
`selected_because`, so a caller who asked for `stable` and got APBS can see it
was the surface model and not an absent install.

**Naming: `stable`, deliberately not `accurate`.** Above a two-atom solute this
corpus has no ground truth — all 37 closed forms are Born ions and Kirkwood
spheres — so nothing here can promise a closer answer. What is measured is how
little the answer moves when the solute is rotated. `tests/test_preferences.py`
asserts the name, because the wrong one is a claim we cannot support.

**Nothing changed defaults.** `apbs` is still the default backend and preferences
are opt-in; naming a backend always overrides one.

### `validate` asks the backends that can answer — items 1–3 above, taken

**Done 2026-08-16, at Charlie's direction, in its own PR as planned.** The three
items that change what `validate` *means* are fixed together, because they turn
out to be one question asked twice: **what is actually being compared, and who
is comparing it.**

**The result is more coverage for less time**, which is not the trade the
recorded items implied. The default went from **13 cases compared in 36 s to 17
in 26 s** — dedupe saves more than per-backend selection costs, and the one
remaining `SKIP` is a genuine crash rather than a design refusal.

- **A refusal drops the backend, not the case.** `_backends_answering` asks
  every selected backend's published preconditions *before* anything solves, so
  a boundary-element solver declining a one-atom solute no longer discards the
  four finite-difference answers to the same question. A case is skipped only
  when fewer than `MIN_BACKENDS` remain. **The row says `not asked: <backend> —
  <reason>`**, because a backend that sat a case out must not read as one that
  agreed.
- **Identical systems are solved once and say so.** `_identical_systems` groups
  by a content fingerprint; the row prints `same system as …`. The saying-so is
  the half that matters — three identical rows read as a measurement of the
  probe, and M4 measured the probe worth +5.72% on ALA-GLY, so "no difference"
  is exactly the wrong conclusion to hand a reader.
- **The fingerprint hashes array bytes, not `repr`.** `repr` elides a large
  numpy array, so two proteins agreeing at the ends would share a fingerprint
  and one would be reported under the other's name — the failure the grouping
  exists to prevent, inverted. Guarded with a 2000-atom pair whose `repr`s are
  equal and whose fingerprints differ.

**The solve-time crash stays a case-level skip, and the distinction is
deliberate.** `aspartate-residue` — TABI-PB aborting on a structure it accepted
— cannot be predicted from preconditions, and by the time it surfaces the work
is spent. A documented refusal and an unexpected crash are different events and
now print differently.

**Item 4 is not fixed and is not obviously fixable.** `CaseTier` encodes APBS
wall time by construction and says so in its own docstring; `validate` pays the
slowest backend's. Making the tier mean "cheap for every backend" would either
re-tier the corpus against its documented meaning or need a second, per-backend
cost model. Left as a known asymmetry, with the mitigation being that the
default tier is now small enough that the worst case is bounded anyway.

**A guard lesson, third occurrence in two PRs.** The first test written for the
refusal fix called `_backends_answering` directly — so replacing `_validate`'s
call to it with `selected, {}`, which reverts the entire change, left the suite
green. **A helper test is not a wiring test.** The stub now records the backend
*list* each solve was handed, and all three mutations redden. See
[[sashimi-guards-that-guard-nothing]].

### The port is finished, and it says the port was the wrong target

**2026-08-21.** The remaining two surface families — `_radially_reachable` and
`_vertex_reachable` — are compiled, so all three are. They are bit-identical per
family and end to end, and the milestone the port was opened to hit is **missed
by a factor of two and a half**: the projection was ~4.7× on a whole solve, the
measurement is **1.73×**.

The gap is the finding, and it was available before any code was written. It is
the **fifth** wrong projection in this milestone, so the pattern is worth naming
rather than the number: *every one of them projected a whole-solve time from a
profile taken before the previous optimisation landed.*

#### What the port bought, measured

Per family, on the finest lattice at 1.0 Å, minimum of three CPU samples, masks
identical node for node:

| family | fas2, 61 aa | actin-monomer, 375 aa | serum albumin, 1,156 aa |
|---|---|---|---|
| radial | **25.4×** | **28.1×** | **28.4×** |
| toroidal *(M7)* | 9.7× | 9.6× | 9.0× |
| vertex | **17.9×** | **16.2×** | **17.2×** |

The two new ones beat the rim loop by 2–3×, and for the reason M7 already
recorded from the other side: the rim loop was *batched* numpy before it was
compiled, so its call overhead had been paid down once already. The radial
family never was — it ran a windowed sub-box copy and a broadcast per atom, and
18,242 atoms is 18,242 of each.

Whole solves, CPU seconds, energies identical to the last digit:

| structure | pure numpy | M7 (rim only) | all three | overall |
|---|---|---|---|---|
| fas2, 61 aa | 5.94 | — | 3.10 | 1.92× |
| actin-monomer, 375 aa | 54.70 | — | 30.16 | 1.81× |
| serum albumin, 1,156 aa | 149.33 | 113.12 | **86.09** | **1.73×** |

So the two families added 1.31× on top of the rim loop's 1.32×, and the whole
extra is now worth 1.73× at the top of the working range rather than the 4.7×
the spike projected. **1,156 residues is 86 s, not the ~34 s forecast** — still
between DelPhi C++ (53.8 s) and protean's `coulombic` (201.4 s), and still the
only no-binary option in that pair that solves the Poisson-Boltzmann equation.

#### Where the time is now, and why nobody saw it

Profiled at 1,156 residues with all three families compiled — 86.00 s, sixteen
lattices, one geometry:

| stage | s | share | built |
|---|---|---|---|
| `_probe_seats` | 23.73 | **27.6%** | once per solve |
| `_neighbours` | 12.54 | **14.6%** | once per solve |
| `_rims` | 10.86 | **12.6%** | once per solve |
| multigrid solve and assembly | ~23 | ~26.8% | — |
| `_toroidally_reachable` | 8.82 | 10.3% | per lattice |
| `inside_union_of_spheres` | 6.12 | 7.1% | per lattice |
| `_radially_reachable` | 0.57 | 0.7% | per lattice |
| `_vertex_reachable` | 0.19 | 0.2% | per lattice |

**The one-time reduced-surface construction is 55% of a solve and none of it is
compiled.** The three families the port was aimed at are **13%** together.

It was invisible because of how the split was taken. The M7 profile attributed
cost to `_vertex_reachable` and `_toroidally_reachable`, and both of those
*call* the one-time builders on their first invocation through a
`cached_property` — so `_probe_seats`' 23.7 s was reported inside the vertex
family's 18.6%, and `_rims` inside toroidal's 56%. The roadmap then read
"`_vertex_reachable` is the next-largest target" off a number that was **92%
somebody else's work**: the vertex family proper is 0.2%.

**The general form, which is not specific to this profiler.** A cached lazy
build inside a hot function is charged to the *first caller*, and the first
caller is chosen by call order rather than by cost. Any wall-clock or CPU
attribution that is not exclusive of callees will report a one-time cost as a
per-iteration one. Build the geometry before the timing window, or measure
exclusively — the exclusive re-measurement here took ten minutes and would have
redirected two milestones.

#### One honest limit on the bit-identity gate

The gate is exact equality per family, and it is real: a control mutation
stopping the radial family marking anything reddens five tests. But three
*subtle* mutations move **not one node** across 614,476 undecided nodes on two
proteins at two resolutions — the two boundary-equality flips (`<` for `<=` on a
blocker radius, on a seat radius) and, more interestingly, **hoisting
`radius / length` out of the radial projection**, which is precisely the
floating-point association trap M7 records the rim kernel's review catching.

So that discipline is a precaution no fixture here demonstrates the need for. It
is kept — the cost is a comment and the failure it prevents is silent — but the
roadmap should not go on citing it as a caught bug. Same shape as the M1b
correction: a rule that read as evidence was a rule that had never been
measured.

#### What is next, in evidence order

1. **`_probe_seats`, 27.6%.** A Python loop over 18,242 atoms, each doing
   `triu_indices` over ~60 neighbours, a sorted-key membership test, a batched
   trilateration and a legality broadcast — roughly forty numpy calls per atom
   for a few dozen floating-point operations of geometry. The same disease the
   radial family had, and the radial family came back 28×.
2. **`_neighbours`, 14.6%.** A `np.linalg.norm` per candidate pair inside a
   triple-nested Python loop, about nine million of them at this size.
3. **`_rims`, 12.6%.** ~550,000 pair iterations of ~15 numpy calls each.
4. **The multigrid solve, ~27%**, which is the floor none of this touches and
   which becomes the largest single item the moment items 1–3 land.

Items 1–3 are 55% and are all one-time, so unlike everything in M7 they do not
depend on lattice count or resolution. **Projecting from the 9–28× the compiled
families measured, 86 s would land near 45 s** — and that projection is the
sixth in this section, so it is to be measured per builder and believed
afterwards.

**Two things make items 1–3 harder than the families were, and both are about
what the kernel returns.** The families write booleans into a mask, so a node
claimed by any feature is the node claimed by the first and association only has
to be *arranged*, never proved. These builders return **float geometry** — rim
circles and seat coordinates — that later stages compare against radii, so a
last-bit difference is a different surface. And their outputs are ragged and
sized by the answer, which in numba means either a counting pass or a growable
buffer where the numpy version had `list.append`.

### The one-time build, compiled — and a square that was not one

**2026-08-21, the block after the port.** The three builders the port's
re-measurement pointed at are compiled: `_neighbours`, `_rims` and
`_probe_seats`, 55% of a solve between them and none of it touched before.

| builder | share, before | speed-up | how identity is checked |
|---|---|---|---|
| `_probe_seats` | 27.6% | 6.0× / 10.2× | `array_equal` on the seat coordinates |
| `_neighbours` | 14.6% | **93× / 116×** | the neighbour lists, element for element |
| `_rims` | 12.6% | ~12× | origins, normals, radii and blocker sets |

*(Two structures per row where two were measured: fas2 at 61 residues and
actin-monomer at 382.)*

**End to end**, CPU seconds at 1.0 Å, interleaved and best of three, energies
identical to the last digit:

| structure | pure numpy | with the extra | ratio | ratio after the port alone |
|---|---|---|---|---|
| fas2, 61 aa | 5.82 | 1.39 | **4.19×** | 1.92× |
| actin-monomer, 375 aa | 62.52 | 18.86 | **3.31×** | 1.81× |
| serum albumin, 1,156 aa | 155.21 | **45.99** | **3.37×** | 1.73× |

**The projection held this time, and that is worth recording after five that did
not.** The port's section closed by saying "projecting from the 9–28× the
compiled families measured, 86 s would land near 45 s". It is 45.99 s. What
changed is not the arithmetic but what it was taken from: the earlier five all
extrapolated a whole-solve time from a profile that predated the previous
optimisation, and this one extrapolated from an *exclusive* profile of the tree
it was projecting about.

`_neighbours` is the outlier for a reason worth keeping:
it was a `np.linalg.norm` **per candidate pair** inside a triple-nested Python
loop — about nine million numpy calls at 18,242 atoms for three subtractions of
arithmetic each. Nothing about it was vectorised, so nothing about it had been
paid down. `_probe_seats` is at the other end because its numpy version was
already batched per atom *and* because the kernel walks its triple loop twice.

#### The finding: a scalar `** 2` is not a multiplication

`_rims` would not come out identical. Twelve of 11,380 rim circles differed in
the last bit — three origins and nine radii — and the cause is not in the
kernel:

**`x ** 2` on a scalar is a call to the platform's `pow`, and this platform's
`pow` is not correctly rounded.** It disagrees with `x * x` for **26 of 27,799**
atom separations on fas2 and for 38 of the same set's `along` values. A
multiplication is correctly rounded by IEEE 754, so `x * x` is the *more
accurate* spelling; `pow` is the one that was shipping. numba lowers `** 2` to
a multiplication and has no way to reproduce libm, so no amount of care in the
kernel could have matched the reference.

Three properties of this are worth separating, because they point in different
directions:

- **It is not a numba fact.** `_rim`'s output — and therefore every recorded
  corpus energy that passes through a rim — depended on whose libm ran it. The
  corpus asserts exact values and CI runs Linux where these numbers were taken
  on macOS, so this was a live portability hazard that the port merely
  *exposed*. Two libms agreeing to date is not the same as a number being
  well defined.
- **The array case is safe**, and that is why this survived. numpy fast-paths
  `array ** 2` to `np.square`, which is the multiplication. Every other square
  in `surface.py` is on an array; the three in `_rim` were the only scalars.
- **The fix changes a recorded answer or it does not, and that had to be
  measured rather than argued.** `corpus verify --backend debye --tier full` on
  the pure-numpy path: **all 58 cases reproduce.** The ulp never reaches a
  recorded digit. Had one moved, it would have been a result change to bring to
  Charlie rather than a recording to regenerate.

The guard against a reintroduced `**` is written as the *invariant* — `_rim`'s
radius equals the explicitly-multiplied formula — and not as
`assert x ** 2 != x * x`, which would be asserting somebody's libm and would
redden CI the day that libm improved.

#### The sequel, three days later: a distance that went through BLAS

**2026-08-22.** The `pow` finding above was not the only place an exact number
rested on somebody else's arithmetic, and the second one was worse. `main` went
red on `ubuntu-latest, full` at `698cbfc` — a README-only merge — with
`test_the_compiled_rims_are_identical_to_the_last_bit` failing on origins and
normals, **while the identical kernel code had passed on the push before it.**

`np.linalg.norm` on a 1-D array is `sqrt(x.dot(x))`, and `dot` is BLAS.
**OpenBLAS dispatches its `ddot` kernel by CPU microarchitecture at run time**,
so the summation order — and the last bit of every distance taken this way —
depends on which machine the job landed on. Two GitHub runners of the same
operating system disagree. That is why it looked flaky and was not: it is
deterministic per CPU and varies across them.

**Two call sites, and the one CI caught was the harmless one.**

- `_rim` built the rim's origin and normal from it. A rim circle out by an ulp
  rarely moves a node, which is why the corpus never noticed.
- `_neighbours` used it as `gap < inflated[i] + inflated[j]` — a **threshold**.
  One ulp there flips whether two atoms are neighbours, which changes the rims,
  the seats and therefore the surface. **That can move an energy rather than a
  last bit, and it had been there since M4.** Nothing had reason to catch it: a
  threshold only bites for a pair sitting within an ulp of touching, and the
  corpus was recorded and verified on machines that happened to agree.

Both now spell the distance out. The corpus reproduces all 58 cases on both
paths afterwards, so nothing recorded moved *on this platform* — which was a
real question rather than a formality, given the threshold.

**What made it confirmable rather than merely plausible** was a second failure
in the same run: `test_the_rim_does_not_square_a_scalar_with_pow`, the
invariant guard written for the `pow` finding, still derived its `separation`
with `np.linalg.norm` after `_rim` had stopped. It failed on Linux and passed
on macOS with both values printed —

    assert 2.664419145590815 == 2.6644191455908146

— which is the divergence on one line of arithmetic rather than inferred from a
surface. *A guard written for one platform-dependence found the next one by
being wrong in the same way.*

**The audit that followed, because two instances is a pattern and not a
coincidence.** Every BLAS-routed number in the package, and whether it can reach
an answer: `sources.py` takes an axis-wise norm, which is `add.reduce` rather
than `dot`, so it is deterministic; `backend.py`'s energy genuinely is
`np.dot(charges, phi)`, but the corpus compares with a per-case `rtol` and the
bit-identity tests run both paths in one process where the dispatch is fixed;
`analysis.py`'s extremum spacing is a threshold on a norm, noted and not
exercised against exact recordings; `field.py`'s direction normalisation feeds
tolerance-checked samples. **The two that mattered were the two feeding an exact
comparison and a threshold**, which is the question to ask of each one rather
than replacing them all.

**The general rule, now stated twice over.** An exactly-recorded number must not
pass through a library whose arithmetic is chosen at run time — `pow` from libm,
`ddot` from BLAS. Neither is a bug in those libraries: both are free to
associate as they like, and both do it differently on different machines. The
requirement that found both was **bit-identity between two independent
implementations**, which is a stronger test of the *first* implementation than
anything written against it alone.

*A test-quality note that cost a CI round trip.* `assert np.array_equal(origin,
other[0])` prints `assert False` and nothing else, so the first failure said
only that rim geometry differed. It now names the rim, the field and the ulp
distance, and prints both values. **A test of last bits has to print the last
bits.**

#### Two mistakes inside the kernels, both caught before they ran

- **An out-of-bounds write that only the last rim could reach.** The first draft
  of the rim kernel wrote each blocker straight to `blocker_flat[blocker_offset[kept]
  + found]` as it found them — but a rim can be found *swallowed* after some of
  its blockers are known, and a swallowed pair that follows the last kept rim
  indexes `blocker_offset` one past its end. Compiled, with bounds checking off,
  that is a silent write into whatever follows. Blockers go through a scratch
  buffer and are copied on success.
- **A frame raised from the wrong atom.** `_probe_seats` sorts each atom's
  neighbours before pairing them, and the sort is load-bearing rather than
  tidy: it decides which of the three atoms the trilateration frame is built
  from. The kernel first walked the bin-order list, which produces the same two
  seats mathematically and different ones in the last bits. `_Spheres` now
  carries a `sorted_testable_table` for exactly this.

Both are the same class as the `PAIR_BATCH` lesson: **the thing to be afraid of
in this module is not a wrong formula, it is a right formula fed from the wrong
index.**

#### Where the time is now

At 1,156 residues the solve is 46 s, and the shape has inverted again. Geometry
was 92% before any of this and the multigrid solve was 11%; the solve is now the
largest single item in a debye run, and `inside_union_of_spheres` — a numpy
loop nobody has looked at, marking each sphere over its own index window — is
next. Neither is a call-overhead problem, so neither is another numba port:
**the remaining levers are algorithmic.** debye still has no focusing, which is
where APBS's near-linear exponent comes from. *That last clause is withdrawn;
M9 below has the measurement. What debye lacks is not focusing's resolution
but the cheap boundary condition focusing licenses.*

*And the numba clause is withdrawn with it. `inside_union_of_spheres` was
compiled the next day as `kernel.mark_union` (#62, 2026-08-22) — the last
uncompiled loop in the geometry — taken because the window loop is bounded by
the volume the spheres occupy rather than by call count, which is what made it
worth porting after all. What is left after it is the multigrid solve, and that
lever is algorithmic.*

**And the quality gap is now much larger than the speed gap.** debye's pose
dispersion is 1.416% against APBS's 0.764% at protein scale — unchanged by any
of this, because every kernel here is bit-identical by construction. ~~Two blocks
of work have gone into speed and the discretization lever named at M1c is still
unspent.~~

**Spent at M8 and M8a (2026-08-22), and the framing above is the half that did
not survive.** Pose dispersion is the *phase-dependent* part of the
discretization error and not the whole of it — Q0 below — so "debye's dispersion
is 1.416% against APBS's 0.764%" is a real gap and is not the gap the ramp
closes. What the ramp closes is the systematic half.

### The order changed: functionality before shipping

**2026-08-12, at Charlie's direction.** sashimi was born out of early protean
work, and the goal is for it to *replace* that functionality rather than sit
beside it. So the ordering is now: make it work, integrate it with protean and
mcpymol, and ship it afterwards. **Phase 5's PyPI release and phase 6
distribution are deferred** — not cancelled, and nothing here forecloses them.
A git or path dependency is enough for protean to consume sashimi, so the
release was never the blocker it looked like.

What that reordering exposed, by reading the thing being replaced
(`protean/src/protean_mcp/analysis/electrostatics.py`, 585 lines):

| protean has | sashimi has |
|---|---|
| `prepare` — pdb2pqr, charges and radii | `sashimi.prep`, `sashimi.pqr` |
| `run_apbs` — its own subprocess driver | `sashimi.apbs`, and three more backends |
| `write_dx` / `read_dx` / `sample` | `sashimi.dx`, `sashimi.analysis` |
| `coulombic` — a **field** with no binary installed | **nothing** |

That last row is the whole reordering. protean's default backend is screened
Coulomb precisely because it needs nothing installed, and it produces a grid,
because colouring a surface needs one. `sashimi.gb` needs nothing installed too
— and returns an energy, refusing `want_potential` outright. So sashimi can
replace protean's APBS path today and cannot replace its default at all.

**Hence phase 8 moves up.** A clean-room pure-Python PB solver is the honest
answer to "a potential field on a machine with no binary", and it is the only
one that does not put a known-wrong approximation on the critical path of a
consumer. Porting screened Coulomb into sashimi was considered and rejected for
that reason: it ignores the low-dielectric interior and the reaction field —
the two things PB exists to model — and a shipped approximate tier is hard to
withdraw once something depends on it. debye's acceptance gate is unchanged and
now matters more: `sashimi corpus verify --backend debye`, against a corpus
that is 64 cases with closed forms in it.

**Two decisions taken with it**, both previously open in §14:

- **The MCP surface grows a `backend` parameter now**, rather than when debye
  exists as §10 assumed. Four backends shipped, CI exercised all four, and
  `sashimi_solve` hardcoded `ApbsSolver()` — so the tier that needs no binary,
  the only one guaranteed to be present, was the least reachable thing in the
  package. `sashimi.backends` is the registry that made this one edit rather
  than three, and debye registers there in a single line.
- **The default surface model becomes `molecular`** — decided here, landed in
  the change after this one. It resolves §14's last question in the
  direction phase 7 pointed: it is the only model every shipped backend
  supports, where `smoothed-molecular` is APBS's alone, so a default request
  refuses on three of four backends. Measured cost of the switch: **0.80% on
  ALA-GLY and 2.35% on hen lysozyme** — smaller than the 1.0–1.6% the two
  reference-tier families differ by on a dipeptide, larger on a protein, and an
  order of magnitude below the 25.7% a surface model is worth (§5), which is the
  comparison that carries the decision. It is separated because changing
  `SolventModel`'s dataclass default rewrites the question every corpus case
  relying on it is asking: those cases must name `smoothed-molecular`
  explicitly first and be verified bit-identical before the default moves, and
  that is a change the corpus should be able to prove on its own.

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

### M8 and M8a (Q0 and Q1) — the instrument, and the interface scheme it graded

*Two names for one piece of work, and the phase table uses the other pair.* The
questions were posed as Q0 and Q1 and the results are recorded in the table as
**M8** and **M8a**; nothing said so, and every other milestone in §12 has a row.
The heading now carries both.

**2026-08-22.** Two blocks of performance work left debye's discretization
untouched, by construction: every kernel was required to be bit-identical, and a
bit-identical change cannot improve accuracy. This is the other axis.

#### Q0 — what a pose spread does and does not measure

`sashimi.invariants` said "the spread across poses is that backend's
discretization error on that structure". **It is the part of that error that
depends on grid phase**, and on debye it is the smaller part. Three findings,
each of which changed a claim in this repository.

**A pose spread cannot see a systematic error.** On `ala-gly` at 0.5 Å the
dispersion is 0.88% while the mean sits several percent from where its own
refinement extrapolates. A scheme can be beautifully phase-stable about the
wrong answer, and a gate on dispersion alone would call that an improvement. So
`grade_refinement` now reports the other half: solve at `h`, `h/2`, `h/4`, fit
`E = limit + C h^order`, and report both.

**The extrapolator is worth about half a percent, and that is measured rather
than assumed.** Graded on the Born sphere, where the limit is a closed form: it
lands 0.08–0.48% from exact. It is a better number than any single solve and it
is not a gold standard.

~~It beats the finest rung it was built from every time (0.45–0.81%).~~
**False, and re-measured 2026-08-24 over 24 Born windows spanning two radii and
four widths.** Of the sixteen that survive the repaired `converging`, **three
land no closer to exact than the finest rung they were built from** —
`r=3.0, w=0.25` at 0.949% against that rung's 0.892%, `r=3.0, w=0.5` at 0.596%
against 0.306%, and `r=3.0, w=1.0` at 1.275% against 0.415%. The extrapolation
usually earns its two extra solves and cannot be promised to.

~~`converging` guards it — a 2 Å sphere on a ladder starting at 1 Å reads
−172.6, −177.0, −171.7, and Richardson on that is 19.6% wrong. The guard refuses
it.~~ **It guarded that one case and not the class.** As shipped it tested
`abs(a) > abs(b) > 0.0` — magnitude only — which admits a ladder that has barely
moved and one that oscillates outright. Over the same 24 windows it let through
an extrapolation **269.688%** from exact, and a width sweep found it labelling a
`fas2` ladder `converging=True` whose limit was **+2902 kJ/mol**, a positive
polar solvation energy that linear response forbids. Two clauses were added
2026-08-24 and the worst surviving error over the Born set falls to **1.341%**:

- **Same sign.** Under `E(h) = L + C h^p` both differences are `C` times a
  positive quantity, so sharing a sign is an identity of the model being fitted;
  opposite signs refute it. `abs()` was exactly what discarded that, and the
  docstring already claimed to reject "oscillating".
- **Shrinkage of at least `MIN_SHRINKAGE = 1.25`.** The correction is
  `1/(ratio − 1)` times the last difference, unbounded as the ratio approaches
  1. The +2902 ladder's ratio was **1.0327** and its correction 30.6× its own
  last difference. The floor is chosen from a measured gap — the Born ratios run
  1.0327, then 1.3968, 1.4430, 1.6520 — not from the case it must reject, and
  any floor in 1.10–1.333 keeps the same sixteen windows.

Both are needed and neither subsumes the other: the two worst Born disasters are
caught by *different* clauses, 269.688% at ratio 1.0327 with matching signs and
7.078% at ratio 8.75 with opposite ones. **Deliberately not an order band** — an
order floor is ladder-dependent, implying `p ≥ 1.0` at a refinement ratio of 2
and `p ≥ 1.5` at 1.5874, and this document already records the fitted order as a
property of the ladder.

**The fitted order is not trustworthy at all.** On the same two spheres it reads
1.95, 1.66, 1.62, 0.96 and 0.52 depending only on which three spacings are
chosen. *An earlier note in this session called debye's convergence "first order
at h^1.17" on the strength of one ladder; that number is a property of the
ladder and the claim is withdrawn.* What survives is the ordering: the
systematic error is larger than the phase-dependent one.

**And the translation in `posed` does nothing.** `POSE_SHIFT_CELLS` carried a
careful rationale about probing grid phase. It cannot: `size_grid` builds the
box from `pqr.center()` and `pqr.extent()`, so translating the solute
translates the lattice with it and every atom keeps its position relative to its
own nodes. Measured on `ala-gly` at 0.5 Å, shifts of 0.25, 0.50 and 0.75 cells
move the energy by 0.0, 0.0 and one ulp; the rotation moves it by 0.96 kJ/mol.
**A pose spread here is a rotation spread.** One consequence: a spherically
symmetric solute has no rotation either, so the metric is identically zero on a
Born ion — the metric and the closed forms cover disjoint sets of structures,
which is why Q1 below had to be graded twice.

#### The noise floor `converging` does not need, and why `limit` never read the spacings

**2026-08-25.** The repaired guard was left with one open objection: it still
accepts fits whose steps sit below the backend's own phase noise, so `order` and
`limit` could be fitted to jitter. That was taken up and **the clause should not
ship**, for three reasons in descending strength.

**One: the amplifier is already bounded, algebraically.** `order` is
`log|d_prev / d_last| / log(step)`, so `step ** order` is *identically*
`|d_prev / d_last|` and the Richardson correction collapses to
`d_last / (ratio - 1)`. With `MIN_SHRINKAGE = 1.25` the ratio is at least 1.25,
so

    |limit - E_finest| <= |d_last| / 0.25 = 4 x |d_last|

for any ladder this guard accepts. A step at the noise floor moves the limit by
at most four noise units. **The failure a noise floor would prevent is already
excluded by the clause beside it.**

The bound is attained rather than generous. Of the shrink ratios recorded beside
`MIN_SHRINKAGE`, the nearest keep — 1.3968 — licenses an amplifier of **2.5202**
where the rejected 1.0327 would have licensed **30.58**; over 50,000 randomised
accepted ladders the worst observed is **3.9877**. The identity itself holds to
**8e-11** relative in double precision, which is what "exact in real arithmetic"
costs when `x ** (log r / log x)` is evaluated in IEEE.

**Two: there is no gap to choose a floor from, and the floor is anti-correlated
with accuracy.** The sixteen survivors' relative last steps run **0.0247% to
2.7903% continuously**, and the four fits closest to the exact answer (0.002%,
0.025%, 0.027%, 0.033%) hold four of the five *smallest* steps, while the worst
survivor (1.341%) sits mid-list at 0.2605%. Any floor that rejects the worst
rejects all four best. `MIN_SHRINKAGE` was chosen from a measured gap; this
observable does not have one.

**Three: the premise is weaker than it was stated, though not empty.** The
comparison behind it read a step at one spacing against a pose spread measured
at another. Matched — the pose spread taken at the *coarser* of the two rungs
the difference spans, twelve poses, `ala-gly`, both surfaces, three widths:

| surface | w | \|d_last\| | pose std | \|d_last\| / std | / peak-to-peak |
|---|---|---|---|---|---|
| van der Waals | 0 | 7.230 | 1.679 (0.737%) | 4.31× | 1.17× |
| van der Waals | 0.25 | 1.751 | 0.926 (0.423%) | **1.89×** | 0.56× |
| van der Waals | 0.5 | 1.473 | 0.449 (0.206%) | 3.28× | 0.85× |
| molecular | 0 | 6.140 | 2.003 (0.915%) | 3.07× | 0.75× |
| molecular | 0.25 | 1.875 | 0.905 (0.429%) | 2.07× | 0.55× |
| molecular | 0.5 | 1.650 | 0.379 (0.180%) | **4.36×** | 1.09× |

**Above the noise on the statistic this repository gates on, and not by much:
1.89–4.36× of the dispersion.** Against the peak-to-peak *range* the steps are
0.55–1.17×, i.e. at or below it — but `PoseSpread` already says why the range is
the wrong statistic ("the range of a small sample is dominated by its two
extremes... quote it, but gate on `dispersion`"), and a floor built on it would
be built on the volatile summary.

**A phase spread must be quoted with the configuration it was taken at**, which
is the part of this worth keeping. `DEBYE_POSE_DISPERSION = 0.0091` is the
repository's one recorded number and it is the `molecular`, hard, 0.5 Å cell of
the table above — 0.915%, which is that row and not a universal. Used as a floor
across the other five it is 1.1–5.1× too large. *An earlier draft of this
paragraph said "4–25×", comparing it against rungs it was never measured at;
the table above refutes it and is the reason the claim is stated this way now.*

**And a fourth thing fell out that is not about the floor at all: `limit` does
not depend on the spacings.** Because `step` cancels, `Refinement.limit` is a
function of the three energies alone. So `_achieved_spacing`, added because
"Richardson divides by the refinement ratio, so the ratio has to be the real
one", earns its keep on **`order` only**; M8a's orders moving 0.10–0.16 between
the requested and achieved conventions is the whole of its effect.

*Exact in real arithmetic and not in IEEE double — over 20,000 randomised
accepted ladders two spacing triples return a different `limit` float on 0.66%
of them, worst relative difference 7e-15. The independence is a property of the
formula, not of its evaluation.*

The corollary matters more than the trivia: the extrapolation is exact only on a
**geometric** ladder. Under `E(h) = L + C h^p` the ratio of successive
differences is `r2^p (r1^p - 1) / (r2^p - 1)`, which equals `r^p` only when
`r1 == r2`; the achieved `ala-gly` ladder is 0.8695 / 0.4545 / 0.2432, ratios
1.913 and 1.869. **No choice of `step` repairs that, because `step` cancels** —
a non-geometric ladder needs `p` solved for rather than read off. Recorded
rather than fixed: the residual is small next to the 0.08–0.48% the extrapolator
is graded at, and the honest first move is to keep ladders geometric.

#### Q1 — the scheme was the whole result

M1c recorded harmonic dielectric averaging taking the Born error from 0.853% to
0.107%, and left it open because the real-structure check used APBS, which
shares the assignment under test. Re-run with references that do not.

**The first implementation failed, and how it failed is the finding.**
"Smoothing over a band of `w` cells" reads naturally as averaging the indicator
over a box of whole cells and blending harmonically. That smears the interface
across three cells, and on the Born closed form it is **worse than the hard
assignment at every rung but one** — 0.862% against hard's 0.806% at 0.25 Å,
against M1c's reported 0.107%. The hard baseline reproduced (0.806% against
0.853%), so the setup was right and the scheme was wrong.

**Ramping the fraction across a single cell, from the exact signed distance,
reproduces M1c's magnitude.** `min_i(|x - c_i| - r_i)` is exact for a union of
spheres **outside** it, which is where this evidence was taken — *inside it is
only an upper bound, and the ramp read it as a depth until 2026-08-25; see "The
other half of that bug" below* — and the solute fraction is a linear ramp over
one cell, the two
dielectrics are blended harmonically, which is the textbook mean for flux normal
to a layered interface.

| a | h | hard | ramp 0.25 | ramp 0.5 |
|---|---|---|---|---|
| 2.0 | 0.500 | 4.569% | **0.087%** | 0.747% |
| 2.0 | 0.250 | 1.450% | 0.138% | **0.185%** |
| 3.0 | 0.500 | 1.492% | 0.441% | **0.310%** |
| 3.0 | 0.250 | 0.806% | **0.100%** | 0.130% |
| 3.0 | 0.125 | 0.453% | 0.098% | **0.060%** |

**Recomputed from the five rows above**: hard ÷ ramp is **3.4–52.5×** in the
`w = 0.25` column and **4.8–7.8×** in the `w = 0.5` column the tests exercise.
*"4–17×" was the figure this line carried and it is the range of neither.* ~~and the width is load-bearing: a whole cell is worse than half a
cell, and two cells is worse than not ramping at all.~~

*The width claim had no table behind it and one spacing-specific test. A sweep on
2026-08-24 was built on a broken observable and is withdrawn below; what replaces
it is fixed-h error against a reference, and there the ordering is neither
monotone nor the same on both axes.*

**The reference-free gate agrees**, which matters because the closed form is a
sphere and the metric is blind to spheres. Pose dispersion on `ala-gly`, twelve
poses, van der Waals:

| h | hard | ramp 0.25 | ramp 0.5 |
|---|---|---|---|
| 1.000 | 2.950% | 1.817% | **0.537%** |
| 0.500 | 0.737% | 0.423% | **0.206%** |
| 0.250 | 0.152% | 0.067% | **0.027%** |

**3.6–5.6× better**, and the means converge sooner: hard is still moving 6.2
kJ/mol between its last two rungs where the ramp moves 1.4. **The ramp at 0.5 Å
is closer to converged than hard at 0.25 Å** — worth a factor of two in
resolution, which is eight times the work.

#### What shipped, and what did not

`DebyeOptions.dielectric_smoothing` is **off by default and the default is
bit-identical** — the corpus reproduces all 58 cases unchanged. It is a knob
rather than a default for one reason: **it is implemented for `van-der-waals`
only, and the shipped default surface is `molecular`.** The ramp needs a signed
distance, and `min_i(|x - c_i| - r_i)` is the distance to a union of spheres
*from outside* — only a bound from inside, which is the other half of the same
defect and is recorded below — where the solvent-excluded surface has no such
formula at all, being three families of patch with a distance function of its
own. Ramping against the wrong surface would return a
number that looked like an improvement, so `dielectric_faces` refuses instead.

**That distance is the next piece of work, and it is the whole of it.** The
evidence that it is worth building is above: on the boundary where the distance
*is* available, a one-cell ramp is 4.8–7.8× better at `w = 0.5` against an exact
reference — 3.4–52.5× at `w = 0.25`, recomputed from the Q1 table above, where
this line used to say "4–17×" — and
3.6–5.6× better on a reference-free one. Nothing about that argument is specific
to the van der Waals surface; only the distance function is.

#### M8a — the solvent-excluded distance, and what it did to the convergence order

**Met 2026-08-22.** M8's ramp needed a signed distance and had one only for a
union of spheres, so it refused the default surface. It no longer does.

**The distance is not a second construction.** This module's header reads
*solvent is the accessible set `A` dilated by the probe*, so the signed distance
to the boundary is `probe - dist(x, A)` — and the three families already compute
`dist(x, A)` and throw it away, each finding a candidate legal probe centre and
asking only whether it is within the probe. `ReducedSurface.signed_gap` asks how
far it is. `sign(gap)` reproduces `inside` **node for node on both surfaces and
both fixtures**, which is the bar, and it is a test rather than a claim.

#### The result: the convergence order, on both surfaces

Richardson over 1.0 / 0.5 / 0.25 Å on `ala-gly`, against **achieved** spacings
rather than requested ones:

| surface | hard | ramp 0.25 | ramp 0.5 |
|---|---|---|---|
| van der Waals | ~~0.32~~ **refused** | **2.31** | **2.48** |
| molecular | 1.01 | **2.39** | **2.38** |

**Re-measured 2026-08-24 and the baseline column is withdrawn.** The
`converging` guard these numbers were read through bounded only the *magnitude*
of successive Richardson corrections, never their sign and never how far they
had to shrink. Repaired — see `sashimi.invariants.MIN_SHRINKAGE` — it **refuses
the hard van der Waals ladder outright**: its corrections shrink by 1.2031×,
under the 1.25 floor, so **no order exists to quote and the 0.32 never described
a fit.** The molecular baseline survives at 1.009 against the 1.05 recorded.
The ramped columns reproduce within 0.13–0.17 of what was recorded.

*Two conventions were also crossed.* This section says "against achieved
spacings"; `tests/test_debye_m4.py` was passing the **requested** 1.0/0.5/0.25 —
a ratio of exactly 2, where `size_grid` lands on 0.8695/0.4545/0.2432 for ratios
1.913 and 1.869. Every order moves 0.10–0.16 between the two. The test now uses
achieved spacings, as this section always claimed.

**What survives, and it is the conclusion rather than the number.** The ramped
orders are 2.31–2.48 on a ladder the repaired guard accepts, against a hard
scheme that reads 1.009 on one surface and cannot be fitted at all on the other.
An interface treatment ceasing to bound the accuracy is still the right reading.
What cannot be said is "from 0.32 to 2.65", because the left-hand number is not
a measurement.

~~Three cross-checks say the limits are real rather than fitted: the two widths
agree to 0.14% on van der Waals and 0.16% on molecular.~~ **Struck — that
agreement is structurally guaranteed and is not evidence.** `dielectric.py:185`
sets `width = min(smoothing * min(spacing), surface_radius)`, a width in *cells*,
so the physical band is `w·h` and vanishes as `h → 0`. Every width therefore
discretizes the *same* sharp-interface continuum problem and their limits must
agree whatever the ramp does at finite h. A width sweep on 2026-08-24 was built
on exactly this mistake and could not have answered the question it was posed —
see "The width sweep, and the observable that could not discriminate" below.

#### The width sweep, and the observable that could not discriminate

**2026-08-24.** The ramp's width has never been swept, and `ROADMAP.md`'s claim
that "a whole cell is worse than half a cell, and two cells is worse than not
ramping at all" had no table behind it. A sweep was run over
`{0.125, 0.25, 0.5, 1.0, 2.0}` cells on `ala-gly` and `fas2`, both surfaces,
three rungs each, on ladders chosen for **consistent achieved ratios** — 1.8821
and 1.8873 for `ala-gly`, 1.5874 and 1.5874 for `fas2`.

**The sweep cannot answer the question it was built to ask, and no refinement of
it could.** Its criterion was whether the extrapolated *limits* agree across
widths, on the reasoning that disagreement would mean the width is a fitted
parameter. But `dielectric.py:185` puts the width in **cells**, so the physical
band `w·h` vanishes as `h → 0` and every width discretizes the same
sharp-interface problem. Agreement of limits is therefore an algebraic
consequence, not evidence — the hypothesis lives at *fixed* h, where callers
are, and the limit is blind to it. **A threshold was named and the observable
could not separate the alternatives**, which is the same failure M9 recorded
three times and is recorded here a fourth.

Two readings were drawn from the sweep before that was noticed, and **both were
wrong**:

- *"The limits drift monotonically with width, so the width is fitted."* The
  drift is extrapolation error. The raw `ala-gly` energies are **non-monotone**
  in `w` at all five rungs while the fitted limit column is strictly monotone.
  On a 5-rung ladder the spread across `w ∈ {0.25, 0.5, 1.0}` collapses from
  **1.69%** on the window the sweep used to **0.062%** one rung finer — a factor
  of 27, which is a fit leaving the pre-asymptotic regime, not physics.
- *"A plateau exists, so the ramp is rescued."* The plateau is guaranteed by the
  paragraph above and proves nothing either.

**What the sweep did establish** is that the instrument was unsound. Every
fitted order in it is unstable between adjacent windows — `3.975 → 1.057`,
`2.127 → 2.691`, `1.865 → 2.484`, `2.489 → 1.221` — and coarse windows return
limits of **−123.07** and **−151.72** against a converged ≈ −208, all labelled
`converging=True`. Those labels are what prompted the `MIN_SHRINKAGE` repair
above. All 24 rows of limit, order and drift are withdrawn.

**The question the sweep should have asked is fixed-h error against a reference
that is not itself the ramp**, and on the one geometry that has a closed form
the ramp wins: on the Born sphere at the finest rung, hard reads 0.624% (a=3)
and 0.853% (a=2) against `w=0.5`'s **0.069%** and **0.137%** — 4.6–9.0×.

*Three statements of one quantity now sit in this section and they do not agree:
4.8–7.8× at `w = 0.5` recomputed from Q1's table above, the 4–17× that
recomputation struck, and this 4.6–9.0×. Nor do the ratios here follow from the
percentages beside them — 0.624/0.069 is 9.0 and 0.853/0.137 is 6.2 — and
neither percentage appears in Q1's table, so this is a different sweep whose
lattice is not named, which is the one thing §12's own rule says a near-field
number must carry. It is flagged rather than rewritten: the configuration behind
it is not recoverable from anything checked in. Quote Q1's table.* That is
a stronger claim than the convergence order it replaces and it does not rest on
a quantity this document has already retracted twice.

**One caveat that disarms the Born evidence for the surface that ships.** On a
lone sphere the solvent-excluded surface *is* the van der Waals surface —
`tests/test_debye_m4.py::test_a_lone_sphere_has_the_same_two_boundaries` asserts
the two masks are equal node for node, with no tolerance. So **every Born-based
validation of the ramp exercises M8a's solvent-excluded distance not at all**,
and ~~prices its cost at ~1× where a real solute pays 11–20×. The measurement
that would decide the default is fixed-h error on a protein against a reference,
and it has not been run.~~

**prices its cost at ~1× where a real solute paid 11–20× when this was written
and pays about 2.5× now (#75, #77).** The caveat itself stands and is worth
keeping in front of anyone quoting the Born evidence for the ramp. What has
changed is that the measurement it names — fixed-`h` error on a real solute
against a reference — is *reachable* without a second solver: the ramp's band is
`w·h` and vanishes with `h`, so the two schemes share a continuum limit and each
scheme's own refinement ladder is an admissible referee for the other. That
route was not available when this paragraph was written, because the reference
was assumed to have to be a different code.

#### Three defects, and the shape they had in common

*A second implementation of one definition found all of them*, which is the
third time that has been the mechanism in this milestone.

- **The radial family wrote its distances into a copy.**
  `np.minimum(..., out=block[fancy_index])` — fancy indexing returns a copy, so
  every value landed in a temporary. The surface came back saturated wherever
  that family was the only one to reach a node, and `sign(gap) == inside` is
  what showed it.
- **`_window` is a bounding box, not a sphere.** `radius - span` is *negative*
  for nodes in the box and outside the sphere, so a "distance" pointed the wrong
  way and `signed_gap` returned values above the probe it is capped by. The
  boolean family tolerates those nodes because it only asks whether a projection
  is legal; a minimum does not.
- **The width is in cells, and a coarse multigrid level's cell is not the fine
  one's.** `build_levels` hands every level the same cell count, so at a 1.0 Å
  request the coarsest level's 6.4 Å spacing turns a 0.25-cell ramp into 1.6 Å —
  wider than the probe, past which the gap field saturates and the ramp stops
  being a ramp. Measured before the clamp: solvent faces came back at **23.0
  instead of 78.54**. The probe is the range the distance carries, so it is the
  width's ceiling.

#### A finding that was withdrawn the same day, and how

The first version of this section reported that **the optimal ramp width is
surface-dependent** — 0.5 cells for van der Waals, 0.25 for molecular, whose
re-entrant patches are tighter — on the strength of molecular's order collapsing
to 0.18 at 0.5 cells. It read like physics and it was a bug.

`signed_gap` filled the whole van der Waals interior with a saturated value,
because `inside` never asks about that region: everything inside the union is
solute whatever the answer. A **distance** cannot take that shortcut — the
solvent-excluded surface touches the van der Waals surface wherever the probe
touches an atom, so a node just inside it is just inside the boundary. The
consequence was a **one-sided ramp**: blended on the solvent side, clamped on
the solute side, which displaces the interface outward by about half the ramp
width. A wider ramp displaced it further, so `molecular` looked width-sensitive
and `van-der-waals` — which had a real two-sided distance from `_union_gap` —
did not.

*Two-sided it was, and that is all this paragraph claims. It is **not** a
distance: inside the union `min_i(|x - c_i| - r_i)` is an upper bound, and the
`van-der-waals` ramp was reading a bound as a depth for as long as it existed.
See "The other half of that bug" below — the same subsection, one lattice
further in.*

With the region fixed, **both widths work on both surfaces and the orders agree
to within 0.01**. There is no surface-dependent width.

*What is worth keeping from the episode* is that pose dispersion alone would
have ratified the bug: at 0.5 cells on `molecular` the dispersion improved 2.2×
while the answer walked away from its own limit. Q0's second half is what
refused it, and a `/code-review` pass is what found the cause.

#### The field axis, measured — and what the instrument can and cannot say

**2026-08-25.** M1c set this as the explicit precondition and no ramp validation
since has measured a field: *"harmonic at w = 0.5 barely improves the worst
near-field error even while it transforms the energy, and debye's consumer reads
the field."* It has been measured now, on three geometries with two references — one of them
independent (the Born closed form) and one of them **not** (debye itself at
finer resolution, in both schemes). That second one is the whole subject of
"The instrument's limit" below, and calling it independent would be the error
this section exists to avoid. **No admissible measurement here shows it improving the near field on a real
solute, and above `w = 0.5` it degrades it** — while improving the energy on the
same fixture, the same surface and the same lattice by a factor of five to nine.
The two axes disagree and protean consumes the one the ramp does not win.

*Stated that carefully because the sphere arm does not say the same thing as the
molecular one, and an earlier draft of this sentence read "does not improve the
near field anywhere" over a table two paragraphs below it showing a 21–23%
improvement at `w = 0.75`. The two statistics on that one geometry disagree with
each other; see Arm A.*

*The width that would actually ship, `w = 0.5`, is the one this instrument
cannot resolve. "The instrument's limit" below is that finding and it is the
part of this section most worth reading.*

##### What was measured, and the two traps it had to clear

Every near-field number in this document has had to clear the same two traps, so
they are stated before the tables rather than after.

**Grid phase**, which swings every finite-difference backend's near field 5–21×
— larger than anything separating the solvers. Handled by sweeping the padding
at fixed resolution and reporting worst *and* median over the sweep rather than
the swing ratio, because M1c's lesson is that the ratio flatters a change that
only raises the floor.

*What that sweep varies, precisely, is the achieved **spacing** — and so `a/h`,
which §12 records as the predictor. It does **not** vary the solute's position
within a cell. `size_grid` returns an odd `n` on every axis and sets
`origin = center - fglen/2`, so the solute's centre lands exactly on a node at
every padding and every resolution; measured, the fractional offset is 0.0 on
all three axes at all five paddings. This is Q0's finding about `posed`'s
translation — the lattice follows the solute — arriving a third time, and an
earlier draft of this section called the sweep a phase sweep without checking.*

**The referee sharing the bias under test.** APBS and DelPhi C++ both assign the
dielectric hard at face centres, so neither can referee an interface change, and
TABI-PB reaches one protein at one mesh density. What is left is refinement, and
it is admissible for a reason specific to this knob: the ramp's band is `w·h` and
vanishes with `h`, so the two schemes discretize the same sharp-interface
continuum problem and share a limit. **Both schemes are run as the referee**, at
two fine resolutions each, so that a verdict can be checked at both ends of the
bracket — including the ramp's own fine solve, which is the referee that should
favour it. How far that bracket can actually be trusted turned out to be the
substantive question, and it is answered below rather than assumed here.

The referee's own behaviour, on `ala-gly`, RMS over the shell:

| | 0.20 → 0.15 | 0.15 → 0.12 | 0.12 → 0.10 |
|---|---|---|---|
| same scheme, hard | 0.00068 | 0.00039 | 0.00022 |
| same scheme, ramp | 0.00094 | 0.00069 | 0.00040 |

| the two schemes at one resolution | 0.20 | 0.15 | 0.12 | 0.10 |
|---|---|---|---|---|
| hard against ramp | 0.00392 | 0.00263 | 0.00221 | 0.00189 |

Each scheme converges; the two are still 0.0019 apart at `h = 0.1` and closing
only about first order. **So the yardstick is ambiguous at the level of the
coarse hard error itself** — 0.0045 — and the verdict below is claimed outright
only where the gap between the two candidates comfortably exceeds that.

**And the shared limit was a theorem here rather than a measurement. It has now
been measured, and it holds.** The band is `w·h` and vanishes with `h`, so the
two schemes discretize one continuum problem — that is M8a's argument. The
*energy* ladder bore it out; the **field** ladder did not, which left open the
possibility that the comparison below was meaningless rather than merely
bounded. `studies/field_axis/shared_limit.py` and `consistency_born.py` settle
it, and the settling took two measurements because the first was not decisive on
its own.

**The test that needs no referee.** At each `h` the two schemes solve on the
*identical* lattice, so `δ(h) = ‖φ_ramp(h) − φ_hard(h)‖` over a fixed physical
shell involves no reference and no cross-lattice comparison. Two null controls
first, because a norm that fails to reach zero is also what a broken measurement
channel looks like: at `w = 1e-5`, where the ramp is bit-identical to hard, and
at a uniform dielectric, where there is no interface to blend, **δ is exactly
0.000000 at all six rungs**. On `ala-gly`, six rungs to 21.4 M points:

| h | 0.4873 | 0.3399 | 0.2436 | 0.1751 | 0.1384 | 0.1077 |
|---|---|---|---|---|---|---|
| δ | 0.011762 | 0.006664 | 0.004713 | 0.003147 | 0.002607 | 0.002110 |

δ is **all shape**: the constant-offset part is ~1e-5 throughout, so the two
schemes do not differ by a monopole. But it does not extrapolate to zero.
`Refinement` refuses two of the four windows outright and returns limits *above*
the last measured value on the other two, so the shipped extrapolator cannot
settle it. Fitting the two models instead: `δ = C·h^p` gives `p = 1.13` with an
RMS residual of 4.9e-4, and `δ = δ∞ + C·h^p` gives **δ∞ = 0.0015** at `p = 1.78`
with 1.4e-4 — **3.6× better for one extra parameter**.

**That is not enough to conclude anything, and the reason is worth keeping.** A
non-zero δ∞ would mean one of the two schemes is not a consistent discretization
of this problem on the field — the O(1)-at-the-interface class §12 records at M0
and that interface methods exist for. It is also exactly what a slowly-converging
δ looks like over six points and three parameters. **The observable cannot
separate them**, which is this document's recurring failure and is why the second
measurement exists.

**The Born sphere separates them, because there the reference is exact.**
`‖φ_X(h) − φ_exact‖` on the same shell construction, `a = 3 Å`, six rungs:

| h | hard | w = 0.25 | w = 0.5 | w = 1.0 |
|---|---|---|---|---|
| 0.500 | 0.000610 | 0.001782 | 0.001758 | 0.001693 |
| 0.250 | 0.000592 | 0.000617 | 0.000536 | 0.000560 |
| 0.140 | 0.000282 | 0.000298 | 0.000232 | 0.000225 |
| 0.109 | **0.000196** | 0.000223 | **0.000173** | **0.000145** |

**Every column falls, none flattens.** Fitted over the whole ladder the orders
are 1.02 (hard) and 1.56–1.63 (ramped). **So both schemes are consistent: each
converges to the exact field, and neither carries a non-vanishing interface
error.** The field axis is bounded, not meaningless, and the comparison below
stands.

*Two things not to read into that table.* It is **one lattice per rung and not a
phase sweep**, so the fine-rung ordering — where the ramp comes out ahead of hard
— is a lottery ticket and not a result; Arm A sweeps 21 paddings for exactly this
reason. And a lone sphere's solvent-excluded surface is its van der Waals
surface, so this says nothing about re-entrant geometry. Consistency is a
property of the scheme, and that is all it is being asked.

**What stays open is narrow and named.** Given both schemes are consistent, the
economical reading of `ala-gly`'s δ is slow convergence rather than different
limits — but it decays at ~0.85 where each scheme converges faster, and nothing
here explains that. One candidate mechanism, recorded as a hypothesis and not a
finding: on a molecular surface the two schemes reach the boundary through
*different code* — the ramp through `signed_gap`'s three distance families, the
hard branch through `inside`'s three boolean ones. `sign(gap)` reproduces
`inside` node for node, but the distance path carries degeneracies the boolean
path does not (see the rim-axis subsection), and nothing has established that
the faces they affect vanish as a fraction of the band as `h → 0`.

##### Arm A — the Born sphere, against the exact potential

Worst-direction error at a **fixed physical radius** `a + 1.5 Å`, 21 paddings
from 10 to 20 Å at a 0.5 Å request, `van-der-waals`:

| a | w | worst | median | best | swing |
|---|---|---|---|---|---|
| 2.0 | 0 (hard) | 3.290% | 2.210% | 1.234% | 2.67× |
| 2.0 | 0.5 | 2.928% | 2.585% | 2.065% | 1.42× |
| 2.0 | 0.75 | **2.557%** | 2.256% | 1.565% | 1.63× |
| 2.0 | 1.0 | 2.706% | 2.192% | 1.921% | 1.41× |
| 3.0 | 0 (hard) | 2.444% | 1.968% | 0.433% | 5.65× |
| 3.0 | 0.5 | 2.162% | 1.863% | 1.354% | 1.60× |
| 3.0 | 0.75 | **1.880%** | **1.563%** | 1.425% | 1.32× |
| 3.0 | 1.0 | 2.021% | 1.847% | 1.309% | 1.54× |

The worst case improves 12–23% at `w ∈ [0.5, 1.0]` and the swing collapses — but
**the swing collapses because the floor rises**, 1.234% → 2.065% and 0.433% →
1.354%. That is M1c's shape, reproduced with a wider sweep. `w = 0.25` and
`w = 2.0` are neutral to worse on both radii and are omitted for space.

**The same geometry graded by Arm B's statistic says the improvement is a tail
effect.** RMS over a 2–3 Å shell against the exact potential, same 21-padding
class:

| a | w | worst | median | best | worst / median / best **of the per-padding ratio** |
|---|---|---|---|---|---|
| 2.0 | 0.5 | 0.00231 | 0.00184 | 0.00137 | 1.80× / 1.19× / 0.73× |
| 3.0 | 0.5 | 0.00195 | 0.00164 | 0.00131 | 2.88× / 0.96× / 0.69× |
| 3.0 | 0 (hard) | 0.00202 | 0.00156 | 0.00061 | — |

*The last column is the worst, median and best of the ratio taken **padding by
padding**, not the ratio of the three columns beside it — those would read 0.97
and 1.05 at `a = 3`. Two different statistics over the same 21 lattices, and the
header used to say neither.*

Median unchanged, floor doubled. **By this statistic the ramp is roughly neutral
on a sphere and raises the best case it used to have** — where by the
worst-direction statistic above it improves the worst by 12–23%. One geometry,
one exact reference, two summaries that disagree; the section says so rather
than picking one.

##### Arm B — a real solute, against a refined referee in both schemes

RMS over a shell **2–3 Å outside the solvent-accessible surface** — a fixed
physical band, so the same shell is graded at every resolution, and guaranteed
more than two coarse cells outside the solvent-excluded surface at both. Ratios
are `ramp ÷ hard`, one per referee (fine-hard | fine-ramp).

`ala-gly`, molecular, coarse 0.5 Å, five paddings 10–14 Å (achieved h 0.4672 to
0.4986, which is a full cell-fraction of phase):

| w | ratio range over phase and referee |
|---|---|
| 0.25 | 0.62× – 1.68× |
| 0.5 | **1.53× – 2.92×** |
| 0.75 | 2.41× – 4.54× |
| 1.0 | 3.16× – 6.63× |
| 2.0 | 6.42× – 12.92× |

`fas2` (906 atoms), molecular, one padding, four referees:

| w | coarse 1.0 Å | coarse 0.5 Å |
|---|---|---|
| 0.25 | 0.78× – 0.86× | 0.79× – 1.49× |
| 0.5 | 0.90× – 1.07× | **1.22× – 2.57×** |
| 0.75 | 1.12× – 1.31× | 2.04× – 3.72× |
| 1.0 | 1.33× – 1.55× | 2.98× – 4.96× |
| 2.0 | 1.87× – 2.10× | 6.13× – 9.08× |

**Monotone in width on both structures.** The scope is much narrower than the
tables look, and the subsection below is why rather than a caveat on it.

- **`ala-gly` carries the verdict.** It is the only arm refereed at 4× the
  coarse spacing, which is the regime the next subsection shows is needed.
- **`fas2` carries direction only.** Its referees are **1.93× and 2.40× finer**
  than the coarse solve — the regime where a referee reports its own scheme's
  bias, by the measurement below. The scripts print `-> NO` on the
  discriminability check for all eight of its rows and this section does not
  overrule them.
- **`w = 0.25` straddles** — 0.62–1.68× and 0.78–1.49× — and is not settled at
  any resolution.
- At protean's **1.0 Å** the eight `w ≤ 0.5` gaps run **0.0017–0.0061** against
  an ambiguity of 0.0057, so the largest of them *exceeds* it and the rest do
  not: nothing is claimed there either. *An earlier draft quoted the range as
  0.0017–0.0054, which is not the data's.*

So what is claimed is `w ≥ 0.75` **on `ala-gly`**, where the separation is
2.4–13× against every referee at `h ≤ 0.12`, and 2.2× if the `h = 0.15` rung is
kept in scope; at `w = 0.5` the sign is
settled and the magnitude is a bracket; below that, nothing. *An earlier draft
claimed "1.4–13× at `w ≥ 0.75` … at every referee resolution, every phase, both
structures". The 1.4 floor is `fas2`'s 1.118× rounded the wrong way, and `fas2`
is the arm that cannot carry it.*

##### The instrument's limit, found by trying to break it

Two controls were run to refute the verdict above. One killed an objection; the
other bounded the claim, and the bound is the more useful half.

**Interpolation contributes exactly nothing, and that is measured rather than
argued.** The referee is read at coarse-node coordinates by trilinear
`value_at`, which is the obvious way for this whole comparison to be an
artefact — a smoother coarse field could look worse against an interpolated
reference for reasons that have nothing to do with accuracy. `size_grid` is a
pure function of the structure and the `GridSpec`, so paddings exist where the
fine lattice nests *exactly* in the coarse one: at 9, 11, 13, 15 and 17 Å a
0.5 Å request gives `n` and a 0.25 Å request gives `2n − 1` on every axis over
the same box, so every coarse node **is** a fine node. Read by index instead of
by interpolation, at two paddings and both referee schemes:

    index vs interp, RMS over the shell: 0.000000   (referee magnitude 0.38)

and every ratio agrees to three figures. **The objection is dead.**

**What does move the verdict is how fine the referee is, and at `w ≤ 0.5` it
moves the sign.** Same coarse solves, same shell, padding 11, only the referee
changed:

| referee | w = 0.25 | w = 0.5 | w = 0.75 | w = 1.0 | ambiguity |
|---|---|---|---|---|---|
| 0.25, hard | 1.02× | 2.18× | 3.22× | 4.10× | 0.00471 against a signal of 0.00451 — **104%** |
| 0.25, ramp | **0.35×** | **0.74×** | 1.38× | 1.93× | |
| 0.15, hard | 1.09× | 2.31× | 3.38× | 4.28× | 0.00273 against 0.00436 — **63%** |
| 0.15, ramp | 0.52× | 1.33× | 2.16× | 2.86× | |
| 0.12, hard | 1.13× | 2.37× | 3.46× | 4.38× | 0.00223 against 0.00428 — **52%** |
| 0.12, ramp | 0.62× | 1.53× | 2.41× | 3.16× | |
| 0.10, hard | 1.14× | 2.38× | 3.46× | 4.38× | 0.00189 against 0.00430 — **44%** |
| 0.10, ramp | 0.69× | 1.68× | 2.60× | 3.39× | |

**A referee only 2× finer than the candidates says whatever its own scheme
says.** Each scheme's fine solve sits nearest its own coarse solve because they
share a construction — which is exactly what "a reference that shares the bias
under test" means. §12 has recorded that trap three times against *other*
codes; this is the first time it has been caught **inside one solver**, between
two settings of one knob.

**It resolves in one direction as the referee refines, and the direction is the
whole of what can be said.** The hard-referee column is converged by `h = 0.15`
— 2.31 / 2.37 / 2.38 at `w = 0.5` — while the ramp-referee column climbs
monotonically toward it, 0.74 / 1.33 / 1.53 / 1.68, and has not arrived. That is
the shared bias being squeezed out, and it is squeezed out **upward**: every
referee from `h = 0.15` down puts `w = 0.5` above 1, and the one still moving is
moving that way. Read as a bracket, `w = 0.5` sits somewhere in **1.7–2.4×** and
the lower end is a lower bound rather than an estimate. `w = 0.25` brackets
**0.69–1.14×** and straddles 1, so it is undecided outright.

**So the honest scope is this.** At `w ≥ 0.75` **on `ala-gly`** the separation is
**2.4–13×** and clears the ambiguity at every referee at `h ≤ 0.12` and every
phase — 2.2× if the `h = 0.15` rung is kept in scope, where the ramp referee
reads 2.16 at `w = 0.75`. `fas2` agrees in direction only, and the 2×-finer
referee cannot discriminate at any width. At `w = 0.5`
the bracket is 1.7–2.4× and does not contain 1, so the *sign* is settled and the
magnitude is not; only the 2×-finer referee, whose ambiguity exceeds its own
signal, disagrees. **`w = 0.25` is undecided outright** — its bracket straddles.
And the cross-scheme spread never falls below about 40% of the signal at any
resolution reachable here, which is the ceiling on all of it.

**And `w = 0.5` is the width that matters**, because it is the energy optimum.
What would settle it is a reference that is not debye at all: a TABI-PB exterior
field — `tabipb/vtk.py` already parses the surface `phi` and `dphi/dn` it needs
and `SolveResult` drops them — or a Kirkwood potential for an off-centre charge,
which `sashimi.analytic.kirkwood_potential` has provided since #85 and which has
not yet been pointed at `w = 0.5`. Until it is, the field axis is graded at
`w ≥ 0.75` and *bounded* at `w = 0.5`.

##### The two axes disagree on one fixture, and that is the finding

`ala-gly`, molecular, four rungs, the same solver and the same lattices the field
was graded on:

| achieved h | points | hard | w = 0.25 | w = 0.5 | w = 0.75 | w = 1.0 |
|---|---|---|---|---|---|---|
| 0.8695 | 35,937 | −230.4391 | −219.8791 | −218.1461 | −222.8878 | −231.9687 |
| 0.4545 | 240,825 | −218.6277 | −211.0382 | −210.4139 | −211.8714 | −214.8247 |
| 0.2432 | 1,537,305 | −212.4877 | −209.1629 | −208.7639 | −209.0317 | −209.5954 |
| 0.1230 | 11,735,977 | −210.5782 | −208.6384 | **−208.2861** | −208.3020 | −208.3832 |

All five columns converge on ≈ −208.3, which is the shared limit the field
argument also rests on. Against it, at 0.4545 Å the hard scheme is **10.34
kJ/mol** out and the ramp at `w = 0.5` is **2.13** — **4.9× closer**; at 0.2432 Å
it is 4.20 against 0.48, **8.8× closer**. *And `w = 0.5` is the energy optimum,
beating 0.25 and 0.75 at every rung — so the width that is best for the energy is
the one that is about twice as bad on the field.*

**A volume integral and a boundary value are not the same quantity, and an
interface scheme is allowed to trade one for the other.** That is what this one
does. Q0 already separated the phase-dependent error from the systematic one and
warned against gating on the wrong half; this is the same lesson on a second
axis, and the observable — not the threshold — is again where it turns.

##### The decision

**`dielectric_smoothing` stays off by default**, and the reason has moved from
coverage to measurement — but the measurement that carries it is the *absence of
a gain*, not the size of the loss. Stated exactly:

- On the one geometry with an **exact** reference the two summaries disagree:
  the worst-direction error at a radius improves 12–23% at `w ∈ [0.5, 1.0]`,
  while by shell RMS the median is unchanged and the floor doubles. **No clean
  gain, and no clean loss.**
- On `ala-gly`'s molecular surface, refereed at 4× the coarse spacing, it
  **degrades** the near field at `w ≥ 0.75` by 2.4–13×, against every referee and
  every lattice. `fas2` agrees in direction and its referees are too coarse to
  count.
- At `w = 0.5` — the width that would actually ship, because it is the energy
  optimum — every referee at `h ≤ 0.15` says worse by 1.3–2.9× and the 2×-finer
  one says better. **Bounded, not settled.**
- The two schemes **do** share a limit on the field — both converge to the exact
  Born potential, measured, so the axis is bounded rather than meaningless. That
  was open when this decision was first written and is now closed.

So: no admissible evidence that the ramp helps the field, clear evidence it hurts
at the wide end, and an instrument that cannot resolve the width that matters or
confirm its own premise. **None of that is a case for moving a default** — and
the axis is *not* discharged. It is bounded, and the bound is the deliverable.

The knob is worth keeping and worth recommending to a caller who wants a
*solvation energy* on a coarse grid: 4.9–8.8× at half the resolution survived
every attempt here to break it. It should not be recommended to a caller
colouring a surface, which is protean, which is what this project exists to
serve.

**What would settle it**, in the order that would do it fastest:

- **A reference that is not debye.** Every field number above is debye against
  debye, and the subsection above measures exactly how far that can be trusted.
  A **TABI-PB exterior field** is the one genuinely independent option and
  `tabipb/vtk.py` already parses the surface `phi` and `dphi/dn` it needs —
  `SolveResult` drops them. A **Kirkwood potential** for an off-centre charge is
  the other, and `sashimi.analytic.kirkwood_potential` has been the field half
  since #85 — so this lever is now available and unpulled, not absent. Either
  would decide `w = 0.5` in an afternoon.
- **A scheme that is sub-cell at the interface and hard at the sample.** The
  degradation is monotone in `w` and vanishes as `w → 0` bit-identically, so the
  two effects are separable in principle; nothing measured says the energy gain
  *requires* the field loss.
- **A second protein.** Two structures agree here and one of them is a
  20-atom dipeptide.

#### The other half of that bug: a bound the van der Waals ramp read as a depth

**2026-08-25.** The subsection above fixed the *region* `signed_gap` computed
over and left the `van-der-waals` branch alone, because that branch "had a real
two-sided distance from `_union_gap`". Two-sided it was. A distance it was not.

`min_i(|x - c_i| - r_i)` measures to the nearest sphere **surface**, and inside
a union that surface may be one a neighbour has swallowed, in which case the
real boundary is further off. It is exact outside the union and an **upper
bound** inside it — never an equality. The counterexample is two balls of radius
2 at `(±1, 0, 0)`: at the origin it returns **−1.0000000** where the signed
distance to the boundary is **−1.7320508**, the intersection circle out at
`sqrt(3)`. The depth is under-reported by 42.3%.

`dielectric.py` reads that number as a depth — `clip(0.5 - gap / (2w), 0, 1)` —
so a face the bound puts inside the ramp band and the truth puts well past it
was handed a **blended dielectric instead of solid solute**. On `fas2` at 0.5 Å
with `w = 0.5` cells, **19.0% of the interior band faces** (8,255 of 43,487
across the three staggered lattices) have a swallowed foot point.

**Why nothing caught it.** `sign(gap)` is graded node for node against `inside`
on both surfaces, and the repair only ever makes an interior value *deeper* — so
the boolean the whole construction is validated on cannot see it. **`|gap|` had
no test at all.** That is the same shape as the three defects listed above: an
oracle that grades the classification and not the quantity.

**The repair is not a second construction, for the same reason M8a's was not.**
The three families compute `dist(x, A)` for the accessible set `A`, and a union
of spheres is the solvent-excluded surface of a **zero probe** — so with the
inflation removed `A` *is* the complement of the union, the rims become the
spheres' own intersection circles, the seats their triple points, and
`-dist(x, A)` is the interior distance. `ReducedSurface._bare` is that solute
with the probe taken away and `_union_signed_gap` keeps the minimum outside,
where it is already exact. The families take the reach as an argument now
instead of reading the probe, because the bare surface's probe is zero and a rim
search of zero radius finds nothing.

Graded against a closed form for two overlapping balls, written out
independently in the test: **exact**, 0.0 over 35,821 graded nodes, where the
bound is out by up to **0.6715 Å** on the same lattice. *An earlier draft quoted
3.6e-15 and 0.46 Å; those are from a random-cluster check, not from the fixture
the test runs, and the test's own guard asserts the bound is out by more than
0.5 — which 0.46 would have reddened.*

**What it moves, measured.** `fas2`, van der Waals, `w = 0.5` cells:

| h | hard | ramp, bound | ramp, distance | move | as a share of the ramp's offset |
|---|---|---|---|---|---|
| 1.0 Å | −2810.169 | −2583.990 | **−2539.203** | +44.79 (1.73%) | 19.8% |
| 0.5 Å | −2523.057 | −2379.524 | **−2369.461** | +10.06 (0.42%) | 7.0% |

The move shrinks with `h` because the band shrinks with it. **Both branches
converge to one limit**, which they must — the width is in cells, so the
physical band vanishes as `h → 0`. Five rungs on `ala-gly`, van der Waals,
`w = 0.5`, and the last rung clamped on `max_points` (requested 0.0625 Å,
achieved 0.1033 Å, 19.7 M points):

| achieved h | points | hard | ramp, bound | ramp, distance |
|---|---|---|---|---|
| 0.8695 | 35,937 | −236.5184 | −225.5166 | −225.1586 |
| 0.4545 | 240,825 | −227.8202 | −218.1488 | −218.0692 |
| 0.2432 | 1,537,305 | −220.5902 | −216.6757 | −216.6482 |
| 0.1230 | 11,735,977 | −219.1037 | −216.4673 | −216.4580 |
| 0.1033 | 19,750,185 | −218.4740 | −216.4666 | −216.4596 |

The two ramp columns agree to **0.007 kJ/mol (0.003%)** at the finest rung, and
the repaired one is nearer the shared plateau at the **first three** rungs. *At
rungs four and five the two columns differ by 0.009 and 0.007 kJ/mol while each
column's own rung-to-rung drift is the same size, and the plateau is estimated
from those same two columns — so the comparison there is circular and is not
claimed.*

**The hard column is the one that has not arrived.** At 19.75 M points it is
2.01 kJ/mol from the finest ramp rung; the ramp at 240,825 points is 1.61 from
the same place. *That is a 1.25× edge, not the "82× fewer points" an earlier
draft claimed — the ramp does not settle at 240,825, it drifts a further 1.61
kJ/mol after it, and the 2.01 is measured against the ramp at 19.75 M.* The
direction is the original finding and the margin is much smaller than stated.

**The cost went down, not up.** The bound was `O(nodes × atoms)` with every atom
evaluated against the whole lattice — 6.8 s on `fas2` at 0.5 Å, most of what the
van der Waals ramp cost. Clamping it at the fill the band cannot read past is
what lets each sphere use its own index window, exactly as
`inside_union_of_spheres` does, and the repair then runs on a thin shell.
Min-of-3, interleaved, warm JIT, `fas2` van der Waals:

| | ramp ÷ hard, 1.0 Å | ramp ÷ hard, 0.5 Å |
|---|---|---|
| before | 2.64× | 2.62× |
| after | 2.41× | **1.28×** |

Ratios rather than seconds, because §12 already records identical code varying
1.9× on load; the hard column is the control and it reads 0.46/0.45 s and
3.58/3.60 s across the two trees.

*The bound is the same share of the ramp at both resolutions — 0.87 s of 1.26 s
at 1.0 Å and 6.8 s of 9.9 s at 0.5 Å, about 69% each. What differs is the price
of what replaces it: the three families cost ~0.55 s at 1.0 Å against the 0.87 s
they remove, and ~1.06 s at 0.5 Å against 6.8 s. An earlier draft said "at 1.0 Å
the bound was never the bottleneck", which is not what the profile says.* *A first pass at this table compared a tree with the compiled
kernel against one without — the worktree had not been synced with the `fast`
extra — which flattered the change. Check `kernel.available()` on both sides of
a before-and-after.*

#### And the same reach was wrong one branch over

`gap = probe - nearest`, so a consumer reading `|gap| < band` is reading
`nearest` out to `probe + band` — but the solvent-excluded branch searched its
rims and seats only to `probe`. A rim further than one probe away can still be a
band node's nearest legal feature, and missing it leaves `nearest` too large,
which reads as *deeper into the solute* than the node is. Exactly the shape
above, on the default surface.

`tests/test_debye_kernel.py::test_the_distance_reaches_past_the_probe_where_the_boolean_twin_stops`
already asserts that the population between one probe and the fill is not empty
— that is the precondition for the compiled kernel's missing probe cull. What
was missing was *searching* it. Measured on `fas2` at `w = 0.5` cells, summed
over the three staggered lattices: **51 faces** move at 1.0 Å and **95** at
0.5 Å — 1 to 34 per lattice — worst fraction change **0.38**, and the energy
moves **0.024%** and **0.012%**, at **no cost** (2.65× → 2.46× and 1.93× → 1.85×
of hard, inside the noise). Both counts grow with the width: at `w = 1.0` cells
it is 763 and 817 faces and a worst change of 0.44. Small, and it is the same
defect: a search radius smaller than what the consumer reads.

**A limitation recorded rather than closed.** A node on a rim's own axis is
equidistant from every point of that circle, so there is no projection to test
for legality and all three families decline it — `usable = length > DEGENERATE`.
It keeps the saturated fill, and `_toroidally_reachable` declines it in the same
place, so `inside` and `sign(gap)` stay consistent and nothing catches it there
either. A rim's axis is the line through two overlapping atom centres, so the
set is measure zero on real coordinates — but **the counterexample above sits
exactly on it**, which is why the closed-form test offsets its lattice and a
second test pins the hole's shape. Closing it means asking whether *any* point
of the circle is legal rather than one, in the numpy loop and in
`kernel.distance_rims` alike, and that is a change with a bit-identity contract
attached.

**Nothing default moves.** `dielectric_smoothing` is off by default and
`signed_gap` has exactly one caller — the ramp — so all 58 debye corpus
recordings reproduce unchanged. What moved is every answer the knob produces,
and on `van-der-waals` that is the surface all of M8's evidence was taken on.

**And the Born ion does not move either, which is the point.** A convex sphere
on its own has no second surface to swallow the first, so the bound *is* the
distance there. Checked across the change on twelve configurations — two radii,
two spacings, three widths — every energy reproduces **to the last digit**. So
M8's headline evidence for the van der Waals ramp, 4.6–9.0× against the Born
closed form, was taken on the one geometry where this defect cannot appear. That
is the same blindness `test_a_lone_sphere_has_the_same_two_boundaries` records
for M8a's solvent-excluded distance, hitting a second construction.

### M9 — a boundary that does not cost `O(nodes × atoms)`

> **Read this first. Most of this section is the history of a plan that was
> measured and dropped.** M9 was "focusing, for the boundary and not the
> resolution" until 2026-08-22, when the step (c) review priced a coarse
> pre-solve at **2.0–2.5 s** on serum albumin against a strided exact face at
> **0.24 s**, at no better accuracy and for a second grid hierarchy.
> **The milestone keeps its purpose — the
> `O(nodes × atoms)` boundary is still what makes debye superlinear — and loses
> its mechanism.** Steps (a) and (b) are merged and still stand. The argument
> below for *why the boundary must change* is intact and worth reading; the
> argument for *a coarse grid as the way to change it* is superseded by
> **"The review of step (c), and why focusing is not being built"** at the end
> of this section. Where the two disagree, the review measured it and this text
> did not.

**Named 2026-08-22, and the name is the argument.** Both incumbents focus:
`mg-auto` solves a fine box at `extent + 2 × padding` and a coarse box at
**`CFAC × fglen`** — 1.7 times the *fine box*, not 1.7 times the extent, which
is about 2.1× the extent on serum albumin — carrying the coarse solution inward
as the fine grid's Dirichlet data. *An earlier draft of this section said
`CFAC × extent`. Under that rule 69 of the corpus's 100 cases fail containment
and APBS 3.4.1 hard-aborts with "Finest mesh has fallen off the coarser
meshes"; `apbs/grid.py` has it right and this section did not.* The usual reason to want that is *resolution* — detail where the
chemistry is, without paying for it everywhere. **That is not why debye would
want it**, and calling this "focusing" without the qualifier would send the next
reader looking for an active site.

#### What focusing is not

It is not why APBS is faster, and this document said it was. "APBS's near-linear
exponent is focusing" has been repeated three times here on no measurement.
Measured, at 1,156 residues and a 1.0 Å request:

| | points per solve | achieved spacing |
|---|---|---|
| APBS | **2,679,201** | 0.843 × 0.827 × 0.805 Å |
| debye | 1,625,505 | 0.963 × 0.973 × 0.991 Å |

APBS solves **1.65× more points, at a finer spacing than it was asked for**, and
it solves more of them — each `elec` block runs coarse then fine, and an energy
needs two blocks. That is roughly three times debye's total grid work, and APBS
still finishes in **11.94 s against debye's 33.0 s**. *Focusing costs an extra
solve; it cannot be the source of the speed.* What is left as the explanation is
the thing that was always true and never needed a mechanism: APBS's multigrid is
compiled and thirty years tuned.

#### What focusing is for, here

`sashimi.apbs.input` asks for **`bcfl sdh`** — *single* Debye-Hückel. APBS places
the entire molecule at the boundary as **one sphere carrying the net charge**,
which is `O(boundary nodes)` and independent of how many atoms there are. It can
afford something that crude *because* focusing moves that boundary out to 1.7×
the fine box — where the monopole is still 5–61% RMS wrong, but where the error
is exponentially damped by salt before it reaches the fine face (4.5 Debye
lengths on albumin) and where the `l ≥ 1` content it omits falls as
`(r/R)^l = 0.588^l` — and then refines inward.

debye has no focusing, and the claim that it therefore *cannot afford* a crude
boundary was asserted here and is **false as stated**: `sdh` on debye's existing
box costs only **0.37–1.83%** of the energy, takes albumin 51.1 s to 28.3 s, and
moves whole-solve scaling from atoms^1.176 to atoms^0.972. **It passes this
milestone's exit criterion as first written, in ten lines** — which is a fact
about the criterion, not about the design.

The real argument is geometric and is stronger. At `padding = 10 Å` debye's box
face sits at `r_max/|R| = 0.89–1.25` — **inside the solute's own circumscribing
sphere** for every structure above ~2,000 atoms — and no single-centre expansion
converges there at any order. Measured on fas2 the reference-state error runs
18.2% (monopole) → 5.2% (+dipole) → 4.5% (+quadrupole) → 2.2% (+octupole):
stalling, not converging. *That* is why the multi-atom sum is there, and why a
multipole expansion is not the cheap way out.

**The two states are not equally at risk and the plan treated them as one.** On
albumin the reference (κ = 0, uniform dielectric) boundary has an RMS of
**117.5 kT/e** against the screened state's **0.0398** — three thousand times —
so every bit of a crude boundary's damage lands in the reference state, which is
also the expensive one. A distance cutoff, the obvious cheap fix, works only on
the screened half.

It sums the screened tail over **every atom at every face node** — the
equivalent of APBS's `mdh` — and section 12's measurement of that is:

    82,050 face nodes × 18,242 atoms = 1.50 billion pairs
    43% of a solve, scaling as atoms^1.45

with face-nodes × atoms predicting exactly 1.45. *The share read 36% here until
2026-08-22; re-measured idle it is 12.94 s of a 30.80 s solve = 43.0%, and this
document's own other pair (12.95 s of 33.0 s) already implied 39.2%. The node
count, the pair count and the exponent all reproduce exactly — 82,050 face nodes,
1.4968e9 pairs, a three-point CPU fit of 1.486 and a face-nodes × atoms fit of
1.460.* **Every other stage in debye is
near-linear now**, so that single term is what makes the whole solver
superlinear, and it grows to dominate as structures get bigger.

So the trade the two solvers have made is the opposite of how this document has
described it:

| | boundary condition | extra solve |
|---|---|---|
| APBS | one sphere, `O(nodes)` | yes — the coarse grid |
| debye | every atom, `O(nodes × atoms)` | no |

**debye's single box was recorded as the simpler design** — `size_grid`'s
docstring says "there is no coarse grid and no focusing, which is the one place
debye is structurally simpler than both incumbents: the box is solved directly,
so `padding` is the whole boundary story." That is true, and the simplicity is
what forces the expensive boundary. ~~The cost of not having a coarse grid is
larger than the coarse grid.~~

***Struck 2026-08-22 by the step (c) review, which measured both halves of it.***
The cost of not having a coarse grid is **0.24 s** on serum albumin — that is what
the exact multi-atom sum costs once it is evaluated on a strided sub-lattice of the
face instead of every node — and the marginal cost of *having* one is **2.2–2.5 s**,
priced on an idle machine with both of focusing's own optimisations granted in
advance. The sentence is wrong by **a factor of about nine**, in the direction that
flattered the plan. *An earlier draft of this correction said nineteen, from the
review's first pass; the audit that re-measured it on an idle machine is the number
above.* See "The review of step (c)" below.

#### Why this is the better structural change

An earlier draft of the accuracy plan proposed a reaction-field split — solve for
the smooth reaction potential so the singular self-energy never reaches the grid,
which is what makes DelPhi's energies four thousand times sharper on a sphere.
That remains attractive and it has a trap: the naive split's source term is
`O(grid points × atoms)`, which is the same shape as the boundary problem it
would sit next to. It needs a cost model before any physics.

Focusing's trap is a different one, and the first draft of this section walked
into it: **a coarse box needs a coarse `build_levels`, not just a coarse
multigrid solve.** `build_levels` calls `dielectric_faces` and `screening_nodes`
at every level, so a second box is a second *geometry* — ~~47–55% of a run by
M7's own row~~ — where the linear algebra it was priced against is 11%.
***Corrected 2026-08-22 by the step (c) audit: that figure is misattributed and
it overstates the trap by about 2.5×.** M7's row says 86% at 0.5 Å and 92% at
1.0 Å; the 47–55% belongs to a different table and names the **one-time**
`ReducedSurface` build — precisely the part a shared surface removes and a second
box does **not** repeat. The per-lattice share a coarse hierarchy genuinely
duplicates is **18.3%** by that table's own rows and **21.2%** measured. Linear
algebra re-measures at 12.8%. This over-pricing is the root of the two focusing
costs the first review pass also got wrong, and it is the sentence that should be
read sceptically first.* Priced the way
`mg-auto` does it, with one shared `dime` for both boxes, focusing is a **net
loss on every structure measured**: 1.265× / 1.203× / 1.168× / 1.012–1.117× on
fas2 / 1a63 / actin / mache. The coarse grid has to be genuinely coarse and the
`ReducedSurface` has to be shared across both boxes, and neither is optional. It is also the most conservative option available:
it changes *where the Dirichlet data comes from*, not what equation is being
solved, and both incumbents have run it for decades.

#### The review found a free win ahead of the milestone, and three profiles had missed it

`_blocker_table(rims)` and the stacked `origins` / `normals` / `ring_radii` are
**pure functions of `rims`**, which is itself a `cached_property` — and
`build_levels` rebuilt all four on **every one of a solve's sixteen lattices**,
re-stacking 245,456 rows of an answer it already had.

Measured by A/B in one process on serum albumin, eight `inside()` calls:
**12.26 s recomputed against 10.62 s memoized**, so about **3.3 s of a 33 s
solve**. Bit-identical — the corpus reproduces all 58 cases — and it needs no
new grid, no re-recording and no accuracy argument.

**Why three profiling passes walked past it.** Every profile in this milestone
attributed cost *by function*: `decide_rims`, `build:rims`, `family:toroidal`.
This work is inside `_toroidally_reachable`, so it was attributed to the rim
family, which is exactly where a reader would expect rim work to be. What a
by-function profile cannot show is that a stage's input **does not depend on the
loop it sits in** — and that is the same lesson M7's `cached_property`
mis-attribution taught in the other direction, where a one-time cost was charged
to a per-lattice caller. *Ask what a stage depends on, not only where it runs.*

It is also a prerequisite rather than a detour: a coarse box means a second
hierarchy asking for the same rim geometry, and paying for it twice per lattice
is what would make that second hierarchy unaffordable.

#### The `sdh` baseline, measured — and the criterion that catches it

**2026-08-22, M9 step (b).** `sashimi.debye.sources.single_debye_huckel_solute`
builds the solute as one sphere at its centroid carrying the net charge, with
the circumscribing radius. It exists to be **measured against**, because the
review found that `sdh` on debye's existing box passes M9's original exit
criterion in ten lines — and a milestone a ten-line change passes is not stating
its milestone.

Against the exact multi-atom sum, same box, same lattice, `molecular` at 1.0 Å.
The near field is sampled on a shell 3–4 Å outside the atoms, which is what a
consumer colours a surface with:

| structure | energy | near-field r | sign | magnitude | boundary cost |
|---|---|---|---|---|---|
| fas2, 906 atoms | 0.981% | 0.9986 | 98.2% | ~~0.983~~ **1.0447** | 0.19 s → **0.00 s** |
| 1a63, 2,065 | 1.530% | 0.9999 | 99.6% | ~~1.008~~ **0.9870** | 0.55 s → **0.00 s** |
| serum albumin, 18,242 | 0.370% | 0.9912 | 97.0% | ~~1.001~~ **1.0506** | 12.95 s → **0.01 s** |

***The magnitude column was re-measured 2026-08-22 and does not reproduce.***
Energy, `r` and sign all reproduce to the digit; magnitude does not, under any of
four estimators tried (RMS ratio, mean-|·| ratio, regression slope, median ratio
— all four agree with each other and none gives 0.983). The corrected column
matters, because **`sdh` breaches the 1.02× magnitude clause on fas2 and albumin**
— which the original column concealed, and which is a *second* reason the gate
should catch it.

**The boundary stage disappears** — 12.95 s to 0.01 s at 1,156 residues, the
whole of the only superlinear term debye has. That is what makes `sdh` the
baseline rather than a curiosity.

**And it fails the rewritten accuracy gate on two of three structures**, at
`r ≥ 0.999` and sign agreement `≥ 99%`. That is the point: the *original* gate —
the Born ion and `ala-gly` pose dispersion — passes it trivially, because a
one-atom solute makes `sdh` and `mdh` bit-identical and a net-neutral dipeptide
makes an `sdh` boundary identically zero. ~~**The rewritten criterion was checked
against the very approximation that gamed its predecessor, and it catches it.**~~

***Struck 2026-08-22. It catches `sdh` on the observable measured here and not on
the observable the criterion named.*** The rewritten criterion said
`residue_potentials`; the numbers in this table are the **3–4 Å shell**. Graded on
`residue_potentials`, `sdh` scores r = 0.999996 / 100.000% on fas2, 0.999999 /
100.000% on 1a63 and 0.999980 / 99.654% on albumin — **it passes on all three**,
because residue pooling averages away the near-surface `l ≥ 1` error that is its
entire defect. The criterion now names the shell explicitly. *This is the second
time the same ten-line change has passed a bar in this milestone, and both times
the threshold was right and the **observable** was the half left unpinned.*

So M9's bar is now a number rather than an aspiration: **beat r = 0.9912 and
97.0% sign agreement on serum albumin while keeping a boundary that costs
0.01 s.** Whether focusing can is the open question; `sdh` is what it has to
beat, and the energy column says the target is not far away — 0.37% on the
largest case, where the near field is the half that is wrong.

*One asymmetry worth carrying into the design.* The error is not where size
suggests: albumin's energy error is the **smallest** of the three (0.370%) and
its near-field correlation the **worst** (0.9912). Net charge is why —
albumin carries −30 e where fas2 carries +4.05, so the monopole that `sdh` keeps
describes albumin's far field better while its neglected `l ≥ 1` content is
larger in absolute terms near the surface. **A gate on energy alone would have
ranked these three backwards.**

#### What it will cost, stated before it is built

It changes recorded answers, and this is the axis where the bit-identity
discipline that carried three blocks of performance work gives no cover at all.
The boundary values move, so every corpus energy moves, and each has to be
re-recorded against a reference that does not share the approximation. **Not
M8's refinement limit**, which was on that list and has been struck: it shares
debye's box, and with the exact boundary its own limit is padding-dependent
(−213.00 / −206.47 / −211.23 at padding 5 / 10 / 20 Å), so it cannot referee a
change to what the box edge is held at. `grade_refinement` varies only
`resolution`, never `padding`, so a boundary-model error passes through
Richardson untouched — measured, it reproduces the bias in its own extrapolated
limit to four significant figures and reports `converging = True`. TABI-PB and
the closed forms remain.

The accuracy question is genuinely open in both directions. A coarse solve is a
*better* boundary than a monopole and a *worse* one than an exact multi-atom
sum, so debye may lose accuracy at the boundary and gain it back in being able
to afford a tighter box or a finer grid for the same wall clock. **That trade is
the milestone**, and it is why M9's exit criterion above requires the accuracy
half as well as the scaling half.

#### The review of step (c), and why focusing is not being built

**2026-08-22.** Step (c) was written up as a design — a coarse box at
`CFAC × fglen` with an `sdh` boundary, a genuinely coarse lattice, a shared
`ReducedSurface`, and the fine face by interpolation — and put through the same
adversarial pattern that rewrote this milestone's exit criterion. Six lenses,
**37 objections, 6 selected, 1 refuted, 5 survived**; every survivor ran code.
The measurements lens died mid-run and was re-run afterwards as a dedicated
audit of the numbers the other five produced — **and it overturned three of
them and found a defect none of the five had looked for.** Both passes are
recorded here, because the difference between them is the more useful artifact.

**The decision: build a strided exact face, do not build the coarse grid.** But
*not* for the reason the first pass gave.

##### What focusing actually costs, after the audit

Evaluating the existing exact multi-atom sum on a **strided sub-lattice of the
fine face** and interpolating up — no second box, no second hierarchy, no coarse
solve — is the change being made. Both candidates were then measured on an
**idle machine** (1-minute load 2.3–4.0 on 8 cores, against ~11 during the first
pass), min-of-N, `process_time`:

| | boundary cost, albumin | near-field r | sign | magnitude | energy vs exact |
|---|---|---|---|---|---|
| `mdh`, every node (today) | 12.94 s | — | — | — | — |
| `sdh`, one sphere (step b) | 0.006–0.008 s | 0.9912 | 97.0% | 1.0506 | 0.370% |
| coarse pre-solve, CFAC 1.7/3× | 2.23–2.46 s | 0.999999 | 99.987% | 1.0002 | −0.0253% |
| **strided exact face** | **0.24 s** | **1.000000** | **99.996%** | **0.9997** | **+0.0681%** |

End to end, with the shared `ReducedSurface` granted to focusing as its design
asks:

| | fas2 | 1a63 | albumin |
|---|---|---|---|
| exact `mdh` baseline | 1.21 s | 2.42 s | 31.37 s |
| coarse pre-solve | 1.18 s | 2.15 s | 20.19 s |
| **strided exact face** | **1.01 s** | **1.88 s** | **18.37 s** |

**The first pass priced focusing at 9.3 s and 19×; it is 2.0–2.5 s and 8–9×.**
End to end the first pass reported albumin at 86.6 s against 44.4 and called
focusing a net loss on two rungs. Both are wrong: focusing is **1.12× / 1.13× /
1.55× faster than the untouched baseline** on the three rungs, and the strided
face beats it by **10–17%**, not by 50–95%. On albumin that margin (1.8 s) is
barely outside the ±1.5 s run-to-run spread of the fine solve itself. The audit
could not reproduce the first pass's coarse-box costs at any load factor — the
same machine ran its `ReducedSurface` build 1.13× while its coarse geometry ran
4.1× and its coarse solve 5.3× — so **something other than contention differs,
and it is not yet explained.**

**So the case against a coarse grid is a complexity case, not a speed case.** A
second grid hierarchy, plus a `padding`-dependent boundary knob with no rule
(`cglen = CFAC × fglen` makes 1a63's error 0.531 / 0.485 / 0.413 / 0.286 /
0.230% at padding 3 / 4 / 5 / 8 / 10 Å, crossing the 0.5% gate between 4 and 3,
while `protocol.py` validates only `padding >= 0`), buys 10–17% over a change
that is one knob on the existing sum. **Stating it as a 2× speed win would not
survive the next re-measurement**, and this document has now made that mistake
twice in this milestone.

##### The exit criterion is passed by `sdh`, for the second time

This is the audit's most consequential finding and no lens looked for it,
because every lens graded on the 3–4 Å shell.

The criterion above names **`residue_potentials`**. Graded on
`residue_potentials` — same box, same lattice, `molecular` at 1.0 Å, against the
exact `mdh` solve:

| structure | keys | r | sign | magnitude | verdict |
|---|---|---|---|---|---|
| fas2 | 63 | 0.999996 | 100.000% | 0.9997 | **passes** |
| 1a63 | 130 | 0.999999 | 100.000% | 1.0001 | **passes** |
| serum albumin | 578 | 0.999980 | 99.654% | 0.9997 | **passes** |

**`sdh` passes the rewritten accuracy gate on all three structures.** The
0.9912 / 96.985% recorded above as catching it are from the **3–4 Å shell**,
which the criterion does not name. Residue pooling — six probe points per atom,
averaged per residue, 1,156 albumin residues collapsing to 578 keys — suppresses
exactly the near-surface `l ≥ 1` error that is `sdh`'s entire defect. Worse,
`residue_potentials` returns r = 1.000000 / 100% / 1.0000 for **every** boundary
variant tested, crude ones included: *it cannot discriminate between boundary
models at all.*

So the sentence recorded above — "**The rewritten criterion was checked against
the very approximation that gamed its predecessor, and it catches it**" — is
**false as written.** It catches `sdh` on the observable the review happened to
measure and not on the observable the criterion names. The criterion has been
amended in the phase table to name the 3–4 Å shell explicitly and to demote
`residue_potentials` to a regression anchor.

*This is the third time a bar in this milestone has been passed by a monopole,
and the second time by the same ten-line change.* The pattern is not carelessness
about thresholds — the thresholds were right both times. It is that **the
observable was never pinned down as tightly as the threshold was**, and an
unnamed observable defaults to whichever one flatters the result. **A gate is a
threshold *and* an observable, and the observable is the half that has failed
twice.**

`sdh` also passes the 0.5% energy clause on albumin (0.370%); only fas2 (0.981%)
and 1a63 (1.530%) fail it, and neither is a "large elongated charged case" — a
phrase that occurs once in this document and matches no corpus case, now
replaced by the three named structures.

##### Four measured corrections to the plan above

- **`sdh`'s legitimacy does not come from the circumscribing sphere.** The
  argument was that the coarse face at CFAC 1.7 sits outside it, where a
  single-centre expansion converges. There is no knee there: on 1a63 the
  crossing `r_min/R_circ = 1.0` falls at CFAC ≈ 1.20, and the local power-law
  exponent is **3.69 immediately below and 3.62 immediately above** — a smooth
  ~`CFAC^−3` decay straight through. The derived figure `0.52–0.74` is
  arithmetically correct (`cglen = CFAC × fglen`, so the half-width scales
  exactly, to within ~3% of lattice rounding) and **attached to an inference it
  does not support.** The law is the decay, not the sphere.
- **Salt is not the damper in the state that takes the damage.** `backend.py`
  sets `ionic_strength = 0.0` for the reference state, whose face carries an RMS
  of **117.59 kT/e against the screened state's 0.039579** on albumin — a factor
  of **2,971**, which reproduces. "Exponentially damped by salt before it reaches
  the fine face" is APBS's defence of `bcfl sdh` and **it does not apply where
  the error lands**.
- **The reference state already costs nothing.** `dielectric_faces` returns a
  constant when the two dielectrics match and `screening_nodes` returns zeros at
  zero ionic strength, so the reference hierarchy materialises **no**
  `ReducedSurface` at all — 0.12% of the solvated build on albumin. M4 took this
  win and its docstring records it. What is genuinely unclaimed is smaller and
  different: that hierarchy still allocates 68–112 MB of constant coefficient
  arrays and runs the general variable-coefficient 7-point operator.
- **Sharing the surface is therefore not an independent win either.** Only one
  hierarchy per solve ever touches a surface, so a `surface=` parameter on
  `build_levels` buys nothing unless a coarse *solvated* box exists. It is
  load-bearing for focusing — not sharing costs albumin 19.6 s against 2.8 s —
  and worthless without it. **If focusing goes, it goes.**

##### Two shipped invariants that any replacement disarms

Neither is focusing-specific; both apply to the strided face, and nothing tests
either.

- **The `InputError` that refuses an atom too close to the box face** is computed
  from the fine face's own distance matrix, inside `debye_huckel_boundaries`. Any
  scheme that stops building that matrix at every node removes it — and a strided
  sampler visiting 756 of 82,050 face nodes **can only fire by luck**. The
  exposure is not the singularity the guard documents (the 7.1e6 kT/e it was
  written against) but **silent charge loss**: `_solve_state` zeroes the
  right-hand side on all six faces, so an atom exactly on the face has **20.00%**
  of the structure's total |q| discarded from the source term, falling linearly
  as `1 − d/h` to **0.00% at one full cell**, while `trilinear_weights` objects
  only when an atom is strictly outside the box. Sixteen padding/structure
  combinations were tried and the guard never fires, so no test covers the case
  it exists for. Replace it with an explicit `O(atoms)` check — per-axis min over
  `coords ± radii` against the box edge — which needs no `O(nodes × atoms)`
  matrix at all. *This is a guard whose only implementation is a side effect of
  an expensive thing being removed.*
- **`content_address` cannot see a boundary change.** `_resolved` hardcodes
  `"boundary_condition": "multiple Debye-Hückel on the box face"` and carries no
  further boundary field, and neither candidate changes the box — so
  `grid.as_diagnostics()` is byte-identical across the change. Measured: `mdh`
  −218.627720, `sdh` −215.510188 and a zeroed boundary −215.510188, **1.4260%
  apart, all addressing `63fb17d4943d`** — one file. M8a added
  `dielectric_smoothing` to `_resolved["debye"]` for exactly this reason and said
  so in a comment; whatever knob replaces the boundary must go there too.

##### The referee plan is not sufficient, whichever design is built

> **Read "The referee gap, closed from recordings already in the repo" below
> before quoting the count here.** It stands as written — the word is
> *independent*, and APBS and DelPhi share debye's face-centre assignment, so
> neither is one. What #72 established is that the relationship between the 58
> and the two same-family referees they already had was never written down, and
> that writing it down is worth more than the count suggested.

Of 58 debye recordings, **6** have a TABI-PB counterpart and **22** carry a gated
closed form; **30 have neither, including all 16 above 906 atoms** —
serum-albumin, hca, protein-1a63, lysozyme, barnase, fkbp-apo, fkbp-dmso and
barstar, on both surfaces. TABI-PB's coverage tops out at fas2 itself; its other
two cases are 260 and 20 atoms. *(Counts as of 2026-08-24. Six molecular
recordings were added above 906 atoms on 2026-08-27 — see "The ceiling above 906
atoms was a property of the recordings, not the tool" — taking the counterparts
to 12, "neither" to 24 and the above-906 remainder to 10, all of them
`van-der-waals`, `hca` or `serum-albumin`. The reasoning below is unaffected:
what changed is coverage, not the argument about what a shared discretization
can referee.)* The 22 closed forms are **one-charge geometries,
blind by construction**: born-ion is a single atom, so
`single_debye_huckel_solute` returns its input bit-identically, and kirkwood is a
3 Å cavity with exactly one charge in it — swapping `mdh` for `sdh` moves them
0.0000–0.1077% against gate rtols of 0.010–0.090, where *deleting* the boundary
moves the same cases 3.7–32.5%. They can see that a boundary exists; they cannot
see which multi-centre scheme made it.

**The referee this milestone actually has is the exact `mdh` sum on the same box
and lattice**, which reaches all 58 cases and which the criterion already names
while the cost section omitted it. It must be described honestly as a
differential against the incumbent rather than an independent oracle. Nothing
shipped even performs the six TABI-PB comparisons: `verify_case` returns a family
`Discrepancy` and compares nothing else when the families differ, so those are
done by hand.

**And the fas2-at-0.5%-against-TABI-PB clause is not passable by any correct
design**, which is why it has been struck. debye with the *exact* boundary is
−2.8749% from TABI-PB and with an `sdh` boundary −1.7850%, so the whole span of
boundary models is **1.090 pp against the 2.375 pp the bar needs**; holding the
exact boundary and sweeping `h` gives −3.59 / −2.87 / −1.15 / −0.63% at 0.6 /
0.5 / 0.4 / 0.35 Å. The residual is discretization. It is also outside the whole
FD family — APBS −1.273%, DelPhi +1.525% on the same case, against a shipped
`MAX_SPREAD` of 0.10 over measured spreads of 2.30–4.02%. *(A genuinely crude
boundary is further still: a zeroed boundary lands at **+7.633%**, so `sdh` is
not "the crudest possible" — it is a good monopole.)*

##### What replaces step (c)

A strided exact face, with the stride **specified in ångströms as a `linspace`,
not as a node stride**. The first pass recommended ångströms while warning it
cost ~1.4× in face error; **the audit settled that and the warning is wrong.**
The 1.4× was a cost-mismatch — 448 samples against 756. At matched cost on
albumin:

| sampler | samples | CPU | screened face | reference face | solved: r / sign / energy |
|---|---|---|---|---|---|
| node stride k=16 | 756 | 0.235 s | 11.787% | 0.597% | 1.000000 / 99.996% / +0.0681% |
| uniform 15.6 Å | 448 | 0.142 s | 15.192% | 0.856% | 0.999999 / 99.994% / +0.1039% |
| **uniform 12.0 Å** | **766** | **0.237 s** | **10.168%** | **0.503%** | **1.000000 / 100.000% / +0.0598%** |

At matched cost the ångström sampler is **1.16× better on the screened face and
1.19× better on the reference face**, and the same holds on 1a63. On fas2 a true
`linspace` sampler is **bit-for-bit identical** to the node stride; the residual
1.08–1.14× penalty in the node-snapped variant comes entirely from appending the
endpoint and leaving a short last interval, which is a two-line fix. Graded on
the near field, the ångström variant clears **r ≥ 0.999, sign ≥ 99% and
magnitude 1.02× on all three structures** — fas2 0.999990 / 99.985% / +0.3642%,
1a63 0.999996 / 99.957% / +0.1718%, albumin 0.999999 / 99.994% / +0.1039% — at
1.5× less cost than the node stride on albumin. *The node stride's nominal
"k = 16" was never a uniform 15.6 Å in any case: it must divide `n − 1 = 8m`, so
on albumin it is per-axis pitches of 15.41 / 7.79 / 12.89 Å.*

The knob ships with **`k = 1` bit-identical to today's code**, which is what
makes re-recording tractable: the Born ion, `ala-gly` and the closed-form cases
stay untouched behind a size guard and only the large cases are re-recorded,
where focusing would have moved all 58 at once.

##### What is still open

- **The `atoms^≤1.05` clause is still unmeasured, and is now measurable.** Idle
  three-point fits give **0.914** for the strided face, 0.928 for `sdh` and 1.117
  for exact `mdh` — all suggestive, none decisive, because the two small rungs
  are under 2 s where fixed costs dominate. Settling it needs `sashimi.bench`
  interleaved, ≥5 repeats, on the **full** recorded ladder rather than three
  points, at this load. That is the only clause of this exit criterion nobody
  has ever measured.
- **Why the first pass priced the coarse box 4–5× high is unexplained**, and it
  is not contention. Until it is, treat the focusing numbers here as the ones to
  quote and the first pass's as superseded.
- **Albumin's fine solve varies ±20% run to run** (3.7–5.6 s) at identical
  iteration counts — memory-bandwidth noise that is ±1.5 s on every albumin total
  above. Any future head-to-head at this margin needs interleaved paired A/B,
  not sequential accounting.
- **Nobody argued the case *for* a coarse hierarchy on grounds other than
  boundary cost.** If its long-term value is the hierarchy itself — infrastructure
  for a preconditioner, or for genuine resolution focusing — that case is
  untested, and this decision does not foreclose it.

##### Two process notes worth keeping

**Both objections that died in the first pass died for the same reason: measured
at the wrong condition.** The refuted one checked three of the criterion's six
clauses and never checked the clause `sdh` fails; the downgraded one measured
fas2 at 0.5 Å when the ladder this criterion names is `molecular` at 1.0 Å, which
turned a claimed 1.33× loss at the first rung into a 0.93× win. *State the
condition a claim is made at, and check it against the condition the criterion
names.*

**And the audit pass earned its place by attacking the review rather than the
design.** Five lenses agreed on a 19× margin that was 8×, on an end-to-end gap
that was half what they reported, and on a stride recommendation that was
backwards — because they were measuring the same quantities the same way, and
consensus among agents that share a method is not corroboration. The audit
re-derived from the formula (its Debye-Hückel evaluator matches
`debye_huckel_boundaries` to `max|Δ| = 0.000e+00` before being used) and on an
idle machine. **Where a review's conclusion is going to be written down as a
decision, one pass whose job is to re-measure the review is worth more than a
sixth lens on the design.**

#### M9 shipped: the strided face, and the clause it cannot meet

**2026-08-23.** `sources.debye_huckel_boundaries` now evaluates the exact
multi-atom Debye-Hückel sum on a **sub-lattice of the box face** and interpolates
it up to every face node. Per-axis index sets, shared by all six faces, both
endpoints always included; `plan_face_sampling` chooses them and names the scheme
for provenance. No coarse grid, no second hierarchy, no change to the equation
being solved.

##### The pitch is a distance, and it is 6 Å, not 12

The pitch is specified in ångströms as a `linspace` rather than as a node stride,
for the reason the review measured: a stride must divide `n − 1 = 8m`, so a
nominal "every 16th node" is per-axis pitches of 15.41 / 7.79 / 12.89 Å on
albumin — three resolutions on the three axes of one face.

**The review's recommended 12 Å was measured on serum albumin at `padding = 10`
and fails elsewhere.** Swept across padding on the observable the criterion now
names — the 3–4 Å shell, against the exact sum on the same box and lattice:

| structure | padding | pitch | r | sign | magnitude | energy |
|---|---|---|---|---|---|---|
| fas2 | 3 Å | 12 Å | 0.998798 | 99.25% | 0.9830 | +0.7536% |
| fas2 | 3 Å | 6 Å | 0.999782 | 100.00% | 0.9921 | +0.2945% |
| 1a63 | 3 Å | 12 Å | 0.999400 | 100.00% | 0.9883 | +0.6325% |
| 1a63 | 3 Å | 6 Å | 0.999907 | 100.00% | 0.9954 | +0.2244% |

At `padding = 10` every pitch from 6 to 20 Å passes on all three structures, at
r ≥ 0.999988 and sign 100%. **It is the small box that discriminates, and the
first default was chosen without one.**

**The mechanism is that a fixed distance-pitch scales the wrong way.** Shrink the
box and the face moves closer to the solute, so the field on it varies *faster*
while a fixed pitch buys *fewer* samples, the face having fewer nodes. `padding`
is a caller's knob that `protocol.py` bounds only below. So the pitch is capped
at `PITCH_CLEARANCE_FRACTION = 0.6` of the solute's measured clearance from the
face rather than documented as safe in the range it was swept over — which turns
every failing row above into a pass (fas2 at padding 3 and a requested 12 Å:
r 0.998798 → **0.999989**, energy +0.7536% → **+0.0618%**). `solute_clearance` is
one function serving both the cap and the atom-on-the-face refusal, because both
ask how close the boundary expression gets to the charges it approximates.

Cost is flat above 6 Å — the boundary is 1.2–2.5% of a solve at 6 Å against
0.3–0.65% at 12 Å — so **the entire saving on offer past 6 Å is under two percent
of a solve**, against a gate failure. There was nothing to trade.

**The gate, re-run on the configuration that actually ships.** Every row of the
sweep above predates the clearance cap, so none of them is the shipped default
for serum albumin — the cap takes its requested pitch to 6.0 Å. Re-measured
against the exact sum on the same box and lattice:

| structure | atoms | effective pitch | face samples | r | sign | magnitude | energy |
|---|---|---|---|---|---|---|---|
| fas2 | 906 | 6.00 Å | 678 | 1.000000 | 100.00% | 0.9996 | +0.0521% |
| 1a63 | 2,065 | 6.00 Å | 902 | 1.000000 | 100.00% | 0.9998 | +0.0260% |
| serum albumin | 18,242 | 6.00 Å | 2,546 | 1.000000 | 100.00% | 0.9999 | +0.0149% |

Against bars of r ≥ 0.999, sign ≥ 99%, magnitude within 1.02× and energy within
0.5%. **Met on all three, with two to three orders of magnitude of margin** — and
2,546 samples where the exact face is 82,050 nodes. *Checking the shipped
configuration rather than the swept one is not a formality: the cap changed what
"the default" means after the sweep was taken, and a gate re-run on superseded
settings is the same error as grading against the approximation under review.*

##### The speed clause, measured for the first time — and it is not reachable

M9's exit criterion asks that total CPU fit `atoms^≤1.05` across the ladder.
**Nobody had ever measured it.** Measured now on nine rungs (fas2 906, barstar
1,403, barnase 1,730, lysozyme 1,960, 1a63 2,065, hca 2,482, actin 5,877,
acetylcholinesterase 8,279, serum albumin 18,242), `molecular` at 1.0 Å,
variants interleaved within each repeat, five repeats, on an idle machine
(1-minute load 2.4 → 4.8 on 8 cores):

| | fas2 906 | 1a63 2,065 | actin 5,877 | albumin 18,242 | **9-rung exponent** | 3-rung fit |
|---|---|---|---|---|---|---|
| exact `mdh` (before) | 0.944 s | 2.358 s | 8.946 s | 30.047 s | **1.192 ± 0.050** | 1.156 |
| **strided face (now)** | **0.787 s** | **1.840 s** | **6.305 s** | **17.941 s** | **1.084 ± 0.055** | 1.042 |
| `sdh`, the floor | 0.775 s | 1.821 s | 6.213 s | 17.264 s | **1.075 ± 0.056** | 1.033 |

The strided face is **strictly faster than the baseline on every rung** — 1.18×
to 1.74×, widening with size — and lands **within 1–4% of the `sdh` floor**, so
it captures essentially all of the saving a boundary scheme has to give.

*The measurement is robust to the thing that has spoiled every previous attempt.*
Two independent runs, at 1-minute loads of **4.8 and 18.8** on 8 cores, agree to
**±0.003** on every exponent (exact 1.189 / 1.192; `sdh` 1.074 / 1.075). That is
what min-of-five CPU time bought: contention inflated wall clock by more than an
order of magnitude between the two runs and moved the CPU minima by under half a
percent.

**Two findings, and the second one is about the criterion rather than the code.**

First, the harness justified itself immediately: **the three-rung fit passes
where the nine-rung fit fails** — 1.032 against 1.074 for the same `sdh` data,
because two of the three old rungs sit under 2 s where fixed costs dominate. Every
exponent this document has quoted came from three points.

Second, **`sdh` deletes the boundary almost entirely and still fits `atoms^1.074`
on this ladder.** `sdh` is the floor — `O(face nodes)`, independent of atom count
— so no boundary scheme can score below it, and the strided face cannot either.
The residual superlinearity is geometry, not the boundary and not the linear
algebra, which measures *sublinear* at 0.929.

***Corrected 2026-08-23, and the correction is the more useful result.*** This
section first concluded that `atoms^≤1.05` was therefore unreachable. It is not.
**The ladder is not a homogeneous family**, and the exponent partly measures
that: `2LZT-ASP66` carries a mean radius of **1.031 Å** against ~1.55 for every
other rung — 959 of its 1,960 atoms are under 1.3 Å, a different hydrogen radius
set — and `hca` is **17.8% hydrogen** against ~49% elsewhere, a polar-hydrogen
convention. Both change how many neighbours an atom has, which is what the
geometry stages cost. Refitted over the seven homogeneous rungs:

| | all nine rungs | seven homogeneous rungs |
|---|---|---|
| exact `mdh` | 1.192 ± 0.050 | **1.170 ± 0.029** |
| strided face | 1.084 ± 0.055 | **1.057 ± 0.023** |
| `sdh`, the floor | 1.075 ± 0.056 | **1.048 ± 0.026** |

The standard errors **halve**, and both independent runs agree to 0.001–0.004,
which is what says this is a confound removed rather than points dropped for not
fitting. *The exclusion was decided from the inputs — radius sets and hydrogen
counts — not from the residuals; both structures remain perfectly good corpus
cases, they are just not rungs on one scaling ladder.*

So the floor sits **at** the bar, not above it, and the shipped solver is one
standard error over. The clause is **marginal rather than impossible** — and it
is still a weak gate. The composition effect is **0.027** where M9's own
improvement is **0.113** in the same units — a quarter of the signal, coming from
which structures were chosen rather than from the solver — and load alone moved
earlier three-point fits by 0.23. **What is worth carrying forward
is not the number but the method: an exponent fitted across structures with
different radius conventions is measuring the conventions.**

##### What moved, and what did not

`EXACT_FACE_PAIRS` holds any case under a million (face node × atom) pairs on the
exact sum. Of 58 debye recordings, **20 moved and 38 are bit-identical** — the 20
being every case from `ion-protein-complex` (260 atoms) up, and the 38 including
every fast-tier case, which is why the suite stayed green through the change.
Energies moved **0.011% to 0.115%, median 0.028%**, all inside the 0.5% gate and
all in the same direction.

*A separate, pre-existing staleness surfaced in the rebuild and is included here:
all 58 recordings predate M8a's addition of `dielectric_smoothing` to
`resolved_parameters`, so none carried it. That is provenance, not physics, and
it accounts for the diff on the 38 that did not move.*

##### The one-ULP defect, and why a green suite did not see it

The exact path was **one ULP off on 13,043 of 22,530 face nodes**, and the whole
suite passed.

The cause is not arithmetic. `np.argwhere` returns a transposed, non-contiguous
view with strides `(8, 180240)`; the `np.unique`-based enumeration the strided
path needs returns C-contiguous `(24, 8)`. **The indices are equal element for
element and in the same order.** But `points` inherits the layout, and numpy's
pairwise summation blocks by memory layout, so `norm(..., axis=2)` and
`sum(axis=1)` accumulate in a different order. On `peptide-molecular` that moved
the recorded energy from −218.62772042354118 to −218.62772042354123.

**Two guards were in place and neither could fire.** The corpus compares with a
relative tolerance, so a one-ULP move is invisible to it by construction. And the
new module's own bit-identity test compared `debye_huckel_boundaries` against
*itself* at two pitch values, both through the new code — it could not have
detected divergence from the scheme it named. *A test that compares a function
against itself is not a comparison, and it reads exactly like one.*

The fix is that the exact path **is** the pre-M9 expression rather than something
that agrees with it — `_exact_nodes` exists to hold that call so the property
belongs to the code instead of to a comment.

**And the obvious test for it does not work, which is the more useful half.** The
first attempt anchored on the literal digits from `a0862ce`,
−218.62772042354118, on the reasoning that nothing else can test identity against
code that no longer exists. It passed locally and **CI failed on all three legs**:
linux/amd64 returns **−218.62772042354138**. *Bit-identity in this solver is a
**per-platform** property, so an absolute anchor tests the platform as much as
the code* — a different BLAS and a different pairwise-summation blocking are
enough. Every "bit-identical" claim this document makes should be read as
"bit-identical on one machine", which is still the right discipline for an
answer-preserving change and is not a portable constant.

What is portable is the *scheme*: `_exact_nodes` must be `np.argwhere`'s
transposed view and not a C-contiguous rearrangement of the same values, and a
second test forces the C-contiguous copy through the same call and asserts the
answer **moves** — so if the layout ever stops mattering, the first test is
guarding nothing and says so. Reinstating the defect reddens the pair; the fix
greens it.

##### Two invariants carried rather than dropped

- **The atom-on-the-face refusal is now an explicit `O(atoms)` box check.** It
  was previously a running minimum over the `O(nodes × atoms)` distance block, so
  removing the block removed the guard — and a strided face visiting a few hundred
  of tens of thousands of nodes could only have caught the bad case by luck. Its
  real exposure is not the singularity the old comment described but **20% of a
  structure's charge silently discarded** from the right-hand side, `_solve_state`
  zeroing all six faces.
- **`_resolved` now names the boundary scheme**, including the effective pitch
  after the clearance cap, so `content_address` distinguishes schemes that change
  the answer without changing the box. Before it did, three boundary models
  1.4260% apart in energy shared one saved map.

#### Where the residual cost actually is, and why it is not an exponent problem

**2026-08-23.** With M9 merged, debye fits **atoms^1.057 ± 0.023** over seven
homogeneous rungs against a boundary-free floor of **1.048 ± 0.026**. That is
0.009 of exponent between the shipped solver and the best any boundary scheme
could do, so the obvious next milestone — "attack the residual superlinearity" —
was scoped by measurement before being attempted, and **the measurement says not
to do it.**

Stage-resolved, serum albumin at 1.0 Å, compiled kernel active, exponents fitted
over the homogeneous rungs:

| stage | albumin | % of solve | exponent |
|---|---|---|---|
| `build_levels` | 13.90 s | **73.0%** | 1.146 |
|   `dielectric_faces` | 12.18 s | 64.0% | 1.117 |
|   `screening_nodes` | 1.66 s | 8.7% | 1.108 |
|     `surface.inside`, 16 calls | 13.82 s | 72.6% | 1.150 |
|       `surface.seats` (one-time) | **5.81 s** | **30.5%** | **1.145** |
|       `surface.rims` (one-time) | 0.99 s | 5.2% | 1.119 |
| `solve_system` | 4.34 s | 22.8% | **0.919** |
| boundary | 0.74 s | 3.9% | **1.379** |
| `source_term`, energy read-back | 0.01 s | 0.1% | 0.89 |

**Three things follow, and none of them is "optimise the exponent".**

- **The linear algebra is sublinear and is 23% of the solve.** Every earlier
  draft of this document that worried about multigrid cost was worrying about
  the one stage that is not a scaling problem.
- **Geometry is 73%, and it splits almost evenly** between a *one-time* surface
  build (7.2 s, of which `_probe_seats` is 5.8 s) and the *per-lattice* masks
  `inside` evaluates on each of 16 lattices (6.6 s). Both are near-linear. What
  is left is a **constant factor**, and a constant factor in 73% of the runtime
  is worth more than the 0.009 of exponent above it.
- **`_probe_seats` is linear in what it produces and superlinear in what it
  considers.** Seats come out at **atoms^0.953** — the output is linear — while
  candidate triples go as **atoms^1.112** and CPU as 1.145. It is enumerating
  more than it keeps, and the gap between 1.112 and 1.145 is memory locality
  rather than arithmetic. *An algorithmic fix would target the candidate set, not
  the seat set.*

**The boundary keeps the highest exponent even after M9 — 1.379 — and that is
worth writing down rather than celebrating.** The strided face cut the
boundary's *magnitude* by 50× and left its *shape* alone: samples grow with the
box face while atoms grow with the volume, so the product still climbs faster
than either. At 3.9% of a solve that is invisible; at ten times albumin it comes
back. The pitch cap is the knob that would answer it, and nothing needs doing
until a structure that size exists in the corpus.

*So there is no M10 here.* The honest next targets are the ones that are not
about speed at all — §12's referee gap, where 30 of 58 debye recordings have no
independent check and none above 906 atoms has any, and phase 5's release.

*Updated 2026-08-25. The gap is narrower and more precisely stated than that.
Every one of the 58 already had **two same-family referees** in this repository
and nobody had written the relationship down; #72 did, and the section below
carries what it says. What was genuinely open is **no cross-family referee above
906 atoms** — TABI-PB's *recordings* topped out at `fas2` — and **no
*independent* referee for an interface change**, which is the ramp's problem and
not a coverage one. *The first is settled: six recordings were added above 906,
and the ten that remain are covered by a recorded decision — see "Decision:
same-family referees are accepted above 906 atoms, for ten cases". The second
is still open, and is not a coverage problem.* *Whether reaching higher was a cost question or a
feasibility one was unmeasured. **It is measured now, and it is cost**: TABI-PB
solves every `molecular` case up to 2,482 atoms. See "The ceiling above 906
atoms was a property of the recordings, not the tool" below — what topped out at
`fas2` was `tests/corpus/tabipb/`.*

### The ceiling above 906 atoms was a property of the recordings, not the tool

**2026-08-27.** Four places in this document said TABI-PB's coverage "tops out
at `fas2`", and the referee gap was scoped around that: *no cross-family referee
above 906 atoms*, with the honest caveat attached that **whether reaching higher
was a cost question or a feasibility one was unmeasured** — "no protein above
`fas2` has ever been meshed by this project".

It has now been meshed. **TABI-PB records every `molecular` case in the corpus up
to 2,065 atoms**, on a laptop, at the `mesh_density` the corpus already uses:

| case | atoms | wall | TABI-PB kJ/mol | vs APBS | debye vs TABI-PB |
|---|---|---|---|---|---|
| `fas2` | 906 | 49.9 s | −2020.7 | +1.26% | −2.82% |
| `barstar` | 1,403 | 59.8 s | −3026.3 | +0.31% | −2.20% |
| `fkbp-apo` | 1,663 | 98.0 s | −2118.5 | +1.41% | −3.69% |
| `fkbp-dmso` | 1,673 | — | −2112.8 | +1.39% | −3.62% |
| `barnase` | 1,730 | 139.9 s | −2631.0 | +1.39% | −4.57% |
| `lysozyme` | 1,960 | 161.5 s | −4961.6 | +0.78% | −3.52% |
| `protein-1a63` | 2,065 | 300.0 s | −4998.8 | +0.71% | −2.79% |

So the sentence to retire is "TABI-PB tops out at `fas2`". What topped out at
`fas2` was `tests/corpus/tabipb/`, and the distance between those two statements
is the whole of this section. **The gap above 906 atoms is a cost question**, and
`tests/corpus/tabipb/` now holds twelve files instead of six.

**`hca` is the boundary, and it is recorded as one rather than accommodated.**
2,482 atoms **completed once at 576.3 s and timed out on a second attempt at
identical settings**. `DEFAULT_TIMEOUT` is 600 s, so the margin was 4%, which is
not a margin — it is a case that passes on a quiet machine. Moving the timeout
would have made the table above read cleanly and hidden that; a limit relaxed
until a marginal case fits is the same edit as a limit removed, which is the
thread §12 already carries under guards that guard nothing.

**How far the cost ceiling actually is, stated as loosely as the data allows.**
Over the six timed rungs the wall clock goes as **`atoms^1.95`**. Including
`hca`'s single successful run it goes as `atoms^2.41` — and the difference
matters, because extrapolating to `serum-albumin` at 18,242 atoms gives **4.8
hours** on the first exponent and 19 on the second. **Neither number should be
quoted.** The fit spans a factor of 2.3 in atoms and the extrapolation is a
factor of 8.8 beyond it, off a power law whose exponent moves by 0.46 when one
questionable point is added. What the data supports is that `serum-albumin` is
**hours rather than minutes**, and that how many is not determined by anything
measured here.

**What the seven rungs say.** `E_delphi > E_tabipb > E_apbs > E_debye`, on 7 of
7. `test_the_three_codes_bracket` already asserts `E_delphi > E_apbs > E_debye`
on all 18 cases at or above 906 atoms, and is careful in its own docstring that
those three **share a discretization** — all of them assign the dielectric hard
at face centres, so none is independent of the others. A boundary-element solver
has no volumetric lattice at all. Inserting one into the bracket is a different
claim from widening the bracket, and the ordering survives it.

**The finding, and it is not the coverage.** **debye's low bias survives a
lattice-free referee.** It sits 2.2–4.6% below TABI-PB at every size, same sign
on every rung. Until now debye had only been compared against codes sharing its
face-centre assignment — the shared-bias trap `debye/dielectric.py` warns about —
so the deviation could not be separated from the convention. It can now, and it
does not go away.

**And it overturned a reading that had one data point.** #73 pinned each grid
code against TABI-PB and recorded that **debye was closest of the three on five
of six cases and furthest on `fas2-molecular`**, "the only real protein among
them" — concluding that *no story about a uniformly better or worse solver fits
both halves*. Six more proteins supply the story, and the exception was the
beginning of it:

| | debye vs TABI-PB | closest of the three |
|---|---|---|
| five solutes ≤ 260 atoms | **0.48–0.86%** | debye, all five |
| seven proteins ≥ 906 atoms | **2.20–4.57%** | never debye, all seven |

**The split is at size, not at `fas2`.** debye is nearest the lattice-free
answer on every small solute and furthest on every protein, without exception in
either direction. Among the proteins APBS is nearest on six of seven and
`barnase-molecular` is DelPhi's — asserted too, so that a change reordering the
*other* two codes while leaving debye last does not pass unread. This is what
`test_debye_is_closest_on_small_solutes_and_furthest_on_every_protein` now
holds, in place of the `fas2` special case.

*A finding that rested on n=1 was not wrong to record — it was the whole sample
there was. It was wrong to read as "no story fits", and what fixed that was six
more rows rather than a better argument.*

**Asserted as an ordering, for two reasons.** The first is the one
`test_the_three_codes_bracket` gives: a magnitude bar against a referee is
exactly the trap this section warns about, and an ordering is not something a
tolerance can be widened to admit. The second is specific to this backend —
every recording here is at a **single mesh density**, and this document is
already explicit that one density is "a rung and not a limit". The magnitudes
are not converged. The ordering does not depend on their being converged.

**A struck claim that turned out to be true, and was still right to strike.**
The note above records "a session note claiming a 1,403-atom TABI-PB solve is
not a recording". `barstar` is 1,403 atoms and solves in 59.8 s, so the number
was right. Striking it was still correct: an unsourced number is not evidence,
and the remedy was never to trust it harder. It was to run it.

**What is left of the referee gap.** `serum-albumin` and `hca`, and **every
`van-der-waals` case above 906**. *Corrected 2026-08-28, and the correction
matters: an earlier version of this paragraph said van der Waals was out of a
surface solver's reach "permanently rather than expensively", by lumping it in
with `smoothed-molecular`. That is true of `smoothed-molecular` and `gaussian`,
which are volumetric dielectrics a BEM solver has no equivalent for — the
protocol behaving correctly. **It is not true of `van-der-waals`**, which
`sashimi.tabipb.options` declines for a different reason entirely: it "would be
`srad 0`, and NanoShaper does not return from it in any reasonable time on a
dipeptide". That is the mesher, not the method. Re-checked 2026-08-28 and the
observation still holds — 90 s and no return on a 20-atom peptide — but a
triangulator that handles a zero probe would unblock it, where nothing unblocks
a smoothed dielectric.* Of 58 debye recordings, the
count with neither a TABI-PB counterpart nor a gated closed form falls from 30
to 24, and above 906 atoms from 16 to 10. The decision this section was blocking
on shrinks accordingly: not "accept same-family referees above 906 atoms", but
"accept them for the surfaces a boundary-element solver cannot express, and for
the two cases that are hours away".

**Still unexplained, and it prices the rest.** debye and the DelPhi backend
agree on the Kirkwood field to 2.7e-5 relative — five significant figures —
while APBS sits ~0.4% from both. If those two share a discretization convention
then the two same-family referees are one, and §14 records that as worth knowing
before either referees the other. Nothing here settles it; the cross-family rungs
sidestep it rather than answer it.

### The referee gap, closed from recordings already in the repo

**2026-08-24 (#72, #73, #74).** Two sections above name "30 of 58 debye
recordings have no independent check" as the largest remaining accuracy
question. The count is right and the word doing the work is *independent* —
APBS, DelPhi C++ and debye all assign the dielectric hard at face centres, so
none of them is one for the other. What was missed is separate and cheaper:
**every one of the 58 already had two same-family referees in this repository,
and nobody had written the relationship down.** An APBS recording sits in
`tests/corpus/` and a DelPhi C++ one in `tests/corpus/delphi/` for all 58, of the
same manifest case, reporting the same `POLAR_SOLVATION` term.

`DEBYE_DEVIATION` is 58 cases × 2 referees of exact arithmetic on checked-in
constants — **no solving and no binary**, so it runs on CI's `none` leg and is
platform-independent where an assertion on a freshly-solved number is not. Like
`GB_DEVIATION` it is the one check that survives
`corpus build --backend debye --force`: a rebuild makes every other test in that
file agree with the new numbers and makes this one fail by however far debye
moved. **Signed**, because the sign is the finding.

**What the 58 rows say, none of it previously recorded.**

- **debye sits below both incumbents rather than scattering around them.** More
  negative than DelPhi C++ on **58 of 58**, without exception, and than APBS on
  **39 of 58**. The nineteen it is not are the one- and two-atom synthetic
  geometries and the smallest molecules; `ion-protein-complex-vdw` agrees with
  APBS to the table's precision.
- **The three codes bracket.** `E_delphi > E_apbs > E_debye` on all **18** cases
  at or above 906 atoms — nine structures, both surfaces, no exception. *The
  count above reads "16 above 906 atoms" and this reads 18; the difference is
  strictly-greater against at-or-equal, which is `fas2`'s two rows. Both are
  right and neither should be silently harmonised into the other.* Asserted
  as an *ordering* rather than a magnitude, deliberately: an ordering is not
  something a tolerance can be widened to admit, and a magnitude bar here would
  be the shared-bias trap `debye/dielectric.py` warns about.
- **The two surfaces disagree in opposite directions, and that is the sharpest
  thing in the table.** Over the twelve pairs where rolling the probe is worth
  more than 1% — `corpus.probe_worth`, the quantity M4 was gated on —
  `|debye − DelPhi|` is larger on **`van-der-waals` in 12 of 12**, while
  `|debye − APBS|` is larger on **`molecular` in 11 of 12**. Two codes, two
  surfaces, the asymmetry running the opposite way for each. A uniform bias in
  any one solver cannot produce that: it locates the residual in how the three
  **construct the boundary**, not in how they solve on it.
- **`serum-albumin-vdw` is the argmax against both referees**, −9.09% against
  APBS and −17.51% against DelPhi, with `serum-albumin` second on both at −6.64%
  and −10.41%. The corpus's largest solute is its widest disagreement, and it is
  the sole exception to the surface asymmetry. Excluding both its rows the
  widest real structure is `barnase-molecular` at −3.12% against APBS and
  `fkbp-dmso-vdw` at −8.87% against DelPhi. Pinned rather than excluded, so a
  change that makes it ordinary fails and gets read.

**The one independent pairing this repository has is debye against TABI-PB**,
which shares neither the discretization nor the dielectric assignment.
`paired_cases(a, b)` (#73) exists to express it, where the `cross_backend_cases`
it replaces hardcoded APBS on the left and so could not state any pair not
involving APBS. Measured: **debye is closest to TABI-PB on five of six cases and
furthest on `fas2-molecular`**, the only real protein in that set. Both halves
are pinned, because either one changing is a story somebody should tell.

**And the one independent referee is not converged.** Its `fas2-molecular`
recording is a single mesh density, `sdens 2.0`. §12 records above that a
boundary-element energy converges *with* mesh density — the ALA-GLY ladder reads
−221.57 → −211.39 over `sdens` 1.5 → 4.0 — so one density is a rung and not a
limit, and on the one protein in that set there is no ladder at all. **Do not
gate against it**, which is also the retrospective justification for M9 striking
its "`fas2` within 0.5% of its TABI-PB recording" clause: the referee was
unsettled as well as the bar being unpassable.

**What is left of the gap, precisely.** Not "30 recordings with no check" —
**no cross-family referee above 906 atoms**, and **no *independent* referee for
an interface change**. TABI-PB's coverage topped out at `fas2` itself; its other
cases were 260 and 20 atoms, and `tests/corpus/tabipb/` held six files. *It now
holds twelve, reaching 2,065 atoms — the ceiling was the recordings, not the
tool, and the section above carries the ladder.*

*Two things not to overstate here. Whether reaching higher is a cost question or
a feasibility one is **unmeasured** — no protein above `fas2` has ever been
meshed by this project, and a session note claiming a 1,403-atom TABI-PB solve
is not a recording. And "no referee of any kind" is too strong: the ramp sections
below establish that each scheme's own refinement ladder referees the other, for
a reason specific to a band that vanishes with `h`. What that route cannot do is
be independent, and how far it can be trusted is itself measured there.*

**Four count floors were replaced while this was written (#74)**, and it is the
guards-that-guard-nothing thread again. `assert len(born) >= 9`,
`assert len(kirkwood) >= 3` and two siblings each happened to equal the real
count when written, which is the most misleading place for a floor to sit: it
reads as exact and stops guarding the moment one more case lands. They are now
the properties they were standing in for — set equality against the recorded
directory, and the arm structure the Born sweep is supposed to have.

*The 58 × 2 table also carries ordinary readings worth keeping: the Born ladder
is about `a/h` and not about debye — against APBS the `molecular` rungs run
−5.56% at a = 1 Å, −2.42% at 2, −1.08% at 4 and −0.18% at 6, which is a
monotone `a/h` sequence and not a property of the solver; Kirkwood's `07` rungs
jump to +2.4% on both surfaces where `03` and
`05` sit inside 0.35%, which is the off-centre charge nearing the boundary; and
the small molecules are **0.46 to 1.52 kJ/mol** on small numbers — methanol's
+3.67% is 0.97 kJ/mol against −26.3.*

### Phase 9 — the case for replacing protean's default, measured on both axes

**2026-08-21.** Phase 9 is "protean consumes `SolveResult`", and the substantive
question inside it is whether debye should replace protean's *default* backend
rather than merely its APBS path. protean's own PLAN.md decision 8 (2026-08-09)
reads "screened Coulomb is the default; APBS is an optional backend", chosen
because APBS needs a binary — and that decision predates debye, which is the
thing built to answer it. So it is re-opened here with measurements rather than
with the argument that was already in §12.

Both axes were measured against protean's **own** code, `coulombic` from
`protean_mcp.analysis.electrostatics`, run in protean's virtualenv against debye
in sashimi's. Neither environment was modified. Python 3.13.12 both, numpy 2.5.1
against 2.5.2.

#### Speed: the crossover is ~244 residues, not ~1,290

CPU seconds, protean's defaults (1.0 Å, 10 Å padding), alternating so machine
drift hits both sides, minimum of three passes. debye is asked for the field
only — `want_energy=False` skips the uniform-dielectric solve, which is what a
consumer colouring a surface actually needs:

| structure | aa | atoms | `coulombic` | debye, field | debye + energy | ratio | debye's grid |
|---|---|---|---|---|---|---|---|
| fas2 | 59 | 906 | 0.51 | 1.19 | 1.39 | 2.34× | 1.39× more points |
| 1a63 | 161 | 2,065 | 2.15 | 3.05 | 3.56 | 1.42× | 1.43× |
| actin-monomer | 382 | 5,877 | 14.94 | 11.60 | 13.68 | **0.78×** | 1.32× |
| acetylcholinesterase | 538 | 8,279 | 19.14 | 14.54 | 16.61 | **0.76×** | 1.22× |
| serum albumin | 1,156 | 18,242 | 110.08 | 37.18 | 44.59 | **0.34×** | 1.14× |

`coulombic` scales as **atoms^1.76**, debye as **atoms^1.15**, and they cross at
**3,845 atoms — about 244 residues.** protean's working range is 250 to 1,200
residues, so **debye is the faster of the two across essentially the whole of
it**, and three times faster at the top.

**Two fairness notes, both against debye and both left in.** debye solves on
1.14–1.43× *more* grid points than `coulombic` at the same requested spacing,
because `size_grid` steps in eights and pads from radius-inflated bounds. And
`coulombic` does its distance matrix in float32 where debye is float64
throughout. The crossover would move further in debye's favour if either were
normalised away, so neither was.

*The projection this replaces was mine and it was wrong.* Fitting the two
endpoints already in this section gave ~155 residues; the measured answer is
244, 57% higher. Same direction, wrong number — which is the seventh projection
in this document to miss, and the reason this section exists at all.

#### Quality: debye converges on APBS, `coulombic` cannot

protean's PLAN.md records what its approximation costs, measured on 1UBQ against
APBS on a shell 3 Å outside the atoms: **Pearson r = 0.958, sign agreement
94.1%, magnitudes about 1.6× low.** That protocol is re-run here with debye
added, on 1UBQ prepared by `sashimi.prep` (1,231 atoms, net charge 0.000, no
warnings, nothing rebuilt). The shell is stated rather than eyeballed: lattice
nodes outside the van der Waals union inflated by 3.0 Å and inside it inflated
by 4.0 Å, sampled by trilinear interpolation from each backend's own solve.

| | nodes | debye r | sign | magnitude | `coulombic` r | sign | magnitude |
|---|---|---|---|---|---|---|---|
| 1.0 Å | 6,211 | 0.998 | 97.8% | **1.05×** | 0.977 | 93.8% | 1.74× low |
| 0.8 Å | 12,903 | 1.000 | 98.9% | **1.03×** | 0.979 | 94.4% | 1.76× low |
| 0.6 Å | 29,725 | 1.000 | 99.1% | **1.04×** | 0.979 | 94.6% | 1.75× low |

The `coulombic` row reproduces protean's own recorded figures closely — 0.977
against 0.958, 93.8% against 94.1%, 1.74× against 1.6× — from a different shell
definition and a different preparation, which is the check that this protocol is
theirs in substance.

**The refinement axis is what makes this decisive, and it is why three
resolutions were run rather than one.** debye *converges* on APBS: r → 1.000,
sign agreement → 99.1%, magnitude ratio 1.05 → 1.04×. `coulombic` is **flat** —
its r does not improve and its magnitudes stay 1.75× low no matter how fine the
grid. That is the signature of the difference being **physical rather than
discretization**: one uniform dielectric has no reaction field to converge to.
protean's own module docstring already says so — "no free energy should ever be
derived from the Coulombic field" — and this puts a number on the part it does
not claim, the field itself.

*A caveat this project has been burned by, discharged rather than waved at.*
M1b's correction was that a cross-lattice comparison of the near field is
worthless, because grid phase swings it 5–21×. Each backend here solved on its
own lattice — APBS 65³ where debye chose 57×57×65 at 1.0 Å — so the three
resolutions are also three different lattice pairings, and the numbers move by
0.002 in r and 0.02× in magnitude across them. The shell sits 3 Å out, well
outside the region where phase dominates.

#### What this settles and what it does not

**Settled: on protean's working range debye is both the more accurate field and
the faster one**, and it needs no binary, which was the whole charter. Below
~244 residues `coulombic` is quicker — 0.51 s against 1.19 s on a 61-residue
peptide — and that is the honest cost of the switch, at a scale where both are
sub-second.

**Not settled here:** whether protean keeps `coulombic` at all. Nothing in this
measurement says a fast indicative field is worthless; it says it should not be
the default for a range where a correct one is cheaper. That is protean's call
to record in protean's plan, and this section exists to be the evidence when it
is made.

### TABI-PB's surface potential was kJ/mol/e, labelled kT/e

§4 fixes the units at the protocol boundary — Å, kT/e, kJ/mol — "so debye never
inherits APBS's unit conventions by accident". TABI-PB was inheriting its own.
Its VTK carries the potential in kJ/mol/e, `sashimi.tabipb` applied no
conversion, and `SurfacePotential` documents kT/e, so six recordings carried a
field a factor of **RT = 2.479** too large.

**Why nothing caught it.** TABI-PB reports its energy in kJ/mol and needs no
conversion there — `run.py` says so in a comment — and the potential is *also*
per mole. So the one backend that was right about energy was wrong about field
in the same breath, and the energy self-test cannot see it: it compares
energies, and every energy was correct to the last recorded digit, before and
after this fix. The corpus verified the wrong field against itself for as long
as it has existed.

The field was never graded against anything else either, and that was by
construction. `_analytic_field_summary` excludes `SurfacePotential` from the
closed-form grade, reasoning that "a boundary-element answer lives *on* the
interface, which is the one place this measurement is meaningless". That is true
of a **grid** — sampling a discretized dielectric at the interface is O(1) wrong,
which is what `field.MIN_CELLS_OUT = 2` exists for — and false of a BEM solver,
whose surface potential *is* the primary unknown and whose closed form is exact
and continuous across the boundary. The new test grades on the interface and
lands within 1.75% at `sdens` 3, converging as h². A grid-specific
fact had been generalized to a family where it does not hold, and that exclusion
removed the one check that would have caught this.

It reached callers, not just recordings: `sashimi_solve` reports surface
statistics under `potential_kT_e`, so the MCP tool published the same wrong unit
in the same words.

The contract was never ambiguous, though. `bem_stub.StubBemSolver` — written in
phase 4 to prove the protocol admits a BEM answer, before one existed — divides
volts by kT/e and has always been right. The reference implementation of the
type got the unit correct and the real backend did not, which makes this a
backend bug rather than a spec that failed to say what it meant.

**What caught it.** A dimensional property, not a reference. kT/e carries a 1/T
and kJ/mol/e does not, so the two peptide recordings at 298.15 K and 277 K must
differ by 298.15/277 = 7.64% beyond the physics. They differ by 0.37% — the
screening response alone, at fixed salt. That is a negative: it says the values
are not kT/e at the requested temperature, and since they are byte-identical
across the two, not kT/e at any temperature. It does not say what they are.

**What pinned the factor.** A Born sphere, because outside a centred monopole
the potential is exactly `q / (4 pi eps0 eps_s r)` and every mesh vertex has a
known answer. NanoShaper refuses fewer than four atoms, so the sphere is four
coincident-to-0.01-Å ones at the vertices of a regular tetrahedron with the
charge split equally — the union is a sphere and the symmetry cancels the
dipole. Graded per-vertex against the closed form at that vertex's own radius:

| a | sdens 3 | 5 | 8 | 12 |
|---|---|---|---|---|
| 3.0 Å | 1.01751 | 1.01087 | 1.00660 | 1.00440 |
| 4.0 Å | 1.01307 | 1.00804 | 1.00497 | 1.00333 |

(ratio to RT; 1.0 means the factor is exactly RT.)

At `sdens` 3 the ratio misses RT by 1.75% — close enough to be the
unit and far enough to be something else, which is why the ladder exists rather
than a single calibration. The excess times the density is constant down each
row (5.25, 5.43, 5.28, 5.28 and 3.92, 4.02, 3.97, 4.00), so it is h²
discretization and extrapolates to zero. **The factor is exactly RT, not a
fitted number.** `studies/tabipb_units/born_sphere.py`.

**The fix** is `backend.to_kt_per_e`, one division at the point where the
temperature is known. `vtk.py` stays a format reader and now says so: the
`SurfacePotential` it returns is in the file's units and is not yet
protocol-conformant. Two tests gate it — one against the closed form, one on the
temperature property that found it — and both fail by 152% and by the whole
7.64% respectively if the conversion is removed.

**The corpus diff is the conversion and nothing else.** All six energies and all
six vertex counts are bit-identical; every one of the twenty-four surface
statistics moved by exactly RT at that case's own temperature, including the
277 K case, where the divisor is 2.3031 rather than 2.4790.

**Still open, two things.** `normal_derivative` is parsed, carried through
`run.py`, and dropped at `backend.py` because `SurfacePotential` has no field for
it. §2 names it as half of what a BEM solver natively produces, so the type
should grow one — a protocol change, deliberately not bundled with a
corpus-changing units fix. It is in the same kJ/mol/e per Å and will need the
same divisor — **and it is the interior derivative, not the exterior one, which
is measured and designed against two sections below.**

And the exclusion above should be narrowed from the family to the measurement:
grading a BEM answer against a closed form on the interface is sound, and only
the *grid* sampling is not. The blocker is a case to do it on — the corpus's
analytic-field cases are Born ions, and a one-atom solute is refused by the
mesher, which is why this section's evidence is a four-atom tetrahedron built by
hand rather than a corpus case.

### The field reference stops being a Born ion

§7's field check had twelve cases and every one was a centred charge — the one
geometry where the field is spherically symmetric, so nothing the corpus graded
could see a solver wrong about *direction*. That is the same gap the energy side
closed at M2, when `kirkwood_solvation_energy` arrived because the Born ion "is
symmetric in every way a solver could be wrong about".

`kirkwood_potential` is the field half of that expression. Matching phi and
eps dphi/dr at r = a, with the source expanded as sum_n d^n r^-(n+1) P_n:

    phi_out(r, theta) = (q / 4 pi eps0) * sum_n
        (2n+1) / (n eps_p + (n+1) eps_s) * d^n / r^(n+1) * P_n(cos theta)

The same matching produces the interior coefficients whose sum at the charge is
`kirkwood_solvation_energy`'s series, so the two are one system solved twice
rather than two transcriptions that have to be kept in step.

**Three identities pin it, and between them they reach every term.** At d = 0
every term above the monopole carries d^n = 0 and what is left is
`born_potential` exactly, fixing n = 0. Far out, the excess over Born is the
dipole with its own coefficient, `3 eps_s d cos(theta) / ((eps_p + 2 eps_s) r)`,
fixing n = 1 — the first place the dielectric contrast enters. And at
eps_p = eps_s the coefficient collapses to `(2n+1)/(eps(2n+1))` and the series
becomes the Legendre generating function, giving plain Coulomb at the charge's
*actual* position: that one fixes the whole sum, and it checks the geometry
rather than the physics, because a reference that put the charge at the centre
would pass every symmetric test in the file and fail only this.

The Legendre recurrence was wrong on first write — P_2 read 0.02035 against
-0.29465 — and none of the reasoning above would have caught it. Checking the
recurrence against numpy before trusting the expression is what did.

**1,338 recorded samples were graded for free.** The 28 Kirkwood recordings —
four offsets, two surface models, three backends — already carry 50 probe
potentials each and had only ever been compared with themselves.
`tests/test_kirkwood_field.py` grades them against the closed form with no
solver in the room, so it runs in the binary-free tier where most of CI lives:
median error 0.52%, worst single sample 3.62%.

**Attached to the eight sharp-boundary rungs**, `AnalyticField` gained an
`offset_a` and a direction-aware evaluator. A radial reference broadcasts
through it, so all twelve Born field cases are byte-identical across the change
— the discipline `exact_at` already records for salt — and `exact_kT_e` stays a
flat list for them, nesting only where the direction axis carries information.

What the attachment measured, worst sample over eight directions at `a + k*h`:

| d/a | APBS | debye | DelPhi |
|---|---|---|---|
| 0.3 | 3.402% | 2.229% | 2.229% |
| 0.5 | 2.940% | 2.144% | 2.144% |
| 0.7 | 7.109% | 2.753% | 2.753% |
| 0.9 | 20.959% | 1.860% | 1.855% |

**APBS's spread is a charge-proximity effect, and it constrains the sampling
rule.** `a + k*h` was designed to clear the dielectric interface; an off-centre
charge adds a second thing to clear. The near pole at d/a = 0.9 lands 0.71 A
from a point charge on APBS's achieved 0.203 A lattice, and stepping that sample
out walks the error down steeply — 20.96%, 8.27%, 4.71% at gaps of 0.71, 1.11
and 1.52 A. debye's worst at the same rung sits on the *far* pole and never
moves off 1.86%, so its near-pole error is smaller still at a shorter gap. The
likely mechanism, offered as a reading rather than a measurement: APBS
discretizes a point charge over a `chgm spl4` stencil about two cells wide, and
0.71 A is inside it. ***Measured 2026-08-28 and wrong in direction*** — narrowing
the stencil to `chgm spl0` takes the error from 20.96% to 29.55%, where this
reading predicts an improvement. The section below carries what the effect
actually is, and it is grid phase. `cells_out` stays at (2, 4, 8) regardless, because the near
sample is the one this reference exists to take — a ladder that stepped back
until every backend looked alike would be measuring the step-back.

**An observation this turned up, explained 2026-08-28.** debye and the DelPhi
backend agree with each other on the Kirkwood field to three digits at every
rung above, and on the recorded probes to 2.7e-5 relative — five significant
figures — while APBS sits ~0.4% from both. It is not a recording mix-up: a
mix-up would be bit-identical and these are not. The question was whether it is
a shared discretization convention or a shared ancestry, because §12's referee
work assumes the two are independent.

**They are independent. What they share is the lattice.**

Two explanations in the *physics* were measured, and both fail:

| | worst sample, d/a = 0.9 |
|---|---|
| APBS, `chgm spl4` (default) | 20.96% |
| APBS, `chgm spl0` — the trilinear assignment debye uses | **29.55%** |
| APBS, `bcfl mdh` — the boundary condition debye uses | 20.95% |
| debye | 1.86% |
| DelPhi | 1.85% |

Charge assignment is not it: `spl0` is APBS's trilinear option and moving to it
takes APBS *further* from debye, not nearer. The boundary condition is not it
either: `mdh` is debye's own and moves APBS by 0.006%.

**What is left is where the lattice falls.** The three backends size a box by
different rules — debye needs `n = 8m + 1`, DelPhi needs an odd `gsize`, APBS
needs `2^k + 1` — and *every `8m + 1` is odd*, so at the corpus's 0.25 Å debye
and DelPhi both land on 105 points exactly while APBS cannot and rounds to 129.
But the coincidence goes further than the spacing, and this is the part that
settles it:

| requested h | backend | shape | achieved h | origin x | charge cell-phase | worst |
|---|---|---|---|---|---|---|
| 0.203125 | APBS | 129³ | 0.203125 | −11.65 | **0.6462** | 20.96% |
| 0.203125 | debye | 129³ | 0.203125 | −13.00 | **0.2923** | 4.10% |
| 0.203125 | DelPhi | 129³ | 0.203125 | −13.00 | **0.2922** | 4.10% |

**At an identical shape and an identical spacing APBS still differs fivefold,
because its origin differs.** debye and DelPhi both take the solute's box
unshifted; APBS re-centres, so the charge sits 0.65 of a cell from a lattice
plane where for the other two it sits 0.29. §12 already records that the near
field swings 5–21× on grid phase. That is sufficient on its own, and no
agreement about physics is needed to produce the five figures.

**The converse confirms it.** The shared lattice is a fact about the corpus's
chosen resolutions, not a property of the two codes, and where their point-count
rules diverge so do their answers: at 0.20 Å they land on 137 and 131 points and
read 2.04% and 1.85%; at 0.1875 Å on 145 and 141, and read **7.75% and 27.53%**.
A factor of three, between the two codes whose agreement was the puzzle.

**Two consequences, and the second is the one that matters.**

1. §12's referee work may keep its assumption. debye and DelPhi are independent
   codes and their energies differ by percent, not by 1e-5 — the confound is
   specific to a near-field probe on a shared lattice, and does not reach the
   energy comparisons the referee tier is built on.
2. **A near-field comparison between debye and DelPhi measures the lattice, not
   the solvers.** On a shared grid it cannot distinguish them, so it is not
   evidence either way about their discretizations. Anything that wants to
   compare their interface treatment has to vary the phase or step off the
   shared lattice deliberately.

**And it retires a reading recorded above.** The `chgm spl4` explanation for
APBS's charge-proximity error — "0.71 Å is inside a stencil about two cells
wide" — was offered as a reading rather than a measurement, correctly labelled.
It is now measured and it is wrong in *direction*: narrowing the stencil to
`spl0` makes the near-pole error worse, from 20.96% to 29.55%, where the reading
predicts it should improve.

**It also makes "APBS is 11× worse than debye" a statement about one lattice.**
At 0.1875 Å the same measurement puts DelPhi at 27.53% against APBS's 15.99% —
the ordering inverts. The 11× is real at the rung it was taken on and is not a
property of the codes. `studies/lattice_phase/debye_delphi_agreement.py` carries
all of it, and `tests/test_kirkwood_field.py::TestWhyDebyeAndDelphiAgree` pins
the shared lattice as pure grid arithmetic, so a change to either sizing rule
fails rather than silently invalidating this section.

### Decision: same-family referees are accepted above 906 atoms, for ten cases

**2026-08-28.** §12's oldest open accuracy question was "30 of 58 debye
recordings have no independent check and none above 906 atoms has any". It was
blocked on two things that were assumed rather than measured, and both have now
been measured — the ceiling (above) and the independence of the same-family pair
(below). This records the decision they unblock.

**The decision.** For the **ten** debye recordings above 906 atoms that still
have no cross-family referee, the two same-family referees already in the
repository are accepted as the check. No further backend is pursued for them.

**What makes that defensible now, and did not before.** The worry was never that
APBS and DelPhi say nothing — it was that they might not be independent of debye
or of each other, in which case agreement is shared bias rather than
confirmation. Three measurements bound that:

- **debye and DelPhi are independent.** Their five-figure agreement on the
  Kirkwood field was a shared *lattice*, not a shared discretization; two
  candidate physics explanations were tested and both fail, and the agreement
  breaks the moment their point-count rules diverge. The confound is specific to
  a near-field probe on a shared grid and does not reach the energies, which
  differ by percent.
- **What two same-family referees can say has been written down** (#72):
  `E_delphi > E_apbs > E_debye` on all 18 cases at or above 906 atoms, and a
  probe-worth asymmetry running opposite for each code, which locates the
  residual in how the three construct the boundary rather than in how they solve
  on it.
- **Where a cross-family referee *is* obtainable, it agrees with them.** On the
  seven `molecular` rungs from 906 to 2,065 atoms, inserting a solver with no
  volumetric lattice leaves the ordering intact —
  `E_delphi > E_tabipb > E_apbs > E_debye`, 7 of 7 — and debye stays furthest.
  So extrapolating the same-family reading to the cases a BEM solver cannot be
  run on is an **evidenced** extrapolation rather than an assumption. That is
  the whole argument, and it is worth more than the ten recordings it licenses.

**What the decision does not license.** Not a magnitude bar against a
same-family referee — that is the shared-bias trap `debye/dielectric.py` warns
about, and #73's pins are two-sided *records* rather than bounds. And not a
near-field comparison between debye and DelPhi as evidence about either's
discretization: on a shared lattice that measures the box.

**Where the extrapolation is weakest, stated because it is the reason to revisit
this.** Eight of the ten are `van-der-waals` and **every cross-family
confirmation above is on `molecular`**. #72 measured the two surfaces
disagreeing in *opposite directions* — `|debye − DelPhi|` larger on
`van-der-waals` in 12 of 12 and `|debye − APBS|` larger on `molecular` in 11 of
12 — so crossing from one surface to the other is exactly where this reasoning
has least support. The decision is taken with that on the record rather than
around it.

**None of the ten is out of reach in principle**, which is the other reason to
expect this to be revisited:

| what blocks it | cases | what would unblock it |
|---|---|---|
| the mesher | eight `van-der-waals` | a triangulator that returns at `srad 0`; NanoShaper does not, on 20 atoms, in 90 s |
| cost | `hca-molecular` (2,482 atoms) | a budget above `DEFAULT_TIMEOUT`; it completed once at 576.3 s and timed out twice |
| cost | `serum-albumin` (18,242 atoms) | hours, on an exponent too weakly determined to say how many |

*A `smoothed-molecular` case would be the one genuinely permanent gap, and there
are none here: debye supports `molecular` and `van-der-waals` only, so no
`smoothed-molecular` case is among the 58 at all.*

**What reopens this.** Any of the three unblocks above; a change to how debye
constructs its boundary, since that is where #72 located the residual; or a
fourth solver family. Until then the ten are checked by two codes that are
independent of debye and of each other, and the ordering they agree on is the
claim — not the magnitudes.

### The exterior field evaluator, designed against a measurement

**Not built. This section is the design, and the reason it is written down before
any code is that two of its three physical premises turned out to be wrong when
measured, and both were wrong in ways that read as correct.**

The motivation is §12's oldest open item. `_analytic_field_summary` grades a
field against a closed form, and closed forms exist for two geometries; every
other question about the near field is settled by comparing solvers, and **APBS,
DelPhi C++ and debye all assign the dielectric hard at face centres, so none of
them is one for the other.** A boundary-element answer has no volumetric lattice
at all. It is the only prospective referee in this repository that does not share
that construction — which is what makes it worth the care below.

**What TABI-PB's `NormalPotential` actually is.** The interior normal derivative
`dphi_1/dn`, in kJ/(mol e Å), and *not* the exterior one. The measurement that
settles it is a sweep of the solute dielectric on the four-atom Born sphere from
`studies/tabipb_units/born_sphere.py`, because the two candidates differ by
exactly `eps_s/eps_p` and nothing else in the problem does:

| `eps_p` | ratio to `-q/(4 pi eps0 eps_s a^2)` | `eps_s/eps_p` | ratio to `-q/(4 pi eps0 eps_p a^2)` |
|---|---|---|---|
| 1.0 | 79.219 | 78.54 | 1.00864 |
| 2.0 | 39.608 | 39.27 | 1.00862 |
| 4.0 | 19.803 | 19.635 | 1.00854 |

The exterior column tracks `eps_s/eps_p` and the interior column does not move.
The residual 1.0086 is discretization and goes as h²: **1.00862, 1.00393,
1.00247, 1.00156** at `sdens` 3, 5, 8, 12, converging on exactly 1.

Two consequences. The divisor is `RT`, the same one the potential needed, so
`to_kt_per_e` extends to it unchanged. And **an acceptance gate written against
`eps_s` fails by a factor of 39 at sashimi's defaults while looking like a
textbook identity** — the same shape of error as the units bug above, one
derivative along. The exterior derivative is recovered from continuity of the
flux, `eps_p dphi_in/dn = eps_s dphi_out/dn`, and never read from the file.

**The representation formula, corrected.** For the exterior domain with `n`
outward from the solute, Green's second identity gives

    phi(x) = integral over Gamma of [ phi dG/dn - G dphi_out/dn ] dS,   G = 1/(4 pi R)

The kernel is the plain Laplace one. Putting `eps_s` in `G` is wrong: the
exterior problem is homogeneous, so the representation is purely geometric and
the dielectric enters only through the Cauchy data. Screened, `G` becomes
`exp(-kappa R)/(4 pi R)` and nothing else changes.

Both errors were measured rather than argued, on a real mesh with centroid
quadrature, outward normals and area weights:

| what was evaluated | ratio to the exact Born potential |
|---|---|
| the sign-flipped, `eps_s`-weighted kernel | **-0.50395** |
| that, with only the derivative convention fixed | -0.01283 |
| the expression above | **+1.00790** |

The first two are not near-misses; they are `-1/eps_p` and a cancellation. A
formula of this shape either reproduces a monopole or it does not.

**What it would be good for, measured before it is built.** The corrected
evaluator's exterior error at `a + 1.5 Å` is **1.278%, 0.790%, 0.502%, 0.338%**
at `sdens` 3, 5, 8, 12.

**That error is not quadrature error, and this is the finding that decides how
the tool can be used.** It equals TABI-PB's own Gauss-law flux error to three
significant figures at every density; it is *bit-identical* under a 16x-refined
quadrature rule; and it is flat from `a + 0.5 Å` to `a + 30 Å`, with the same
sign in every direction. It is a spurious monopole fixed by the discrete surface
data. **Neither refining the quadrature nor standing further off does anything
about it** — the only knob is mesh density, and the referee inherits the
accuracy of the code it is refereeing.

**So it is a referee for magnitudes, not yet for the interface question.** The
sub-cell ramp signal it would have to resolve on `ala-gly` is 0.45-1.6% of a
referee field magnitude of 0.38 (§12's field-axis entry), against 0.338% of its
own at `sdens` 12. That is the same order, not a decade below it. Worth stating
plainly rather than discovering after the work: **this does not obviously settle
the ramp question, which is the thing it was wanted for.** It is a genuine
independent check on near-field magnitude, and `ala-gly` carries net charge
0.0000 e, where the spurious monopole has nothing to couple to — a caveat that
happens to point the right way, and one to test rather than assume.

**Two ceilings.** *Corrected 2026-08-27: the first of these was a property of
the recordings and not of the tool — TABI-PB reaches 2,482 atoms, so an exterior
evaluator built on it reaches there too.* As written: TABI-PB's recordings
topped out at `fas2`, 906 atoms, which is exactly where the referee gap begins,
so this appeared to deepen the referee below the gap rather than closing it. And there is no Kirkwood fixture
for it: `kirkwood_pqr` emits two atoms and the mesher refuses fewer than four,
so an off-centre BEM reference needs the same hand-built tetrahedron trick, or a
different geometry.

**What building it costs.** `normal_derivative` on `SurfacePotential` (a protocol
change, §2 already names it as half of what a BEM solver produces), the same
`RT` divisor extended in `to_kt_per_e`, the continuity factor applied once, and
a `sashimi.tabipb.field` module carrying the expression above. The parse side is
already done and already tested. `vtk.py` needs nothing: `parse_vtk` forwards
`normal_derivative` today and `tests/test_tabipb.py` already asserts it survives.

***Shipped 2026-08-28, with the field named for its side.*** The protocol field
is **`interior_normal_derivative`**, not the bare `normal_derivative` budgeted
above, and the continuity factor is applied by
`SurfacePotential.exterior_normal_derivative(solvent)` rather than at each call
site. The reason is the one this section already gives: the two sides differ by
`eps_s/eps_p` and the difference is invisible in the array, so an unqualified
name is a 39× error that reads as a textbook identity. It is a *name* and not an
`EnergyTerm`-style discriminator because the two sides are an exact
multiplication where two energy terms are not — nothing has to be recorded
per-instance, only fixed once and made unmissable. **`vtk.py` did need one
thing** after all, though not parsing: the `SurfacePotential` is constructed
there, so the field is populated there, in the file's own units, and
`to_kt_per_e` divides both blocks by `RT`. *The gate that carries this is the
`eps_p` sweep, and it needs a second leg: with `eps_s` held fixed, a hardcoded
`1/39.27` cancels the `eps_p` dependence exactly as a correct conversion does,
so the sweep must also vary `eps_s`, where the hardcode reads 1.000 against a
required 1.9635.* `tests/test_tabipb_normal_derivative.py` and
`tests/test_normal_derivative_side.py` carry it.

### `residue_potentials` reported one number for two chains

**2026-08-26.** `sashimi.analysis.residue_potentials` is one of exactly two
things phase 9's consumer reads, and on any multi-chain structure it was
answering about a residue that does not exist.

The key was `"<resName> <resSeq>"`, taken from the first two fields of the
per-atom label. That is not a residue identifier: residue 58 of chain A and
residue 58 of chain B share it. On `tests/data/1ao6.pqr.gz` — serum albumin,
18,242 atoms, a dimer — **578 keys stood for 1,156 residues**, 576 of them
merging atoms more than 15 A apart. The worst, `SER 58`, averaged 22 atoms
spread over **115.1 A** and reported the mean as one residue's environment.

**The chain column is the obvious fix and it is not sufficient**, because the
files this tool is pointed at are the ones that lack it. `pqr.py` reads fields
from the end of the line precisely because the chain is optional, and 1ao6 has
ten fields and two chains. Worse, `prepare_structure` was *creating* that
shape: **pdb2pqr drops the chain ID unless given `--keep-chain`**, which sashimi
never passed, so every structure prepared through the MCP server arrived
unable to distinguish its own chains.

So the fix is in three parts, and each is separately verified by reverting it:

1. `prepare_structure` passes `--keep-chain`. Information-only — on a two-chain
   fixture the coordinates, charges, radii, labels and `format_pqr` output are
   bit-identical with and without it.
2. `PQRData` carries `chains`, a **separate** optional tuple rather than a
   fourth field in `labels`. `format_pqr` recovers the three names by splitting
   the label and falls back to `UNK`/`X` on any other count, so widening the
   label would have silently renamed every atom it writes — and the chain is
   still never *written*, because a chain column shifts every field after it,
   which is the fixed-column failure that cost DelPhi a year of wrong acetate
   charges.
3. Grouping is by **contiguous run** of `(resName, resSeq, chain)`. One
   residue's atoms are contiguous in a PQR, so a new residue begins where that
   tuple changes. This needs no chain IDs and invents none.

Where the file names no chains, the label carries a synthesized *segment*
ordinal — `#2:SER 58`, incremented at each numbering restart — and the `#`
is deliberate: it is an inference from a `resSeq` that failed to advance, not a
chain ID, and must not be dressed up as one. On every multi-chain structure in
`tests/data` the segment and the run agree on the partition; the run is the
primitive and the segment is only how a group is named. A single-chain structure
keeps the bare labels it always reported, **including when the file names its
one chain** — prefixing there distinguishes nothing and would churn every
recording, notably `test_debye_m6.py`'s residue ranking on `fas2`.

**The boundary, asserted rather than discovered later:** two single-residue
chains, adjacent in the file, identically numbered and unnamed. The run boundary
then coincides with nothing and only the coordinates say there are two.
Splitting on distance instead would mean choosing a threshold the file gives no
basis for, so item 1 is what closes that case.

Second-order finding, not acted on: `ion-protein-complex.pqr` is a
coarse-grained two-chain model with the same disease at 260 atoms, and
`fas2.pqr` numbers `NTE 544` and `THR 544` as distinct residues — both handled
by the run rule for free.

### What "nothing but the upload" turned out to mean

**2026-08-27.** Phase 5 was recorded as complete bar the upload, and the
manifest genuinely was complete — name, version, licence, classifiers, urls,
both console scripts. The first `uv build` of this project still failed.

**The sdist was 48.9 MB and could not be unpacked.** hatchling had swept
`.pydelphi/` into it — the pyDelPhi virtual environment the README tells you to
create in the repository root — and a virtualenv contains absolute symlinks into
whoever built it. `uv build` unpacks its own sdist to build the wheel from, so
the build failed on its own artifact, which is the good case: the bad case is a
release that installs on the machine that made it and nowhere else.

`.pydelphi/` was untracked and **not** in `.gitignore`, which is the same class
of hole that put two `.DS_Store` files in the tree. It is ignored now, but the
durable fix is that `[tool.hatch.build.targets.sdist]` carries an **allowlist**:
an exclude list has to anticipate the next such directory, and there will be
one. Verified as an allowlist rather than assumed — an untracked, un-ignored
probe directory does not ship. 48.9 MB to 2.0 MB.

One refinement the check turned up: hatchling globs a bare `README.md` at any
depth, so `studies/README.md` shipped while the other fifty tracked files under
`studies/` did not. Leading slashes anchor the patterns to the root, and the
selection is now deliberate rather than whatever the glob happened to reach.

**What the release workflow checks, and why each one.** The tag must equal
`pyproject.toml`'s version, because PyPI never lets a version number be reused
and a mismatched tag cannot be corrected afterwards. The built *wheel* must
import in a clean interpreter with no project on the path — an editable install
would pass with a `src/` layout the wheel does not ship. Then the binary-free
tier runs, since that is the part that breaks by packaging rather than by
physics; the full suite has already run on the main the tag is cut from.

Publishing is by **Trusted Publishing**: PyPI verifies a short-lived OIDC token
minted for the workflow rather than a long-lived API token in repository
secrets. Nothing to leak and nothing to rotate. It requires a pending publisher
configured on PyPI before the first upload — the one step that is not in this
repository.

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
| 5 | Does the protocol graduate to `pb-protocol`? | ~~At debye, not before.~~ **Superseded 2026-08-13: debye starts in-repo as `sashimi.debye`**, so extraction waits for a consumer other than sashimi. The seam is still a closed set guarded by a test, which is what keeps the deferral cheap — see §10 and §12 |

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
  Confirmed a second time on 2026-08-26 and on a different solute: the
  four-atom Born sphere of `studies/tabipb_units/` crashes at `sdens` 1.0 with
  the same `stoul: no conversion`, so the floor is the mesher's and not the
  dipeptide's. The error message is what makes this survivable rather than the
  default being right.
- ~~**Does `SMOOTHED_MOLECULAR` belong as the default?**~~ **No — resolved
  2026-08-12**, and the trigger was making the backend selectable: a default
  that three of four backends refuse is a default that hides the other three.
  `MOLECULAR` replaces it, measured at 0.80% on ALA-GLY and 2.35% on hen
  lysozyme. This entry first argued that was "inside the 1.0–1.6% two
  reference-tier families already differ by", which is arithmetically false for
  the lysozyme figure and was written that way in the decision it justified.
  The accurate comparison: the switch is smaller than that band on a dipeptide
  and somewhat larger on a protein, and both are an order of magnitude below the
  25.7% between `molecular` and `van-der-waals` (§5) — which is the number that
  says a surface model is the largest modelling choice in the calculation. That
  is the ground the decision rests on. Refusing rather than substituting was
  still right and stays right; what changed is which model the request starts
  from. **Landed 2026-08-13**, after the corpus-explicitness step it needed
  first; §12 records what the two steps measured.
