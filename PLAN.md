# sashimi — Implementation Plan

*Thinly sliced Poisson: a maintained, MCP-native wrapper around APBS for biomolecular electrostatics.*

## 1. Goals and non-goals

Sashimi provides a clean Python interface and FastMCP server for computing electrostatic potential maps from protein structures, using the frozen conda-forge APBS 3.4.1 binary as its solver backend. It exists so that mcpymol and protean never touch APBS input files, temp directories, or OpenDX parsing directly — and so that when debye (a native LPBE solver) eventually exists, it can be swapped in behind the same interface without changing a line of client code.

Explicit non-goals: sashimi does not compile, vendor, or patch APBS source. It does not expose the FEM, geoflow, BEM, PBAM, or PBSAM solvers — only the `mg-auto` finite-difference path, which covers visualization-grade and standard solvation-energy work. It does not attempt nonlinear PBE support in v1 (the API leaves room for it, but LPBE is the contract debye will initially honor). It is not a general APBS input-file generator; it deliberately exposes a narrower, physically-parameterized surface.

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

The protocol layer is the load-bearing decision. Everything above it speaks in physical terms (angstroms, molar ionic strength, dielectric constants); everything below it speaks APBS (dime, cglen, fglen, chgm spl4). Debye replaces only the bottom-left box.

## 3. Package layout

```
sashimi/
├── pyproject.toml            # uv-managed; runtime deps: numpy, pdb2pqr, pydantic, fastmcp
├── uv.lock                   # the single lockfile; APBS comes from brew/apt
├── src/sashimi/
│   ├── protocol.py           # PQRData, GridSpec, PotentialGrid, SolveResult, Solver
│   ├── pqr.py                # PQR read/write (own parser — trivial format, no deps)
│   ├── dx.py                 # OpenDX read/write ↔ numpy + metadata
│   ├── apbs/
│   │   ├── backend.py        # ApbsSolver implements Solver
│   │   ├── input.py          # GridSpec → mg-auto input file (jinja-free f-strings)
│   │   ├── grid.py           # physical grid intent → legal dime/cglen/fglen (psize logic)
│   │   ├── run.py            # subprocess, tmpdir, timeout, stdout parsing, error mapping
│   │   └── discover.py       # binary discovery: $SASHIMI_APBS_PATH → which() → conda env
│   ├── prep.py               # pdb2pqr subprocess wrapper (PDB → PQRData)
│   ├── corpus.py             # golden-corpus build/verify (the debye validation target)
│   └── mcp/server.py         # FastMCP tools
├── tests/
│   ├── test_dx.py, test_pqr.py, test_grid.py     # pure unit tests, no binary
│   ├── test_solver.py        # @pytest.mark.apbs — requires binary
│   ├── test_analytic.py      # born ion vs closed-form
│   └── corpus/               # checked-in golden summaries (JSON, not raw grids)
└── .github/workflows/ci.yml  # micromamba installs apbs + pdb2pqr, runs full suite
```

## 4. The protocol layer (the debye contract)

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
class PotentialGrid:
    values: np.ndarray              # (nx, ny, nz), kT/e
    origin: np.ndarray              # (3,), Å
    spacing: np.ndarray             # (3,), Å  (uniform per-axis)
    # methods: to_dx(path), value_at(xyz), stats()

@dataclass
class SolveResult:
    potential: PotentialGrid
    energy_kj_mol: float | None     # total polar solvation energy when requested
    backend: str                    # "apbs-3.4.1" | "debye-x.y" — provenance travels
    diagnostics: dict               # iterations, residual, wall time, grid actually used

class Solver(Protocol):
    def solve_lpbe(self, pqr: PQRData, grid: GridSpec,
                   solvent: SolventModel = SolventModel(),
                   *, compute_energy: bool = False) -> SolveResult: ...
```

Design notes. Units are fixed at the protocol boundary (Å, kT/e, kJ/mol) so debye never inherits APBS's unit conventions by accident. `GridSpec` deliberately has no `dime`: legal PMG dimensions (n = c·2^(l+1) + 1, i.e. 65, 97, 129, 161, 193, 225, 257 …) are an APBS implementation detail computed in `apbs/grid.py`, which reimplements the essentials of the classic `psize.py` sizing logic — coarse grid at ~1.7× molecular extent for the focusing boundary, fine grid at extent + 2·padding, smallest legal dime meeting the resolution target, capped by `max_points`. Debye, grid-flexible, will just honor `resolution` and `padding` directly. The `diagnostics` dict reports the grid actually used so callers can detect when the resolution target was relaxed.

## 5. APBS backend specifics

**Input generation** targets one template: `mg-auto`, `lpbe`, `bcfl sdh`, `chgm spl4`, dielectric/ion maps per `SolventModel`, `calcenergy total` when requested, `write pot dx`. Roughly 40 lines of f-string; no need for the abandoned `apbs` Python package's input dataclasses.

**Execution** (`run.py`): each solve runs in a fresh `TemporaryDirectory`; write `mol.pqr` + `input.in`, run with a configurable timeout (default 300 s), capture stdout/stderr. APBS exits 0 on some failures, so success is verified structurally: the expected `pot.dx` exists and parses, and stdout is scanned for `Vpmg` error signatures. Failures raise typed exceptions (`ApbsNotFound`, `GridTooLarge`, `ConvergenceFailure`, `ApbsCrash`) carrying the tail of stdout — this maps directly onto actionable MCP error messages later. Energy is parsed from the `Global net ELEC energy` line and converted to kJ/mol.

**DX I/O** (`dx.py`): own reader/writer (~80 lines) rather than a `gridData`/MDAnalysis dependency — the format is trivial (header with counts/origin/deltas, then 3 floats per line, C-order) and owning it keeps sashimi's dependency tree at numpy + pydantic + fastmcp. Round-trip fidelity is unit-tested, and the writer exists so debye's output can be exported to DX for PyMOL/ChimeraX regardless of backend.

**Binary discovery**: `$SASHIMI_APBS_PATH` env var wins, then `shutil.which("apbs")`, then an active conda env as a courtesy. APBS is compiled, so no Python installer can provide it and `which` is the normal answer; it arrives via `brew install apbs` or `apt install apbs`, both of which ship exactly 3.4.1. `SolveResult.backend` records the resolved *path* alongside the version, so which binary produced a given result is always recoverable. On failure the error message includes the platform one-liner to fix it.

Discovery's version probe must run inside a temp dir: **APBS writes an `io.mc` log into the working directory on every invocation, `apbs --version` included**. Solving is already covered by the per-solve `TemporaryDirectory` above, but a probe from the server's cwd litters the user's repo. `io.mc` is gitignored as a backstop.

**Structure prep** (`prep.py`): subprocess wrapper around pdb2pqr 3.7.1 — installed from PyPI, not conda-forge, which tops out at 3.6.1 (`--ff=AMBER`, `--titration-state-method=propka --with-ph 7.0` optional). Subprocess, not its Python API — pdb2pqr's internal API is not a stability contract, and process isolation means a pdb2pqr hang can't take down the MCP server. Returns `PQRData` plus a structured summary of warnings (missing heavy atoms, debumped residues), which the MCP layer surfaces instead of burying.

## 6. FastMCP server

Tools are prefixed `sashimi_`, use Pydantic schemas, and return structured content plus a short human-readable summary. All are `readOnlyHint: false` only where they write files; none are destructive.

`sashimi_prepare_structure(pdb_path, ph=7.0, forcefield="AMBER") → pqr_path + warnings` — runs pdb2pqr; the returned warnings summary is the important part (an agent should know three sidechains were rebuilt before trusting downstream energies).

`sashimi_solve(pqr_path, resolution=0.5, padding=10.0, ionic_strength=0.15, solute_dielectric=2.0, compute_energy=False, output_dx=None) → stats + dx_path + energy` — the workhorse. Flat, physically-named parameters (agents handle these better than nested config objects); response carries grid stats (min/max/mean potential, dimensions, spacing actually achieved), the DX path for PyMOL loading, energy when requested, and backend provenance.

`sashimi_potential_at(dx_path, points) → values` — trilinear interpolation of a saved map at arbitrary coordinates; cheap way for an agent to ask "what's the potential at this residue's CB" without a re-solve.

`sashimi_compare_maps(dx_a, dx_b) → diff stats` — grid-aware difference statistics (RMSD, max abs diff, correlation). Useful for mutant-vs-wildtype questions now, and doubles as the debye-vs-apbs validation tool later.

Deliberately not included: a PDB-fetching tool (mcpymol already owns structure acquisition) and raw APBS-input passthrough (defeats the abstraction; if someone needs `fe-manual`, they need APBS, not sashimi).

Transport is stdio, matching your existing FastMCP fleet. Solves at default resolution on a ~300-residue protein run in seconds; no async job queue needed in v1, but the 300 s timeout plus a clear timeout error keeps a pathological grid from wedging the server.

## 7. Testing and the golden corpus

Three test tiers. Pure unit tests (PQR parsing, DX round-trip, dime legality, psize math) run anywhere with no binary. Analytic tests solve the Born ion — a single +1 charge, 3 Å radius sphere — and check the computed polar solvation energy against the closed-form Born expression ΔG = −(q²/8πε₀r)(1/εₚ − 1/εₛ) within a **1%** grid-discretization tolerance, at two resolutions to confirm the error shrinks with spacing. This is the test that catches unit-conversion bugs, which are the entire failure mode of wrapper projects.

Two calibration facts from the Phase 0 spike, both of which will otherwise produce a test that fails for the wrong reason. First, **pin the analytic value in-repo rather than quoting APBS's**: the APBS `examples/born` README states −230.62 kJ/mol, while the closed-form expression above with current CODATA constants gives **−228.61 kJ/mol**. That 0.87% spread straddles a 0.5% gate — measured against the CODATA value `smol` errs 0.176% and `mol` 0.509%; against APBS's, 0.70% and 0.37% — so each reference fails a different variant. Sashimi computes its own analytic value from named constants and asserts at 1%. Second, **probe only well outside the dielectric boundary**: at exactly r = a = 3 Å the potential is off by 71% (4.063 vs 2.379 kT/e) because the point sits on the smoothed discontinuity, and the grid center is the +5501 kT/e point-charge singularity. From 3.75 Å outward agreement is 0.5–1.1%. Probe points live at r ≥ 1.25 a. Integration tests run 4–6 cases drawn from the APBS `examples/` set (born, solv, protein-RNA subset) end-to-end.

The golden corpus is a first-class deliverable, not a test artifact. `sashimi corpus build` runs a fixed manifest of cases (structure + GridSpec + SolventModel, seeds pinned) and writes, per case, a JSON summary: grid geometry, energy, potential min/max/mean/std, and potential values at ~50 pinned probe points per structure. Summaries are checked into the repo; raw grids are reproducible on demand. `sashimi corpus verify --backend X` re-runs the manifest against any `Solver` and diffs within stated tolerances. Day one this is a regression net for sashimi itself (and for any future conda-forge APBS rebuild); the day debye exists, `sashimi corpus verify --backend debye` is its acceptance test, with APBS ground truth baked in and no APBS installation required.

CI: GitHub Actions running `uv run pytest` against the committed `uv.lock`, full suite on `ubuntu-latest` and `macos-latest`. uv covers Python exactly; APBS comes from conda-forge via micromamba, whose only job in CI is fetching that one binary. conda-forge is chosen over the system package managers for provenance — macOS APBS exists only in the third-party `brewsci/bio` tap, and Debian's build is MPI-enabled and names its output `potential-PE0.dx` rather than `potential.dx` (`find_potential` accepts either, and the binary-free tests cover both).

Because APBS is no longer version-pinned by a lockfile, `tests/test_corpus.py` carries that weight: it asserts the discovered version and re-solves the Born ion against checked-in energies and probe values (`tests/corpus/born-sashimi.json`, regenerated deliberately via `scripts/build_corpus.py`). A drifted system binary fails there, in kJ/mol. This is the phase 0/1 slice of the golden corpus below, pulled forward to cover exactly what the lockfile used to.

**Settled in Phase 0 — `osx-arm64` is fine.** conda-forge ships an `osx-arm64` build of apbs 3.4.1; it installs and runs natively on Apple silicon, and the Born ion reproduces the published energies to seven significant figures (`smol` −229.0124252387, `mol` −229.7735526282, vs APBS's −229.0124 / −229.7736). No Rosetta, no local build, no README caveat. Provisional ground truth is checked in at `tests/corpus/born-phase0.json`. Add `osx-arm64` to the CI matrix as a cheap regression guard.

## 8. Phasing

**Phase 0 — spike (half a day). ✅ Done.** Pixi env with apbs 3.4.1 (conda-forge) + pdb2pqr 3.7.1 (PyPI); born example reproduces published energies on `osx-arm64`; energy, grid geometry, and five probe points locked in `tests/corpus/born-phase0.json`. Also confirmed the DX contract §5 assumes — 65³, uniform 0.1875 Å spacing, C-order, 3 floats per line, kT/e.

**Phase 1 — core library (2–3 days). ✅ Done.** `protocol.py`, `pqr.py`, `dx.py`, `apbs/` backend; 49 tests green (31 of them binary-free). Exit criterion met: `ApbsSolver().solve_lpbe(pqr, GridSpec())` on the Born ion returns −230.03 / −228.87 / −228.56 kJ/mol at 0.41 / 0.20 / 0.16 Å spacing against a closed form of −228.61 — 0.62% → 0.11% → 0.02%, converging monotonically, which is what rules out a systematic unit error.

One structural note: `GridTooLarge` and `ConvergenceFailure` are backend-neutral and live in `sashimi/errors.py`, while `ApbsNotFound` and `ApbsCrash` subclass them under `sashimi.apbs`. A native solver hits the first pair too; only the second pair is APBS's business. §5's four exception names all exist, just split across that boundary.

**Phase 2 — MCP server (1–2 days).** The four tools, error mapping, MCP Inspector pass, registration in your Claude config alongside mcpymol. Exit criterion: from a chat session, prepare a PDB and produce a DX that PyMOL renders as a sane surface potential.

**Phase 3 — corpus + CI (1–2 days).** Corpus manifest, build/verify commands, checked-in summaries, GitHub Actions green. Exit criterion: a deliberate unit-conversion bug is caught by `corpus verify`.

**Phase 4 — integration (ongoing, small).** mcpymol grows a convenience that chains `sashimi_solve` → load DX → `ramp_new`/surface coloring; protean consumes `SolveResult` for whatever energetics it needs. Sashimi itself should go quiet after this — a wrapper that needs constant attention has failed at its one job.

## 9. Risks and mitigations

The conda-forge apbs package going unbuildable on future platforms is the long-tail risk; mitigation is that the recipe is small and forkable, and the corpus means a re-validated fork is cheap to trust. (The acute version of this — no `osx-arm64` build — was retired in Phase 0: the build exists and is verified. `linux-64`, `osx-64`, `osx-arm64`, and `win-64` are all published for 3.4.1.) APBS's silent-failure modes (exit 0 on error) are handled by structural output verification rather than exit codes. pdb2pqr's occasionally opinionated rebuilding of structures is surfaced, not hidden, via the warnings channel. Grid memory blowups on large complexes are capped by `max_points` with an error that states the achievable resolution. And the abandoned upstream `apbs` Python package sharing the import name is avoided by never depending on it — sashimi's namespace is its own.

## 10. Debye handoff checklist (written now, so sashimi stays honest)

Debye ships as a separate repo implementing `sashimi.protocol.Solver` (protocol types either stay importable from sashimi or graduate to a tiny shared `pb-protocol` package — decide only when debye starts). Its acceptance gate is `sashimi corpus verify --backend debye` within tolerances. The MCP server grows a `backend` parameter defaulting to auto-selection, and `SolveResult.backend` provenance means every downstream artifact already records which solver produced it. Nothing in mcpymol or protean changes.
