# sashimi

*Thinly sliced Poisson: a maintained wrapper around APBS for biomolecular electrostatics.*

Distributed as **`sashimi-electro`**; imported as **`sashimi`**. The plain
`sashimi` name on PyPI belongs to an unrelated, dormant library.

Computes electrostatic potential maps and polar solvation energies from charged
structures, using the [APBS](https://www.poissonboltzmann.org/) 3.4.1 binary
as its solver backend. APBS is never vendored, patched, or built from source —
it comes from your system package manager.

```python
from sashimi import FiniteDifferenceRequest, GridSpec, SolventModel
from sashimi.apbs import ApbsSolver
from sashimi.pqr import read_pqr

result = ApbsSolver().solve(
    FiniteDifferenceRequest(
        structure=read_pqr("protein.pqr"),
        solvent=SolventModel(ionic_strength=0.15),
        grid=GridSpec(resolution=0.5, padding=10.0),
    )
)

print(result.energy_kj_mol)              # -1234.5 (kJ/mol)
print(result.provenance.summary())       # apbs-3.4.1 sha256:83acdfb0818a
result.potential.to_dx("potential.dx")   # load in PyMOL / ChimeraX
result.potential.value_at([[12.0, 3.4, -1.0]])
```

The request type is per solver family: a `FiniteDifferenceRequest` carries a
grid, a `BoundaryElementRequest` carries a mesh density and no grid at all, so a
request a backend cannot honor is unrepresentable rather than merely rejected.

## Install

Two steps, because APBS is a compiled binary that no Python installer can
provide. Install it from your system package manager first:

```sh
# Recommended — same channel and build CI tests against, linux-64 and osx-arm64
micromamba create -n apbs -c conda-forge apbs=3.4.1
export SASHIMI_APBS_PATH="$(micromamba run -n apbs which apbs)"
```

Any conda-family tool works for that (`conda`, `mamba`, `pixi`); it manages
nothing Python here, it just fetches one binary. If you would rather use a
system package manager:

```sh
brew tap brewsci/bio && brew install brewsci/bio/apbs   # macOS, native arm64
sudo apt install apbs    # Ubuntu 24.04+ / Debian 12+ (22.04 has only 3.0.0)
```

Both work, but neither is as well-governed: APBS is **not** in homebrew-core —
`brewsci/bio` is a third-party tap that recent Homebrew refuses to load without
an explicit trust step — and Debian's build differs observably from it. CI
therefore tests the conda-forge build on both platforms.

That difference is real and worth knowing: Debian's APBS is MPI-enabled and
writes its grid as `potential-PE0.dx`, where the Homebrew and conda-forge
builds write `potential.dx`. Same version, same input, same contents.
`sashimi.apbs.run.find_potential` accepts either.

Then everything else, which is pure Python:

```sh
uv sync                       # creates .venv from uv.lock
uv run pytest                 # full suite
uv run pytest -m "not apbs"   # skips everything needing the binary
```

`SASHIMI_APBS_PATH` overrides binary discovery; otherwise `which apbs` wins,
with an active conda environment as a fallback.

### If you are doing real electrostatics with `debye`, install the extra

**`debye` — the built-in solver that needs no binary at all — is roughly seven
times faster with `sashimi-electro[fast]`, and you almost certainly want it if
you are working on proteins rather than peptides.**

```sh
uv sync --all-extras                 # or: pip install 'sashimi-electro[fast]'
```

**It is an extra rather than a dependency because it is large: ~145 MB**, since
it pulls in numba and llvmlite — several times the size of everything else here
combined. That is a real cost on a laptop, in a container image and in CI, and
it is not a decision to make on your behalf when many callers solve one small
structure and never notice the difference.

What it buys, and where. Between 86% and 92% of a `debye` solve is classifying
grid points against the solvent-excluded surface, and the extra swaps that loop
for a compiled one — measured at **6.8x on a 382-residue protein and 7.0x on a
1,156-residue one**. On this machine that is a 1,156-residue solve going from
about two and a half minutes to about half a minute. It changes nothing else:
**the energies are bit-identical either way**, which `tests/test_debye_kernel.py`
asserts on real geometry rather than assuming, and CI runs the pure-numpy path
on two of its three legs and the compiled one on the third.

If you are unsure whether you have it, `sashimi_capabilities` reports
`acceleration.compiled_surface_kernel` and, when it is missing, says so in one
sentence. `SASHIMI_NO_NUMBA=true` turns it off without uninstalling anything (`1`, `yes`
and `on` work too; `false`, `0`, `no` and `off` leave it on).

Nothing else in sashimi is affected — APBS, DelPhi, TABI-PB and `gb` do not use
it, and `debye` is correct without it, only slower.

Both package managers above ship exactly the 3.4.1 this project is frozen
against — but unlike a lockfile, neither *holds* it there. `tests/test_corpus.py`
is what replaces that pin: it asserts the discovered version and re-solves the
Born ion against checked-in energies and probe values. If a `brew upgrade`
moves APBS underneath you, that test tells you in kJ/mol. Regenerate the corpus
deliberately with `sashimi corpus build --force` — never to turn a red test green.

## Development

```sh
uv run pre-commit install   # once, per clone
uv run pytest && uv run ruff check . && uv run mypy
```

Ruff and mypy (strict) run as pre-commit hooks and again in CI, both through
`uv run`, so the hook and CI use the one version pinned in `uv.lock` — there is
no second set of tool versions to drift. The binary-free test tier runs on
push; CI runs the full suite including the APBS-marked tests.

### Repository settings

Two settings the checked-in workflows depend on, neither of which lives in the
repo:

1. **Settings → General → Allow auto-merge.** Without it, the Dependabot
   auto-merge workflow fails at `gh pr merge --auto`.
2. **A ruleset on `main` requiring the `ci-ok` status check.** This is what
   makes auto-merge *wait*. `ci-ok` is a single job gating the whole matrix, so
   the required check name stays stable when the matrix changes.

Both are configured on this repository. They need a public repository or a
paid plan; on a private repo on the Free plan the ruleset API returns 403 and
`allow_auto_merge` silently stays `false`.

If either is ever missing, the auto-merge workflow skips with a warning rather
than merging. That guard is not decoration: `gh pr merge --auto` does **not**
fail when auto-merge is unavailable — it merges the PR immediately, ignoring
CI. PR #1 landed on `main` with a red build exactly that way.

Dependabot covers GitHub Actions and the `uv` ecosystem, which updates
`pyproject.toml` and `uv.lock` in the same commit. It does not cover APBS —
no ecosystem tracks system packages — which is what `tests/test_corpus.py`
guards instead.

## Layers

`sashimi.protocol` is the load-bearing boundary. Above it everything speaks
physics — angstroms, molar ionic strength, dielectric constants, kT/e, kJ/mol.
Below it, `sashimi.apbs` speaks APBS — `dime`, `cglen`, `fglen`, `srfm`. A
future native solver implements `Solver` and replaces only the bottom layer, so
no APBS concept may leak upward.

Scope is deliberately narrow: the `mg-auto` finite-difference path with the
linearized PBE. The FEM, geoflow, BEM, PBAM and PBSAM solvers are not exposed,
and there is no raw APBS-input passthrough — that would defeat the abstraction.

`sashimi.delphi` is the second backend, and the test of whether that boundary
was real. It required no protocol change at all, absorbing a cubic grid, a
Gaussian-cube map in Bohr, energies in kT and a temperature parameter in
*Celsius* below the same `Solver` interface.

`sashimi.tabipb` is the harder test. It is a **boundary-element** solver: no
grid at all, and its answer is a potential on a triangulated dielectric
interface rather than a volume. That is the split the protocol was shaped around
— `BoundaryElementRequest` and `SurfacePotential` were designed in phase 4 for a
solver that did not exist yet — and it too needed no change. Three solvers
across two families agree on ALA-GLY to **2.08%**, with the boundary-element
answer falling between the two finite-difference ones.

A BEM backend answers different questions, and sashimi says so rather than
pretending otherwise: there is no volume to interpolate, so `sashimi_potential_at`
does not apply, and a single-sphere solute cannot be triangulated at all.

`sashimi.gb` is the odd one out twice over. It is a **Generalized Born**
approximation rather than a discretization of the equation, and it runs **in
this process** — no binary, no environment variable, nothing to install, so it
is the one backend that cannot be missing. Roughly 300 lines of numpy: hen
lysozyme in 0.196 s against APBS's 6.79 s, landing 1.48% from the APBS/DelPhi
consensus.

Being an approximation is declared, not implied. `AccuracyTier` in provenance
says so, and `sashimi validate` reports it separately rather than averaging it
into the spread — otherwise the tolerance would have to be wide enough to
accommodate an approximation, which is wide enough to hide a real regression in
the solvers being compared.

Two things about it are worth knowing before using it:

- **It answers on the molecular surface**, though its integral runs over van der
  Waals spheres. The rescaling exists to carry one onto the other. Declared the
  intuitive way it sits 31% from APBS instead of 4.7%.
- **It substitutes mbondi radii by default.** pdb2pqr emits Lennard-Jones radii,
  including exactly 0 for hydroxyl hydrogens — fine for a grid solver, an
  infinite self-energy for a method that divides by radius. Using them as given
  costs 35% on a protein. The substitution is counted in the result's
  diagnostics, and `GbRadii.AS_GIVEN` turns it off.

`sashimi.debye` is the clean-room solver, and the point of it is the
combination: it needs **no binary** and it is in the **reference tier**. `gb`
needs nothing installed but approximates the equation; every other backend that
discretizes it needs a compiled program. So on a machine with no APBS you get
real Poisson-Boltzmann rather than known-wrong physics — which is the gap that
reordered this roadmap, since the consumer it exists for has no binary
available.

About 2,000 lines of numpy: finite-volume flux balance with the dielectric
sampled at face centres, cloud-in-cell charge assignment, a Debye-Huckel
boundary on the box face, and multigrid-preconditioned CG whose coefficients are
re-discretized from the geometry at each level. It solves the linearized
equation on the van der Waals and solvent-excluded boundaries, and **declines
`smoothed-molecular` and `gaussian` on purpose** — those are APBS's harmonic
averaging and DelPhi's Gaussian dielectric, and reproducing them would be
claiming another code's discretization rather than the equation. `corpus verify`
reports the cases it declines as refusals rather than as gaps.

It is graded against the incumbents rather than against a round number: 0.85%
from the Born closed form, within 1.5% of the Kirkwood series, and on the
solvent-excluded surface it sits *between* DelPhi C++ and APBS on all twelve
real structures measured from 906 to 8,279 atoms.

### Using DelPhi

Neither DelPhi flavour has a package, so both are opt-in and neither is a
dependency:

```bash
# the C++ reference build — compile from the Clemson tarball, then:
export SASHIMI_DELPHI_PATH=/path/to/delphicpp_release

# or the same lab's pure-Python reimplementation, which installs anywhere
uv venv --python 3.13 .pydelphi
uv pip install --python .pydelphi/bin/python git+https://github.com/shaileshp51/pyDelPhi
export SASHIMI_DELPHI_PATH="$PWD/.pydelphi/bin/pydelphi-static"
```

pyDelPhi is driven as a subprocess and never imported: it is AGPL-3.0 where
sashimi is MIT, and that is the boundary APBS already sits behind. The C++ build
is faster and additionally supports a van der Waals boundary; pyDelPhi needs no
compiler and runs anywhere, including `linux-aarch64`, where no APBS exists.

**The backends do not agree about which surface models exist**, and surfacing
that is the most useful thing a second backend does. `sashimi_capabilities`
reports which models the installed backends actually share, because a spread
computed across mismatched surface definitions is a modelling difference
misreported as a solver disagreement. All five shipped backends share
`molecular`, which is why it is the default; `smoothed-molecular` is APBS-only,
so a solve that asks for it refuses on the other four rather than silently
substituting, and `van-der-waals` is APBS, DelPhi and debye only. On the models
they share, APBS and DelPhi agree to 2.4% on hen lysozyme.

See [ROADMAP.md](ROADMAP.md) for the full design and phasing.

## Status

Phases 0–4 and 7 are done, and phase 5 needs only its PyPI release. The core
library is validated against the closed-form Born ion, converging monotonically
as the grid refines (0.62% → 0.11% → 0.02% at 0.41 / 0.20 / 0.16 Å spacing).

Five backends now span three solver families — finite difference, boundary
element, analytic — and the protocol absorbed all five without changing shape:
two enum members, no new types. They agree on ALA-GLY to 3.65% across families,
and on hen lysozyme to 1.97%.

**Two of the five need nothing installed**, and since they share a surface model
you can cross-validate on a machine with no APBS, no DelPhi and no mesher.
`sashimi.debye` is the newer one: a clean-room finite-difference solver in numpy
that discretizes the linearized Poisson-Boltzmann equation rather than
approximating it, on the van der Waals and solvent-excluded boundaries.

The MCP server exposes nine tools over stdio:

| Tool | Question it answers |
|---|---|
| `sashimi_capabilities` | what can this installation do? |
| `sashimi_validate_inputs` | would this solve work, and what would it cost? |
| `sashimi_prepare_structure` | PDB → PQR, and what did pdb2pqr rebuild? |
| `sashimi_solve` | the potential map and solvation energy |
| `sashimi_potential_at` | the potential at these coordinates |
| `sashimi_potential_extrema` | where are the strongest patches? |
| `sashimi_potential_in_sphere` | what is the field in this pocket? |
| `sashimi_residue_potentials` | which residues sit in negative potential? |
| `sashimi_compare_maps` | how do two maps differ? |

Run it with `sashimi-mcp`, or register it:

```json
{ "mcpServers": { "sashimi": { "command": "uv",
    "args": ["run", "--project", "/path/to/sashimi", "sashimi-mcp"] } } }
```

The golden corpus is a first-class feature:

```sh
sashimi corpus verify              # the standard tier, diffed against record
sashimi corpus verify --tier full  # all 98 cases
sashimi corpus build --force       # re-record, deliberately
```

Ninety-eight cases, summaries in `tests/corpus/`, split by measured wall time
into `fast` (what `pytest` runs), `standard` (a CI step per push) and `full` (on
demand). It is the regression net for the unpinned APBS, and it is the
acceptance gate the clean-room solver was held to: `sashimi corpus verify
--backend debye` needs no APBS installed, and passes.

Many of those cases are on a surface model more than one solver supports, so
they carry answers from more than one backend: `tests/corpus/delphi/` holds
fifty-six, `tests/corpus/debye/` fifty-six from the clean-room solver,
`tests/corpus/gb/` thirty from the in-process Generalized Born tier, and
`tests/corpus/tabipb/` six from the boundary-element one. No backend answers
the whole corpus — debye declines APBS's harmonic averaging and DelPhi's
Gaussian dielectric by name, and `corpus verify` reports those as refusals
rather than as gaps. Comparing
those files against `tests/corpus/` needs no binary at all, which is where the corpus says something
neither recording could alone — the two reference-tier families agree to
1.0–1.6%, and the approximate tier ranges from 0.7% to 28%.

With two backends installed, they can be checked against *each other*:

```sh
sashimi validate                              # the fast tier, every installed backend
sashimi validate --tier full                  # every case; tens of minutes with five backends
sashimi validate --case lysozyme-molecular    # a named case, whatever tier it is in
sashimi validate --backend apbs --backend delphi --surface molecular
```

```
  ok    born-ion-coarse          3 backends agree: 2.30% spread (-234.000 to -228.609 kJ/mol,
                                 tolerance 10%); approximations as documented from the
                                 reference: gb 1.76% (tolerance 15%)
          apbs           -234.000 kJ/mol  (polar-solvation)
          delphi         -228.609 kJ/mol  (polar-solvation)
          gb             -235.681 kJ/mol  (polar-solvation)  [approximate, 1.76% from reference]
          debye          -232.213 kJ/mol  (polar-solvation)
          not asked: tabipb — tabipb needs at least 4 atoms to triangulate a surface and this structure has 1
          same system as born-ion-molecular, born-ion-vdw once the molecular surface is applied — solved once
```

Two lines there are the point of the tool. **`not asked`** means that backend
declined *this* system and the others were compared without it — a boundary
element solver has no mesh for a one-atom solute, and that is not a reason to
throw away the answers of the four solvers that do. **`same system as`** means
the named cases differ only in a surface model that `validate` had to override
to compare across backends, so they are one question and were solved once;
without that line their identical energies read as a measurement.

`validate` **exits non-zero when any compared case disagrees**, which on a
fully-installed machine today means `peptide-low-solvent-dielectric`: the three
finite-difference backends land within 1.8% while TABI-PB reads 19.4% away and
`gb` 46.6%. That is a real result about a low solvent dielectric, not a broken
install. A case is only `SKIP`ped when fewer than two backends can take it, or
when one crashes on a structure it had accepted.

Most of `validate` is about **refusing to answer**. A spread only means "the
solvers disagree" if the surface model, the equation and the *reported energy
term* were all held fixed, and none of those differences shows up in the number.

That last one is not hypothetical. DelPhi's headline energy is the polarization
term alone, where APBS's is a difference against an ion-free reference and so
carries the mobile-ion atmosphere; they coincide only at zero salt. sashimi asks
the C++ DelPhi for the matching quantity instead of comparing the two — but
pyDelPhi cannot report it, so with that flavour installed a salted comparison is
refused, and says why.

Energies are compared directly, potentials by sampling both maps at the same
physical coordinates, since two backends never produce the same grid.

### Choosing a backend by what you want, not by name

`sashimi_solve` takes `prefer` as well as `backend`, for the caller who knows
what they want from the answer but not which program gives it:

| | leads with | why |
|---|---|---|
| `fast` | APBS | 12.4 s where pyDelPhi is 27.3 s at 1,156 residues, and it answers every surface model |
| `stable` | pyDelPhi | least sensitive to where the lattice falls — 0.15–0.52% under rigid rotation against APBS's 0.42–1.07%, measured on a *coarser* grid in every comparison |
| `portable` | debye | ships with sashimi, no install step; the only option on `linux-aarch64`, where conda-forge has no APBS |

**It resolves against what is installed *and* what the request needs.** pyDelPhi
has no van der Waals boundary, so `prefer="stable"` on one falls through to
DelPhi C++ and the result says so in `selected_because`. Naming `backend`
explicitly always wins. `sashimi_capabilities` prints the whole table for the
machine it is on.

**`stable`, not `accurate`** — above a two-atom solute nothing here has a
reference answer, so what is measured is how little the answer moves when the
solute is rotated, which is discretization noise and not distance from truth.

The two DelPhi builds are separately addressable as `delphi-cpp` and `pydelphi`;
`delphi` still means whichever one is installed.

### Measuring a change to the solver

`debye` runs in this process, so it is the one backend whose speed is sashimi's
own problem. `sashimi bench` is the instrument its performance claims are made
on, and every part of it is a reaction to a measurement that lied:

```sh
sashimi bench --structure tests/data/apbs-examples/fas2.pqr
sashimi bench --structure tests/data/apbs-examples/fas2.pqr --against ../sashimi-main
sashimi bench --structure tests/data/apbs-examples/fas2.pqr --backend apbs
```

**`--backend` times any installed solver**, not only `debye`, and CPU time
includes subprocesses — `time.process_time()` cannot see a child, so timing APBS
or DelPhi without `getrusage(RUSAGE_CHILDREN)` measures nothing but this
process's bookkeeping. That is how every backend row in ROADMAP.md §12 is
produced. Two caveats it will tell you about: `--resolution` means nothing to
`gb`, which discretizes nothing, or to `tabipb`, whose cost knob is mesh
density; and on a platform without POSIX `resource` the command warns that
subprocess time is not being counted.

**Wall clock cannot measure this.** Interleaved on one machine, alternating
between two revisions of the same case, *identical* code read 61.8 / 79.1 /
42.5 s against 52.2 / 44.6 / 60.5 s — a 1.9x spread with the ranges overlapping
completely. That is other processes taking wall-clock time away from the run,
which does not change how many CPU-seconds the work costs. So `bench` reports
**CPU time**, takes the **minimum of N** rather than a mean, and with
`--against` **interleaves** the two revisions one sample at a time so that a
drift in machine state lands on both. The same pair that read as a 40%
regression on wall clock reads 44.96 s against 42.04 s here.

`--against` takes any checkout of this repository — a `git worktree` at the
revision you are comparing to. It runs there with that tree's own lockfile, and
it does **not** require the baseline to contain `sashimi bench`, which would
make every revision before this one unmeasurable.

**The energy is part of the measurement.** A solver can be made arbitrarily fast
by computing something else, so both sides report their energy to full precision
and `bench` exits non-zero when they differ. An answer-preserving change is held
to *bit*-identical, not to a tolerance — which is the bar the batched surface
work in M7 has to meet.

Not yet released to PyPI. See [ROADMAP.md](ROADMAP.md) for where this is
heading — it is the single planning document, covering the protocol, the
multi-backend future, distribution and `debye`.
