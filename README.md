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
misreported as a solver disagreement. All three shipped backends share
`molecular`; `smoothed-molecular` — sashimi's default — is APBS-only, so a
DelPhi solve at defaults refuses rather than silently substituting. On the
models they share, APBS and DelPhi agree to 2.4% on hen lysozyme.

See [ROADMAP.md](ROADMAP.md) for the full design and phasing.

## Status

Phases 0–4 are done and phase 5 is nearly so. The core library is validated
against the closed-form Born ion, converging monotonically as the grid refines
(0.62% → 0.11% → 0.02% at 0.41 / 0.20 / 0.16 Å spacing). The protocol admits a
boundary-element backend without APBS-shaped concessions, proven by a stub that
returns surface potentials through the same `SolveResult`.

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
sashimi corpus verify              # re-solve the manifest, diff against record
sashimi corpus build --force       # re-record, deliberately
```

Five cases, summaries in `tests/corpus/`. It is the regression net for the
unpinned APBS today, and the acceptance gate for a second backend later —
`sashimi corpus verify --backend debye` needs no APBS installed.

With two backends installed, they can be checked against *each other*:

```sh
sashimi validate                   # every case, every installed backend
sashimi validate --backend apbs --backend delphi --surface molecular
```

```
ok    born-ion-coarse   2 backends agree: 2.30% spread (-234.000 to -228.609 kJ/mol)
        apbs           -234.000 kJ/mol  (polar-solvation)
        delphi         -228.609 kJ/mol  (reaction-field)
SKIP  born-ion-salt: refusing to report a spread:
  - energy terms differ at 0.15 M ionic strength. The difference between them is
    exactly the mobile-ion contribution, which is nonzero here…
```

Most of `validate` is about **refusing to answer**. A spread only means "the
solvers disagree" if the surface model, the equation and the *reported energy
term* were all held fixed, and none of those differences shows up in the number.
APBS reports a polar solvation energy; DelPhi reports a reaction-field energy;
they coincide only where there are no mobile ions. Comparing them at
physiological salt would report a definitional gap as a solver disagreement,
so it refuses — and says which one it is.

Energies are compared directly, potentials by sampling both maps at the same
physical coordinates, since two backends never produce the same grid.

Not yet released to PyPI. See [ROADMAP.md](ROADMAP.md) for where this is
heading — it is the single planning document, covering the protocol, the
multi-backend future, distribution and `debye`.
