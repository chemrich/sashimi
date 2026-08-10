# sashimi

*Thinly sliced Poisson: a maintained wrapper around APBS for biomolecular electrostatics.*

Computes electrostatic potential maps and polar solvation energies from charged
structures, using the [APBS](https://www.poissonboltzmann.org/) 3.4.1 binary
as its solver backend. APBS is never vendored, patched, or built from source —
it comes from your system package manager.

```python
import numpy as np
from sashimi import GridSpec, PQRData, SolventModel
from sashimi.apbs import ApbsSolver
from sashimi.pqr import read_pqr

result = ApbsSolver().solve_lpbe(
    read_pqr("protein.pqr"),
    GridSpec(resolution=0.5, padding=10.0),
    SolventModel(ionic_strength=0.15),
    compute_energy=True,
)

print(result.energy_kj_mol, result.backend)   # -1234.5 kJ/mol, apbs-3.4.1
result.potential.to_dx("potential.dx")        # load in PyMOL / ChimeraX
result.potential.value_at([[12.0, 3.4, -1.0]])
```

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
deliberately with `uv run python scripts/build_corpus.py` — never to turn a red
test green.

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

See [PLAN.md](PLAN.md) for the full design and phasing.

## Status

Phases 0–2 are done. The core library is validated against the closed-form
Born ion, converging monotonically as the grid refines (0.62% → 0.11% → 0.02%
at 0.41 / 0.20 / 0.16 Å spacing), and a slice of phase 3's golden corpus is in
place early to guard the unpinned APBS.

The MCP server exposes four tools — `sashimi_prepare_structure`,
`sashimi_solve`, `sashimi_potential_at`, `sashimi_compare_maps`. Run it over
stdio with `sashimi-mcp`, or register it:

```json
{ "mcpServers": { "sashimi": { "command": "uv",
    "args": ["run", "--project", "/path/to/sashimi", "sashimi-mcp"] } } }
```

Phase 3 adds the golden corpus as a real feature:

```sh
sashimi corpus verify              # re-solve the manifest, diff against record
sashimi corpus build --force       # re-record, deliberately
```

Five cases, summaries in `tests/corpus/`. It is the regression net for the
unpinned APBS today, and the acceptance gate for a second backend later —
`sashimi corpus verify --backend debye` needs no APBS installed.

See [ROADMAP.md](ROADMAP.md) for where this is all heading; PLAN.md is the
narrower APBS implementation plan beneath it.
