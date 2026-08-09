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
brew install apbs        # macOS, native on Apple silicon
sudo apt install apbs    # Ubuntu 24.04+ / Debian 12+ (22.04 has only 3.0.0)
```

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

Phases 0 and 1 are done: the core library solves and is validated against the
closed-form Born ion, converging monotonically as the grid refines
(0.62% → 0.11% → 0.02% at 0.41 / 0.20 / 0.16 Å spacing). A slice of phase 3's
golden corpus is in place early, guarding the unpinned APBS. The FastMCP server
(phase 2) is next.
