# sashimi

*Thinly sliced Poisson: a maintained wrapper around APBS for biomolecular electrostatics.*

Computes electrostatic potential maps and polar solvation energies from charged
structures, using the conda-forge [APBS](https://www.poissonboltzmann.org/)
3.4.1 binary as its solver backend. APBS is never vendored, patched, or built
from source.

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

The environment is pixi-managed, which is what pins APBS:

```sh
pixi install
pixi run test          # full suite
pixi run test-fast     # skips everything needing the binary
```

`SASHIMI_APBS_PATH` overrides binary discovery. Otherwise the active
pixi/conda environment is preferred over a system-wide `apbs`, so a Homebrew
install cannot silently shadow the pinned one.

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
(0.62% → 0.11% → 0.02% at 0.41 / 0.20 / 0.16 Å spacing). The FastMCP server
(phase 2) and golden corpus (phase 3) are next.
