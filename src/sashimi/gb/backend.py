"""GbSolver — the first backend that is not a subprocess.

Two firsts, and they are related. This is the first backend that *approximates*
the Poisson-Boltzmann equation rather than discretizing it, so it is the first
to declare `AccuracyTier.APPROXIMATE` and the reason that enum exists. And it is
the first solver that runs in this process: no binary to discover, no input file
to write, no output to parse, nothing to install. ROADMAP.md section 8 expected
PyGBe to prove the protocol was transport-agnostic; it proves it by having no
transport at all.

What that costs, stated rather than hidden:

- **`binary_path` and `binary_sha256` are `None`.** Provenance has allowed that
  since phase 4 and no real backend had ever exercised it. The identity of the
  code that produced the number is this package's own version.
- **There is no `timeout`.** A function call cannot be interrupted the way
  `subprocess.run(timeout=)` can. The substitute is that the cost is knowable
  before it is paid: the work is O(N^2) with no iteration and no convergence
  criterion, so it cannot fail to terminate — it can only be large, and
  `GbOptions.chunk_size` bounds the memory that largeness costs.
- **It returns no potential.** Generalized Born computes solvation energy from
  effective radii; there is no field to sample and constructing one would be
  invention.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from sashimi.errors import UnsupportedRequest
from sashimi.gb.energy import debye_kappa, polar_solvation_energy
from sashimi.gb.options import GbOptions, check_surface
from sashimi.gb.radii import effective_radii, input_radii
from sashimi.protocol import (
    AccuracyTier,
    EnergyTerm,
    Provenance,
    SolveRequest,
    SolveResult,
)

__all__ = ["BACKEND_VERSION", "GbSolver"]

# This backend's own answer changes only when this package changes, so its
# identity is a version rather than a checksum of somebody else's binary.
BACKEND_VERSION = "1"


@dataclass
class GbSolver:
    """Generalized Born solvation energies, in process.

    Takes the base `SolveRequest`: Generalized Born needs neither a grid nor a
    mesh, so it is `Solver[SolveRequest]` and — the protocol's request type
    being contravariant — accepts either family's request without either family
    having to know it exists.
    """

    options: GbOptions = field(default_factory=GbOptions)

    @property
    def label(self) -> str:
        return f"gb-{self.options.model.value}-{BACKEND_VERSION}"

    def solve(self, request: SolveRequest) -> SolveResult:
        solvent = request.solvent
        # Both refusals land before any work, as with every other backend.
        check_surface(solvent.surface_model)
        if request.want_potential:
            raise UnsupportedRequest(
                "Generalized Born computes a solvation energy from effective "
                "radii and has no potential field to report. Ask a "
                "finite-difference backend for a map, or set want_potential=False."
            )

        started = time.perf_counter()
        _, n_substituted = input_radii(request.structure, self.options)
        radii = effective_radii(request.structure, self.options)
        energy = polar_solvation_energy(
            request.structure.coords,
            request.structure.charges,
            radii,
            solvent,
            self.options.chunk_size,
        )
        wall_seconds = time.perf_counter() - started

        provenance = Provenance(
            backend=self.label,
            # No binary: nothing was discovered, nothing was executed. This is
            # the case `Provenance` allowed for and had never been handed.
            binary_path=None,
            binary_sha256=None,
            resolved_parameters={
                "surface_model": solvent.surface_model.value,
                "solute_dielectric": solvent.solute_dielectric,
                "solvent_dielectric": solvent.solvent_dielectric,
                "ionic_strength": solvent.ionic_strength,
                "temperature": solvent.temperature,
                "gb": {
                    "model": self.options.model.value,
                    "radii": self.options.radii.value,
                    "offset": self.options.offset,
                    "minimum_radius": self.options.minimum_radius,
                    "element_screening": self.options.use_element_screening,
                    "debye_kappa_inverse_a": round(debye_kappa(solvent), 6),
                },
            },
            wall_seconds=round(wall_seconds, 4),
            # The Debye-Huckel factor is applied inside the pair sum, so this
            # energy moves with ionic strength and carries the mobile-ion
            # contribution the way APBS's difference-of-blocks does.
            energy_term=EnergyTerm.POLAR_SOLVATION,
            accuracy_tier=AccuracyTier.APPROXIMATE,
        )

        result = SolveResult(
            provenance=provenance,
            energy_kj_mol=energy if request.want_energy else None,
            potential=None,
            diagnostics={
                "family": "analytic",
                "n_atoms": request.structure.n_atoms,
                # Not cosmetic: a caller comparing this against a PB energy is
                # entitled to know the two were not handed identical radii. Under
                # the default mbondi set most atoms differ, and that substitution
                # is worth a third of the answer on a protein.
                "n_radii_substituted": n_substituted,
                "effective_radius_min_a": float(np.min(radii)),
                "effective_radius_max_a": float(np.max(radii)),
                "effective_radius_mean_a": float(np.mean(radii)),
                "accuracy_tier": (
                    "an approximation to the Poisson-Boltzmann equation, not a "
                    "discretization of it; expect tens of percent from a PB solver"
                ),
            },
        )
        result.check_satisfies(request)
        return result
