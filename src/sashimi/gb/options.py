"""Generalized Born knobs, and the mapping from solver-neutral concepts onto them.

The parameter sets are Amber's, named as Amber names them, because that is where
their published validation lives. Inventing our own constants would make every
comparison against the literature meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sashimi.errors import UnsupportedRequest
from sashimi.protocol import SurfaceModel

__all__ = [
    "DEFAULT_MINIMUM_RADIUS",
    "DEFAULT_OFFSET",
    "MBONDI_RADII",
    "SUPPORTED_SURFACES",
    "GbModel",
    "GbOptions",
    "GbRadii",
    "check_surface",
]


class GbModel(StrEnum):
    """Which effective-radius formula to use.

    All three share the same pairwise descreening integral and differ only in
    how it becomes a radius. OBC2 is Amber's `igb=5` and the default here: it is
    the most accurate of the three against Poisson-Boltzmann for proteins, which
    is the comparison this backend exists to be measured by.
    """

    HCT = "hct"  # Amber igb=1: the raw integral, no rescaling
    OBC1 = "obc1"  # Amber igb=2
    OBC2 = "obc2"  # Amber igb=5


# alpha, beta, gamma for the OBC tanh rescaling. HCT applies none.
OBC_PARAMETERS: dict[GbModel, tuple[float, float, float]] = {
    GbModel.HCT: (0.0, 0.0, 0.0),
    GbModel.OBC1: (0.8, 0.0, 2.909125),
    GbModel.OBC2: (1.0, 0.8, 4.85),
}

# **The molecular surface, though the integral is over van der Waals spheres.**
# The construction is not the surface, and reading it as one is a mistake this
# project has already made once, with pyDelPhi's `surfmethod=vdw` (section 12).
# Descreening integrates over the union of vdW spheres, so `van-der-waals` is
# the intuitive declaration — and it is wrong. The OBC tanh rescaling exists
# precisely to correct the union of spheres toward the solvent-excluded volume,
# because the union has interstitial holes the real boundary does not, and the
# alpha/beta/gamma parameters were fit so that the result reproduces
# Poisson-Boltzmann *on the molecular surface*.
#
# Measured on hen lysozyme, 1,960 atoms, mbondi radii, against APBS:
#
#     molecular       -3879.1 GB vs -4071.1 APBS    4.72%
#     van-der-waals   -3879.1 GB vs -5650.3 APBS   31.35%
#
# Declaring the surface the method's *inputs* look like rather than the one its
# *answer* corresponds to would have reported a 31% modelling mismatch as the
# approximation's error.
SUPPORTED_SURFACES: frozenset[SurfaceModel] = frozenset({SurfaceModel.MOLECULAR})

# Amber's intrinsic-radius offset. Subtracted from every radius before the
# descreening integral and added back into the vdW radius that scales the tanh.
DEFAULT_OFFSET = 0.09  # angstroms

# **pdb2pqr's radii are not Generalized Born radii**, and this is not a detail.
# They are Lennard-Jones parameters: AMBER gives hydroxyl and sulfhydryl
# hydrogens a radius of exactly 0, because their volume is subsumed into the
# heavy atom they hang off, and nonpolar hydrogens 0.6 A. A grid solver spreads
# their charge over grid points and never notices. Generalized Born divides by
# the radius, so an atom carrying +0.42 e on a zero radius contributes an
# infinite self-energy — hen lysozyme has twenty of them, holding +8.34 e.
#
# Amber ships the mbondi sets for exactly this reason, and the difference is not
# cosmetic. Measured on hen lysozyme against APBS on the molecular surface:
#
#     mbondi radii            -3879.1 vs -4071.1     4.72%
#     pdb2pqr AMBER radii     -2567.7 vs -3976.6    35.43%
#
# So mbondi is the default: shipping the alternative means shipping a backend
# that is wrong by a third out of the box. It is a substitution of the caller's
# input, so it is recorded in provenance and counted in diagnostics, and
# `GbRadii.AS_GIVEN` turns it off for anyone who wants the strictly identical
# solute the other backends were handed.
DEFAULT_MINIMUM_RADIUS = 0.8  # angstroms; mbondi's H-bonded-to-O/S value


class GbRadii(StrEnum):
    """Which radii to descreen over."""

    MBONDI = "mbondi"  # Amber's Generalized Born set, what the parameters were fit with
    AS_GIVEN = "as-given"  # the structure's own radii, floored at `minimum_radius`


# mbondi, by element. Hydrogen bonded to nitrogen should be 1.3 rather than 1.2,
# and is not distinguished here: PQR carries no bonding, and inferring it from
# atom names is the guess this file declines to make elsewhere. The 4.72%
# measured above is with every hydrogen at 1.2.
MBONDI_RADII: dict[str, float] = {
    "H": 1.2,
    "C": 1.7,
    "N": 1.55,
    "O": 1.5,
    "F": 1.5,
    "S": 1.8,
    "P": 1.85,
}

# Amber's element-dependent screening factors (mbondi/bondi sets). Only the six
# elements that make up proteins are listed, and anything else takes 1.0 —
# see `screening_factors` for why nothing more ambitious is attempted.
ELEMENT_SCREEN: dict[str, float] = {
    "H": 0.85,
    "C": 0.72,
    "N": 0.79,
    "O": 0.85,
    "P": 0.86,
    "S": 0.96,
}


@dataclass(frozen=True)
class GbOptions:
    """Backend-specific overrides. Defaults reproduce Amber's igb=5.

    `offset` is the one worth understanding. Amber shrinks every intrinsic
    radius by 0.09 A before descreening, which is a fitted correction rather
    than a physical length, and it means an isolated atom's effective radius is
    *not* its van der Waals radius. Setting it to zero makes a single sphere
    reduce to the Born formula exactly, which is how this backend is calibrated
    (`tests/test_gb.py`), and costs a few percent of accuracy on real solutes.

    `radii` is the one that decides whether the answer is worth having, and
    `minimum_radius` is the floor that applies when it is `AS_GIVEN`. See
    `DEFAULT_MINIMUM_RADIUS` for both, and for the measurements.
    """

    model: GbModel = GbModel.OBC2
    radii: GbRadii = GbRadii.MBONDI
    offset: float = DEFAULT_OFFSET
    minimum_radius: float = DEFAULT_MINIMUM_RADIUS
    # Uniform screening factor for atoms whose element cannot be read off the
    # PQR label. 1.0 is the no-scaling limit.
    default_screen: float = 1.0
    use_element_screening: bool = True
    # Rows of the pairwise matrix computed at once. Bounds peak memory at
    # roughly chunk * n_atoms * 8 bytes; see `sashimi.gb.radii`.
    chunk_size: int = 512

    @property
    def obc_parameters(self) -> tuple[float, float, float]:
        return OBC_PARAMETERS[self.model]


def check_surface(model: SurfaceModel) -> None:
    """Refuse a surface this method does not construct, before anything runs."""
    if model in SUPPORTED_SURFACES:
        return
    raise UnsupportedRequest(
        f"Generalized Born has no equivalent of the {model.value!r} surface model. "
        "Its parameters were fit to reproduce Poisson-Boltzmann on the molecular "
        "surface and it reproduces nothing else: on a van der Waals boundary the "
        "same answer is 31% out on hen lysozyme, and a Gaussian or spline-smoothed "
        "dielectric is a grid construction it has no way to represent. Request "
        "'molecular', which every backend here supports."
    )
