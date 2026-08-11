"""Effective Born radii by pairwise descreening.

The one quantity Generalized Born is really about. An atom's effective radius is
its distance to the dielectric boundary *as seen through the rest of the
solute*: an atom on the surface keeps nearly its van der Waals radius, one
buried in the core acquires a large one and is barely solvated. Everything else
in this backend is Coulomb's law.

The integral below is the standard Hawkins-Cramer-Truhlar approximation — the
solute volume treated as a union of spheres, and each sphere's contribution to
the r^-4 integral outside atom i evaluated in closed form. It is derived rather
than quoted, and `tests/test_gb.py` checks it against direct numerical
quadrature of the same integral, because a misremembered sign here is exactly
the kind of error that produces plausible numbers.
"""

from __future__ import annotations

import numpy as np

from sashimi.gb.options import ELEMENT_SCREEN, MBONDI_RADII, GbOptions, GbRadii
from sashimi.protocol import FloatArray, PQRData

__all__ = [
    "descreening_integral",
    "effective_radii",
    "elements",
    "input_radii",
    "screening_factors",
]


def elements(structure: PQRData) -> list[str]:
    """One element symbol per atom, or "" where it cannot be read.

    PQR carries no element column, so this is the first alphabetic character of
    the atom name — the PDB convention, and reliable for the elements proteins
    are made of. It is *not* reliable for two-letter elements: `CL` and `CA` are
    chlorine and calcium as ions but chlorine-substituted carbon and the alpha
    carbon in a residue, and nothing in the name disambiguates them. Callers
    treat "" as "leave this atom alone" rather than guessing.
    """
    found = []
    for i in range(structure.n_atoms):
        label = structure.labels[i] if i < len(structure.labels) else ""
        atom_name = label.rsplit(" ", 1)[-1] if label else ""
        found.append(next((c for c in atom_name.upper() if c.isalpha()), ""))
    return found


def screening_factors(structure: PQRData, options: GbOptions) -> FloatArray:
    """Per-atom screening factors, from the element when it can be read.

    Anything not recognised takes `default_screen`, the no-scaling limit, which
    errs toward a smaller effective radius rather than a wrong one.
    """
    factors = np.full(structure.n_atoms, options.default_screen, dtype=np.float64)
    if not options.use_element_screening:
        return factors
    for i, element in enumerate(elements(structure)):
        if element in ELEMENT_SCREEN:
            factors[i] = ELEMENT_SCREEN[element]
    return factors


def descreening_integral(
    coords: FloatArray, intrinsic: FloatArray, scaled: FloatArray, chunk_size: int = 512
) -> FloatArray:
    """The HCT integral I_i for every atom, in inverse angstroms.

    `intrinsic` is rho_i, the offset-corrected radius of the atom being
    descreened; `scaled` is s_j * rho_j, the screening radius of the atoms doing
    the descreening. Both are needed because the integral is not symmetric.

    Computed `chunk_size` rows at a time. The full pairwise matrix is N^2 and
    hen lysozyme is 1,960 atoms — 30 MB, harmless — but a 20,000-atom complex
    would be 3.2 GB, and an in-process backend that exhausts memory takes the
    caller down with it. There is no subprocess boundary here to absorb that.
    """
    n = len(coords)
    totals = np.zeros(n, dtype=np.float64)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        rho_i = intrinsic[start:stop, None]  # (chunk, 1)
        a = scaled[None, :]  # (1, n), the screening spheres

        deltas = coords[start:stop, None, :] - coords[None, :, :]
        r = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))

        # An atom does not descreen itself, and two atoms at identical
        # coordinates have no well-defined separation to integrate over.
        rows = np.arange(start, stop)
        self_pair = np.zeros_like(r, dtype=bool)
        self_pair[np.arange(stop - start), rows] = True
        usable = (r > 0) & ~self_pair

        upper = r + a
        # Everything closer than rho_i is inside the atom being descreened and
        # contributes nothing: the integral runs over the solvent-excluded
        # volume *outside* it.
        contributes = usable & (upper > rho_i)

        safe_r = np.where(contributes, r, 1.0)
        safe_upper = np.where(contributes, upper, 2.0)
        lower = np.maximum(rho_i, np.abs(safe_r - a))

        integral = 0.5 * (
            1.0 / lower
            - 1.0 / safe_upper
            + np.log(lower / safe_upper) / (2.0 * safe_r)
            + ((safe_r**2 - a**2) / (4.0 * safe_r)) * (1.0 / safe_upper**2 - 1.0 / lower**2)
        )

        # Atom i sitting inside sphere j: the shell between rho_i and (a - r) is
        # wholly enclosed, so its whole surface counts rather than a cap of it.
        enclosed = contributes & (a - safe_r > rho_i)
        shell_outer = np.where(enclosed, a - safe_r, 1.0)
        integral = integral + np.where(enclosed, 1.0 / rho_i - 1.0 / shell_outer, 0.0)

        totals[start:stop] = np.sum(np.where(contributes, integral, 0.0), axis=1)

    return totals


def input_radii(structure: PQRData, options: GbOptions) -> tuple[FloatArray, int]:
    """The radii this backend will actually descreen over, and how many it changed.

    The count is returned rather than logged because it belongs in the result: a
    caller comparing this energy against a Poisson-Boltzmann one is entitled to
    know the two solvers were not handed identical radii, and under the default
    `GbRadii.MBONDI` most of them were not. See `DEFAULT_MINIMUM_RADIUS` for why
    the structure's own radii are the wrong ones for this method.
    """
    if options.radii is GbRadii.MBONDI:
        radii = np.array(
            [
                MBONDI_RADII.get(element, given)
                for element, given in zip(elements(structure), structure.radii, strict=True)
            ],
            dtype=np.float64,
        )
    else:
        radii = structure.radii
    # The floor is redundant under mbondi, which assigns nothing below 1.2, and
    # load-bearing under AS_GIVEN, where zero-radius hydrogens arrive.
    radii = np.maximum(radii, options.minimum_radius)
    return radii, int(np.count_nonzero(radii != structure.radii))


def effective_radii(structure: PQRData, options: GbOptions) -> FloatArray:
    """Effective Born radii in angstroms, one per atom.

    An isolated atom gets `radius - offset` exactly: with nothing to descreen it
    the integral is zero, the tanh rescaling vanishes with it, and what remains
    is the intrinsic radius. That is why `offset=0` makes a lone sphere reduce to
    Born exactly, and why the default does not.
    """
    resolved, _ = input_radii(structure, options)
    intrinsic = resolved - options.offset
    if np.any(intrinsic <= 0):
        raise ValueError(
            f"offset {options.offset} A is not smaller than the minimum radius "
            f"{options.minimum_radius} A, leaving an atom with a non-positive "
            "intrinsic radius"
        )

    scaled = screening_factors(structure, options) * intrinsic
    integral = descreening_integral(structure.coords, intrinsic, scaled, options.chunk_size)

    alpha, beta, gamma = options.obc_parameters
    if alpha == 0.0:
        # HCT: the integral is the correction, unrescaled.
        inverse = 1.0 / intrinsic - integral
    else:
        psi = integral * intrinsic
        vdw = intrinsic + options.offset
        inverse = 1.0 / intrinsic - np.tanh(alpha * psi - beta * psi**2 + gamma * psi**3) / vdw

    # A deeply buried atom can drive the inverse radius to zero or below, which
    # is the approximation running out rather than a physical radius. Amber caps
    # it at 30 A; the alternative is a negative radius and an infinite energy.
    return np.where(inverse > 1.0 / 30.0, 1.0 / np.maximum(inverse, 1e-12), 30.0)
