"""M3a — the ion-exclusion region's geometry, and the guard that scopes it.

`screening_nodes` carries the proof that `dilate(SES, r) == dilate(vdw, r)` for
`r >= probe`. These are the invariants that proof buys, asserted on a real
structure rather than trusted, plus the one that says the guard is a guard: below
the probe the two constructions genuinely diverge, and the branch that survives
there is the dilated one.

**Every check here needs the coarsest multigrid level, not just the finest.**
`build_levels` re-discretizes at every level, and the defect this milestone
removed was invisible on a fine grid and total on a coarse one — a 2.0 A ball
dilated on a 6.4 A lattice reaches nothing at all.

Needs no binary.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sashimi.debye.dielectric import screening_nodes
from sashimi.debye.grid import axis_coordinates, grid_hierarchy, size_grid
from sashimi.debye.surface import ReducedSurface, inside_union_of_spheres
from sashimi.pqr import read_pqr
from sashimi.protocol import GridSpec, SolventModel, SurfaceModel

PEPTIDE = read_pqr("tests/data/ala-gly.pqr")
SALTED = SolventModel(ionic_strength=0.15, surface_model=SurfaceModel.MOLECULAR)


def levels(resolution=0.9, padding=10.0):
    return grid_hierarchy(size_grid(PEPTIDE, GridSpec(resolution=resolution, padding=padding)))


@pytest.mark.parametrize("ion_radius", [1.4, 2.0, 3.0])
def test_no_mobile_ions_inside_the_solute_at_any_multigrid_level(ion_radius):
    """C-G4's first half. Bit-exact — an IEEE-754 predicate, not a tolerance.

    At or above the probe the union-inflated region provably contains the
    solvent-excluded one, so no node the surface calls solute may carry
    screening. Checked at every level `build_levels` will produce, because the
    coarsest is where a lattice-quantised construction fails and the finest is
    where it looks fine.
    """
    solvent = dataclasses.replace(SALTED, ion_radius=ion_radius)
    surface = ReducedSurface(PEPTIDE, solvent)
    for level in levels():
        screening, bulk = screening_nodes(level, PEPTIDE, solvent, surface)
        assert bulk > 0.0
        solute = surface.inside(axis_coordinates(level))
        leaked = int(np.count_nonzero(screening[solute]))
        assert leaked == 0, (
            f"{leaked} solute nodes carry mobile-ion screening at spacing "
            f"{level.spacing}; ions are inside the dielectric body"
        )


def test_below_the_probe_the_union_would_leak_which_is_why_the_guard_exists():
    """C-G4's second half, without which the first cannot fail.

    A gate that only ever asserts zero proves nothing unless something can make
    it non-zero. Below the probe the union-inflated region does *not* contain the
    solvent-excluded one, so taking that branch there would put mobile ions
    inside the solute — and the shipped code does not, because it falls back to
    the dilation. Both halves are asserted: the hazard is real, and it is avoided.
    """
    solvent = dataclasses.replace(SALTED, ion_radius=0.0)
    surface = ReducedSurface(PEPTIDE, solvent)
    level = levels()[0]
    axes = axis_coordinates(level)
    solute = surface.inside(axes)

    would_leak = np.logical_and(
        solute,
        ~inside_union_of_spheres(axes, PEPTIDE.coords, PEPTIDE.radii + solvent.ion_radius),
    )
    assert int(np.count_nonzero(would_leak)) > 0, (
        "the union path is supposed to be wrong below the probe; if it is not, "
        "the guard is decoration and the branch should go"
    )

    screening, _ = screening_nodes(level, PEPTIDE, solvent, surface)
    assert int(np.count_nonzero(screening[solute])) == 0, (
        "the shipped path leaked below the probe — the guard is not routing"
    )


def test_the_exclusion_region_stops_moving_with_the_lattice_above_the_guard():
    """C-G3, on the observable that is actually monotone.

    The dilated construction quantises its reach to the lattice, so its exclusion
    volume was h-dependent: 82.2% of exact at 0.80 A, 97.6% at 0.19 A, and the
    residual is not even monotone in the *energy*, because it tracks whether
    `ion_radius` is commensurate with the spacing. The union test has no such
    term. Graded on the volume, which is monotone, rather than on an energy that
    is not.
    """
    solvent = dataclasses.replace(SALTED, ion_radius=2.0)
    surface = ReducedSurface(PEPTIDE, solvent)
    volumes = []
    for resolution in (0.9, 0.6, 0.4):
        grid = size_grid(PEPTIDE, GridSpec(resolution=resolution, padding=10.0))
        screening, _ = screening_nodes(grid, PEPTIDE, solvent, surface)
        cell = float(np.prod(grid.spacing))
        volumes.append(float(np.count_nonzero(screening == 0.0)) * cell)

    spread = max(volumes) / min(volumes)
    assert spread <= 1.02, (
        f"the ion-exclusion volume moved {spread:.4f}x across a 2.25x refinement "
        f"({volumes}); above the guard it is an exact sphere test and must not"
    )


def test_the_two_surface_models_build_the_same_exclusion_region_above_the_guard():
    """C-G1. The theorem's whole content, on a real molecule.

    `van-der-waals` and `molecular` disagree about the *dielectric* boundary --
    the probe rolls, and ala-gly's crevices are real. They must still agree about
    where ions cannot go, because above the probe both dilate to the same region.
    Bit-identical, since above the guard both take the same exact test.
    """
    grid = size_grid(PEPTIDE, GridSpec(resolution=0.6, padding=10.0))
    molecular = dataclasses.replace(SALTED, ion_radius=2.0)
    vdw = dataclasses.replace(molecular, surface_model=SurfaceModel.VAN_DER_WAALS)

    from_molecular, _ = screening_nodes(grid, PEPTIDE, molecular)
    from_vdw, _ = screening_nodes(grid, PEPTIDE, vdw)
    assert np.array_equal(from_molecular, from_vdw)

    # And the dielectric boundaries really do differ, or the check above is
    # comparing a surface model with itself.
    axes = axis_coordinates(grid)
    solute_molecular = ReducedSurface(PEPTIDE, molecular).inside(axes)
    solute_vdw = ReducedSurface(PEPTIDE, vdw).inside(axes)
    assert not np.array_equal(solute_molecular, solute_vdw)


def test_a_negative_ion_radius_is_refused():
    """It silently produced an empty exclusion region until 2026-08-28."""
    with pytest.raises(ValueError, match="ion_radius"):
        SolventModel(ion_radius=-1.0)
