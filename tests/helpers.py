"""Shared narrowing helpers.

`SolveResult.potential` is a `PotentialGrid | SurfacePotential | None` union as
of phase 4, which is the point: a caller must say which shape it expects. In
tests that expect APBS, saying so once is better than an `isinstance` at every
attribute access.
"""

from sashimi.protocol import PotentialGrid, SolveResult, SurfacePotential

__all__ = ["surface", "volume"]


def volume(result: SolveResult) -> PotentialGrid:
    """The result's volumetric potential. Fails loudly if it is not one."""
    grid = result.potential
    assert isinstance(grid, PotentialGrid), (
        f"expected a volumetric potential, got {type(grid).__name__}"
    )
    return grid


def surface(result: SolveResult) -> SurfacePotential:
    """The result's surface potential. Fails loudly if it is not one."""
    mesh = result.potential
    assert isinstance(mesh, SurfacePotential), (
        f"expected a surface potential, got {type(mesh).__name__}"
    )
    return mesh
