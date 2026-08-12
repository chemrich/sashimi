"""Shared narrowing helpers.

`SolveResult.potential` is a `PotentialGrid | SurfacePotential | None` union as
of phase 4, which is the point: a caller must say which shape it expects. In
tests that expect APBS, saying so once is better than an `isinstance` at every
attribute access.
"""

import os
from collections.abc import Callable

import pytest

from sashimi.errors import BackendUnavailable
from sashimi.protocol import PotentialGrid, SolveResult, SurfacePotential

__all__ = ["installed_or_skip", "surface", "volume"]


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


def installed_or_skip[T](discover: Callable[[], T], env_var: str) -> T:
    """A discovered binary, or a skip — but never a skip that hides a mistake.

    Absent is the normal case and skipping is right. Being *pointed at* a binary
    that then fails to discover is a broken installation, and skipping there
    would report the same green result as a working one.

    **A marker is not a guard.** `@pytest.mark.tabipb` selects and deselects; it
    does not skip. A test that needs a binary must ask for it through here, or it
    runs everywhere and fails wherever the binary is absent — which is exactly
    how this helper came to exist: a marked MCP test passed locally, passed on
    the Linux leg that builds TABI-PB, and failed on macOS, which does not.
    """
    try:
        return discover()
    except BackendUnavailable as exc:
        if os.environ.get(env_var):
            raise
        pytest.skip(str(exc))
