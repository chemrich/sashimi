"""The side of the interface a normal derivative belongs to, and its guards.

`dphi/dn` is discontinuous across the dielectric boundary: `eps_p (dphi/dn)_in =
eps_s (dphi/dn)_out`, so interior and exterior differ by `eps_s/eps_p` — 39.27 at
the protocol's defaults — and nothing in the array says which one it holds. This
file is the binary-free half of that guard; `tests/test_tabipb_normal_derivative.py`
carries the half that needs a real solve.

**Why a conversion helper rather than a ratio at each call site.** A hardcoded
1/39.27 is right at the defaults and wrong everywhere else, and a test that only
ever runs at the defaults cannot see the difference. Every check below that
varies a dielectric exists to catch that specific shape of error.
"""

from __future__ import annotations

import numpy as np
import pytest

from sashimi.protocol import SolventModel, SurfacePotential


def surface_with(derivative: np.ndarray | None) -> SurfacePotential:
    m = 4 if derivative is None else len(derivative)
    return SurfacePotential(
        vertices=np.zeros((m, 3)),
        values=np.linspace(1.0, 2.0, m),
        interior_normal_derivative=derivative,
    )


def test_the_field_is_optional_and_absent_by_default():
    """A finite-difference backend has no normal derivative and never will."""
    assert surface_with(None).interior_normal_derivative is None
    assert surface_with(None).exterior_normal_derivative(SolventModel()) is None


def test_a_mismatched_length_is_refused_and_the_message_names_the_field():
    """Exact integer comparison, and the field named — a truncated VTK block
    would otherwise propagate silently into `stats()` and the corpus summary."""
    with pytest.raises(ValueError, match="interior_normal_derivative"):
        SurfacePotential(
            vertices=np.zeros((3, 3)),
            values=np.ones(3),
            interior_normal_derivative=np.ones(2),
        )


def test_the_exterior_side_is_the_interior_one_scaled_by_the_dielectric_ratio():
    interior = np.array([-77.0, -38.0, -154.0, -12.0])
    solvent = SolventModel()  # eps_p 2.0, eps_s 78.54
    exterior = surface_with(interior).exterior_normal_derivative(solvent)
    assert exterior is not None
    np.testing.assert_allclose(exterior, interior * (2.0 / 78.54), rtol=0, atol=0)
    # The factor at the defaults, stated so a reader can check it by eye.
    np.testing.assert_allclose(interior / exterior, 39.27, rtol=1e-12)


@pytest.mark.parametrize("solute_dielectric", [1.0, 2.0, 4.0, 20.0])
def test_the_ratio_is_read_from_the_request_and_not_hardcoded(solute_dielectric):
    """The gate a default-only test cannot be.

    A hardcoded 1/39.27 passes at `eps_p = 2, eps_s = 78.54` and fails at every
    other pair. Varying `eps_p` over a factor of twenty is what separates reading
    the request from remembering the default.
    """
    interior = np.array([-100.0, -50.0])
    solvent = SolventModel(solute_dielectric=solute_dielectric, solvent_dielectric=78.54)
    exterior = surface_with(interior).exterior_normal_derivative(solvent)
    assert exterior is not None
    np.testing.assert_allclose(exterior, interior * (solute_dielectric / 78.54), rtol=1e-15)


def test_the_exterior_side_moves_with_the_solvent_dielectric_too():
    """The mirror of the check above: `eps_s` is the other half of the ratio, and
    a conversion that reads only `eps_p` would pass every test above this one."""
    interior = np.array([-100.0])
    lo = surface_with(interior).exterior_normal_derivative(SolventModel(solvent_dielectric=40.0))
    hi = surface_with(interior).exterior_normal_derivative(SolventModel(solvent_dielectric=78.54))
    assert lo is not None and hi is not None
    np.testing.assert_allclose(lo / hi, 78.54 / 40.0, rtol=1e-12)


def test_stats_carries_the_derivative_only_when_there_is_one():
    """Unconditional keys would put a null in front of every caller reading a
    finite-difference result. Both consumers select by name, so this changes no
    recording either way — it is about what a caller sees."""
    keys = set(surface_with(None).stats())
    assert not any("normal_derivative" in k for k in keys)

    with_derivative = surface_with(np.array([-1.0, -2.0, -3.0, -4.0])).stats()
    assert with_derivative["interior_normal_derivative_mean"] == pytest.approx(-2.5)
    assert set(surface_with(None).stats()) < set(with_derivative)
