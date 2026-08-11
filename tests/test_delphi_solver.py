"""DelPhi behaviour that needs a real executable.

Gated behind `@pytest.mark.delphi` and skipped where none is installed, which
is most places: neither flavour has a package, so this tier runs where someone
has built the C++ program or pip-installed pyDelPhi. Which flavour is present
changes what can be asserted — they do not support the same surface models —
so the tests ask the binary what it is rather than assuming.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from sashimi.delphi import DelphiSolver, discover_delphi
from sashimi.delphi.discover import DelphiFlavour, DelphiNotFound
from sashimi.delphi.options import SUPPORTED_SURFACES
from sashimi.errors import UnsupportedRequest
from sashimi.protocol import (
    Equation,
    FiniteDifferenceRequest,
    GridSpec,
    PQRData,
    SolventModel,
    SurfaceModel,
)
from tests.born_reference import born_solvation_energy
from tests.helpers import volume

pytestmark = pytest.mark.delphi


@pytest.fixture(scope="module")
def binary():
    """The installed DelPhi, or a skip — but never a skip that hides a mistake.

    Absent is the normal case and skipping is right. Being *pointed at* a DelPhi
    that then fails to run is a broken installation, and skipping there would
    report the same green result as a working one.
    """
    try:
        return discover_delphi()
    except DelphiNotFound as exc:
        if os.environ.get("SASHIMI_DELPHI_PATH"):
            raise
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def ion():
    return PQRData(
        coords=np.zeros((1, 3)),
        charges=np.array([1.0]),
        radii=np.array([3.0]),
    )


def _supported_surface(binary) -> SurfaceModel:
    """A surface model this flavour can actually run."""
    supported = SUPPORTED_SURFACES[binary.flavour]
    for candidate in (SurfaceModel.MOLECULAR, SurfaceModel.VAN_DER_WAALS, SurfaceModel.GAUSSIAN):
        if candidate in supported:
            return candidate
    pytest.skip(f"{binary.flavour.value} supports no surface model sashimi can request")


def _request(structure, binary, **kwargs) -> FiniteDifferenceRequest:
    solvent = SolventModel(
        solute_dielectric=1.0,
        ionic_strength=0.0,
        surface_model=_supported_surface(binary),
    )
    return FiniteDifferenceRequest(
        structure=structure,
        solvent=dataclasses.replace(solvent, **kwargs.pop("solvent", {})),
        grid=GridSpec(resolution=0.5, padding=10.0),
        **kwargs,
    )


def test_discovery_reports_a_flavour_and_version(binary):
    assert binary.flavour in (DelphiFlavour.CPP, DelphiFlavour.PYDELPHI)
    assert binary.version[0].isdigit()
    assert binary.label.startswith(binary.flavour.value)
    assert len(binary.sha256) == 64


def test_born_ion_matches_the_closed_form(binary, ion):
    """The analytic calibration APBS is held to, applied to the other backend.

    Runs for both flavours: `_supported_surface` picks the molecular surface,
    which both express, and which has a sharp dielectric boundary the closed
    form applies to. Only a Gaussian dielectric would have to be skipped here,
    and no flavour falls back to one.
    """
    result = DelphiSolver().solve(_request(ion, binary))
    expected = born_solvation_energy(3.0, solute_dielectric=1.0)

    assert result.energy_kj_mol == pytest.approx(expected, rel=0.01)


def test_potential_grid_is_returned_in_angstroms(binary, ion):
    result = DelphiSolver().solve(_request(ion, binary))
    grid = volume(result)

    assert grid.values.ndim == 3
    assert grid.shape[0] == grid.shape[1] == grid.shape[2]  # DelPhi's box is cubic
    # A 3 A ion with 10 A padding is a ~26 A box; in Bohr this would read ~49.
    assert 20 < float(np.ptp(grid.origin) + grid.spacing[0] * (grid.shape[0] - 1)) < 40


def test_provenance_records_the_flavour_and_the_mapped_parameters(binary, ion):
    result = DelphiSolver().solve(_request(ion, binary))
    resolved = result.provenance.resolved_parameters

    assert result.provenance.backend == binary.label
    assert result.provenance.binary_sha256 == binary.sha256
    assert resolved["delphi"]["flavour"] == binary.flavour.value
    # The units the temperature was written in, not just its value: the same
    # number means two different temperatures across the flavours.
    assert resolved["delphi"]["temper_units"] in ("celsius", "kelvin")


def test_energy_only_request_skips_the_map(binary, ion):
    result = DelphiSolver().solve(_request(ion, binary, want_potential=False))

    assert result.energy_kj_mol is not None
    assert result.potential is None


def test_unsupported_surface_is_refused_before_running(binary, ion):
    """The default surface model is APBS-only, and no flavour may fake it."""
    request = FiniteDifferenceRequest(
        structure=ion,
        solvent=SolventModel(surface_model=SurfaceModel.SMOOTHED_MOLECULAR),
        grid=GridSpec(resolution=0.5, padding=10.0),
    )
    with pytest.raises(UnsupportedRequest, match="smoothed-molecular"):
        DelphiSolver().solve(request)


def test_nonlinear_is_refused(binary, ion):
    request = dataclasses.replace(_request(ion, binary), equation=Equation.NONLINEAR)
    with pytest.raises(UnsupportedRequest):
        DelphiSolver().solve(request)


def test_the_reported_energy_is_the_reaction_field_term_only(binary, ion):
    """DelPhi's energy does not move with salt, and that is not a bug.

    The natural assertion is that mobile ions change the answer, which is what
    APBS's difference-of-blocks shows. DelPhi reports the polarization term
    alone: measured here, -92.22 kT from the C++ build and -228.611 kJ/mol from
    pyDelPhi at both 0 M and 0.5 M. The salt does reach the solver — the C++
    build reports a Debye length of 4.307 A at 0.5 M — so this pins a
    definitional difference between the backends, not a parameter that failed to
    arrive.

    Both flavours, deliberately. An earlier version asserted they differed here,
    on a measurement taken from pyDelPhi's Gaussian path against the C++ build's
    molecular one; on the same surface model they agree exactly.
    """
    salted = DelphiSolver().solve(_request(ion, binary, solvent={"ionic_strength": 0.5}))
    plain = DelphiSolver().solve(_request(ion, binary))

    assert salted.energy_kj_mol == pytest.approx(plain.energy_kj_mol, rel=1e-6)
    assert "polarization only" in salted.diagnostics["energy_term"]
