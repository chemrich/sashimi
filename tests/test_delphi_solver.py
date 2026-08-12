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

import sashimi.delphi.backend as backend_module
from sashimi.delphi import DelphiSolver, discover_delphi
from sashimi.delphi.discover import DelphiFlavour, DelphiNotFound
from sashimi.delphi.options import SUPPORTED_SURFACES
from sashimi.errors import SolverError, UnsupportedRequest
from sashimi.pqr import parse_pqr
from sashimi.protocol import (
    EnergyTerm,
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


def test_the_reported_energy_matches_the_declared_term(binary, ion):
    """Whether salt moves the answer is a property of the term being reported.

    The C++ build is asked for the ion-inclusive quantity, so mobile ions must
    change its answer and make it more favourable — screening stabilises the
    charge. pyDelPhi has no ion-atmosphere column and reports the reaction field
    alone, which is salt-independent by construction.

    Asserting each flavour against its own declared `EnergyTerm` is the point:
    `sashimi.validate` trusts that declaration, so a backend whose number does
    not behave like the term it claims would silently corrupt every comparison.
    """
    salted = DelphiSolver().solve(_request(ion, binary, solvent={"ionic_strength": 0.5}))
    plain = DelphiSolver().solve(_request(ion, binary))
    assert salted.energy_kj_mol is not None
    assert plain.energy_kj_mol is not None

    if binary.flavour is DelphiFlavour.CPP:
        assert salted.provenance.energy_term is EnergyTerm.POLAR_SOLVATION
        assert salted.energy_kj_mol < plain.energy_kj_mol
        assert "mobile-ion atmosphere" in salted.diagnostics["energy_term"]
    else:
        assert salted.provenance.energy_term is EnergyTerm.REACTION_FIELD
        assert salted.energy_kj_mol == pytest.approx(plain.energy_kj_mol, rel=1e-6)
        assert "polarization only" in salted.diagnostics["energy_term"]


def test_delphi_reading_a_different_structure_is_caught_rather_than_solved(binary, monkeypatch):
    """Structural verification of the output, not trust in the exit code.

    DelPhi parses PQR by fixed column, so a field one place to the right is not
    an error to it — it is a different number, and it solves happily on it. This
    reproduces the writer that shipped until 2026-08-12, where a four-character
    residue name shifted every column after it: acetate arrived as two charged
    atoms carrying +80.84 e where the file says seven and -1, and the run
    returned -865,205 kJ/mol against APBS's -196.90, with nothing raised.

    The same discipline ROADMAP.md §13 applies to APBS, which also exits 0 on
    failure: check the output against what was asked, rather than the status.

    C++ only, and that is the honest shape of the guard rather than a
    convenience: it reads DelPhi's printed echo of the charges it assigned, and
    pyDelPhi reports through a CSV and prints no equivalent line. Asserting the
    refusal on a flavour that cannot produce it would be asserting something
    else — measured on CI, where pyDelPhi rejects this input at the parameter
    file instead, which is a different failure that happens to look like a pass.
    """
    if binary.flavour is not DelphiFlavour.CPP:
        pytest.skip("the guard reads a line only the C++ build prints")

    def minimum_width_writer(pqr) -> str:
        lines = []
        for i in range(pqr.n_atoms):
            res_name, res_seq, atom_name = pqr.labels[i].split()
            x, y, z = pqr.coords[i]
            lines.append(
                f"ATOM  {i + 1:5d} {atom_name:>4s} {res_name:>3s} {res_seq:>5s}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f} {pqr.charges[i]:7.4f} {pqr.radii[i]:6.4f}"
            )
        return "\n".join([*lines, "TER", "END", ""])

    monkeypatch.setattr(backend_module, "format_pqr", minimum_width_writer)
    acetate = parse_pqr(
        "\n".join(
            f"ATOM  {i:5d} {name:>4s} TARG     1    "
            f"{i:8.3f}{0.0:8.3f}{0.0:8.3f} {charge:7.4f} {1.88:6.4f}"
            for i, (name, charge) in enumerate(
                [("C1", -0.137), ("C2", 0.78), ("O6", -0.91), ("O7", -0.91)], start=1
            )
        )
    )

    with pytest.raises(SolverError, match="read a different structure"):
        DelphiSolver().solve(
            FiniteDifferenceRequest(
                structure=acetate,
                solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
                grid=GridSpec(resolution=0.5, padding=6.0),
                want_energy=True,
                want_potential=False,
            )
        )
