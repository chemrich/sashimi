"""DelPhi backend, binary-free tier.

Everything here is pure arithmetic and text: grid sizing, parameter-file
generation, cube parsing, unit conversion and the surface-model mapping. It
runs on any machine, including the platforms where no DelPhi exists — which is
most of them, since neither flavour has a package.

The tests that matter most are the ones pinning conventions that fail
*silently*: the PQR column layout, the Celsius temperature statement, and the
convergence parameters without which DelPhi never terminates. Each of those was
found by running the real program, and each would otherwise produce a confident
wrong number.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

from sashimi.delphi.cube import parse_cube
from sashimi.delphi.discover import DelphiFlavour
from sashimi.delphi.grid import MIN_GSIZE, odd_gsize, size_grid
from sashimi.delphi.input import ABSOLUTE_ZERO_C, build_input, resolved_parameters
from sashimi.delphi.options import (
    SUPPORTED_SURFACES,
    DelphiOptions,
    check_equation,
    resolve_surface,
)
from sashimi.delphi.run import parse_cpp_energy, parse_csv_energy
from sashimi.delphi.units import BOHR_TO_ANGSTROM, kt_to_kj_per_mol
from sashimi.errors import GridTooLarge, MalformedStructure, UnsupportedRequest
from sashimi.pqr import format_pqr, parse_pqr, read_pqr
from sashimi.protocol import Equation, GridSpec, SolventModel, SurfaceModel

BORN_ION_PQR = "ATOM      1  I   ION     1       0.000   0.000  0.000  1.00  3.00\n"

FLAVOURS = (DelphiFlavour.CPP, DelphiFlavour.PYDELPHI)


@pytest.fixture
def born():
    return parse_pqr(BORN_ION_PQR)


@pytest.fixture
def peptide():
    return read_pqr(Path(__file__).resolve().parent / "data" / "ala-gly.pqr")


# --- the PQR column trap -----------------------------------------------------
#
# DelPhi has two PQR readers with different column arithmetic, and picking the
# wrong one loses the radius rather than raising. These tests encode the layout
# so a future change to `format_pqr` cannot break the DelPhi backend silently.


def _atof(text: str) -> float:
    """C's `atof`, which is what DelPhi calls on each field.

    The distinction matters and is the whole failure mode: Python's `float()`
    raises on "4 1.824", so a Python reimplementation of DelPhi's parser would
    have *caught* this. `atof` takes the longest leading numeric prefix and
    returns 0.0 when there is none, so DelPhi reads 4.0 and carries on.
    """
    match = re.match(r"\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else 0.0


def _delphi_pqr4_fields(line: str) -> tuple[float, float]:
    """Charge and radius as DelPhi's `pqr4` reader takes them.

    From `io_pdb.cpp::readPqr4File`: charge is `substr(54, 8)`, radius is
    `substr(62, 7)`.
    """
    return float(line[54:62]), float(line[62:69])


def _delphi_pqr_fields(line: str) -> tuple[float, float]:
    """The same, as DelPhi's `pqr` reader takes them: `substr(54, 7)` and
    `substr(61, 7)`. This is the reader sashimi must *not* ask for."""
    return _atof(line[54:61]), _atof(line[61:68])


def test_sashimi_pqr_is_read_correctly_by_delphis_pqr4_reader(peptide):
    """`in(modpdb4, format=pqr)` must recover the charges and radii exactly."""
    lines = [ln for ln in format_pqr(peptide).splitlines() if ln.startswith("ATOM")]
    assert len(lines) == peptide.n_atoms

    for i, line in enumerate(lines):
        charge, radius = _delphi_pqr4_fields(line)
        assert charge == pytest.approx(peptide.charges[i], abs=5e-5)
        assert radius == pytest.approx(peptide.radii[i], abs=5e-5)


def test_delphis_plain_pqr_reader_would_misread_the_same_file(peptide):
    """Why `input.py` says `modpdb4` and never `pdb`.

    The `pqr` reader's radius field is one column to the left, so it picks up
    the last digit of the charge and stops at the space — an ALA-GLY radius of
    1.824 A parses as 4.0. The solve then proceeds on wrong-sized atoms with no
    error anywhere. This test asserts the misreading is real, so the reason the
    backend pins `modpdb4` survives contact with a future refactor.
    """
    line = next(ln for ln in format_pqr(peptide).splitlines() if ln.startswith("ATOM"))
    good_charge, good_radius = _delphi_pqr4_fields(line)
    _, bad_radius = _delphi_pqr_fields(line)

    assert good_radius == pytest.approx(peptide.radii[0], abs=5e-5)
    assert good_charge == pytest.approx(peptide.charges[0], abs=5e-5)
    # The measured misreading: 1.824 A becomes 4.0 A, silently.
    assert bad_radius == pytest.approx(4.0)


def test_input_uses_the_four_digit_pqr_statement(born):
    text = build_input(
        size_grid(born, GridSpec()),
        SolventModel(surface_model=SurfaceModel.MOLECULAR),
        flavour=DelphiFlavour.CPP,
    )
    assert "in(modpdb4," in text
    assert "format=pqr)" in text


# --- temperature units -------------------------------------------------------


@pytest.mark.parametrize(
    ("flavour", "expected"),
    [(DelphiFlavour.CPP, 298.15 - ABSOLUTE_ZERO_C), (DelphiFlavour.PYDELPHI, 298.15)],
)
def test_temperature_is_celsius_for_cpp_and_kelvin_for_pydelphi(born, flavour, expected):
    """The C++ parser ends `fTemper -= dAbsoluteZero`; pyDelPhi's is kelvin.

    Writing 298.15 to the C++ build runs the solve at 571.3 K and reports
    -48.13 kT where the right answer is -92.22. Both programs accept it.
    """
    model = SurfaceModel.GAUSSIAN if flavour is DelphiFlavour.PYDELPHI else SurfaceModel.MOLECULAR
    text = build_input(
        size_grid(born, GridSpec()),
        SolventModel(temperature=298.15, surface_model=model),
        flavour=flavour,
    )
    written = float(next(ln for ln in text.splitlines() if ln.startswith("temper")).split("=")[1])
    assert written == pytest.approx(expected)


def test_resolved_parameters_record_the_temperature_units(born):
    resolved = resolved_parameters(
        size_grid(born, GridSpec()),
        SolventModel(surface_model=SurfaceModel.MOLECULAR),
        DelphiOptions(),
        flavour=DelphiFlavour.CPP,
        equation=Equation.LINEAR,
    )
    assert resolved["delphi"]["temper_units"] == "celsius"
    assert resolved["temperature"] == 298.15  # the physical value, unconverted
    assert resolved["delphi"]["temper_written"] == pytest.approx(25.0)


# --- convergence -------------------------------------------------------------


def test_convergence_parameters_are_always_written(born):
    """DelPhi defaults to linit=0 and maxc=0.0 and then never terminates."""
    for flavour in FLAVOURS:
        model = (
            SurfaceModel.GAUSSIAN if flavour is DelphiFlavour.PYDELPHI else SurfaceModel.MOLECULAR
        )
        text = build_input(
            size_grid(born, GridSpec()), SolventModel(surface_model=model), flavour=flavour
        )
        assert "linit" in text
        assert "maxc" in text


def test_zero_max_delta_phi_is_rejected():
    """A threshold of zero is what makes DelPhi iterate forever."""
    with pytest.raises(ValueError, match="iterates forever"):
        DelphiOptions(max_delta_phi=0.0)


# --- grid --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("wanted", "expected"), [(3.2, 5), (5.0, 5), (6.0, 7), (65.0, 65), (65.1, 67)]
)
def test_gsize_is_odd_and_at_least_the_minimum(wanted, expected):
    assert odd_gsize(wanted) == expected


def test_grid_box_covers_the_solute_plus_padding(peptide):
    spec = GridSpec(resolution=0.5, padding=10.0)
    grid = size_grid(peptide, spec)

    assert grid.gsize % 2 == 1
    assert grid.box_length >= float(np.max(peptide.extent())) + 2 * spec.padding - 1e-9
    # Spacing is achieved, not requested: the box is quantised by an odd gsize.
    assert max(grid.spacing) <= spec.resolution + 1e-9


def test_max_points_relaxes_resolution_rather_than_shrinking_the_box(peptide):
    spec = GridSpec(resolution=0.1, padding=10.0, max_points=51**3)
    grid = size_grid(peptide, spec)

    assert grid.n_points <= spec.max_points
    assert grid.gsize % 2 == 1
    assert max(grid.spacing) > spec.resolution  # relaxed, and visibly so
    assert grid.box_length >= float(np.max(peptide.extent())) + 2 * spec.padding - 1e-9


def test_impossible_point_budget_raises(peptide):
    with pytest.raises(GridTooLarge, match="max_points"):
        size_grid(peptide, GridSpec(resolution=0.5, padding=10.0, max_points=MIN_GSIZE**3 - 1))


# --- surface mapping ---------------------------------------------------------


def test_smoothed_molecular_is_refused_by_both_flavours():
    """sashimi's default surface is APBS-only; substituting would be a 25% error."""
    for flavour in FLAVOURS:
        with pytest.raises(UnsupportedRequest, match="smoothed-molecular"):
            resolve_surface(SurfaceModel.SMOOTHED_MOLECULAR, 1.4, flavour)


def test_van_der_waals_collapses_the_probe_on_cpp():
    resolved = resolve_surface(SurfaceModel.VAN_DER_WAALS, 1.4, DelphiFlavour.CPP)
    assert resolved.probe_radius == 0.0
    assert resolved.gaussian is False


def test_molecular_keeps_the_probe_on_cpp():
    resolved = resolve_surface(SurfaceModel.MOLECULAR, 1.4, DelphiFlavour.CPP)
    assert resolved.probe_radius == pytest.approx(1.4)


def test_pydelphi_refuses_van_der_waals_and_says_why():
    """Its `vdw` still applies the probe, and prbrad=0 aborts it inside numba."""
    with pytest.raises(UnsupportedRequest, match="probe"):
        resolve_surface(SurfaceModel.VAN_DER_WAALS, 1.4, DelphiFlavour.PYDELPHI)


def test_every_flavour_shares_the_molecular_surface_with_apbs():
    """The precondition for cross-validation, pinned rather than assumed.

    This test previously asserted the opposite — that APBS and pyDelPhi shared
    nothing — on the reading that `surfmethod=vdw` was a van der Waals surface.
    It is the probe-rolling *construction*, and with a probe it reproduces the
    C++ build's `molecular` answer to the last printed digit. Both DelPhi
    flavours therefore share `MOLECULAR` with APBS, and that is what makes
    `tests/test_cross_validation.py` able to run at all.
    """
    from sashimi.apbs.options import SURFACE_KEYWORD  # noqa: PLC0415

    apbs_models = set(SURFACE_KEYWORD)
    for flavour in FLAVOURS:
        shared = apbs_models & SUPPORTED_SURFACES[flavour]
        assert SurfaceModel.MOLECULAR in shared, f"{flavour.value} shares nothing with APBS"


def test_pydelphi_maps_molecular_onto_its_vdw_construction():
    """`vdw` plus a probe is the molecular surface, and the probe must survive."""
    resolved = resolve_surface(SurfaceModel.MOLECULAR, 1.4, DelphiFlavour.PYDELPHI)

    assert resolved.keyword == "vdw"
    assert resolved.probe_radius == pytest.approx(1.4)
    assert resolved.gaussian is False


def test_nonlinear_is_refused_but_named_as_a_sashimi_limit():
    with pytest.raises(UnsupportedRequest, match="ROADMAP"):
        check_equation(Equation.NONLINEAR, DelphiFlavour.CPP)


# --- cube parsing ------------------------------------------------------------


def _cube(counts=(2, 2, 2), step=0.944863, origin=(-1.0, -2.0, -3.0), values=None):
    n = counts[0] * counts[1] * counts[2]
    values = values if values is not None else list(range(n))
    lines = [
        "cube written by a test",
        "Gaussian cube phimap(water)",
        f"    1 {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}",
        f"   {counts[0]} {step:12.6f}     0.000000     0.000000",
        f"   {counts[1]}     0.000000 {step:12.6f}     0.000000",
        f"   {counts[2]}     0.000000     0.000000 {step:12.6f}",
        "    1     0.000000     0.000000     0.000000     0.000000",
    ]
    lines += [" ".join(f"{v:.6e}" for v in values[i : i + 6]) for i in range(0, n, 6)]
    return "\n".join(lines) + "\n"


def test_cube_geometry_is_converted_from_bohr():
    """Atomic units by definition; read as angstroms every map is 1.89x too small."""
    grid = parse_cube(_cube())

    assert grid.spacing == pytest.approx([0.944863 * BOHR_TO_ANGSTROM] * 3)
    assert grid.spacing[0] == pytest.approx(0.5, abs=1e-6)  # what scale=2 means
    assert grid.origin == pytest.approx(np.array([-1.0, -2.0, -3.0]) * BOHR_TO_ANGSTROM)


def test_cube_values_are_c_ordered_like_dx():
    grid = parse_cube(_cube(counts=(2, 2, 2), values=list(range(8))))
    assert grid.values[0, 0, 0] == 0
    assert grid.values[0, 0, 1] == 1  # z fastest
    assert grid.values[1, 0, 0] == 4  # x slowest


def test_truncated_cube_is_rejected():
    text = _cube(counts=(2, 2, 2))
    truncated = "\n".join(text.splitlines()[:-1])
    with pytest.raises(MalformedStructure, match="truncated"):
        parse_cube(truncated)


def test_angstrom_cube_is_refused_rather_than_guessed():
    """Negative counts are Gaussian's angstrom flag; misreading it is a 1.89x error."""
    text = _cube().replace("   2 ", "   -2 ", 1)
    with pytest.raises(MalformedStructure, match="positive"):
        parse_cube(text)


# --- units and energy parsing ------------------------------------------------


def test_kt_converts_to_kj_per_mol():
    assert kt_to_kj_per_mol(1.0, 298.15) == pytest.approx(2.4789, abs=1e-4)
    # The Born ion measured here: -92.22 kT at 298.15 K.
    assert kt_to_kj_per_mol(-92.22, 298.15) == pytest.approx(-228.6, abs=0.2)


def test_cpp_energy_is_parsed_from_stdout():
    stdout = " Energy> Corrected reaction field energy               :       -92.22 kT\n"
    assert parse_cpp_energy(stdout) == pytest.approx(-92.22)


def test_missing_cpp_energy_is_none():
    assert parse_cpp_energy("Energy> Coulombic energy : 0.00 kT") is None


def test_pydelphi_energy_is_read_from_its_csv(tmp_path):
    """Preferred over stdout: four decimals rather than the printed two."""
    path = tmp_path / "outputs.csv"
    path.write_text(
        "# E_coul: ... | E_rxn_corr_tot: total.corrected_reaction_field_energy\n"
        "LABEL\tE_coul\tE_grid_w\tE_rxn_w\tE_rxn_corr_tot\tE_grid_tot\n"
        "pdbid\t0.0000\t1692.3961\t-92.5527\t-92.5527\t1692.3961\n"
    )
    assert parse_csv_energy(path) == pytest.approx(-92.5527)


def test_absent_csv_is_none(tmp_path):
    assert parse_csv_energy(tmp_path / "nothing.csv") is None


# --- flavour differences in the generated file -------------------------------


def test_boundary_condition_is_a_code_for_cpp_and_a_name_for_pydelphi(born):
    cpp = build_input(
        size_grid(born, GridSpec()),
        SolventModel(surface_model=SurfaceModel.MOLECULAR),
        flavour=DelphiFlavour.CPP,
    )
    py = build_input(
        size_grid(born, GridSpec()),
        SolventModel(surface_model=SurfaceModel.GAUSSIAN),
        flavour=DelphiFlavour.PYDELPHI,
    )
    assert "bndcon             = 4" in cpp
    assert "bndcon             = coulombic" in py


def test_only_cpp_asks_for_energies_explicitly(born):
    solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR)
    cpp = build_input(size_grid(born, GridSpec()), solvent, flavour=DelphiFlavour.CPP)
    assert "energy(s,c)" in cpp

    py = build_input(
        size_grid(born, GridSpec()),
        dataclasses.replace(solvent, surface_model=SurfaceModel.GAUSSIAN),
        flavour=DelphiFlavour.PYDELPHI,
    )
    assert "energy(s,c)" not in py  # computed unconditionally, written to CSV


def test_salt_and_ion_radius_reach_the_file(born):
    text = build_input(
        size_grid(born, GridSpec()),
        SolventModel(ionic_strength=0.15, ion_radius=2.0, surface_model=SurfaceModel.MOLECULAR),
        flavour=DelphiFlavour.CPP,
    )
    assert "salt               = 0.150000" in text
    assert "ionrad             = 2.0000" in text
