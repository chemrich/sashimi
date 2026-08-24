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
from pathlib import Path

import numpy as np
import pytest

import sashimi.delphi.backend as backend_module
from sashimi.corpus import MANIFEST, load_summary, verify_case
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
    """`smoothed-molecular` is APBS-only, and no flavour may fake it."""
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


# --- the golden corpus, for the flavour that can reproduce it ----------------

DELPHI_DIRECTORY = Path("tests/corpus/delphi")

# Split by measured cost on osx-arm64, 2026-08-12. DelPhi's cost is its cubic
# grid, which follows the bounding box rather than the solute: `fas2-molecular`
# is 906 atoms and takes 11.5 s where `lysozyme-molecular` is 1,960 and takes
# 5.8 s. So this is measured per case, like every other tier assignment here.
DELPHI_PER_PUSH = (
    "born-ion-molecular-eps2",  # 0.13 s
    "born-ion-molecular",  # 0.16 s
    "peptide-molecular-no-salt",  # 0.20 s
    "peptide-molecular",  # 0.21 s
    "peptide-molecular-cold",  # 0.21 s
    "peptide-molecular-high-salt",  # 0.22 s
    "methanol-molecular",  # 0.95 s — one of the three the column bug destroyed
    "methoxide-molecular",  # 1.00 s
    "acetate-molecular",  # 1.13 s — and another
    "acetic-acid-molecular",  # 1.49 s — and the third
    "aspartate-residue-molecular",  # 1.50 s
    # The sharp-boundary ladder (ROADMAP.md section 12, M0). Every one of these
    # carries a `delphi_rtol` — a closed-form tolerance three orders tighter
    # than the shared one, because DelPhi is 0.0006% from Born where APBS is
    # 2.36% — and a per-backend tolerance nothing re-solves is not a tolerance.
    # They were recorded and then left out of this list at first, which made all
    # sixteen of them dead weight: the tight gates existed and nothing drove
    # them. Cheap enough that "too slow per push" was never the reason.
    "born-ion-molecular-r1",  # 0.08 s
    "born-ion-molecular-r2",  # 0.10 s
    "born-ion-molecular-negative",  # 0.13 s
    "born-ion-molecular-divalent",  # 0.14 s
    "born-ion-molecular-eps4",  # 0.13 s
    "born-ion-molecular-salt",  # 0.13 s
    "born-ion-molecular-high-salt",  # 0.13 s
    "born-ion-vdw",  # 0.13 s
    "born-ion-molecular-r4",  # 0.16 s
    "born-ion-molecular-r6",  # 0.23 s
    "kirkwood-molecular-07",  # 1.09 s
    "born-ion-vdw-fine",  # 1.11 s
    "kirkwood-molecular-09",  # 1.11 s
    "kirkwood-molecular-05",  # 1.13 s
    "born-ion-molecular-fine",  # 1.13 s
    "kirkwood-molecular-03",  # 1.14 s
)  # 15.3 s in total

# Everything recorded that is not re-solved per push. **Derived, where this and
# the tuple above were both hand-kept lists** — and between them they named 35
# of the 58 recordings in `tests/corpus/delphi/`, so twenty-three sat in
# neither: re-solved by nothing here and named by nothing here. Six of those
# carry a *tight* per-backend closed-form tolerance (`born-ion-vdw-r1` and
# `-r6` at 0.001, the four `kirkwood-vdw-*` rungs at 0.003 to 0.01) which
# `tests/test_corpus_manifest.py` does check against the recordings, but which
# no binary in this file was ever asked to reproduce.
#
# Taking the complement is what makes the split total: a recording added later
# joins this set by existing, instead of falling out of both and looking like
# neither. The fast list stays hand-kept because it encodes measured cost, and
# nothing in the manifest knows what a case costs.
#
# These are verified on demand rather than per push — 83 s in total —
# with `sashimi corpus verify --backend delphi --tier full --directory
# tests/corpus/delphi --case <name>`. Their presence and their agreement with
# the APBS recordings are checked without a binary in
# `tests/test_corpus_manifest.py`.
DELPHI_ON_DEMAND = tuple(
    sorted(
        {path.stem for path in DELPHI_DIRECTORY.glob("*.json")} - set(DELPHI_PER_PUSH),
    )
)


def test_every_delphi_recording_is_either_re_solved_or_named():
    """The partition is total, and it is asserted rather than assumed.

    "Too slow to check per push" decaying into "quietly absent" is the shape of
    the bug that let the whole DelPhi tier skip while CI stayed green. A case in
    neither tuple is a third state that reads like neither.
    """
    recorded = {path.stem for path in DELPHI_DIRECTORY.glob("*.json")}

    assert set(DELPHI_PER_PUSH) <= recorded, set(DELPHI_PER_PUSH) - recorded
    assert set(DELPHI_PER_PUSH) | set(DELPHI_ON_DEMAND) == recorded
    assert not set(DELPHI_PER_PUSH) & set(DELPHI_ON_DEMAND)


@pytest.mark.parametrize("name", DELPHI_PER_PUSH)
def test_delphi_reproduces_its_recorded_corpus_answer(binary, name):
    """A third reference tier, recorded from the C++ build and verified by it.

    **C++ only, and that is a measurement rather than a preference.** The
    recordings were made on osx-arm64; CI verified all nineteen of them against
    a linux-64 build of the same source at full corpus tolerance, so a C++
    recording is portable. pyDelPhi fails fifteen of the nineteen — energies
    0.047% to 0.426% out, the Born ion's potential minimum 2.5%, grid origins
    differing in their last digits. That is not wrong: 0.4% is far tighter than
    the 2.3% between DelPhi and APBS. It is a different implementation of an
    iterative solver, and ~43x the 1e-4 a recording is held to.

    Both figures are the *nineteen-case* bound. Re-measured over the whole
    corpus on 2026-08-20, the band is 0.001-1.257% across the 35 cases both
    flavours answer, and the worst of those is ~125x the tolerance. Either way
    the conclusion is the same and the arithmetic is now right: the earlier
    "4,000x" divided a percentage by a fraction and was wrong by 100x.

    So the flavours are not interchangeable *as sources of a recorded number*,
    while remaining interchangeable as backends. pyDelPhi keeps the behavioural
    tier above; it does not verify numbers another program produced.
    """
    if binary.flavour is not DelphiFlavour.CPP:
        pytest.skip("pyDelPhi is 0.001-1.257% from the C++ build; it cannot verify its recordings")

    case = next(c for c in MANIFEST if c.name == name)
    recorded = load_summary(case, DELPHI_DIRECTORY)

    assert verify_case(DelphiSolver(), case, recorded) == []


def test_the_expensive_delphi_cases_are_recorded_even_though_pytest_skips_them():
    """83 s of solving a per-push suite has no business repeating.

    Named here so "too slow to check per push" cannot decay into "quietly
    absent" — the shape of the bug that let the DelPhi tier skip every test
    while CI stayed green.
    """
    for name in DELPHI_ON_DEMAND:
        case = next(c for c in MANIFEST if c.name == name)
        assert load_summary(case, DELPHI_DIRECTORY)["energy_kj_mol"] < 0
