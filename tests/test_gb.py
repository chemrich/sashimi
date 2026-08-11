"""Generalized Born, binary-free by construction.

Every other backend's solver tests carry a marker and skip without an
executable. This one has nothing to install, so it runs everywhere and cannot
silently skip — which, given that this project has twice shipped a tier that
skipped while CI stayed green, is worth stating.

The physics is checked two ways that do not depend on any other solver being
present: against the Born closed form, which Generalized Born reduces to exactly
for a single sphere, and against direct numerical quadrature of the descreening
integral, which was derived here rather than quoted. `tests/test_gb_reference.py`
covers the comparison against real Poisson-Boltzmann solvers.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from sashimi.errors import UnsupportedRequest
from sashimi.gb import GbModel, GbOptions, GbSolver
from sashimi.gb.energy import debye_kappa
from sashimi.gb.options import GbRadii
from sashimi.gb.radii import descreening_integral, effective_radii, input_radii
from sashimi.protocol import (
    AccuracyTier,
    EnergyTerm,
    PQRData,
    SolventModel,
    SolveRequest,
    SurfaceModel,
)
from tests.born_reference import born_solvation_energy

VDW_ION_RADIUS = 3.0


def ion(radius: float = VDW_ION_RADIUS, charge: float = 1.0) -> PQRData:
    """The corpus Born ion: one sphere, one charge, a closed-form answer."""
    return PQRData(
        coords=np.zeros((1, 3)),
        charges=np.array([charge]),
        radii=np.array([radius]),
        labels=("ION 1 I",),
    )


def solvent(**kwargs) -> SolventModel:
    defaults = {
        "surface_model": SurfaceModel.MOLECULAR,
        "solute_dielectric": 1.0,
        "ionic_strength": 0.0,
    }
    return SolventModel(**{**defaults, **kwargs})


def energy(structure: PQRData, options: GbOptions | None = None, **solvent_kwargs) -> float:
    result = GbSolver(options or GbOptions()).solve(
        SolveRequest(structure=structure, solvent=solvent(**solvent_kwargs), want_potential=False)
    )
    assert result.energy_kj_mol is not None
    return result.energy_kj_mol


# --- the closed form ---------------------------------------------------------


def test_one_sphere_reduces_to_born_exactly():
    """The calibration this backend gets and no other one can.

    Generalized Born's interpolating denominator collapses to the effective
    radius when i == j and there is no second atom, so a lone sphere is the Born
    formula rather than an approximation to it. TABI-PB cannot be pinned this way
    at all — NanoShaper will not triangulate fewer than four atoms — so this is
    the only backend here whose agreement with a closed form is exact rather than
    convergent.

    `offset=0` because Amber's 0.09 A intrinsic-radius correction is a fitted
    shift, not physics; with it the effective radius is deliberately not the van
    der Waals one.
    """
    got = energy(ion(), GbOptions(offset=0.0))
    expected = born_solvation_energy(VDW_ION_RADIUS, 1.0, 1.0, 78.54)

    assert got == pytest.approx(expected, rel=1e-12)


def test_the_offset_moves_the_answer_by_exactly_the_radius_it_removes():
    """Born energy goes as 1/R, so a known radius shift is a known energy shift."""
    shifted = energy(ion(), GbOptions(offset=0.09))
    expected = born_solvation_energy(VDW_ION_RADIUS - 0.09, 1.0, 1.0, 78.54)

    assert shifted == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("charge", [1.0, -1.0, 2.0])
def test_born_reduction_holds_for_other_charges(charge: float):
    got = energy(ion(charge=charge), GbOptions(offset=0.0))
    expected = born_solvation_energy(VDW_ION_RADIUS, charge, 1.0, 78.54)

    assert got == pytest.approx(expected, rel=1e-12)


# --- the descreening integral, against quadrature ----------------------------


def numerical_integral(rho_i: float, r: float, a: float, seed: int = 11) -> float:
    """Monte Carlo of (1/4pi) * integral of |x|^-4 over sphere j, outside rho_i.

    The definition `descreening_integral` claims to evaluate in closed form,
    computed the slow honest way. Atom i sits at the origin; atom j has radius
    `a` and sits at distance `r` along z.
    """
    rng = np.random.default_rng(seed)
    centre = np.array([0.0, 0.0, r])
    points = centre + rng.uniform(-a, a, size=(2_000_000, 3))
    inside_j = np.sum((points - centre) ** 2, axis=1) < a * a
    distance = np.sqrt(np.sum(points**2, axis=1))
    counted = inside_j & (distance > rho_i)
    box_volume = (2 * a) ** 3
    return box_volume * float(np.mean(np.where(counted, distance**-4.0, 0.0))) / (4 * math.pi)


@pytest.mark.parametrize(
    ("rho_i", "r", "a"),
    [
        (1.5, 4.0, 1.7),  # disjoint spheres
        (1.5, 2.0, 1.7),  # overlapping
        (1.9, 3.0, 1.2),  # overlapping, larger descreened atom
        (1.0, 1.0, 2.5),  # atom i inside atom j
        (2.0, 0.6, 3.4),  # atom i deep inside atom j
    ],
)
def test_the_descreening_integral_matches_numerical_quadrature(rho_i: float, r: float, a: float):
    """A misremembered sign here produces plausible numbers, so it is checked.

    The last two cases are the ones worth having: when atom i's centre lies
    inside atom j, part of the shell around i is wholly enclosed and contributes
    its whole surface rather than a spherical cap of it. That term is easy to
    omit and impossible to notice from the energies.
    """
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]])
    analytic = descreening_integral(coords, np.array([rho_i, 1.0]), np.array([0.0, a]))[0]

    assert analytic == pytest.approx(numerical_integral(rho_i, r, a), rel=2e-3)


def test_an_atom_alone_is_descreened_by_nothing():
    integral = descreening_integral(np.zeros((1, 3)), np.array([1.5]), np.array([1.5]))
    assert integral[0] == 0.0


def test_an_engulfed_atom_contributes_nothing():
    """Atom j entirely inside atom i adds no volume outside atom i."""
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]])
    integral = descreening_integral(coords, np.array([3.0, 1.0]), np.array([0.0, 0.5]))
    assert integral[0] == 0.0


def test_chunking_does_not_change_the_answer():
    """The oracle pattern `_solute_mask` needed: chunked must equal unchunked."""
    rng = np.random.default_rng(4)
    n = 200
    coords = rng.uniform(-12.0, 12.0, size=(n, 3))
    intrinsic = rng.uniform(1.0, 2.0, size=n)
    scaled = intrinsic * 0.8

    whole = descreening_integral(coords, intrinsic, scaled, chunk_size=n)
    chunked = descreening_integral(coords, intrinsic, scaled, chunk_size=7)

    np.testing.assert_allclose(chunked, whole, rtol=0, atol=0)


def enclosed_atom(shell_distance: float, n_shell: int = 60) -> PQRData:
    rng = np.random.default_rng(9)
    shell = rng.normal(size=(n_shell, 3))
    shell /= np.linalg.norm(shell, axis=1)[:, None]
    coords = np.vstack([np.zeros(3), shell * shell_distance])
    return PQRData(
        coords=coords,
        charges=np.zeros(len(coords)),
        radii=np.full(len(coords), 1.7),
        labels=tuple(f"RES 1 C{i}" for i in range(len(coords))),
    )


def test_burial_raises_the_effective_radius():
    """The one qualitative claim the method makes about its own output.

    Tested as a monotonic response to burial rather than against a threshold:
    an effective radius has no absolute value to check it against, and the
    "buried atoms exceed surface atoms" phrasing is false for a shell tight
    enough that the shell atoms descreen each other harder than the atom they
    surround. On a real protein the ordering does appear — hen lysozyme's
    surface atoms sit near 4 A against 11-15 A for its core.
    """
    centres = [effective_radii(enclosed_atom(d), GbOptions())[0] for d in (7.0, 6.0, 5.0, 4.0)]

    assert all(b > a for a, b in itertools.pairwise(centres))
    assert centres[-1] > 1.7  # larger than the van der Waals radius it started from


# --- salt --------------------------------------------------------------------


def test_the_debye_length_is_the_textbook_one():
    """7.86 A at physiological salt: the number a units mistake fails."""
    kappa = debye_kappa(SolventModel(ionic_strength=0.15))
    assert 1.0 / kappa == pytest.approx(7.86, abs=0.02)


def test_no_salt_means_no_screening():
    assert debye_kappa(SolventModel(ionic_strength=0.0)) == 0.0


def test_salt_makes_solvation_more_favourable():
    """Mobile ions screen the charge, so |dG| grows with ionic strength."""
    energies = [energy(ion(), ionic_strength=i) for i in (0.0, 0.05, 0.15, 0.5)]
    assert all(b < a for a, b in itertools.pairwise(energies))


# --- radii -------------------------------------------------------------------


def test_zero_radius_hydrogens_are_raised_rather_than_divided_by():
    """pdb2pqr gives hydroxyl hydrogens radius 0; this method divides by radius.

    A grid solver spreads that charge over grid points and never notices. Hen
    lysozyme has twenty such atoms carrying +8.34 e between them, which is an
    infinite self-energy rather than a small error.
    """
    structure = PQRData(
        coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        charges=np.array([-0.4, 0.4]),
        radii=np.array([1.5, 0.0]),
        labels=("SER 1 OG", "SER 1 HG"),
    )

    resolved, n_substituted = input_radii(structure, GbOptions(radii=GbRadii.AS_GIVEN))

    assert resolved.min() == pytest.approx(GbOptions().minimum_radius)
    assert n_substituted == 1
    assert math.isfinite(energy(structure, GbOptions(radii=GbRadii.AS_GIVEN)))


def test_mbondi_is_the_default_and_says_so():
    """Feeding this method pdb2pqr's radii costs 35% on a protein."""
    structure = PQRData(
        coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]),
        charges=np.array([-0.4, 0.4]),
        radii=np.array([1.661, 0.6]),
        labels=("SER 1 OG", "SER 1 HG"),
    )

    resolved, n_substituted = input_radii(structure, GbOptions())

    assert resolved.tolist() == [1.5, 1.2]  # mbondi O and H
    assert n_substituted == 2


def test_an_unrecognised_element_keeps_its_given_radius():
    """Better than guessing: `CA` is calcium as an ion and carbon in a residue."""
    resolved, _ = input_radii(ion(), GbOptions())
    assert resolved.tolist() == [VDW_ION_RADIUS]


# --- what it refuses ---------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [SurfaceModel.VAN_DER_WAALS, SurfaceModel.SMOOTHED_MOLECULAR, SurfaceModel.GAUSSIAN],
)
def test_only_the_molecular_surface_is_answered(model: SurfaceModel):
    """The surface its answer corresponds to, not the one its integral runs over.

    Declaring `van-der-waals` — the intuitive reading, since descreening
    integrates over van der Waals spheres — puts the answer 31% from APBS on hen
    lysozyme, because the OBC rescaling exists precisely to carry the union of
    spheres onto the solvent-excluded volume.
    """
    with pytest.raises(UnsupportedRequest, match="molecular surface"):
        energy(ion(), surface_model=model)


def test_a_potential_field_is_refused_rather_than_invented():
    with pytest.raises(UnsupportedRequest, match="no potential field"):
        GbSolver().solve(
            SolveRequest(structure=ion(), solvent=solvent(), want_potential=True, want_energy=True)
        )


# --- what it reports ---------------------------------------------------------


def test_provenance_names_no_binary_because_there_is_none():
    """The first real backend to exercise what `Provenance` has always allowed."""
    result = GbSolver().solve(
        SolveRequest(structure=ion(), solvent=solvent(), want_potential=False)
    )

    assert result.provenance.binary_path is None
    assert result.provenance.binary_sha256 is None
    assert result.provenance.wall_seconds is not None
    assert result.provenance.backend.startswith("gb-obc2")


def test_provenance_declares_the_approximation():
    result = GbSolver().solve(
        SolveRequest(structure=ion(), solvent=solvent(), want_potential=False)
    )

    assert result.provenance.accuracy_tier is AccuracyTier.APPROXIMATE
    assert result.provenance.energy_term is EnergyTerm.POLAR_SOLVATION


def test_the_model_reaches_provenance_so_two_runs_can_be_told_apart():
    resolved = (
        GbSolver(GbOptions(model=GbModel.HCT))
        .solve(SolveRequest(structure=ion(), solvent=solvent(), want_potential=False))
        .provenance.resolved_parameters
    )

    assert resolved["gb"]["model"] == "hct"
    assert resolved["gb"]["radii"] == "mbondi"
