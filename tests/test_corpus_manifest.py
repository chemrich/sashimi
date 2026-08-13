"""The manifest and its recorded summaries, without running a solver.

`tests/test_corpus.py` needs APBS because it re-solves. Everything here reads
what is already checked in, which makes it the part of the corpus that runs on
every machine — including the assertions that the recorded numbers obey physics
rather than merely obeying themselves.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sashimi.corpus
from sashimi.analytic import born_solvation_energy
from sashimi.apbs.grid import size_grid
from sashimi.corpus import (
    CORPUS_DIR,
    MANIFEST,
    TIER_ORDER,
    AnalyticReference,
    Case,
    CaseTier,
    build_case,
    cases_for_tier,
    load_summary,
    verify_case,
)
from sashimi.protocol import (
    FiniteDifferenceRequest,
    PotentialGrid,
    Provenance,
    SolveRequest,
    SolveResult,
    SolverFamily,
    SurfaceModel,
)


def recorded(name: str) -> dict[str, Any]:
    case = next(c for c in MANIFEST if c.name == name)
    return load_summary(case)


# --- tiers -------------------------------------------------------------------


def test_tiers_are_cumulative():
    counts = [len(cases_for_tier(tier)) for tier in TIER_ORDER]
    assert counts == sorted(counts)
    assert len(cases_for_tier(CaseTier.FULL)) == len(MANIFEST)


def test_every_case_declares_a_tier_that_exists():
    assert all(case.tier in TIER_ORDER for case in MANIFEST)


def test_the_fast_tier_is_not_empty():
    """Whatever else is true, some corpus runs on every push."""
    assert cases_for_tier(CaseTier.FAST)


# --- the recorded numbers, against physics rather than against themselves ----


@pytest.mark.parametrize(
    "case", [c for c in MANIFEST if c.analytic is not None], ids=lambda c: c.name
)
def test_recorded_energies_sit_within_their_analytic_tolerance(case: Case):
    """The check that would fail if a backend had been wrong since the first build.

    Every other corpus assertion compares a solver against its own recording, so
    a number that was wrong on day one stays wrong forever and passes. This one
    compares the recording against the closed form.
    """
    assert case.analytic is not None
    summary = load_summary(case)
    assert summary["analytic"]["relative_error"] <= case.analytic.rtol


def test_solvation_energy_does_not_depend_on_the_sign_of_the_charge():
    """dG goes as q^2, so -1e must reproduce +1e exactly.

    A sign mishandled anywhere between the PQR reader and the energy — in charge
    parsing, in the source term, in the integration — breaks this and nothing
    else in the corpus, because every other case is positively charged.
    """
    positive = recorded("born-ion-coarse")["energy_kj_mol"]
    negative = recorded("born-ion-negative")["energy_kj_mol"]

    assert negative == pytest.approx(positive, rel=1e-9)


def test_solvation_energy_scales_as_the_square_of_the_charge():
    """+2e on the same sphere is exactly 4x +1e, in the recording and in theory."""
    monovalent = recorded("born-ion-coarse")["energy_kj_mol"]
    divalent = recorded("born-ion-divalent")["energy_kj_mol"]

    assert divalent == pytest.approx(4.0 * monovalent, rel=1e-6)


def test_refining_the_grid_moves_the_answer_toward_the_closed_form():
    """The convergence pairs are the corpus's only claim about correctness."""
    for coarse, fine in (
        ("born-ion-coarse", "born-ion-fine"),
        ("born-ion-r1-coarse", "born-ion-r1-fine"),
    ):
        assert (
            recorded(fine)["analytic"]["relative_error"]
            < (recorded(coarse)["analytic"]["relative_error"])
        ), f"{fine} should be closer to exact than {coarse}"


def test_a_lone_sphere_has_the_same_molecular_and_van_der_waals_boundary():
    """Rolling a probe over one sphere cannot carve a re-entrant surface.

    So these two must agree exactly, not approximately, and no other case in the
    corpus can catch a probe applied where it should not be.
    """
    molecular = recorded("born-ion-molecular")["energy_kj_mol"]
    van_der_waals = recorded("born-ion-vdw")["energy_kj_mol"]

    assert molecular == van_der_waals


def test_the_same_boundary_is_not_the_same_dielectric_map():
    """And the exact agreement above is a coarse-grid coincidence.

    The physics is identical — one sphere, one boundary, whatever the probe
    does — so the *exact* answers agree and the test above is right about what
    it claims. What the pair at 0.5 A cannot show is that APBS reaches that
    boundary two different ways: `srad 1.4` with `srfm mol` against `srad 0`.
    Refine to 0.25 A and the two part company, 0.621% from the closed form
    against 0.787%, which is discretization rather than modelling.

    Worth stating because the coarse pair reads as a guarantee and is not one:
    a solver that assumed the two paths were interchangeable would pass at
    0.5 A and be wrong everywhere it mattered.
    """
    molecular = recorded("born-ion-molecular-fine")
    van_der_waals = recorded("born-ion-vdw-fine")

    assert molecular["energy_kj_mol"] != van_der_waals["energy_kj_mol"]
    assert molecular["analytic"]["relative_error"] < van_der_waals["analytic"]["relative_error"]
    # Both still land near exact: this is a fine distinction, not a defect.
    assert van_der_waals["analytic"]["relative_error"] < 0.01


def test_the_surface_model_moves_the_answer():
    """Section 5's 25.7%: the largest modelling choice in the calculation.

    Every corpus case was `smoothed-molecular` until these, so a backend that
    ignored the surface model entirely would have passed the whole corpus.
    """
    energies = {
        name: recorded(name)["energy_kj_mol"]
        for name in ("peptide-default", "peptide-molecular", "peptide-vdw")
    }

    assert len(set(energies.values())) == len(energies), (
        f"surface models must give different answers, got {energies}"
    )


def test_no_two_cases_ask_the_same_question():
    """A case that duplicates another adds runtime and no coverage.

    Caught exactly this: `acetate-molecular` was committed with the surface
    model left at the default, making it a byte-identical rerun of `acetate`.
    """
    seen: dict[tuple[Any, ...], str] = {}
    for case in MANIFEST:
        key = (case.source, case.solvent, case.grid, case.compute_energy)
        assert key not in seen, f"{case.name} is identical to {seen.get(key)}"
        seen[key] = case.name


def test_every_case_names_its_surface_model_rather_than_inheriting_one():
    """The default is sashimi's to change; what a case asks is not.

    Forty-two cases took `smoothed-molecular` from `SolventModel`'s dataclass
    default, so moving that default to `molecular` would have silently rewritten
    the question each one asks — and the recordings would have gone red with
    nothing to say which side was wrong. Reading the source is the only way to
    see this: at runtime an inherited value and a stated one are the same value.

    The source comes from the imported module rather than from a path relative
    to the working directory, so it is the same file `MANIFEST` was built from —
    a cwd-relative read is one `pytest` invocation from parsing nothing, and one
    non-editable install from parsing a different copy than it counts.
    """
    source = inspect.getsourcefile(sashimi.corpus)
    assert source is not None
    manifest = next(
        node.value
        for node in ast.walk(ast.parse(Path(source).read_text()))
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "MANIFEST"
    )
    assert isinstance(manifest, ast.Tuple)
    assert len(manifest.elts) == len(MANIFEST)

    inherited = []
    for position, element in enumerate(manifest.elts):
        assert isinstance(element, ast.Call), f"MANIFEST[{position}] is not a Case(...) call"
        keywords = {k.arg: k.value for k in element.keywords}
        label = keywords.get("name")
        name = (
            label.value
            if isinstance(label, ast.Constant) and isinstance(label.value, str)
            else f"MANIFEST[{position}]"
        )
        solvent = keywords.get("solvent")
        assert isinstance(solvent, ast.Call), f"{name} builds its solvent indirectly"
        if not any(k.arg == "surface_model" for k in solvent.keywords):
            inherited.append(name)

    assert inherited == [], f"cases inheriting the surface default: {inherited}"


def test_both_closed_form_families_reach_a_boundary_every_backend_can_build():
    """The gap that made M0 necessary, and the guard that would have shown it.

    Fifteen of the eighteen closed-form cases sat on `smoothed-molecular` —
    APBS's harmonic averaging, which nothing else implements — and **all four**
    Kirkwood cases did. So the analytic sweep, the only part of the corpus that
    can say *wrong* rather than *changed*, could grade exactly one backend, and
    nothing in the suite said so: every case passed, every count looked healthy.
    A clean-room solver's acceptance ladder was unreachable by construction.

    Counting is not enough — a family entirely on one backend's private surface
    is the failure — so this asks the question per family.
    """
    portable = {SurfaceModel.MOLECULAR, SurfaceModel.VAN_DER_WAALS}
    by_family: dict[str, set[SurfaceModel]] = {}
    for case in MANIFEST:
        if case.analytic is None:
            continue
        family = case.analytic.source.split(":")[0]
        by_family.setdefault(family, set()).add(case.solvent.surface_model)

    assert set(by_family) == {"Born", "Kirkwood"}, f"unrecognised closed-form family: {by_family}"
    for family, models in sorted(by_family.items()):
        assert models & portable, (
            f"{family} closed forms exist only on {sorted(m.value for m in models)}, "
            "so no backend but APBS can be graded against them"
        )


def test_the_sharp_boundary_ladder_is_wide_enough_to_grade_a_solver_on():
    """One case per family would satisfy the guard above and prove nothing.

    A single agreeing number is a coincidence a wrong solver can produce; a
    functional form that has to agree across radius, charge and dielectric is
    not. This is the shape of that claim on the surfaces a new backend builds.
    """
    portable = {SurfaceModel.MOLECULAR, SurfaceModel.VAN_DER_WAALS}
    sharp = [c for c in MANIFEST if c.analytic is not None and c.solvent.surface_model in portable]

    born = [c for c in sharp if c.analytic is not None and c.analytic.source.startswith("Born")]
    kirkwood = [c for c in sharp if c.analytic is not None and c.analytic.source.startswith("Kirk")]

    assert len(born) >= 9, "the Born sweep needs its radius, charge and dielectric arms"
    assert len(kirkwood) >= 3, "M2's ladder is three rungs"
    # And a convergence pair, which is the only way to state "monotonic".
    assert {c.grid.resolution for c in born} >= {0.5, 0.25}


def test_every_recording_describes_the_case_it_answers():
    """The one field in a summary that nothing else can check.

    `verify_case` compares numbers, so a recording's prose can drift from the
    manifest's and stay green forever — and two files had, since the case's
    description was extended without rebuilding it. Harmless in itself, except
    that the description is how a reader learns what a recorded number is *for*,
    and it is the field a hand-edit touches when a case is re-explained rather
    than re-solved. Cheap to check, so it is checked rather than trusted.
    """
    stale = []
    for directory in [CORPUS_DIR, *CROSS_BACKEND_DIRECTORIES.values()]:
        for case in MANIFEST:
            path = directory / f"{case.name}.json"
            if path.is_file() and json.loads(path.read_text())["description"] != case.description:
                stale.append(str(path))

    assert stale == [], f"recordings describing a case differently than the manifest: {stale}"


def test_salt_makes_a_real_solute_more_favourably_solvated():
    """Screening on a structure rather than on a sphere."""
    energies = [
        recorded(name)["energy_kj_mol"]
        for name in ("peptide-no-salt", "peptide-default", "peptide-high-salt")
    ]

    assert all(b < a for a, b in itertools.pairwise(energies)), energies


def test_adding_a_proton_makes_a_protein_more_favourably_solvated():
    """+8e to +9e on the same 1,960 atoms. Solvation grows with net charge."""
    neutral_form = recorded("lysozyme")["energy_kj_mol"]
    protonated = recorded("lysozyme-protonated")["energy_kj_mol"]

    assert protonated < neutral_form


def test_geometry_matters_at_fixed_net_charge():
    """Two lysozymes at +9e, differing only in whether Asp66 is there.

    Both carry the same monopole, so anything separating them is the charge
    *distribution* rather than its total — which is the whole reason to solve
    the equation instead of using Born.
    """
    protonated = recorded("lysozyme-protonated")["energy_kj_mol"]
    deleted = recorded("lysozyme-deleted-residue")["energy_kj_mol"]

    assert protonated != deleted
    # ...and the difference is small next to either, which is what makes
    # charge-state calculations numerically awkward: 0.48% here.
    assert abs(deleted - protonated) / abs(protonated) < 0.05


def test_a_bound_ligand_is_a_small_perturbation_on_a_large_number():
    """FKBP with and without DMSO: 0.26% apart on 2,094 kJ/mol.

    A binding energy is this difference, so it is a few kJ/mol extracted from
    two numbers three orders of magnitude larger — the reason the corpus holds
    energies to 1e-4 rather than to something comfortable.
    """
    apo = recorded("fkbp-apo")["energy_kj_mol"]
    holo = recorded("fkbp-dmso")["energy_kj_mol"]

    assert apo != holo
    assert abs(holo - apo) / abs(apo) < 0.01


def test_the_largest_case_had_its_resolution_relaxed():
    """The guardrail engaging, recorded rather than assumed.

    8,279 atoms asks for 0.5 A and is given coarser, because `max_points` caps
    the grid rather than the atom count. That is why the largest case in the
    corpus costs 15 s and not an hour — and it is only visible because the
    resolved geometry is recorded next to the requested one.
    """
    summary = recorded("acetylcholinesterase")
    requested = summary["grid_spec"]["resolution"]

    assert max(summary["geometry"]["spacing"]) > requested
    assert summary["geometry"]["shape"] == [161, 161, 161]


def test_analytic_references_are_computed_rather_than_quoted():
    """The +2e reference is 4x the +1e one to the last bit, because both came
    from the same expression rather than from a table someone typed."""
    monovalent = next(c for c in MANIFEST if c.name == "born-ion-coarse").analytic
    divalent = next(c for c in MANIFEST if c.name == "born-ion-divalent").analytic
    assert monovalent is not None
    assert divalent is not None

    assert divalent.energy_kj_mol == pytest.approx(4.0 * monovalent.energy_kj_mol, rel=1e-15)


def test_the_salted_case_declines_an_analytic_reference():
    """Deliberate: the two backends disagree 39% on the mobile-ion term, so
    pinning either convention as "the" closed form would encode a choice as physics."""
    assert next(c for c in MANIFEST if c.name == "born-ion-salt").analytic is None


# --- the family-agnostic seam ------------------------------------------------


@pytest.mark.parametrize("case", MANIFEST[:6], ids=lambda c: c.name)
def test_a_case_asked_as_a_system_is_the_same_finite_difference_question(case: Case):
    """`Case.system()` must not quietly change what the corpus has been asking.

    Fifty recorded summaries were built through `Case.request()`; the seam that
    lets other families in has to produce the identical request for the family
    that was there first, or every golden number silently means something else.
    """
    direct = case.request()
    through_system = case.system().request_for(SolverFamily.FINITE_DIFFERENCE)

    # Field by field: `PQRData` holds numpy arrays, so dataclass equality on the
    # whole request raises rather than answering.
    assert type(through_system) is type(direct)
    assert through_system.grid == direct.grid
    assert through_system.solvent == direct.solvent
    assert through_system.equation == direct.equation
    assert through_system.want_energy == direct.want_energy
    assert through_system.want_potential == direct.want_potential
    np.testing.assert_array_equal(through_system.structure.coords, direct.structure.coords)
    np.testing.assert_array_equal(through_system.structure.charges, direct.structure.charges)
    np.testing.assert_array_equal(through_system.structure.radii, direct.structure.radii)


def test_the_analytic_family_gets_a_request_with_no_grid():
    """Which is the point: an analytic backend has no grid to be given one."""
    case = MANIFEST[0]
    request = case.system().request_for(SolverFamily.ANALYTIC)

    assert type(request) is SolveRequest
    assert not hasattr(request, "grid")
    assert request.structure.n_atoms == case.structure().n_atoms


# --- the analytic check catches a wrong answer -------------------------------


class WrongSolver:
    """Returns a plausible grid and an energy off by a factor the tolerance rejects."""

    def __init__(self, scale: float = 1.5):
        self.scale = scale

    def solve(self, request: FiniteDifferenceRequest) -> SolveResult:  # noqa: ARG002 — the protocol's signature
        potential = PotentialGrid(
            values=np.zeros((9, 9, 9)),
            origin=np.zeros(3),
            spacing=np.full(3, 0.5),
        )
        exact = -228.61080098772135
        return SolveResult(
            provenance=Provenance(backend="wrong"),
            energy_kj_mol=exact * self.scale,
            potential=potential,
        )


def test_a_backend_that_is_wrong_from_the_first_build_is_caught():
    """The failure mode a self-recorded corpus cannot see, now visible.

    The reference here is the wrong solver's *own* summary, so every
    recording-based check agrees with itself. Only the closed form disagrees.
    """
    case = next(c for c in MANIFEST if c.name == "born-ion-coarse")
    solver = WrongSolver()
    self_consistent = build_case(solver, case)

    found = verify_case(solver, case, self_consistent)

    assert [d.field for d in found] == ["analytic.energy_kj_mol"]
    assert "from the closed form" in found[0].detail


class RightEnergyWrongFieldSolver:
    """Integrates to the right number and hands back a field that is not the answer.

    The failure the corpus could not see before M0. Every analytic check was on
    the energy, and the field was compared only against its own recording, so a
    solver like this was indistinguishable from a correct one on its first
    build — and the field is the half a viewer displays.
    """

    def solve(self, request: FiniteDifferenceRequest) -> SolveResult:
        grid = size_grid(request.structure, request.grid)
        shape = tuple(grid.dime)
        spacing = np.array(grid.spacing, dtype=np.float64)
        origin = np.array(grid.center, dtype=np.float64) - (np.array(shape) - 1) * spacing / 2.0
        return SolveResult(
            provenance=Provenance(backend="right-energy-wrong-field"),
            # Exactly the Born closed form: the energy gate has nothing to say.
            energy_kj_mol=born_solvation_energy(3.0, 1.0, 1.0, 78.54),
            # A field that is merely plausible — right shape, right geometry,
            # uniformly a fifth of the true potential.
            potential=PotentialGrid(values=np.full(shape, 0.2), origin=origin, spacing=spacing),
        )


def test_a_solver_with_the_right_energy_and_the_wrong_field_is_caught():
    """The axis M0 added, doing the one thing the energy axis cannot."""
    case = next(c for c in MANIFEST if c.name == "born-ion-molecular")
    solver = RightEnergyWrongFieldSolver()
    self_consistent = build_case(solver, case)

    found = verify_case(solver, case, self_consistent)

    assert [d.field for d in found] == ["analytic_field.max_relative_error"]
    assert "closed-form potential" in found[0].detail


def test_no_field_sample_sits_in_a_cell_that_straddles_the_boundary():
    """The rule that makes the measurement mean anything, checked from the files.

    Interpolating across the dielectric interface is O(1) wrong — the potential
    is continuous there and its normal derivative is not — so a sample has to
    land in a cell that does not contain the boundary. The obvious rule, a fixed
    fraction of the radius, does not guarantee that: at 1.05a with 0.25 A
    spacing a 1 A sphere puts the sample *inside* the straddling cell and a 2 A
    sphere puts a cell corner exactly on the boundary. This asserts the rule the
    corpus actually used, from the recorded spacing rather than the requested
    one, because the guardrail relaxes resolution and a sample placed with the
    number that was asked for would drift into the interface unnoticed.
    """
    checked = 0
    for case in MANIFEST:
        if case.analytic_field is None:
            continue
        field = load_summary(case)["analytic_field"]
        spacing = field["spacing_used_a"]
        for radius, cells in zip(field["radii_a"], field["cells_out"], strict=True):
            assert cells >= 2, f"{case.name}: {cells} cells is inside the stencil"
            margin = radius - case.analytic_field.radius_a
            assert margin >= 2 * spacing - 1e-9, (
                f"{case.name}: sample at {radius:.4f} A is {margin:.4f} A beyond a "
                f"{case.analytic_field.radius_a:g} A boundary, under "
                f"{2 * spacing:.4f} A of clearance"
            )
            checked += 1
    assert checked >= 12, "the sampling rule is asserted on too little to mean anything"


def test_the_field_axis_reaches_more_than_one_radius_and_both_surfaces():
    """A field check on one sphere would not have caught the sampling bug.

    The rule fails at *small* radii, where the margin stops beating the spacing,
    so a single 3 A case would have passed while the arm it belongs to was
    broken. Both surfaces, because van der Waals is the one debye climbs first.
    """
    with_field = [c for c in MANIFEST if c.analytic_field is not None]

    assert {c.analytic_field.radius_a for c in with_field if c.analytic_field} >= {1.0, 2.0, 3.0}
    assert {c.solvent.surface_model for c in with_field} == {
        SurfaceModel.MOLECULAR,
        SurfaceModel.VAN_DER_WAALS,
    }


def test_a_case_without_a_closed_form_is_not_checked_against_one():
    case = next(c for c in MANIFEST if c.name == "born-ion-salt")
    solver = WrongSolver()

    assert verify_case(solver, case, build_case(solver, case)) == []


def test_an_analytic_reference_records_how_it_was_derived():
    """A bare number in a summary is unauditable; this says where it came from."""
    reference = next(c for c in MANIFEST if c.analytic is not None).analytic
    assert isinstance(reference, AnalyticReference)
    assert reference.source


# --- what the other backends recorded, from the files alone ------------------
#
# `tests/corpus/gb/` and `tests/corpus/tabipb/` hold answers to corpus questions
# from solvers other than APBS. Re-solving them needs those backends installed —
# a compiler and a mesher, in TABI-PB's case — but *comparing what they
# recorded* needs nothing, and it is where the interesting statement lives:
# `AccuracyTier` claims a reference tier and an approximate one behave
# differently in kind, and these files are the measurement of that claim.

CROSS_BACKEND_DIRECTORIES = {
    "gb": Path("tests/corpus/gb"),
    "tabipb": Path("tests/corpus/tabipb"),
    "delphi": Path("tests/corpus/delphi"),
}

# TABI-PB against APBS on every case both recorded. Measured 2026-08-12.
# A boundary-element solver and a grid solver share no discretization, so this
# is the honest width of "the reference tier agrees with itself".
TABIPB_DEVIATION_CEILING = 0.02


def cross_backend_cases(backend: str) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Every recorded (case, reference summary, other-backend summary) triple."""
    directory = CROSS_BACKEND_DIRECTORIES[backend]
    found = []
    for case in MANIFEST:
        path = directory / f"{case.name}.json"
        if path.is_file():
            found.append((case.name, load_summary(case), load_summary(case, directory)))
    return found


@pytest.mark.parametrize("backend", sorted(CROSS_BACKEND_DIRECTORIES))
def test_every_cross_backend_recording_answers_a_manifest_case(backend: str):
    """A summary whose case was renamed or dropped is a file nothing verifies."""
    directory = CROSS_BACKEND_DIRECTORIES[backend]
    recorded = {path.stem for path in directory.glob("*.json")}

    assert recorded <= {case.name for case in MANIFEST}
    assert recorded, f"no {backend} recordings found in {directory}"


def test_the_boundary_element_tier_agrees_with_the_grid_tier():
    """Two solver families, no shared discretization, and 1.0-1.6% between them.

    This is the corpus's strongest correctness statement, and it needs no closed
    form: TABI-PB meshes a surface where APBS fills a volume, so an error in
    either one's charge handling, units or boundary conditions has no way to
    cancel. It is also the number that makes the approximate tier's 0.7-28%
    (`tests/test_corpus_gb.py`) read as a property of the method rather than as
    corpus noise.
    """
    recorded = cross_backend_cases("tabipb")
    assert len(recorded) >= 6

    for name, reference, surface in recorded:
        deviation = abs(surface["energy_kj_mol"] - reference["energy_kj_mol"]) / abs(
            reference["energy_kj_mol"]
        )
        assert deviation < TABIPB_DEVIATION_CEILING, f"{name}: {deviation:.2%}"


def test_the_expensive_boundary_element_recordings_are_present():
    """Two protein-scale meshes `pytest` deliberately does not re-solve.

    Eight minutes of meshing between them, so they are verified on demand rather
    than per push — which makes "too slow to check here" one step from "quietly
    absent", the exact shape of the bug that let the DelPhi tier skip every test
    while CI stayed green. Their presence is checked here instead.
    """
    recorded = {name: surface for name, _, surface in cross_backend_cases("tabipb")}

    for name in ("fas2-molecular", "ion-protein-complex-molecular"):
        assert name in recorded, f"{name} has no boundary-element recording"
        assert recorded[name]["surface"]["n_vertices"] > 20_000
        assert recorded[name]["energy_kj_mol"] < 0


# The DelPhi tier, from the files alone. Re-solving these needs a DelPhi
# executable; comparing what two solvers recorded needs nothing, and it is the
# comparison that says whether either is right.
DELPHI_DIRECTORY = Path("tests/corpus/delphi")

# APBS against the C++ DelPhi on every case both recorded, measured 2026-08-12.
# Two independent finite-difference codes on different lattices — APBS's
# multigrid `dime`, DelPhi's odd cubic `gsize` — so this band is discretization
# and the definitional difference in what each calls a solvation energy, not
# arithmetic. Deliberately loose: pinning the measured 2% would make it a
# change-detector for physics that is allowed to change.
DELPHI_DEVIATION_CEILING = 0.08


def test_the_two_grid_codes_agree_where_they_can_be_compared():
    """The third reference, and the one that anchors the Born ion.

    DelPhi answers the closed form to -228.609 kJ/mol against -228.611 exactly,
    where APBS is 2.36% out at the same nominal resolution — so this tier is not
    a worse copy of APBS, it is a second opinion that happens to be sharper on
    the one case with an analytic answer.
    """
    recorded = cross_backend_cases("delphi")
    assert len(recorded) >= 19

    for name, reference, other in recorded:
        deviation = abs(other["energy_kj_mol"] - reference["energy_kj_mol"]) / abs(
            reference["energy_kj_mol"]
        )
        assert deviation < DELPHI_DEVIATION_CEILING, f"{name}: {deviation:.2%}"


def test_no_recorded_delphi_answer_is_absurd():
    """The check that would have caught what recording this tier actually found.

    Until 2026-08-12 `format_pqr` wrote minimum-width fields, so a
    four-character residue name shifted every column after it and DelPhi — which
    reads fixed columns — solved on charges that were not in the file. It
    returned -865,205 kJ/mol for acetate against APBS's -196.90, and the
    identical value for acetic acid, which is a different molecule. Two numbers
    agreeing to six decimals for two different structures is the tell.

    **Cases that physics requires to agree are grouped, not compared.** The
    sharp-boundary ladder added three kinds of legitimate coincidence, and a
    plain "all energies distinct" assertion called all of them bugs: solvation
    goes as q^2 so -1e reproduces +1e exactly; a lone sphere has the same
    molecular and van der Waals boundary; and DelPhi's corrected reaction field
    on a sphere does not move with the grid, so a convergence pair agrees to the
    last digit. The key below is what the energy is allowed to depend on —
    geometry, |charge|, and the two dielectric-and-salt parameters — so those
    four cases form one group and any *other* pair sharing an answer still
    fails. Which is what the original bug was: two different molecules.
    """
    energies = {name: other["energy_kj_mol"] for name, _, other in cross_backend_cases("delphi")}
    by_case = {c.name: c for c in MANIFEST}

    def physics_key(name: str) -> tuple[Any, ...]:
        case = by_case[name]
        structure = case.structure()
        return (
            structure.coords.tobytes(),
            np.abs(structure.charges).tobytes(),
            structure.radii.tobytes(),
            case.solvent.solute_dielectric,
            case.solvent.ionic_strength,
        )

    seen: dict[float, str] = {}
    for name, energy in energies.items():
        clash = seen.get(energy)
        if clash is not None:
            assert physics_key(clash) == physics_key(name), (
                f"{name} and {clash} are different structures with one answer: {energy}"
            )
        seen[energy] = name

    for name, energy in energies.items():
        assert -20_000 < energy < 0, f"{name}: {energy}"
