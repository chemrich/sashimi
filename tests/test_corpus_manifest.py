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
from sashimi import backends
from sashimi.analytic import born_solvation_energy
from sashimi.apbs.grid import size_grid
from sashimi.corpus import (
    CORPUS_DIR,
    MANIFEST,
    ROOT_BACKEND,
    TIER_ORDER,
    AnalyticReference,
    Case,
    CaseTier,
    _analytic_field_summary,
    build_case,
    cases_for_tier,
    corpus_dir_for,
    load_summary,
    verify_case,
)
from sashimi.debye import DebyeSolver
from sashimi.field import FIELD_DIRECTIONS
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
    "case",
    [c for c in MANIFEST if c.analytic is not None and c.analytic.gated],
    ids=lambda c: c.name,
)
def test_recorded_energies_sit_within_their_analytic_tolerance(case: Case):
    """The check that would fail if a backend had been wrong since the first build.

    Every other corpus assertion compares a solver against its own recording, so
    a number that was wrong on day one stays wrong forever and passes. This one
    compares the recording against the closed form.

    **`gated` is honoured here and not only in `verify_case`.** It was not at
    first, which made the flag a half-measure: `kirkwood-molecular-09` is
    declared ungatable because both reference codes get *worse* under refinement
    there, and it was still being judged from the files with 2.2e-5 of headroom.
    A rebuild on an APBS that moved the fourth digit would have turned CI red on
    the one case the manifest says nobody should judge.
    """
    assert case.analytic is not None
    summary = load_summary(case)
    assert summary["analytic"]["relative_error"] <= case.analytic.rtol


@pytest.mark.parametrize(
    "case",
    [c for c in MANIFEST if c.analytic is not None and not c.analytic.gated],
    ids=lambda c: c.name,
)
def test_an_ungated_case_is_still_recorded_against_its_closed_form(case: Case):
    """Not judged is not the same as not measured.

    The point of `gated=False` is that the deviation stays visible and diffable:
    a reader should be able to see how far off the method is without the corpus
    either blessing that number or calling it a failure.
    """
    summary = load_summary(case)

    assert summary["analytic"] is not None
    assert summary["analytic"]["relative_error"] > 0.0
    assert summary["analytic"]["source"]


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

    # The arms themselves, not a count of cases. `>= 9` and `>= 3` were the
    # previous form and neither asserted what this docstring claims: fourteen
    # Born cases at one radius, one charge and one dielectric would have passed
    # the first, and that is precisely the coincidence the paragraph above says
    # a single agreeing number cannot rule out. A count is also the wrong shape
    # for a set that grows — 9 was near the truth when written and the truth is
    # 14 now, so five could have been deleted in silence.
    radii = {round(float(r), 3) for c in born for r in c.structure().radii}
    charges = {round(float(q), 3) for c in born for q in c.structure().charges}
    dielectrics = {c.solvent.solute_dielectric for c in born}

    assert len(radii) >= 3, f"the radius arm needs more than {sorted(radii)}"
    assert min(charges) < 0, "no negative charge, so a sign error has nowhere to show"
    assert max(charges) >= 2, "no divalent, so the q^2 scaling is untested here"
    assert len(dielectrics) >= 2, f"the dielectric arm needs more than {sorted(dielectrics)}"
    # And a convergence pair, which is the only way to state "monotonic".
    assert {c.grid.resolution for c in born} >= {0.5, 0.25}

    # M2's ladder, by the offset that defines a rung rather than by how many
    # files carry one. Derived from the geometry — the charged atom's distance
    # from the origin over the sphere radius — so it cannot drift from a prose
    # `source` string, and each rung must exist on both portable surfaces
    # because the probe is what tells the two apart.
    rungs: dict[float, set[str]] = {}
    for case in kirkwood:
        structure = case.structure()
        coords = np.asarray(structure.coords)
        charge_at = int(np.argmax(np.abs(np.asarray(structure.charges))))
        offset = float(np.linalg.norm(coords[charge_at])) / float(max(structure.radii))
        rungs.setdefault(round(offset, 3), set()).add(case.solvent.surface_model.value)

    assert len(rungs) >= 4, f"M2's ladder needs more rungs than {sorted(rungs)}"
    assert max(rungs) >= 0.9, "no rung near the boundary, which is where the method strains"
    assert all(len(surfaces) == 2 for surfaces in rungs.values()), rungs


def test_the_tight_delphi_tolerances_are_actually_reaching_delphi():
    """A per-backend tolerance keyed on a string nothing asserts is not a guard.

    `rtol_for` matches a prefix of `Provenance.backend`, which DelPhi builds as
    `f"{flavour}-{version}"` in `delphi/discover.py`. Reformat that identity —
    `delphi-cpp-8.6`, say — and every tight tolerance silently reverts to the
    shared one that exists to accommodate APBS, with the whole suite still
    green. This reads the identity out of the recordings themselves, so the
    coupling is asserted rather than assumed, and checks the recorded deviations
    actually sit inside the tight bound rather than merely inside the loose one.
    """
    checked = 0
    for name, _, other in paired_cases(ROOT_BACKEND, "delphi"):
        case = next(c for c in MANIFEST if c.name == name)
        if case.analytic is None or not case.analytic.per_backend_rtol:
            continue
        identity = other["backend"]
        tight = case.analytic.rtol_for(identity)
        assert tight < case.analytic.rtol, (
            f"{name}: {identity!r} did not match any per-backend prefix, so the "
            f"tolerance fell back to the shared {case.analytic.rtol}"
        )
        assert other["analytic"]["relative_error"] <= tight, f"{name}: {other['analytic']}"
        checked += 1

    # Every recorded pair whose case declares a per-backend tolerance, derived
    # rather than floored. `>= 15` was written when 15 was near the real count;
    # it is 20 now, so five could have stopped being checked in silence.
    assert checked == sum(
        1
        for case in MANIFEST
        if case.analytic is not None
        and case.analytic.per_backend_rtol
        and (corpus_dir_for("delphi") / f"{case.name}.json").is_file()
    )


def test_the_debye_tolerances_are_actually_reaching_debye():
    """The same guard for the other tight key, which had no equivalent.

    If `DebyeSolver.label` stops starting with `debye`, every bar in the
    manifest reverts to the shared tolerance that exists to accommodate APBS,
    and M1's and M2's gates go green by falling back. Nothing asserted that for
    the `_born` cases at all; M2's own tests happened to cover its three rungs.

    **The assertion is that the declared number is the one applied, not that it
    is tighter.** It was written as `rtol_for(label) < rtol` while every
    per-backend tolerance in the manifest happened to tighten, which made
    "tighter" look like the property being guarded. It is not — the property is
    that the key *matches*, and M5 produced the first widening entry:
    `born-ion-molecular-r4`, where APBS relaxes the request to 0.4375 A and
    debye solves the 0.5 it was asked for, so debye's honest 1.504% sits above
    the 1% APBS earns on a finer lattice. Comparing against `rtol` could not
    tell that from a mistyped key; comparing against the declared value can.

    Derived from the manifest rather than from a list, per the guards file's
    second lesson: a case that gains a `debye_rtol` joins this automatically.
    """
    label = DebyeSolver().label
    checked = tightened = 0
    for case in MANIFEST:
        reference = case.analytic
        declared = dict(reference.per_backend_rtol).get("debye") if reference else None
        if reference is None or declared is None:
            continue
        assert reference.rtol_for(label) == declared, (
            f"{case.name}: {label!r} did not match the 'debye' prefix, so its bar "
            f"fell back to the shared {reference.rtol} instead of the declared {declared}"
        )
        checked += 1
        tightened += declared < reference.rtol

    assert checked >= 5, (
        f"only {checked} case(s) carry a debye bar; M1's Born pair and M2's three "
        "Kirkwood rungs should all be here"
    )
    assert tightened >= 5, (
        f"only {tightened} debye bar(s) are tighter than the shared tolerance. The "
        "milestone bars must be: a gate at the tolerance APBS needs is a gate that "
        "cannot fail."
    )


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

# Derived from the registry rather than listed, since M5 made the layout a
# function of the backend name (`corpus.corpus_dir_for`). Two things follow that
# a hand-maintained dict did not give: a backend registered without recording
# anything fails `test_every_cross_backend_recording_answers_a_manifest_case`
# on its empty directory instead of quietly sitting outside every check here,
# and the mapping cannot disagree with the one `sashimi corpus build` writes to.
# `delphi-cpp` and `pydelphi` are the same two executables `delphi` resolves to,
# registered so a caller can pin the flavour. They record nothing of their own:
# `tests/corpus/delphi/` stays C++-only, as section 7 decided, and a recording
# filed under a flavour name would be the same numbers in a second place.
FLAVOUR_ALIASES = frozenset({"delphi-cpp", "pydelphi"})

CROSS_BACKEND_DIRECTORIES = {
    name: corpus_dir_for(name)
    for name in backends.names()
    if name != ROOT_BACKEND and name not in FLAVOUR_ALIASES
}

# TABI-PB against APBS on every case both recorded. Measured 2026-08-12.
# A boundary-element solver and a grid solver share no discretization, so this
# is the honest width of "the reference tier agrees with itself".
TABIPB_DEVIATION_CEILING = 0.02


def paired_cases(a: str, b: str) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Every recorded (case, summary from `a`, summary from `b`) triple.

    Symmetric in its two arguments, where the `cross_backend_cases` it replaces
    hardcoded APBS on the left. That asymmetry was not a simplification — it was
    the reason every comparison in this file ran against APBS and no other pair
    could be expressed. `corpus_dir_for` already resolves the root backend to
    `tests/corpus/` and every other to a subdirectory of it, so both sides are
    the same lookup and neither is privileged.

    What it unlocks is the pairing this corpus most needed and could not state:
    debye against TABI-PB. Those two share no discretization *and* no dielectric
    assignment, where APBS, DelPhi and debye all sample the dielectric at face
    centres — `sashimi.debye.dielectric` calls reaching for the other two "a
    shared bias". A grid-versus-boundary-element pair is the only comparison in
    this repository that is independent in both respects.
    """
    return [
        (case.name, load_summary(case, backend=a), load_summary(case, backend=b))
        for case in MANIFEST
        if all((corpus_dir_for(name) / f"{case.name}.json").is_file() for name in (a, b))
    ]


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
    recorded = paired_cases(ROOT_BACKEND, "tabipb")
    # Set equality against the directory, not `>= 6`. The floor happened to
    # equal the real count when it was written, which is the most misleading
    # place for one to sit: it reads as exact and silently stops guarding the
    # moment a seventh recording lands.
    assert {name for name, _, _ in recorded} == {
        path.stem for path in corpus_dir_for("tabipb").glob("*.json")
    }

    for name, reference, surface in recorded:
        deviation = abs(surface["energy_kj_mol"] - reference["energy_kj_mol"]) / abs(
            reference["energy_kj_mol"]
        )
        assert deviation < TABIPB_DEVIATION_CEILING, f"{name}: {deviation:.2%}"


# --- the one comparison that shares neither discretization nor dielectric ----
#
# Every other cross-backend number in this repository compares two codes that
# both fill a volume and both sample the dielectric at face centres. APBS,
# DelPhi and debye all do, which is why `sashimi.debye.dielectric` says reaching
# for the first two to grade the third "measures a shared bias".
#
# TABI-PB shares neither. It meshes the dielectric boundary and solves an
# integral equation on it, so a grid code agreeing with it has agreed with
# something that could not have inherited its errors. That makes these eighteen
# ratios the most independent statement the corpus can make, and until now only
# one of the three pairings — APBS's — was checked at all.
#
# Measured 2026-08-23, TABI-PB as the denominator in every row so the three
# grid codes are directly comparable to each other. The pre-existing
# `TABIPB_DEVIATION_CEILING` test above keeps APBS as its denominator and is
# untouched; these numbers therefore differ from it in the third digit by
# convention rather than by disagreement.
#
# What the columns say, and it is not what anyone would have guessed:
#
# - **debye is the closest of the three to TABI-PB on every solute at or below
#   260 atoms** — 0.48% to 0.86% where APBS is 1.01% to 1.56% and DelPhi 1.71%
#   to 3.75%.
# - **And the furthest on every protein at or above 906 atoms** — 2.20% to
#   4.57%, on all seven, without exception.
#
# *Extended 2026-08-27 from six recordings to twelve. At six this read "closest
# on five of six and furthest on `fas2-molecular`, the only real protein among
# them", and concluded that no story about a uniformly better or worse solver
# fitted both halves. Six more proteins say the split is at **size**, and that
# the lone exception was the beginning of the pattern rather than an anomaly in
# it. The original reading is left visible here because a finding that turned
# out to be an n=1 artefact is worth seeing next to what replaced it.*
#
# `tests/test_corpus_debye.py` records `fas2-molecular` sitting -1.53% from APBS
# and -4.41% from DelPhi, so the disagreement is real and not an artifact of the
# denominator.
#
# **Pinned two-sided, with no ceiling.** A bar would have to sit above 2.82% to
# admit `fas2-molecular` today, which is above every other number here and
# therefore not a bar at all; and a ceiling derived from the measurement it
# grades is the threshold-fitted-to-the-result shape ROADMAP.md section 12
# records this milestone being caught by three times. A pin cannot be satisfied
# by loosening it.
CROSS_FAMILY_DEVIATION: dict[tuple[str, str], float] = {
    ("apbs", "barnase-molecular"): 0.01412,
    ("apbs", "barstar-molecular"): 0.00307,
    ("apbs", "fas2-molecular"): 0.01273,
    ("apbs", "fkbp-apo-molecular"): 0.01430,
    ("apbs", "fkbp-dmso-molecular"): 0.01405,
    ("apbs", "ion-protein-complex-molecular"): 0.01006,
    ("apbs", "lysozyme-molecular"): 0.00785,
    ("apbs", "peptide-molecular"): 0.01456,
    ("apbs", "peptide-molecular-cold"): 0.01461,
    ("apbs", "peptide-molecular-high-salt"): 0.01561,
    ("apbs", "peptide-molecular-no-salt"): 0.01447,
    ("apbs", "protein-1a63-molecular"): 0.00713,
    ("debye", "barnase-molecular"): 0.04574,
    ("debye", "barstar-molecular"): 0.02201,
    ("debye", "fas2-molecular"): 0.02817,
    ("debye", "fkbp-apo-molecular"): 0.03692,
    ("debye", "fkbp-dmso-molecular"): 0.03617,
    ("debye", "ion-protein-complex-molecular"): 0.00862,
    ("debye", "lysozyme-molecular"): 0.03519,
    ("debye", "peptide-molecular"): 0.00583,
    ("debye", "peptide-molecular-cold"): 0.00578,
    ("debye", "peptide-molecular-high-salt"): 0.00477,
    ("debye", "peptide-molecular-no-salt"): 0.00729,
    ("debye", "protein-1a63-molecular"): 0.02792,
    ("delphi", "barnase-molecular"): 0.01139,
    ("delphi", "barstar-molecular"): 0.01674,
    ("delphi", "fas2-molecular"): 0.01525,
    ("delphi", "fkbp-apo-molecular"): 0.01853,
    ("delphi", "fkbp-dmso-molecular"): 0.01885,
    ("delphi", "ion-protein-complex-molecular"): 0.01709,
    ("delphi", "lysozyme-molecular"): 0.01298,
    ("delphi", "peptide-molecular"): 0.03652,
    ("delphi", "peptide-molecular-cold"): 0.03652,
    ("delphi", "peptide-molecular-high-salt"): 0.03745,
    ("delphi", "peptide-molecular-no-salt"): 0.03502,
    ("delphi", "protein-1a63-molecular"): 0.01367,
}

# Both sides are checked-in recordings, so every ratio is exact arithmetic on
# two constants and this only has to absorb float formatting.
CROSS_FAMILY_BAND = 0.0005

GRID_BACKENDS = ("apbs", "debye", "delphi")


def test_every_grid_backend_is_pinned_against_the_boundary_element_tier():
    """Set equality, so a seventh TABI-PB recording cannot land unpinned."""
    expected = {
        (backend, path.stem)
        for backend in GRID_BACKENDS
        for path in corpus_dir_for("tabipb").glob("*.json")
        if (corpus_dir_for(backend) / path.name).is_file()
    }
    assert set(CROSS_FAMILY_DEVIATION) == expected


@pytest.mark.parametrize("backend", GRID_BACKENDS)
def test_the_grid_tier_sits_where_it_was_measured_against_the_surface_tier(backend: str):
    """Each grid code against TABI-PB, case by case, two-sided.

    Two-sided on purpose. Moving *closer* to the boundary-element answer is the
    outcome a change to the dielectric treatment would be hoping for, and it is
    exactly the outcome that should be read by a human rather than absorbed
    silently by a one-sided bound.
    """
    off = []
    for name, surface, grid in paired_cases("tabipb", backend):
        found = abs(grid["energy_kj_mol"] - surface["energy_kj_mol"]) / abs(
            surface["energy_kj_mol"]
        )
        expected = CROSS_FAMILY_DEVIATION[(backend, name)]
        if abs(found - expected) > CROSS_FAMILY_BAND:
            off.append(f"{name}: {found:.5f} != {expected:.5f}")

    assert not off, "\n  ".join([f"{backend} moved against TABI-PB:", *off])


def test_debye_is_closest_on_small_solutes_and_furthest_on_every_protein():
    """Where debye stands against the independent family, split by size.

    **This replaces a reading that had one data point.** With six TABI-PB
    recordings the table said debye was closest on five of six and furthest on
    `fas2-molecular`, "the only real protein among them" — and concluded that no
    story about a uniformly better or worse solver fitted both halves. With
    twelve there is a story, and the exception was the beginning of it:

    - debye is **closest of the three on all five solutes at or below 260
      atoms** — 0.48% to 0.86%, against APBS's 1.01% to 1.56%.
    - debye is **furthest of the three on all seven proteins at or above 906
      atoms** — 2.20% to 4.57%, without exception.

    The split is at size, not at `fas2`. That is a sharper claim than the one it
    replaces and it is pinned the same way, in both directions: a change making
    debye uniformly closest fails here, and so does one making it uniformly
    furthest. Neither is a regression by itself; neither should land without
    somebody saying which story is now true.

    Among the proteins APBS is nearest the boundary-element answer on six of
    seven, `barnase-molecular` being DelPhi's. That third-place detail is
    asserted too, because a change that reordered the *other* two codes while
    leaving debye last would otherwise pass unread.
    """
    sizes = {case.name: case.structure().n_atoms for case in MANIFEST}
    ranked = {
        name: sorted(GRID_BACKENDS, key=lambda b: CROSS_FAMILY_DEVIATION[(b, name)])
        for _, name in CROSS_FAMILY_DEVIATION
    }

    small = {n: r for n, r in ranked.items() if sizes[n] <= 260}
    proteins = {n: r for n, r in ranked.items() if sizes[n] >= 906}
    assert len(small) == 5 and len(proteins) == 7, (len(small), len(proteins))

    assert {r[0] for r in small.values()} == {"debye"}, small
    assert {r[-1] for r in proteins.values()} == {"debye"}, proteins
    assert {n for n, r in proteins.items() if r[0] == "delphi"} == {"barnase-molecular"}


def test_the_expensive_boundary_element_recordings_are_present():
    """Two protein-scale meshes `pytest` deliberately does not re-solve.

    Eight minutes of meshing between them, so they are verified on demand rather
    than per push — which makes "too slow to check here" one step from "quietly
    absent", the exact shape of the bug that let the DelPhi tier skip every test
    while CI stayed green. Their presence is checked here instead.
    """
    recorded = {name: surface for name, _, surface in paired_cases(ROOT_BACKEND, "tabipb")}

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

# kJ/mol per atom, above which a recorded energy is wrong rather than large.
# Measured at both ends: 914 is the worst legitimate density in the corpus (a
# divalent Born ion) and 108,151 is the bug this guards (acetate read on shifted
# columns). This is their geometric midpoint.
PER_ATOM_ABSURD = 10_000.0


def test_the_two_grid_codes_agree_where_they_can_be_compared():
    """The third reference, and the one that anchors the Born ion.

    DelPhi answers the closed form to -228.609 kJ/mol against -228.611 exactly,
    where APBS is 2.36% out at the same nominal resolution — so this tier is not
    a worse copy of APBS, it is a second opinion that happens to be sharper on
    the one case with an analytic answer.
    """
    recorded = paired_cases(ROOT_BACKEND, "delphi")
    # Set equality against the directory, not `>= 35`. The floor this replaces
    # was written when 35 was the real count; the directory holds 58 now, so it
    # had stopped guarding twenty-three of them and would have let any of those
    # be deleted without a word. A count floor under a set that grows is a guard
    # with a shelf life, and ROADMAP.md section 12 records this repository
    # hitting the same shape more than once.
    assert {name for name, _, _ in recorded} == {
        path.stem for path in corpus_dir_for("delphi").glob("*.json")
    }

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
    last digit. The key below is what the energy is allowed to depend on, so
    those cases form one group and any *other* pair sharing an answer still
    fails. Which is what the original bug was: two different molecules.

    **Every solvent parameter is in the key, and leaving one out retires a guard
    silently.** A first version omitted temperature, which grouped
    `peptide-molecular` with `peptide-molecular-cold` — cases that differ only
    by 298.15 K against 277 K, and by 0.02 kJ/mol. Had DelPhi regressed to
    ignoring temperature, which is the exact bug that cost a day when it turned
    out to want Celsius, the two would have become bit-identical and this test
    would have called them the same physics and passed.
    """
    energies = {
        name: other["energy_kj_mol"] for name, _, other in paired_cases(ROOT_BACKEND, "delphi")
    }
    by_case = {c.name: c for c in MANIFEST}

    def physics_key(name: str) -> tuple[Any, ...]:
        case = by_case[name]
        structure = case.structure()
        return (
            structure.coords.tobytes(),
            np.abs(structure.charges).tobytes(),
            structure.radii.tobytes(),
            case.solvent.solute_dielectric,
            case.solvent.solvent_dielectric,
            case.solvent.ionic_strength,
            case.solvent.temperature,
            case.solvent.ion_radius,
            case.solvent.surface_radius,
        )

    seen: dict[float, str] = {}
    for name, energy in energies.items():
        clash = seen.get(energy)
        if clash is not None:
            assert physics_key(clash) == physics_key(name), (
                f"{name} and {clash} are different structures with one answer: {energy}"
            )
        seen[energy] = name

    # Per atom, not absolute. The bound is here to catch an energy wrong by
    # orders of magnitude, and "absurd" scales with the solute: the flat
    # -20,000 this used to carry was set when the corpus stopped at 8,279 atoms,
    # and a 1,156-residue protein legitimately reads -38,701. Widening the flat
    # bound to fit it would have loosened the guard for every small molecule at
    # once — including acetate, which is the case the guard exists for.
    #
    # Both ends measured rather than chosen. Across every recording the worst
    # legitimate density is 914 kJ/mol per atom (a divalent Born ion, where the
    # energy goes as q^2 on a single atom) and the gentlest is 1.2 (a protein).
    # The bug is 108,151 per atom — acetate at -865,205 where the answer is
    # -196.90. The bound below is the geometric midpoint of those two, so it
    # sits ~11x above anything real and ~11x below the failure, and it still
    # admits a hypothetical hexavalent ion at ~8,200.
    for name, energy in energies.items():
        atoms = len(by_case[name].structure().coords)
        assert -PER_ATOM_ABSURD * atoms < energy < 0, (
            f"{name}: {energy} over {atoms} atoms is {energy / atoms:.0f} per atom"
        )


def test_the_field_check_samples_every_cubic_symmetry_class():
    """One ray is not a measurement of a staircase, and this is what says so.

    A sphere discretized on a Cartesian grid has the grid's cubic symmetry, so
    its error varies over three direction classes — <100> along the axes, <110>
    through the face diagonals, <111> through the body diagonals. The check
    shipped sampling `centre + r*x_hat` alone, which for a spherically symmetric
    problem reads as an arbitrary-but-harmless choice and is not: it recorded
    APBS's worst case and understated DelPhi's by 2.6x.
    """
    classes = {
        tuple(sorted(abs(round(float(c), 6)) for c in direction))
        for _, direction in FIELD_DIRECTIONS
    }
    assert len(classes) == 3, f"expected three symmetry classes, got {sorted(classes)}"

    # Each class carries more than one representative, so an axis swap or a sign
    # error in the sampling has somewhere to show up.
    for _, direction in FIELD_DIRECTIONS:
        assert float(np.linalg.norm(direction)) == pytest.approx(1.0)
    assert len(FIELD_DIRECTIONS) >= 6
    assert any(float(d[0]) < 0 for _, d in FIELD_DIRECTIONS), "no sign-flipped direction"


def test_a_single_ray_would_have_understated_the_recorded_error():
    """The finding itself, held against the recordings.

    If the sampling ever collapses back to one direction — or to a set that
    misses the diagonals — the recorded worst error falls and this fails. Stated
    as a ratio rather than a named case so it survives the tolerances moving.
    """
    ratios = {}
    for case in MANIFEST:
        if case.analytic_field is None:
            continue
        for label, directory in (("apbs", CORPUS_DIR), ("delphicpp", DELPHI_DIRECTORY)):
            path = directory / f"{case.name}.json"
            if not path.is_file():
                continue
            field = json.loads(path.read_text())["analytic_field"]
            errors = np.array(field["relative_errors"])  # [radius][direction]
            axis = field["directions"].index("+x")
            ratios[f"{label}/{case.name}"] = float(
                field["max_relative_error"] / errors[:, axis].max()
            )

    assert ratios, "no field recordings found"
    worst = max(ratios.values())
    assert worst >= 1.5, (
        "no recorded case is materially worse off-axis than along +x, which is what "
        f"makes a single-ray sample wrong; best ratio seen was {worst:.2f}x. Ratios: "
        f"{ {k: round(v, 2) for k, v in ratios.items() if v > 1.05} }"
    )


def test_the_axis_directions_agree_for_a_centred_sphere():
    """Grid centring, which nothing else here would catch — and it needs no binary.

    A lone sphere sits at the centre of its own bounding box, and every backend
    builds an odd-sized grid, so the atom lands exactly on a node and the four
    <100> samples must read the same value. They agree to under 0.001 percentage
    points in the recordings; a grid centred half a cell off would break this
    while leaving the *worst* error, and so every tolerance, untouched.

    Solved with debye rather than read from a file, because this is a claim
    about the geometry rather than about a recording — and because debye needs
    nothing installed, so it runs on the bare leg where the recordings' own
    backends do not.
    """
    case = next(c for c in MANIFEST if c.name == "born-ion-vdw")
    summary = _analytic_field_summary(case, DebyeSolver().solve(case.request()))
    assert summary is not None

    errors = np.array(summary["relative_errors"])  # [radius][direction]
    axis_columns = [summary["directions"].index(d) for d in ("+x", "+y", "+z", "-x")]
    for radius, row in zip(summary["radii_a"], errors[:, axis_columns], strict=True):
        assert float(np.ptp(row)) < 1e-4, (
            f"the <100> samples disagree by {np.ptp(row):.2e} at r = {radius:.4f} A, "
            "which says the grid is not centred on the sphere"
        )
