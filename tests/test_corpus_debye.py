"""The debye tier of the corpus, on a machine with no binaries at all.

Every other recorded tier needs something installed to re-solve: `tests/corpus/`
needs APBS, `tests/corpus/delphi/` a compiler, `tests/corpus/tabipb/` a mesher
as well. debye needs nothing, which makes it **the one tier that can be
re-solved on every leg of CI** — and until this file it was the one tier nothing
re-solved at all. `test_every_cross_backend_recording_answers_a_manifest_case`
checks filenames; the CI corpus step runs the default backend, which is APBS.
So all 56 recordings could have gone stale silently, which is the same
"a skipped tier and a passing tier look identical" trap `.github/workflows/ci.yml`
already records being burned by.

Nothing here is marked and nothing here can skip, for the reason
`tests/test_debye_solver.py` gives: a suite that quietly skipped debye on a
binary-free machine would be testing the opposite of debye's central claim.

Scoped to the fast tier so the edit-test loop stays a loop. The standard and
full tiers are `sashimi corpus verify --backend debye --tier standard|full`,
which is what M5 was graded on and what a release should run.
"""

from __future__ import annotations

import json

import pytest

from sashimi.corpus import (
    MANIFEST,
    Case,
    CaseTier,
    Tolerances,
    cases_for_tier,
    corpus_dir_for,
    load_summary,
    verify_case,
)
from sashimi.debye import DebyeSolver
from sashimi.protocol import SolverFamily

DEBYE_DIR = corpus_dir_for("debye")

# The recorded fast-tier cases. Derived by asking which recordings exist rather
# than by listing names, so a case recorded later joins without an edit here —
# and `test_every_case_debye_can_answer_is_recorded` is the other half, which
# fails if one it *could* answer is missing.
FAST_RECORDED = [
    case for case in cases_for_tier(CaseTier.FAST) if (DEBYE_DIR / f"{case.name}.json").is_file()
]


def test_the_debye_tier_is_not_empty():
    """Whatever else is true, something here runs."""
    assert FAST_RECORDED, f"no debye recordings in {DEBYE_DIR}"


@pytest.mark.parametrize("case", FAST_RECORDED, ids=lambda c: c.name)
def test_debye_reproduces_its_recorded_answer(case: Case):
    """Re-solve and compare, with no binary installed.

    This is `sashimi corpus verify --backend debye --tier fast` as a test, which
    is M5's exit criterion. The CLI is what a release runs; this is what every
    push runs, on every platform, including the leg with nothing installed.
    """
    found = verify_case(
        DebyeSolver(),
        case,
        load_summary(case, backend="debye"),
        Tolerances(),
        SolverFamily.FINITE_DIFFERENCE,
    )
    assert found == [], f"{case.name}: {'; '.join(str(item) for item in found)}"


def test_every_case_debye_can_answer_is_recorded():
    """A gap in coverage, told apart from a principled refusal.

    debye declines `smoothed-molecular` and `gaussian` on purpose, so most of
    the corpus has no debye recording and never will. What *would* be a gap is a
    sharp-boundary case with no recording, and the two look identical from a
    file listing. Asked of the solver's own supported set, so this cannot drift
    from what it actually builds.
    """
    from sashimi.debye.options import SUPPORTED_SURFACES  # noqa: PLC0415

    missing = [
        case.name
        for case in MANIFEST
        if case.solvent.surface_model in SUPPORTED_SURFACES
        and not (DEBYE_DIR / f"{case.name}.json").is_file()
    ]
    assert not missing, (
        f"debye builds these surfaces but has no recording for: {', '.join(missing)}. "
        "Record them with `sashimi corpus build --backend debye --tier full`."
    )


def test_no_recording_carries_a_tolerance_the_manifest_has_since_changed():
    """The failure mode that shipped in this milestone and nothing caught.

    `build_case` writes the tolerance *that applies to this backend* into each
    summary, calling it "the one a reader of the file needs". Adding a
    `debye_rtol` to a case already recorded leaves that field behind, and
    `corpus build` skips existing files without `--force`, so the recording
    keeps the old number. `verify` grades the freshly-solved answer rather than
    the recording, so nothing goes red — the artifact just quietly disagrees
    with the manifest, and three of them shipped reading
    `relative_error 0.01504` beside `rtol 0.01`.

    Covers every backend's recordings rather than debye's, since the mechanism
    is not debye's.
    """
    from sashimi import backends  # noqa: PLC0415

    stale = []
    for name in backends.names():
        directory = corpus_dir_for(name)
        for case in MANIFEST:
            path = directory / f"{case.name}.json"
            if not path.is_file():
                continue
            summary = json.loads(path.read_text())
            label = summary.get("backend", name)
            for key, reference in (
                ("analytic", case.analytic),
                ("analytic_field", case.analytic_field),
            ):
                block = summary.get(key)
                if reference is None or not block or "rtol" not in block:
                    continue
                expected = reference.rtol_for(label)
                if block["rtol"] != expected:
                    stale.append(f"{name}/{case.name} {key}: {block['rtol']} != {expected}")

    assert not stale, (
        "recording(s) carry a tolerance the manifest has since changed — re-record "
        "with `--force`:\n  " + "\n  ".join(stale)
    )


# --- debye against the two reference-tier backends ---------------------------
#
# Everything above compares debye against its own past, which a solver that was
# wrong on day one passes forever. Twenty of these 58 cases carry a gated closed
# form and six have a TABI-PB counterpart; the other 32 had nothing but their
# own recording until this section, and none of the eighteen above 906 atoms had
# an independent check of any kind.
#
# They did have referees all along, unrecorded: every case here has an APBS
# recording in `tests/corpus/` and a DelPhi C++ one in `tests/corpus/delphi/`,
# both of the same question and both reporting `POLAR_SOLVATION`. The
# relationship between the three files is a fact about the physics rather than
# about any one recording, it needs no binary to check, and — like
# `tests/test_corpus_gb.py`'s equivalent — it is the one check that survives
# `sashimi corpus build --backend debye --force`. A rebuild makes every test
# above agree with the new numbers and makes these fail by however far debye
# moved.
#
# **What this section cannot do, stated here because the temptation is
# structural.** debye, APBS and DelPhi all assign the dielectric hard, from the
# face centre's own side of the surface. `sashimi.debye.dielectric` says so in
# its own module docstring and draws the conclusion: "Reaching for APBS or
# DelPhi here measures a shared bias." So these numbers pin *where debye sits*
# and catch it moving, and they are **not** evidence for or against any change
# to the interface treatment — the sub-cell dielectric ramp above all. A ramp
# that walked these deviations toward zero would be as likely to have found the
# shared bias as to have found the truth. Grading that needs a reference which
# does not itself discretize a volumetric dielectric: a closed form, TABI-PB, or
# a converged refinement study.
#
# Unlike `GB_DEVIATION`, this is reference tier against reference tier — debye
# is `AccuracyTier.REFERENCE` (`sashimi.backends`), not an approximation being
# priced. Three codes that agree on the physics and differ in discretization
# should land close, and the interesting content is in the pattern of where they
# do not.


def deviation(name: str, backend: str) -> float:
    """Where debye sits relative to a reference-tier backend, from two recordings.

    **Signed**, where `tests/test_corpus_gb.py`'s namesake takes an absolute
    value. The sign is the finding here, and an unsigned table would record the
    same 58 magnitudes and lose it: debye is more negative than DelPhi on **58
    of 58 cases, without exception**, and more negative than APBS on 39 of 58.
    The nineteen it is not are the one- and two-atom synthetic geometries, the
    two smallest molecules, and `ion-protein-complex-vdw` — which agrees with
    APBS to 1.1e-07, the closest pair in the corpus. Every real all-atom
    structure is negative against both.

    So debye does not scatter around the incumbents; it sits below them, and the
    ordering `test_the_three_codes_bracket` asserts is the sharper form of that.
    """
    case = next(c for c in MANIFEST if c.name == name)
    reference = float(load_summary(case, backend=backend)["energy_kj_mol"])
    subject = float(load_summary(case, backend="debye")["energy_kj_mol"])
    return (subject - reference) / abs(reference)


# `(deviation from APBS 3.4.1, deviation from DelPhi C++ 8.6)`, measured
# 2026-08-23 from the recordings checked in beside this file. Not a tolerance
# debye is required to meet — a record of what it does, so a move in any of the
# three tiers surfaces as a named case rather than as a diff.
#
# Grouped by what the grouping shows:
#
# - **The Born ladder is about `a/h`, not about debye.** The radius series runs
#   -5.56% at 1 A to -0.18% at 6 A against APBS, and the `-fine` and `r6` rungs
#   are the closest in the whole corpus. That is a sphere resolved by a handful
#   of cells at one end and by dozens at the other; all three codes converge and
#   the coarse end is where they have not yet.
# - **Kirkwood's `07` rungs jump to +2.4% on both surfaces** where `03` and `05`
#   sit inside 0.35%. The off-centre charge nears the boundary and the three
#   discretizations stop agreeing about a charge that is nearly on the interface
#   — the same place `kirkwood-*-09` is recorded ungated for.
# - **Small molecules are a few kJ/mol on a small number**: methanol is +3.7%
#   against APBS on an energy of about -20 kJ/mol, which is a fraction of a
#   kJ/mol of disagreement.
# - **The real structures bracket, 18 of 18 above 906 atoms**:
#   `E_delphi > E_apbs > E_debye`, without exception. Three independent codes
#   ordering themselves the same way on every large solute is a statement about
#   their discretizations, and it is why `test_the_three_codes_bracket` below is
#   an ordering test and not a magnitude one.
# - **The two surfaces disagree in opposite directions**, and this is the
#   sharpest thing in the table. Over the 12 pairs where rolling the probe is
#   worth more than 1%, `|debye - DelPhi|` is larger on `van-der-waals` in
#   **12 of 12**, while `|debye - APBS|` is larger on `molecular` in **11 of
#   12**. debye's surface construction agrees better with APBS's where the probe
#   matters most and worse with DelPhi's. Serum albumin is the sole exception,
#   and it is the sole exception to everything else here too.
# - **Serum albumin is a different case and is pinned separately below.**
DEBYE_DEVIATION: dict[str, tuple[float, float]] = {
    # The Born sphere: one atom, and the ladder is in `a/h`.
    "born-ion-molecular": (+0.00764, -0.01576),
    "born-ion-molecular-divalent": (+0.00764, -0.01576),
    "born-ion-molecular-eps2": (+0.00755, -0.01482),
    "born-ion-molecular-eps4": (+0.00738, -0.01361),
    "born-ion-molecular-fine": (-0.00230, -0.00854),
    "born-ion-molecular-high-salt": (+0.00756, -0.01622),
    "born-ion-molecular-negative": (+0.00764, -0.01576),
    "born-ion-molecular-r1": (-0.05557, -0.01228),
    "born-ion-molecular-r2": (-0.02421, -0.04731),
    "born-ion-molecular-r4": (-0.01079, -0.01497),
    "born-ion-molecular-r6": (-0.00183, -0.00852),
    "born-ion-molecular-salt": (+0.00759, -0.01659),
    "born-ion-vdw": (+0.00764, -0.01576),
    "born-ion-vdw-fine": (-0.00066, -0.00854),
    "born-ion-vdw-high-salt": (+0.00760, -0.01618),
    "born-ion-vdw-r1": (-0.05557, -0.01228),
    "born-ion-vdw-r6": (+0.00004, -0.00852),
    "born-ion-vdw-salt": (+0.00761, -0.01657),
    # Kirkwood: the charge moves off centre, and `07` is where that starts to
    # cost. Both `09` rungs are recorded ungated in `sashimi.corpus`.
    "kirkwood-molecular-03": (-0.00202, -0.00949),
    "kirkwood-molecular-05": (-0.00349, -0.01046),
    "kirkwood-molecular-07": (+0.02381, -0.00909),
    "kirkwood-molecular-09": (+0.01427, -0.03827),
    "kirkwood-vdw-03": (+0.00036, -0.00949),
    "kirkwood-vdw-05": (-0.00015, -0.01046),
    "kirkwood-vdw-07": (+0.02472, -0.00909),
    "kirkwood-vdw-09": (+0.01433, -0.03827),
    # Small molecules, where a fraction of a kJ/mol is a large fraction.
    "methoxide-molecular": (+0.00230, -0.01200),
    "methanol-molecular": (+0.03673, -0.02197),
    "acetate-molecular": (-0.00774, -0.01577),
    "acetic-acid-molecular": (-0.02148, -0.04213),
    "aspartate-residue-molecular": (-0.00624, -0.01689),
    # The dipeptide, across salt and temperature. Salt moves these by 0.14 pp
    # against APBS and by 0.01 pp against DelPhi, so what the arm grades is the
    # screening and not the surface.
    "peptide-molecular": (-0.02069, -0.04396),
    "peptide-molecular-cold": (-0.02069, -0.04390),
    "peptide-molecular-high-salt": (-0.02070, -0.04386),
    "peptide-molecular-no-salt": (-0.02208, -0.04384),
    "peptide-vdw": (-0.00409, -0.04529),
    "peptide-vdw-high-salt": (-0.00407, -0.04525),
    "peptide-vdw-no-salt": (-0.00417, -0.04522),
    # Real structures, 260 to 2,482 atoms. Against APBS these sit inside 3.2%
    # and against DelPhi inside 8.9%, and the second number is the larger on
    # every one of them.
    "ion-protein-complex-molecular": (-0.00146, -0.00862),
    "ion-protein-complex-vdw": (+0.00000, -0.00784),
    "fas2-molecular": (-0.01525, -0.04408),
    "fas2-vdw": (-0.00481, -0.06087),
    "barstar-molecular": (-0.01889, -0.03941),
    "barstar-vdw": (-0.00875, -0.06019),
    "fkbp-apo-molecular": (-0.02230, -0.05650),
    "fkbp-apo-vdw": (-0.01482, -0.08852),
    "fkbp-dmso-molecular": (-0.02181, -0.05608),
    "fkbp-dmso-vdw": (-0.01501, -0.08868),
    "barnase-molecular": (-0.03118, -0.05779),
    "barnase-vdw": (-0.01088, -0.07947),
    "lysozyme-molecular": (-0.02713, -0.04881),
    "lysozyme-vdw": (-0.02034, -0.08323),
    "protein-rna-molecular": (-0.02064, -0.04216),
    "protein-rna-vdw": (-0.01267, -0.05244),
    "hca-molecular": (-0.01839, -0.04822),
    "hca-vdw": (-0.00870, -0.06911),
    # Serum albumin, 18,242 atoms — the corpus's largest solute and the widest
    # disagreement in it, by a factor of two over the next. See
    # `test_the_corpus_largest_solute_is_its_widest_disagreement`.
    "serum-albumin": (-0.06639, -0.10405),
    "serum-albumin-vdw": (-0.09092, -0.17513),
}

# Both sides of every comparison are checked-in recordings, so each deviation is
# exact arithmetic on two constants and this band only has to absorb float
# formatting. That also makes it platform-independent, which an assertion on a
# freshly-solved number could not be: ROADMAP.md section 12 records the same
# debye expression returning -218.62772042354118 on macOS and ...138 on
# linux/amd64. Anything larger than this band is a real move.
DEVIATION_BAND = 0.0005

REFEREES = ("apbs", "delphi")


def test_every_debye_recording_has_a_recorded_deviation():
    """Set equality, so a case added to one tier and not the other is caught.

    Deliberately not `len(...) >= N`. A count floor under a growing set stops
    guarding the moment the set outgrows it, which
    `tests/test_corpus_manifest.py` has an instance of and ROADMAP.md section 12
    records as a repeat defect in this repository.
    """
    recorded = {path.stem for path in DEBYE_DIR.glob("*.json")}
    assert set(DEBYE_DEVIATION) == recorded


@pytest.mark.parametrize("backend", REFEREES)
def test_every_debye_recording_has_two_referees(backend: str):
    """The files this section reads, and the identity of what produced them.

    The backend string is asserted rather than assumed, and the reason is not
    bookkeeping. `sashimi.delphi` ships two flavours into one directory, and
    `sashimi/delphi/run.py` reports `POLAR_SOLVATION` for the C++ build against
    `REACTION_FIELD` for pyDelPhi — different quantities, both plausible, both
    landing in `tests/corpus/delphi/`. A pyDelPhi recording filed there would
    leave the table below silently comparing two different things.

    The APBS half is the same argument for a different reason: `tests/corpus/`
    is the pin ROADMAP.md section 13 leans on now that APBS is not lockfile-held,
    and `tests/test_corpus.py` asserts it under `pytest.mark.apbs` — which
    *skips* on the one CI leg where debye's whole tier runs.
    """
    expected = {"apbs": "apbs-3.4.1", "delphi": "delphicpp-8.6"}[backend]
    directory = corpus_dir_for(backend)

    wrong = []
    for name in sorted(DEBYE_DEVIATION):
        path = directory / f"{name}.json"
        if not path.is_file():
            wrong.append(f"{name}: no {backend} recording at {path}")
            continue
        found = json.loads(path.read_text()).get("backend")
        if found != expected:
            wrong.append(f"{name}: {found!r} != {expected!r}")

    assert not wrong, "\n  ".join(["referee recordings are not what this table read:", *wrong])


@pytest.mark.parametrize("name", sorted(DEBYE_DEVIATION), ids=lambda n: n)
def test_debye_sits_where_it_was_measured(name: str):
    """debye against both reference-tier backends, case by case.

    Two-sided: a deviation that *shrank* fails as loudly as one that grew. That
    is deliberate against a shared-bias referee — moving toward APBS is not
    evidence of an improvement here, so it is not something this test should be
    willing to accept quietly.
    """
    apbs, delphi = DEBYE_DEVIATION[name]

    assert deviation(name, "apbs") == pytest.approx(apbs, abs=DEVIATION_BAND)
    assert deviation(name, "delphi") == pytest.approx(delphi, abs=DEVIATION_BAND)


def test_the_three_codes_bracket():
    """`E_delphi > E_apbs > E_debye`, on every case above 906 atoms.

    Eighteen cases, nine structures, two surfaces each, and no exception. Three
    independently written codes ordering themselves the same way on every large
    solute is a statement about their discretizations rather than about any one
    recording, and it is the strongest thing this section can say without a
    reference — which is also why it is an ordering test and not a magnitude
    one. A magnitude bar here would be the shared-bias trap the section header
    warns about; an ordering is not something a tolerance can be widened to
    admit.

    Reads energies rather than deviations so it fails on the ordering itself,
    not on a derived quantity that could preserve it while both moved.
    """
    big = [
        case
        for case in MANIFEST
        if case.name in DEBYE_DEVIATION and case.structure().n_atoms >= 906
    ]
    assert len(big) == 18, [c.name for c in big]

    out_of_order = []
    for case in big:
        energies = {
            backend: float(load_summary(case, backend=backend)["energy_kj_mol"])
            for backend in ("delphi", "apbs", "debye")
        }
        if not energies["delphi"] > energies["apbs"] > energies["debye"]:
            out_of_order.append(f"{case.name}: {energies}")

    assert not out_of_order, "\n  ".join(["the bracket broke:", *out_of_order])


def test_the_two_surfaces_disagree_in_opposite_directions():
    """The sharpest pattern in the table, and it points at the surface.

    Over the twelve pairs where rolling the probe moves the energy by more than
    1% — `sashimi.corpus.probe_worth`, the quantity M4 was gated on — the
    disagreement with DelPhi is larger on `van-der-waals` in **12 of 12**, and
    the disagreement with APBS is larger on `molecular` in **11 of 12**.

    Two codes, two surfaces, and the asymmetry runs the opposite way for each.
    That is not something a uniform bias in any one solver produces, and it says
    the residual is in how the three construct the boundary rather than in how
    they solve on it.

    `serum-albumin` is the sole exception on the APBS side and is asserted here
    as the exception rather than excluded, so that a change making it ordinary
    fails and gets read.
    """
    from sashimi.corpus import probe_worth  # noqa: PLC0415

    pairs = []
    for case in MANIFEST:
        if case.name not in DEBYE_DEVIATION:
            continue
        # `-molecular` is not always a suffix: `peptide-molecular-no-salt` pairs
        # with `peptide-vdw-no-salt`, and `serum-albumin` carries no token at
        # all. Replacing rather than stripping is what makes both work, and
        # getting it wrong silently drops pairs instead of failing.
        partner = (
            case.name.replace("-molecular", "-vdw")
            if "-molecular" in case.name
            else f"{case.name}-vdw"
        )
        if case.solvent.surface_model.value != "molecular" or partner not in DEBYE_DEVIATION:
            continue
        summaries = {
            key: load_summary(next(c for c in MANIFEST if c.name == key), backend="debye")
            for key in (case.name, partner)
        }
        if abs(probe_worth(summaries[partner], summaries[case.name])) > 1.0:
            pairs.append((case.name, partner))

    assert len(pairs) == 12, pairs

    delphi_wider_on_vdw = [
        m for m, v in pairs if abs(DEBYE_DEVIATION[v][1]) > abs(DEBYE_DEVIATION[m][1])
    ]
    apbs_wider_on_molecular = [
        m for m, v in pairs if abs(DEBYE_DEVIATION[m][0]) > abs(DEBYE_DEVIATION[v][0])
    ]

    assert len(delphi_wider_on_vdw) == 12
    assert sorted({m for m, _ in pairs} - set(apbs_wider_on_molecular)) == ["serum-albumin"]


def test_the_corpus_largest_solute_is_its_widest_disagreement():
    """Serum albumin, pinned deliberately because it is the outlier.

    18,242 atoms, and `serum-albumin-vdw` is the argmax of |deviation| against
    *both* referees — 9.09% against APBS and 17.51% against DelPhi, where the
    next widest real structure is 2.03% and 8.87%. It is also the one case that
    breaks the surface asymmetry above.

    Some of this is resolution rather than physics: debye and APBS do not solve
    on the same lattice, and `h_debye/h_apbs` is 1.1828 here — the largest ratio
    in the corpus. Putting debye on a matched lattice (resolution 0.85 A, the
    corpus padding) narrows -6.639% to -4.687% and -9.092% to -6.778%, so the
    lattice is worth 25-29% of it and 71-75% survives the control. That
    measurement is recorded here rather than run, because running it costs an
    albumin solve at 2.7x the point count.

    Pinned so that a change which fixes this fails and gets explained. Nothing
    here says which of the three codes is right — no reference reaches 18,242
    atoms in this corpus, which is what ROADMAP.md section 12 means by the
    referee gap not being closed at the top end.
    """
    assert max(DEBYE_DEVIATION, key=lambda n: abs(DEBYE_DEVIATION[n][0])) == "serum-albumin-vdw"
    assert max(DEBYE_DEVIATION, key=lambda n: abs(DEBYE_DEVIATION[n][1])) == "serum-albumin-vdw"

    others = [n for n in DEBYE_DEVIATION if not n.startswith("serum-albumin")]
    assert abs(DEBYE_DEVIATION["serum-albumin-vdw"][0]) > 1.6 * max(
        abs(DEBYE_DEVIATION[n][0]) for n in others
    )
