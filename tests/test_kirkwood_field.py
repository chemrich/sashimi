"""Every recorded Kirkwood field, against the closed form. No binary required.

The corpus holds 28 Kirkwood recordings — four charge offsets, two surface
models, three backends — and each carries 50 probe potentials. Until
`kirkwood_potential` existed they had only ever been compared with *themselves*,
which is the failure mode section 7 names: a backend wrong in the field from its
first build stays wrong and passes forever.

This grades them against truth, and it costs nothing. The recordings are already
on disk; no solver runs here, so it is one of the few checks in this project that
says a shipped number is *right* rather than *unchanged*, and it says it in the
binary-free tier where most of CI lives.

**Only probes clear of the dielectric interface are graded**, by the same rule
`AnalyticField` documents: phi is continuous at the boundary but its normal
derivative is not, so a sample inside a cell that straddles it is an ill-posed
question about a grid rather than a solver defect. Two probes of the 50 fall
inside that band on the finest cases and are dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sashimi.analytic import kirkwood_potential
from sashimi.corpus import MANIFEST

CORPUS = Path(__file__).resolve().parent / "corpus"
RADIUS = 3.0  # every Kirkwood case is a 3 A sphere
CELLS_CLEAR = 2  # `sashimi.field.MIN_CELLS_OUT`

# Measured across all 28 recordings; the worst single sample is 3.62% and the
# worst median 1.17%, both at d/a = 0.9 where the charge is nearest the boundary.
# The bars sit above those without being decorative: a unit error is a factor of
# 2.5 and a geometry error tens of percent, so anything this test exists to catch
# clears them by an order of magnitude.
MEDIAN_BAR = 0.02
WORST_BAR = 0.06

RECORDINGS = sorted(
    (p for p in CORPUS.rglob("kirkwood*.json")),
    key=lambda p: (p.parent.name, p.name),
)


def _graded(path: Path):
    """Recorded and exact potentials at the probes clear of the interface."""
    recording = json.loads(path.read_text())
    resolved = recording["resolved_parameters"]
    offset = RADIUS * int(recording["name"].split("-")[-1]) / 10.0
    spacing = max(recording["geometry"]["spacing"])

    points = np.array(recording["probes"]["points"])
    recorded = np.array(recording["probes"]["values_kT_e"])
    radii = np.linalg.norm(points, axis=1)
    clear = radii > RADIUS + CELLS_CLEAR * spacing

    # The charge sits on +x, so cos(theta) is the x-cosine. `kirkwood_pqr` puts
    # the sphere at the origin and this reads the frame back rather than assuming
    # it: a translated structure would show up as a large error, not a silent one.
    cosines = points[clear, 0] / radii[clear]
    exact = np.array(
        [
            kirkwood_potential(
                r,
                cos_theta,
                RADIUS,
                offset,
                1.0,
                solute_dielectric=resolved["solute_dielectric"],
                solvent_dielectric=resolved["solvent_dielectric"],
                temperature=resolved["temperature"],
            )
            for r, cos_theta in zip(radii[clear], cosines, strict=True)
        ]
    )
    return recorded[clear], exact


def test_there_are_recordings_to_grade():
    """The glob is the test's subject, so an empty one must fail rather than pass.

    Every `all()`-shaped assertion below reads as agreement over no samples.
    """
    assert len(RECORDINGS) == 28


@pytest.mark.parametrize("path", RECORDINGS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_a_recorded_kirkwood_field_matches_the_closed_form(path):
    recorded, exact = _graded(path)
    assert len(recorded) > 40, f"only {len(recorded)} probes cleared the interface"

    errors = np.abs(recorded - exact) / np.abs(exact)
    assert float(np.median(errors)) < MEDIAN_BAR
    assert float(np.max(errors)) < WORST_BAR


def test_the_field_gets_harder_as_the_charge_nears_the_boundary():
    """A shape check, not a threshold: the error must *order* by d/a.

    A bar that every rung passes says only that nothing is catastrophically
    wrong. This says the errors behave like a discretization of this geometry —
    the multipole content grows with the offset, and the terms that grow are the
    ones a lattice resolves worst. A reference that ignored the offset entirely
    would sit at a flat error across the four rungs and pass every bar above.
    """
    for backend in ("", "debye", "delphi"):
        medians = []
        for rung in ("03", "05", "07", "09"):
            name = f"kirkwood-vdw-{rung}" if backend else f"kirkwood-{rung}"
            path = CORPUS / backend / f"{name}.json"
            recorded, exact = _graded(path)
            medians.append(float(np.median(np.abs(recorded - exact) / np.abs(exact))))
        assert medians == sorted(medians), f"{backend or 'apbs'}: {medians}"


class TestWhyDebyeAndDelphiAgree:
    """The five-figure agreement is a shared lattice, not a shared discretization.

    ROADMAP.md section 12 recorded the observation and did not explain it: debye
    and the DelPhi backend agree on the Kirkwood field to 2.7e-5 relative while
    APBS sits ~0.4% from both, and "two codes sharing no source agreeing that
    closely is either a shared discretization convention or a shared ancestry in
    one" — which matters because section 12's referee work assumes they are
    independent.

    They are independent. Two candidate *physics* explanations were measured and
    both failed: APBS at `chgm spl0`, which is the trilinear assignment debye
    uses, moves **away** (20.96% to 29.55%), and APBS at `bcfl mdh`, which is
    debye's boundary condition, moves 0.006%. What debye and DelPhi actually
    share is the box: same shape, same spacing and the same origin, so the
    lattice falls identically against the charge. The near-field observable is
    phase-dominated, so that is sufficient on its own.

    These assertions are pure grid arithmetic — no solver, no binary — which is
    the point: the dependency is structural and can be checked anywhere.
    `studies/lattice_phase/debye_delphi_agreement.py` carries the measurement.
    """

    @staticmethod
    def kirkwood_cases():
        return [c for c in MANIFEST if c.name.startswith("kirkwood-vdw-")]

    def test_debye_and_delphi_resolve_to_the_same_lattice(self):
        """If this ever fails, the agreement above needs re-reading rather than fixing."""
        from sashimi.debye.grid import size_grid as debye_grid  # noqa: PLC0415
        from sashimi.delphi.grid import size_grid as delphi_grid  # noqa: PLC0415

        cases = self.kirkwood_cases()
        assert cases, "no kirkwood-vdw cases in the manifest"

        differing = []
        for case in cases:
            structure, spec = case.structure(), case.grid
            mine, theirs = debye_grid(structure, spec), delphi_grid(structure, spec)
            same_shape = mine.shape[0] == theirs.gsize
            same_spacing = mine.spacing[0] == pytest.approx(theirs.spacing[0], rel=1e-9)
            # DelPhi describes its box by centre and side, so the origin is derived.
            theirs_origin = np.asarray(theirs.center) - theirs.box_length / 2.0
            same_origin = np.allclose(np.asarray(mine.origin), theirs_origin)
            if not (same_shape and same_spacing and same_origin):
                differing.append(
                    f"{case.name}: debye {mine.shape[0]}^3 h={mine.spacing[0]:.6f} "
                    f"o={mine.origin} vs delphi {theirs.gsize}^3 "
                    f"h={theirs.spacing[0]:.6f} o={theirs_origin}"
                )

        assert not differing, "\n  ".join(
            ["debye and DelPhi no longer share a lattice:", *differing]
        )

    def test_the_point_count_rules_are_what_make_them_coincide(self):
        """debye needs `8m + 1` and DelPhi needs an odd `gsize`; every `8m + 1` is odd.

        That is the whole coincidence, and it is one-directional: DelPhi can sit
        on point counts debye cannot reach. So the agreement is not guaranteed at
        every resolution, and the study measures two where it breaks — at 0.20 A
        they land on 137 and 131, and at 0.1875 A on 145 and 141, where the worst
        samples are 7.75% and 27.53%. A shared lattice is a fact about the
        corpus's chosen resolutions, not a property of the two codes.
        """
        from sashimi.debye.grid import LATTICE_STEP, _lattice_ceil  # noqa: PLC0415
        from sashimi.delphi.grid import odd_gsize  # noqa: PLC0415

        assert LATTICE_STEP % 2 == 0, "8m + 1 is odd only while the step is even"
        for minimum in (9, 100, 105, 129, 137):
            n = _lattice_ceil(minimum)
            assert (n - 1) % LATTICE_STEP == 0
            assert odd_gsize(n) == n, f"DelPhi cannot sit on debye's {n}"
