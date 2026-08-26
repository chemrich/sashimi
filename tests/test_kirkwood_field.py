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
