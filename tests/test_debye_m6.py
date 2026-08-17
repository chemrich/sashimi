"""M6: the potential field out — what is gated, and what is only recorded.

ROADMAP.md section 12 M6. The milestone has two halves and they need different
treatment, which is the result rather than an inconvenience.

**The DX half is gated**, and the work was removing a circularity rather than
building anything: `write_dx`, `residue_potentials` and debye's `want_potential`
path all predate M6. A round-trip test proves our reader accepts our writer,
which says nothing about a viewer we do not control. The claim worth making is
that our file is *structurally the file those viewers already read* — verified
by diffing our output against one APBS 3.4.1 wrote for the same grid, where
exactly one line differed. `tests/test_dx.py` holds that guard.

**The residue half is met by measurement and recorded rather than gated**,
decided 2026-08-17 by Charlie. On `fas2-molecular` (906 atoms, 63 residues),
every measurement says the same thing: **the disagreement between solvers is the
size of each solver's own grid noise.**

    APBS  against itself, padding 8->11 A   median 0.49 kT/e
    debye against itself, padding 8->11 A   median 0.66 kT/e
    debye against APBS                      median 0.32-0.49 kT/e

    rank, Spearman:  APBS self >= 0.9916   debye self >= 0.9783
                     debye vs APBS 0.9794-0.9842

debye is not failing; the quantity is dominated by discretization. Two things
follow, and they are why there is no gate here:

- **The obvious relational bar has the wrong comparator.** "Agrees with APBS as
  well as APBS agrees with itself" *fails* — 0.9794 < 0.9916 — because a
  cross-solver difference cannot be expected to beat the noise of a solver
  inside it. Against the noisier participant it passes by **0.0011**. A gate
  with that margin reddens for reasons unrelated to debye, which teaches as
  little as a check that cannot fail.
- **Top-N is not gateable at all**: unstable *within* one backend, APBS 9/10 and
  debye 8/10 against themselves across a box change.

So this file pins the relationship the way `test_debye_m3.py` pins its neutral
solute. If debye ever becomes clearly better than the noise — which is what
fractional-volume dielectric averaging is expected to do — the pin fails and
says to revisit the milestone rather than absorbing the improvement silently.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from sashimi import backends
from sashimi.analysis import residue_potentials
from sashimi.corpus import MANIFEST
from sashimi.dx import read_dx, write_dx
from sashimi.protocol import PotentialGrid

CASE = "fas2-molecular"

# debye on `fas2-molecular` at the case's own grid, measured 2026-08-17. The
# ten most negative residues and their means, in order. Recorded at one fixed
# padding: the 8/10 top-N instability above is across *box changes*, so it does
# not apply to re-solving the same question.
RECORDED_TOP_10 = {
    "CYX 565": -9.6241,
    "TYR 604": -8.8136,
    "PRO 599": -8.0149,
    "CYX 596": -7.8840,
    "GLY 587": -7.3324,
    "THR 597": -6.6277,
    "ASP 588": -6.2008,
    "ASP 600": -5.8646,
    "CYX 584": -4.9808,
    "PRO 574": -4.7146,
}

# debye's own box-to-box median, padding 8->11 A, measured 2026-08-17. The
# comparator for "is the cross-backend difference still just noise".
DEBYE_BOX_NOISE_KT_E = 0.6605


def volumetric(result) -> PotentialGrid:
    """The map, narrowed to a volume — which is a real distinction, not a cast.

    A boundary-element backend answers with a `SurfacePotential`, and residue
    means over a triangulation are a different quantity from residue means over
    a grid. M6 is about the volumetric map protean loads, so anything else here
    is a mistake worth failing on.
    """
    potential = result.potential
    assert isinstance(potential, PotentialGrid), (
        f"expected a volume, got {type(potential).__name__}"
    )
    return potential


def case_system(*, want_potential: bool = True):
    case = next(c for c in MANIFEST if c.name == CASE)
    return case.system(want_potential=want_potential)


@pytest.fixture(scope="module")
def debye_residues():
    """One debye solve, shared. It is ~38 s and needs no binary at all."""
    system = case_system()
    solver, family = backends.solver_for("debye")
    result = solver.solve(system.request_for(family))
    return result, {
        r.label: r.value for r in residue_potentials(volumetric(result), system.structure)
    }


def test_debye_writes_a_dx_a_third_party_parser_would_read(tmp_path):
    """The map is the deliverable, so its container is part of being correct.

    A small case rather than the protein: this asserts the container, and the
    container does not know how many atoms produced it. Runs with no binary,
    which is the environment debye exists for.
    """
    case = next(c for c in MANIFEST if c.name == "peptide-vdw")
    solver, family = backends.solver_for("debye")
    result = solver.solve(case.system(want_potential=True).request_for(family))
    potential = volumetric(result)

    path = tmp_path / "debye.dx"
    write_dx(path, potential, comment="debye")

    text = path.read_bytes().decode("ascii")  # raises if a byte is non-ASCII
    nx, ny, nz = potential.values.shape
    # The header grammar APBS emits, which is what makes the file loadable by
    # anything that already loads an APBS map.
    assert f"object 1 class gridpositions counts {nx} {ny} {nz}" in text
    assert f"object 2 class gridconnections counts {nx} {ny} {nz}" in text
    assert f"items {nx * ny * nz} data follows" in text
    assert 'component "data" value 3' in text

    again = read_dx(path)
    np.testing.assert_allclose(again.values, potential.values, rtol=1e-6)
    np.testing.assert_allclose(again.origin, potential.origin)
    np.testing.assert_allclose(again.spacing, potential.spacing)


def test_debye_reproduces_its_recorded_residue_ranking(debye_residues):
    """The field projection a consumer reads, pinned against drift.

    The corpus pins debye's *energy* on this structure, and M0's finding is
    exactly why that is not enough: an energy is one integrated scalar, so a
    solver wrong in the field can reproduce it forever. This pins a different
    projection of the same map — the quantity protean would colour a surface
    with.
    """
    _, values = debye_residues

    ranked = sorted(values, key=lambda label: values[label])[: len(RECORDED_TOP_10)]
    assert set(ranked) == set(RECORDED_TOP_10), (
        "debye's most-negative residues have changed on a fixed grid. Recorded "
        f"{sorted(RECORDED_TOP_10)}, got {sorted(ranked)}."
    )
    for label, recorded in RECORDED_TOP_10.items():
        assert values[label] == pytest.approx(recorded, abs=0.05), (
            f"{label} moved from {recorded} to {values[label]:.4f} kT/e"
        )


@pytest.mark.apbs
def test_the_residue_axis_is_recorded_and_not_judged(debye_residues):
    """Whether the solvers still disagree by no more than the grid does.

    This is the recorded half of M6, held as a live comparison rather than a
    quoted constant so that it keeps meaning something. It asserts the *shape*
    of the finding, in both directions:

    - debye must not drift far from APBS, which would be a real regression;
    - and it must not become *dramatically* closer than debye's own box noise,
      because that would mean the noise floor has moved and the milestone's
      central claim — that this axis is discretization-limited, not
      solver-limited — no longer holds.

    The second is the one to read carefully when it fails. Fractional-volume
    dielectric averaging is expected to cause exactly that, and when it does the
    right response is to revisit ROADMAP section 12 M6 and gate this axis
    properly, not to widen the number here.
    """
    _, debye_values = debye_residues
    system = case_system()
    solver, family = backends.solver_for("apbs")
    apbs = solver.solve(system.request_for(family))
    apbs_values = {r.label: r.value for r in residue_potentials(volumetric(apbs), system.structure)}

    shared = sorted(set(debye_values) & set(apbs_values))
    difference = np.array([abs(debye_values[r] - apbs_values[r]) for r in shared])
    median = float(np.median(difference))

    assert median < 2 * DEBYE_BOX_NOISE_KT_E, (
        f"debye and APBS now differ by a median {median:.4f} kT/e per residue, "
        f"well beyond debye's own {DEBYE_BOX_NOISE_KT_E} kT/e box noise. That is "
        "a regression in the field, not the grid."
    )
    assert median > DEBYE_BOX_NOISE_KT_E / 4, (
        f"debye and APBS now agree to a median {median:.4f} kT/e, far inside "
        f"debye's own {DEBYE_BOX_NOISE_KT_E} kT/e box noise. M6 recorded this "
        "axis instead of gating it *because* the two were the same size — if "
        "that is no longer true the noise floor has moved, which is a real "
        "result. Update ROADMAP.md section 12 M6 and gate the axis rather than "
        "loosening this bound."
    )


def test_the_recorded_comparison_needs_a_common_question(debye_residues):
    """Both backends must be asked the same thing for the number to mean anything.

    Not a formality. M6 measured that a three-way common *lattice* is impossible
    on a protein — DelPhi's grid is isotropic where APBS's and debye's are
    per-axis — and that no spacing is reachable from two different paddings. So
    the one thing this comparison can hold fixed is the request, and it is worth
    asserting that it does.
    """
    result, _ = debye_residues
    system = case_system()

    assert volumetric(result) is not None
    assert system.want_potential is True
    # The structure the potentials are grouped by is the structure that was solved.
    assert system.structure.labels
    assert len(system.structure.labels) == system.structure.n_atoms
    # And the case is a real protein rather than a synthetic sphere, which is
    # what "on a real protein" in the exit criterion means.
    assert system.structure.n_atoms > 500
    assert len(set(system.structure.labels)) > 100


def test_a_case_without_labels_refuses_rather_than_grouping_wrongly():
    """Residue means need labels, and a PQR that lacks them must say so."""
    system = case_system()
    unlabelled = dataclasses.replace(system.structure, labels=())
    grid = PotentialGrid(values=np.zeros((4, 4, 4)), origin=np.zeros(3), spacing=np.ones(3))
    with pytest.raises(ValueError, match="no per-atom labels"):
        residue_potentials(grid, unlabelled)
