"""M4: the solvent-excluded surface, graded on what the probe is worth.

ROADMAP.md section 12 M4. Two kinds of test here, and the split is the whole
design.

**The identities are the real test of the construction.** A lone convex
sphere's solvent-excluded surface *is* that sphere; a zero-radius atom bounds no
volume for a probe to roll around, so adding one changes nothing. Both are exact
at the node with no tolerance, and both caught a wrong implementation during
M4: dilating the set of legal *grid nodes* inflates the 3 A Born ion's effective
radius to 3.0717, and sampling candidate probe centres over each accessible
sphere makes the sample count the answer. `surface.py`'s docstring carries the
numbers. DelPhi C++ honours all three identities; APBS does not, missing them by
0.09-0.33% on the Kirkwood geometry where the exact answer is zero — so
**"closer to APBS" is not the same as "more correct" on this axis**, and the
gate below is deliberately not "match APBS".

**The gate is the probe's worth**, `(E_molecular - E_vdw)/|E_vdw|`, decided by
Charlie on 2026-08-15 over the ROADMAP's original wording. The criterion in the
file said "inside the 2.3% band APBS and DelPhi already occupy", and that 2.3%
traced to a passing remark about pyDelPhi rather than to a measurement: across
the shared molecular cases the band is 0.41% to 5.74%. What replaced it is
relational and carries no constant at all — **debye must be no further from
APBS than DelPhi is** — which is the same shape M3 landed on for the same
reason, that a tolerance loose enough for the incumbents grades nothing.

**Why the probe's worth and not the energy.** Every closed-form case in the
corpus is blind to the solvent-excluded surface by construction, so a solver
answering `molecular` by returning its van der Waals number passes all eighteen
of them exactly. Until M4 added the pairs below, ALA-GLY was the only multi-atom
structure in the corpus carrying both surfaces — and the probe is worth 4% there
against 17% to 36% on a protein, so the peptide was not standing in for protein
scale, it was a different question.

**What is recorded and not gated.** The full protein sweep — twelve structures
from 906 to 8,279 atoms, including protein-RNA, an apo/holo pair and a ligand
complex — is in ROADMAP.md section 12. debye lands strictly between DelPhi and
APBS on every one, two to eight times closer to APBS than DelPhi is, and the
ordering survives a 0.7-0.35 A refinement ladder on fas2. Only the two cheapest
pairs are re-solved here; solving all seven protein pairs would be ten minutes
of every test run to re-derive a table that `corpus verify` already protects for
the incumbents.
"""

from __future__ import annotations

import numpy as np
import pytest

from sashimi.corpus import (
    CORPUS_DIR,
    MANIFEST,
    Case,
    born_ion_pqr,
    kirkwood_pqr,
    load_summary,
    probe_worth,
    surface_pairs,
)
from sashimi.debye import DebyeSolver
from sashimi.debye.grid import axis_coordinates, size_grid
from sashimi.debye.surface import inside_solute, inside_union_of_spheres
from sashimi.pqr import parse_pqr, read_pqr
from sashimi.protocol import (
    FiniteDifferenceRequest,
    FloatArray,
    GridSpec,
    PQRData,
    SolventModel,
    SurfaceModel,
)

DELPHI_DIR = CORPUS_DIR / "delphi"

# The two pairs cheap enough to re-solve on every run. `fas2-vdw` is deliberately
# `standard` rather than `full` for this reason: it is the cheapest protein-scale
# pair in the corpus, and a gate that only ever ran on a twenty-atom peptide is
# the gap M4 existed to close.
GATED_PAIRS = ("peptide", "fas2")

# Radii where a lone sphere's two boundaries must coincide. Spread deliberately
# across the probe: at r = 1 A the probe is larger than the solute, which is
# where a construction that dilates the wrong set fails first.
LONE_RADII = (1.0, 3.0, 6.0)

# Below this the probe's worth is not a quantity worth gating, and the number
# comes from a measurement rather than from taste: `ion-protein-complex` at 260
# atoms carries a total probe worth of ~0.5%, where the fas2 refinement ladder
# moves a single code by 0.5-0.9 points. Signal at or under the noise. Every
# solute at or above this size reads 17% or more, which is one to two orders
# above it. `test_a_small_probe_worth_is_recorded_and_not_judged` holds the
# other end of the argument.
GATEABLE_ATOMS = 900


def _grid_and_axes(structure: PQRData, resolution: float = 0.5) -> list[FloatArray]:
    grid = size_grid(structure, GridSpec(resolution=resolution, padding=10.0))
    return axis_coordinates(grid)


def _both_masks(structure: PQRData, probe: float = 1.4) -> tuple[np.ndarray, np.ndarray]:
    """The van der Waals and molecular masks on one lattice."""
    axes = _grid_and_axes(structure)
    vdw = inside_union_of_spheres(axes, structure.coords, structure.radii)
    molecular = inside_solute(
        axes,
        structure,
        SolventModel(surface_model=SurfaceModel.MOLECULAR, surface_radius=probe),
    )
    return vdw, molecular


@pytest.mark.parametrize("radius", LONE_RADII)
def test_a_lone_sphere_has_the_same_two_boundaries(radius: float):
    """A convex sphere's solvent-excluded surface is that sphere, exactly.

    Not a tolerance and not a near-miss: the probe rolling over a single sphere
    reaches every point outside it, so the two masks must agree at every node.
    This is what the node-dilating implementation failed — it called 72 nodes
    solute on the 3 A ion that a probe demonstrably reaches.
    """
    structure = parse_pqr(born_ion_pqr(radius))
    vdw, molecular = _both_masks(structure)
    extra = int(np.count_nonzero(molecular & ~vdw))
    assert extra == 0, f"r = {radius} A: the probe left {extra} node(s) unreachable"
    assert np.array_equal(vdw, molecular)


def test_a_zero_radius_charge_bounds_no_volume_for_the_probe():
    """Kirkwood's off-centre point charge cannot change the surface.

    The geometry is a sphere plus an atom of radius zero. A zero-radius atom
    bounds no volume, so there is nothing for a probe to roll around and the two
    boundaries must stay identical. DelPhi C++ reproduces this; APBS separates
    its two answers by 0.09-0.33% here, which is its own SES discretization
    rather than physics — recorded in `surface.py` because it is the reason this
    milestone is not gated on matching APBS.
    """
    structure = parse_pqr(kirkwood_pqr(radius=3.0, offset=0.5 * 3.0))
    assert min(structure.radii) == 0.0, "the fixture stopped carrying a zero-radius atom"
    vdw, molecular = _both_masks(structure)
    assert np.array_equal(vdw, molecular)


def test_the_two_boundaries_give_debye_the_same_energy_on_a_sphere():
    """The identity above, carried all the way through to an energy.

    The masks agreeing is the construction being right; the energies agreeing to
    the last bit is that rightness surviving `dielectric_faces`, three staggered
    lattices and the whole multigrid hierarchy. Those are different claims, and
    only the second one would catch a surface rebuilt inconsistently per level.
    """
    structure = parse_pqr(born_ion_pqr(3.0))
    energies = []
    for model in (SurfaceModel.VAN_DER_WAALS, SurfaceModel.MOLECULAR):
        result = DebyeSolver().solve(
            FiniteDifferenceRequest(
                structure=structure,
                grid=GridSpec(resolution=0.5, padding=10.0),
                solvent=SolventModel(surface_model=model, ionic_strength=0.0),
                want_energy=True,
                want_potential=False,
            )
        )
        energies.append(result.energy_kj_mol)
    assert energies[0] == energies[1], f"{energies[0]!r} != {energies[1]!r}"


def test_the_corpus_can_state_the_probes_worth_above_twenty_atoms():
    """The gap M4's corpus work existed to close, asserted so it stays closed.

    Before M4, ALA-GLY was the only multi-atom structure carrying both surfaces
    and every closed-form case was blind to the probe. A molecular case on a
    real protein with no van der Waals sibling cannot contribute to the gate at
    all, which is the state `barnase-vdw` and `fas2-molecular` were each in from
    opposite directions.

    Derived from the manifest rather than from a list of names, so a protein
    case added later without its sibling turns this red instead of silently
    sitting outside the gate.
    """
    paired = {case.name for pair in surface_pairs() for case in pair}
    lonely = [
        case.name
        for case in MANIFEST
        if case.solvent.surface_model in (SurfaceModel.MOLECULAR, SurfaceModel.VAN_DER_WAALS)
        and case.name not in paired
        and case.structure().n_atoms >= GATEABLE_ATOMS
    ]
    assert not lonely, f"solute(s) large enough to gate with no surface sibling: {lonely}"

    proteins = [
        molecular.name
        for _, molecular in surface_pairs()
        if molecular.structure().n_atoms >= GATEABLE_ATOMS
    ]
    assert len(proteins) >= 8, (
        f"only {len(proteins)} protein-scale pair(s); the probe is worth 4% on ALA-GLY "
        "and 17-36% on a protein, so the peptide does not stand in for them"
    )


def test_a_small_probe_worth_is_recorded_and_not_judged():
    """Where the gate stops applying, measured rather than asserted.

    The probe's worth is a difference of two energies, so its own error is
    amplified when the difference is small. On fas2 the refinement ladder moves
    each code by 0.5 to 0.9 points between 0.7 and 0.35 A, and the probe is
    worth 18 — signal far above that noise. On `ion-protein-complex` the probe
    is worth about half a point in total, which is *below* it, and the three
    codes order differently there for the same reason the three-atom molecules
    do. Gating that would grade the lattice.

    Asserted from the recordings so the claim stays true: if this case ever
    grows a probe worth comparable to the proteins', the reason it sits outside
    the gate has evaporated and this turns red.
    """
    worth = probe_worth(
        load_summary(_case("ion-protein-complex-vdw")),
        load_summary(_case("ion-protein-complex-molecular")),
    )
    assert abs(worth) < 2.0, (
        f"{worth:+.3f}% is no longer small against the 0.5-0.9 point lattice swing "
        "measured on fas2 — this case may now be gateable"
    )


def test_the_probes_worth_is_far_larger_on_a_protein_than_on_the_peptide():
    """Why protein-scale pairs had to exist, read out of the recordings.

    If the peptide had stood in for protein scale this would be a small
    difference and the corpus work would have been unnecessary. It is not: the
    probe is worth about four percent on twenty atoms and five to nine times
    that on a folded protein, because buried volume a probe cannot enter grows
    faster than surface does.
    """
    peptide = probe_worth(
        load_summary(_case("peptide-vdw")), load_summary(_case("peptide-molecular"))
    )
    assert 3.0 < peptide < 8.0, peptide

    for name in ("fas2", "barnase", "lysozyme"):
        worth = probe_worth(
            load_summary(_case(f"{name}-vdw")), load_summary(_case(f"{name}-molecular"))
        )
        assert worth > 3.0 * peptide, f"{name}: {worth:.3f}% against the peptide's {peptide:.3f}%"


@pytest.mark.parametrize("stem", GATED_PAIRS)
def test_debye_is_no_further_from_apbs_than_delphi_is(stem: str):
    """The M4 gate, on the pairs cheap enough to re-solve.

    Needs no binary installed: both incumbents' halves are recordings in the
    repository, and only the candidate is solved. That is the property the
    corpus was built for — `corpus verify --backend debye` grading a clean-room
    solver on a machine with no APBS.

    Relational rather than absolute, and deliberately so. APBS over-fills its own
    solvent-excluded surface (see the Kirkwood identity above), so a bar of the
    form "within x% of APBS" would be grading debye against a known bias. "No
    further from APBS than the other reference-tier code is" needs no constant
    and cannot be met by drifting toward either incumbent.
    """
    vdw, molecular = _case(f"{stem}-vdw"), _case(f"{stem}-molecular")
    apbs = probe_worth(load_summary(vdw), load_summary(molecular))
    delphi = probe_worth(load_summary(vdw, DELPHI_DIR), load_summary(molecular, DELPHI_DIR))
    debye = probe_worth(_solve(vdw), _solve(molecular))

    assert abs(debye - apbs) <= abs(delphi - apbs), (
        f"{stem}: debye {debye:+.3f}% is further from APBS {apbs:+.3f}% "
        f"than DelPhi {delphi:+.3f}% is"
    )
    # Recorded rather than gated: debye sat strictly between the incumbents on
    # all twelve structures measured at M4, which is a stronger statement than
    # the gate makes and is not one to freeze into a bar.
    assert min(delphi, apbs) <= debye <= max(delphi, apbs), (
        f"{stem}: debye {debye:+.3f}% left the DelPhi-APBS band "
        f"[{min(delphi, apbs):+.3f}, {max(delphi, apbs):+.3f}] — a real change, "
        "not necessarily a regression; re-measure before adjusting this"
    )


def _case(name: str) -> Case:
    return next(case for case in MANIFEST if case.name == name)


def _solve(case: Case) -> dict[str, float]:
    """debye's answer in the shape `probe_worth` reads recordings in."""
    result = DebyeSolver().solve(case.request())
    assert result.energy_kj_mol is not None, f"{case.name}: the case asked for no energy"
    return {"energy_kj_mol": result.energy_kj_mol}


def test_the_signed_distance_agrees_with_the_boolean_it_is_derived_from():
    """M8a's oracle, checked against M4's — `sign(gap)` must reproduce `inside`.

    `signed_gap` is not a second construction: `inside` reads "solvent is the
    accessible set dilated by the probe", so the signed distance to the boundary
    is `probe - dist(x, A)` and the three families already compute that distance
    before discarding it. Two implementations of one definition, so exact
    agreement is the bar.

    **It found a real defect immediately.** The radial family wrote its
    distances with `np.minimum(..., out=block[fancy_index])`, and fancy indexing
    returns a copy — every value landed in a temporary. The surface came back
    saturated wherever that family was the only one to reach a node, and this
    comparison is what showed it.
    """
    from sashimi.debye.surface import ReducedSurface  # noqa: PLC0415

    for path, resolution in (
        ("tests/data/ala-gly.pqr", 0.5),
        ("tests/data/apbs-examples/fas2.pqr", 1.0),
    ):
        structure = read_pqr(path)
        for model in (SurfaceModel.VAN_DER_WAALS, SurfaceModel.MOLECULAR):
            surface = ReducedSurface(structure, SolventModel(surface_model=model))
            axes = axis_coordinates(size_grid(structure, GridSpec(resolution=resolution)))
            inside = surface.inside(axes)
            gap = surface.signed_gap(axes)
            assert inside.any() and not inside.all(), f"{path} {model} decided nothing either way"
            assert np.array_equal(gap <= 0.0, inside), (
                f"{path} {model}: the distance and the boolean disagree on "
                f"{int(((gap <= 0.0) != inside).sum())} nodes"
            )


def test_the_ramp_raises_the_convergence_order_on_both_surfaces():
    """M8a's gate, and the one a pose spread alone would have failed to catch.

    A hard face-centre dielectric converges at order 0.3 (van der Waals) and
    1.0 (molecular) on this structure. Ramping the solute fraction across a cell
    from the signed distance takes both above 2 — which is the interface
    treatment, not the solver, being what bounded the accuracy.

    **Both widths, on both surfaces**, because an earlier draft of this test
    graded each surface at its own width on the strength of a finding that was a
    bug: `signed_gap` saturated the van der Waals interior, so the ramp was
    one-sided and a wider one displaced the interface further. With a two-sided
    distance the orders agree to within 0.01 across widths. Testing both is what
    would catch that returning.

    The bar is 1.5 — comfortably under the 2.4-2.7 measured, and comfortably
    over the hard scheme it has to beat.
    """
    from sashimi.debye.options import DebyeOptions  # noqa: PLC0415
    from sashimi.invariants import Refinement  # noqa: PLC0415

    ladder = (1.0, 0.5, 0.25)
    structure = read_pqr("tests/data/ala-gly.pqr")

    def order(model: SurfaceModel, width: float) -> float:
        solver = DebyeSolver(options=DebyeOptions(dielectric_smoothing=width))
        energies = []
        for spacing in ladder:
            request = FiniteDifferenceRequest(
                structure=structure,
                solvent=SolventModel(surface_model=model),
                grid=GridSpec(resolution=spacing, padding=10.0),
                want_potential=False,
            )
            answer = solver.solve(request).energy_kj_mol
            assert answer is not None
            energies.append(float(answer))
        grade = Refinement(backend="debye", spacings=ladder, energies=tuple(energies))
        assert grade.converging, f"{model} at width {width} is not converging: {energies}"
        return grade.order

    for model in (SurfaceModel.VAN_DER_WAALS, SurfaceModel.MOLECULAR):
        blunt = order(model, 0.0)
        for width in (0.25, 0.5):
            assert order(model, width) > 1.5 > blunt, (
                f"the ramp no longer raises the order on {model.value} at width {width}"
            )
