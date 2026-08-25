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

import itertools

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
    DIMENSIONS,
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

    proteins = {
        molecular.name
        for _, molecular in surface_pairs()
        if molecular.structure().n_atoms >= GATEABLE_ATOMS
    }
    # Derived from the manifest by a second route rather than floored at 8. The
    # `lonely` check above says no large solute *lacks* a sibling, which passes
    # vacuously if there are no large solutes at all; this says which ones there
    # are. `>= 8` was written when 8 was near the count and is 12 now, so four
    # could have left the gate in silence — and a floor cannot tell "one pair
    # was dropped" from "the manifest shrank on purpose".
    expected = {
        case.name
        for case in MANIFEST
        if case.solvent.surface_model is SurfaceModel.MOLECULAR
        and case.name in paired
        and case.structure().n_atoms >= GATEABLE_ATOMS
    }
    assert proteins == expected
    assert proteins, (
        "no protein-scale pair; the probe is worth 4% on ALA-GLY and 17-36% on "
        "a protein, so the peptide does not stand in for them"
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


def _two_ball_gap(
    points: np.ndarray,
    first: tuple[np.ndarray, float],
    second: tuple[np.ndarray, float],
) -> np.ndarray:
    """The signed distance to a union of two overlapping balls, in closed form.

    Written out here rather than taken from `sashimi.debye.surface`, because the
    point of the test below is to have a second implementation of the definition.
    The boundary of two overlapping balls is the two exposed caps plus the circle
    they meet on, so the distance is a minimum over three features — a foot point
    that the other ball has swallowed is not on the boundary and does not count.
    """
    (centre_a, radius_a), (centre_b, radius_b) = first, second
    axis = centre_b - centre_a
    separation = float(np.linalg.norm(axis))
    unit = axis / separation
    along = (separation * separation + radius_a * radius_a - radius_b * radius_b) / (
        2.0 * separation
    )
    ring = float(np.sqrt(radius_a * radius_a - along * along))
    origin = centre_a + along * unit

    best = np.full(len(points), np.inf)
    for centre, radius, other, other_radius in (
        (centre_a, radius_a, centre_b, radius_b),
        (centre_b, radius_b, centre_a, radius_a),
    ):
        offset = points - centre
        span = np.maximum(np.linalg.norm(offset, axis=1), 1e-12)
        foot = centre + radius * offset / span[:, None]
        exposed = np.linalg.norm(foot - other, axis=1) >= other_radius
        best = np.where(exposed, np.minimum(best, np.abs(span - radius)), best)

    offset = points - origin
    axial = offset @ unit
    radial = np.linalg.norm(offset - axial[:, None] * unit, axis=1)
    best = np.minimum(best, np.sqrt((radial - ring) ** 2 + axial**2))

    inside = (np.linalg.norm(points - centre_a, axis=1) <= radius_a) | (
        np.linalg.norm(points - centre_b, axis=1) <= radius_b
    )
    return np.asarray(np.where(inside, -best, best))


def test_the_union_distance_measures_to_the_union_and_not_to_one_sphere():
    """The van der Waals gap graded against a closed form, on both sides of it.

    **`min_i(|x - c_i| - r_i)` is a distance only outside a union of spheres.**
    It measures to the nearest sphere *surface*, and inside the union that
    surface may be one a neighbour has swallowed, in which case the real
    boundary is further off. Two balls of radius 2 at `(±1, 0, 0)` read
    **-1.0000000** at the origin where the true signed distance is
    **-1.7320508** — the nearest boundary is the circle the two meet on, out at
    `sqrt(3)` — so the depth is under-reported by 42.3%.

    That matters because `dielectric.py` reads this as a depth:
    `clip(0.5 - gap / (2w), 0, 1)`. A face the bound puts inside the band but
    the truth puts well past it was handed a blended dielectric instead of solid
    solute. Measured on `fas2` at 0.5 A with `w = 0.5` cells, **19.0% of the
    interior band faces** had a swallowed foot point, and repairing them moves
    the ramped energy by **+10.06 kJ/mol — 7.0% of the whole ramp-minus-hard
    offset**.

    `sign(gap)` never moved, which is why every existing test passed: the repair
    only ever makes an interior value *deeper*, so the boolean the whole
    construction is graded against cannot see it. `|gap|` had no test at all.

    **The lattice is offset off the axis on purpose**, and the reason is a real
    limitation recorded in the test below this one rather than dodged here.
    """
    from sashimi.debye.surface import ReducedSurface, _union_gap  # noqa: PLC0415

    centre_a, radius = np.array([-1.0, 0.0, 0.0]), 2.0
    centre_b = np.array([1.0, 0.0, 0.0])
    structure = PQRData(
        coords=np.stack([centre_a, centre_b]),
        charges=np.zeros(2),
        radii=np.array([radius, radius]),
    )
    # Offset in y so that no node sits on the rim's axis; see the next test.
    axes = [np.linspace(-4.0, 4.0, 33) + shift for shift in (0.0, 0.0625, 0.0)]
    surface = ReducedSurface(structure, SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS))

    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    exact = _two_ball_gap(
        grid.reshape(-1, DIMENSIONS), (centre_a, radius), (centre_b, radius)
    ).reshape(grid.shape[:DIMENSIONS])
    gap = surface.signed_gap(axes)
    bound = _union_gap(axes, structure.coords, structure.radii)

    # Only where the answer is observable: past `2 * max radius` both the bound
    # and the repair saturate by design, exactly as the solvent-excluded branch
    # does past `2 * probe`.
    graded = np.abs(exact) < 2.0 * radius
    assert int(graded.sum()) > 1000, "the saturation cut left too little to grade"
    assert np.abs(gap[graded] - exact[graded]).max() < 1e-9, (
        f"the distance is {np.abs(gap[graded] - exact[graded]).max():.3e} from the closed form"
    )

    # And the bound it replaces is *not* that answer, or this grades nothing:
    # the two agree outside the union and part company inside it.
    outside = exact > 0.0
    assert np.array_equal(bound[outside], exact[outside]), "the bound is exact outside the union"
    assert np.abs(bound[graded] - exact[graded]).max() > 0.5, (
        "the bound already matched the closed form, so this test cannot fail"
    )


def test_a_node_on_a_rim_axis_is_the_one_place_the_distance_saturates():
    """A limitation of all three families, pinned rather than dodged.

    A node on a rim's axis is equidistant from **every** point of that circle,
    so there is no projection to test for legality and `_toroidal_distance`
    declines it — `usable = length > DEGENERATE`. The node keeps the saturated
    fill, which reads as "deep solute". `_toroidally_reachable` declines it in
    the same place, so `inside` and `sign(gap)` stay consistent and no existing
    test can see this either.

    A rim's axis is the line through two overlapping atom centres, so on real
    coordinates the set is measure zero — but the *recorded counterexample for
    the bound* sits exactly on it: the origin of two balls at `(±1, 0, 0)` is
    where the closed form reads `-sqrt(3)`, the bound reads `-1`, and this reads
    the fill. That is why the test above offsets its lattice, and it is worth
    knowing before the next reader picks a symmetric fixture.

    **What such a node returns is the search reach, not the saturated fill**, and
    that distinction is a repair of its own. The fill asserts `depth >= 2 * reach`,
    which for a node no family reached is an assertion nothing supports. What is
    reported instead is the larger of the bound and the reach — the two things
    actually known — so this node reads `-2.0` against a true `-1.7320508`
    rather than the fill's `-4.0`. It is still an over-report, and it is bounded
    by the reach rather than by twice it.

    *The same fallback is what makes the lone-sphere case exact: there the bound
    is the larger term and it is the truth. See the test below.*

    The assertion is the *shape* of the hole, not its size: every node that
    misses the closed form must lie on the axis. A repair that made the families
    answer these would only tighten it.
    """
    from sashimi.debye.surface import ReducedSurface, _union_gap  # noqa: PLC0415

    centre_a, radius = np.array([-1.0, 0.0, 0.0]), 2.0
    centre_b = np.array([1.0, 0.0, 0.0])
    structure = PQRData(
        coords=np.stack([centre_a, centre_b]),
        charges=np.zeros(2),
        radii=np.array([radius, radius]),
    )
    axes = [np.linspace(-4.0, 4.0, 33) for _ in range(DIMENSIONS)]
    surface = ReducedSurface(structure, SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS))

    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    exact = _two_ball_gap(
        grid.reshape(-1, DIMENSIONS), (centre_a, radius), (centre_b, radius)
    ).reshape(grid.shape[:DIMENSIONS])
    gap = surface.signed_gap(axes)

    missed = (np.abs(gap - exact) > 1e-9) & (np.abs(exact) < 2.0 * radius)
    assert missed.any(), (
        "the symmetric lattice no longer lands on the rim axis, so this test "
        "is not exercising the degeneracy it exists to record"
    )
    off_axis = grid[missed][:, 1:]
    assert np.abs(off_axis).max() == 0.0, (
        f"{int(missed.sum())} nodes miss the closed form and not all are on the rim axis; "
        f"furthest off it by {np.abs(off_axis).max():.3g} A"
    )
    # And every one of them comes back at the larger of the bound and the reach
    # rather than at the fill, so the repair never asserts twice the depth it
    # searched to.
    bound = _union_gap(axes, structure.coords, structure.radii)
    expected = -np.maximum(-bound[missed], radius)
    assert np.array_equal(gap[missed], expected), (
        "a declined node did not come back at max(bound, reach); the fill asserts "
        "a depth no family measured"
    )
    assert (np.abs(gap[missed]) < 2.0 * radius).all()


def test_a_lone_sphere_has_nothing_to_bury_and_so_moves_not_one_bit():
    """Why every Born-based validation of the ramp was blind to this defect.

    A convex sphere on its own has no second surface to swallow the first, so
    `min_i(|x - c_i| - r_i)` *is* the signed distance there and the repair has
    nothing to do. That is not a happy coincidence — it is why the 4.6-9.0x
    accuracy evidence M8 took for the van der Waals ramp, all of it on the Born
    ion, could not have seen a bug that only exists where spheres overlap.
    `test_a_lone_sphere_has_the_same_two_boundaries` records the same blindness
    for M8a's solvent-excluded distance.

    Asserted bit for bit rather than approximately, because the claim is that
    one branch of `signed_gap` degenerates into the other and not that it lands
    nearby. Measured across the change on twelve Born configurations — two
    radii, two spacings, three widths — every energy reproduced to the last
    digit.
    """
    from sashimi.debye.surface import ReducedSurface, _union_gap  # noqa: PLC0415

    for radius in (2.0, 3.0):
        structure = PQRData(
            coords=np.zeros((1, DIMENSIONS)), charges=np.array([1.0]), radii=np.array([radius])
        )
        solvent = SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS)
        grid = size_grid(structure, GridSpec(resolution=0.5, padding=10.0))
        axes = axis_coordinates(grid)
        width = min(0.5 * float(min(grid.spacing)), solvent.surface_radius)

        surface = ReducedSurface(structure, solvent)
        bound = _union_gap(axes, structure.coords, structure.radii)

        gap = surface.signed_gap(axes, band=width)
        band = np.abs(bound) < width
        assert band.any(), f"a = {radius}: no node in the band, so this grades nothing"
        assert np.array_equal(gap[band], bound[band]), (
            f"a = {radius}: {int((gap[band] != bound[band]).sum())} of {int(band.sum())} "
            "band nodes moved on a solute with no concavity to find"
        )

        # **The whole interior, not just the band, and that is the assertion
        # that matters.** Grading the band alone misses the node sitting exactly
        # on the atom centre — which is never in the band, and which the first
        # version of the repair returned `-2a` for against a true `-a`, because
        # `_radial_distance` has no direction to project along there and the
        # saturated fill survived. `band=None` is the path a caller plotting or
        # grading the field takes, so it is the one this has to cover.
        exact = surface.signed_gap(axes)
        inside = bound < 0.0
        assert inside.any()
        assert np.array_equal(exact[inside], bound[inside]), (
            f"a = {radius}: {int((exact[inside] != bound[inside]).sum())} of "
            f"{int(inside.sum())} interior nodes moved, worst by "
            f"{float(np.abs(exact[inside] - bound[inside]).max()):.4g} A"
        )


def test_the_repaired_interior_moves_faces_on_a_real_solute():
    """The control: the two-ball counterexample is not a contrived geometry.

    A closed form needs two spheres to have one, so the test above is as small
    as a test can be. This is the same defect counted on real input, which is
    where every genuine bug in this repository has come from. The bound and the
    distance must differ on a real solute — and they must differ *inside the
    band the ramp reads*, because outside it the value is clipped and a
    difference there costs nothing.
    """
    from sashimi.debye.surface import ReducedSurface, _union_gap  # noqa: PLC0415

    for path, resolution in (
        ("tests/data/ala-gly.pqr", 0.5),
        ("tests/data/apbs-examples/fas2.pqr", 1.0),
    ):
        structure = read_pqr(path)
        solvent = SolventModel(surface_model=SurfaceModel.VAN_DER_WAALS)
        grid = size_grid(structure, GridSpec(resolution=resolution, padding=10.0))
        axes = axis_coordinates(grid)
        width = min(0.5 * float(min(grid.spacing)), solvent.surface_radius)

        surface = ReducedSurface(structure, solvent)
        gap = surface.signed_gap(axes, band=width)
        bound = _union_gap(axes, structure.coords, structure.radii)

        # The repair is one-directional: the true depth is never smaller than
        # the bound, so anything shallower is a defect rather than a difference.
        assert (gap <= bound + 1e-12).all(), (
            f"{path}: the repaired distance is shallower than its own bound on "
            f"{int((gap > bound + 1e-12).sum())} nodes"
        )
        band = (bound > -width) & (bound < 0.0)
        moved = band & (gap != bound)
        assert moved.sum() > 0, f"{path}: the repair changed no band node, so it is not running"
        # The face is only mis-assigned if it leaves the band, and it is those
        # the ramp actually mis-blended.
        escaped = moved & (gap <= -width)
        assert escaped.sum() > 0, (
            f"{path}: {int(moved.sum())} band nodes moved but none left the band, "
            "so no dielectric was wrong and the repair is unobservable here"
        )


def test_pruning_the_band_leaves_every_value_the_ramp_actually_reads():
    """`signed_gap(band=w)` against `signed_gap()`, on the half that is consumed.

    The ramp clamps: `dielectric.py` computes `clip(0.5 - gap / (2w), 0, 1)`, so
    only `|gap| < w` reaches an answer and the exact distance everywhere else is
    unobservable. `band` declines to compute that unobservable part, which is
    most of the lattice — 0.2% to 3.7% of nodes are in the band across the
    lattices below.

    Two assertions, and the second is what stops the first being vacuous. Inside
    the band the two must agree **bit for bit**, not approximately. Outside it
    they must *differ somewhere*: if `band` were ignored — the obvious way for
    this to rot — the pruned field would equal the exact one everywhere and the
    first assertion would pass while the parameter did nothing.

    `sign(gap)` is checked too, because it is what
    `test_the_signed_distance_agrees_with_the_boolean_it_is_derived_from` grades
    the whole construction on and a prune must not move it.

    **Both surfaces, and van der Waals is the one that needed adding.** It ran
    on `molecular` alone for as long as it existed, because the union branch had
    nothing to prune — it returned a closed form over the whole lattice. Now
    that it measures to the union's own rims and seats it has a search radius,
    and *this is the test that says the radius is big enough*: `band=w` and
    `band=None` search different distances, so they can only agree in the band
    if every feature that could win is offered under the narrower one.

    **What it does not say is that either reach is big enough.** It grades the
    two against each other, so a reach that is uniformly too small passes: put
    `look = self.probe` back on both branches — exactly the state before this
    change — and it stays green. What it catches is the two branches searching
    *different* distances, which is the shape a partial fix would have. Run as a
    mutation: `look = far if band is None else self.probe` reddens it on
    `ala-gly` at 0.5 A, by **one node of 1,633**.
    """
    from sashimi.debye.surface import ReducedSurface  # noqa: PLC0415

    pruned_somewhere = False
    for path, resolution in (
        ("tests/data/ala-gly.pqr", 0.5),
        ("tests/data/ala-gly.pqr", 0.35),
        ("tests/data/apbs-examples/fas2.pqr", 1.0),
    ):
        structure = read_pqr(path)
        grid = size_grid(structure, GridSpec(resolution=resolution, padding=10.0))
        axes = axis_coordinates(grid)
        for smoothing, model in itertools.product(
            (0.25, 0.5, 1.0), (SurfaceModel.MOLECULAR, SurfaceModel.VAN_DER_WAALS)
        ):
            solvent = SolventModel(surface_model=model)
            surface = ReducedSurface(structure, solvent)
            width = min(smoothing * float(min(grid.spacing)), solvent.surface_radius)

            exact = surface.signed_gap(axes)
            pruned = surface.signed_gap(axes, band=width)
            where = f"{path} {model} res={resolution} w={width:.4f}"

            in_band = np.abs(exact) < width
            assert in_band.any(), f"{where}: no node in the band, so this grades nothing"
            assert np.array_equal(exact[in_band], pruned[in_band]), (
                f"{where}: {int((exact[in_band] != pruned[in_band]).sum())} of "
                f"{int(in_band.sum())} in-band nodes moved"
            )

            exact_fraction = np.clip(0.5 - exact / (2.0 * width), 0.0, 1.0)
            pruned_fraction = np.clip(0.5 - pruned / (2.0 * width), 0.0, 1.0)
            assert np.array_equal(exact_fraction, pruned_fraction), (
                f"{where}: the consumed ramp fraction moved on "
                f"{int((exact_fraction != pruned_fraction).sum())} nodes"
            )
            assert np.array_equal(np.sign(exact), np.sign(pruned)), f"{where}: sign moved"

            pruned_somewhere = pruned_somewhere or bool((exact != pruned).any())

    assert pruned_somewhere, (
        "`band` changed nothing on any lattice, so the prune is not running and "
        "the agreement above is a comparison of one code path with itself"
    )


def test_the_ramp_raises_the_convergence_order_on_both_surfaces():
    """M8a's gate, and the one a pose spread alone would have failed to catch.

    A hard face-centre dielectric converges at order 1.009 on `molecular` and
    **does not converge at all on van der Waals** — its corrections shrink by
    1.2031x, under `MIN_SHRINKAGE`, so no order exists to quote. Ramping the
    solute fraction across a cell from the signed distance takes both to
    2.31-2.48 — which is the interface treatment, not the solver, being what
    bounded the accuracy.

    *The docstring here used to read "order 0.3 (van der Waals)", matching
    ROADMAP.md section 12's M8a table. That number came from a ladder the
    repaired `converging` refuses; it is withdrawn rather than restated.*

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

    # **Achieved spacings, not requested.** ROADMAP.md section 12 says M8a was
    # measured "against achieved spacings rather than requested ones" and this
    # test did not: it passed `(1.0, 0.5, 0.25)`, a ratio of exactly 2, where
    # `size_grid` actually lands on 0.8695 / 0.4545 / 0.2432 — ratios 1.913 and
    # 1.869. `sashimi.invariants._achieved_spacing` exists for that reason, and
    # reading the exponent against the wrong ratio biases it: every order below
    # moves 0.10-0.16 between the two conventions.
    achieved = tuple(
        float(
            np.prod(
                np.asarray(
                    size_grid(structure, GridSpec(resolution=r, padding=10.0)).spacing, float
                )
            )
            ** (1.0 / 3)
        )
        for r in ladder
    )

    def grade(model: SurfaceModel, width: float) -> Refinement:
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
        return Refinement(backend="debye", spacings=achieved, energies=tuple(energies))

    for model in (SurfaceModel.VAN_DER_WAALS, SurfaceModel.MOLECULAR):
        blunt = grade(model, 0.0)
        # **The hard scheme need not be fittable, and on one surface it is not.**
        # This used to `assert blunt.converging` for every surface, which passed
        # only because `converging` bounded the magnitude of successive
        # differences and nothing else. Repairing that guard (see
        # `sashimi.invariants.MIN_SHRINKAGE`) refuses the hard van der Waals
        # ladder outright: its corrections shrink by 1.2031x, under the 1.25
        # floor, so no order can be read from it at all.
        #
        # That is a *stronger* statement than a low order, not a weaker one, and
        # it is why the comparison below is written as two separate claims
        # rather than one chained inequality. ROADMAP.md section 12 records the
        # consequence: M8a's headline "0.32 -> 2.65" quotes a baseline that the
        # repaired instrument will not fit.
        if blunt.converging:
            assert blunt.order < 1.5, (
                f"the hard scheme on {model.value} now reaches {blunt.order:.3f}, "
                "so the ramp is no longer the thing raising the order"
            )
        for width in (0.25, 0.5):
            ramped = grade(model, width)
            assert ramped.converging, (
                f"{model.value} at width {width} no longer converges: {ramped.energies}"
            )
            assert ramped.order > 1.5, (
                f"the ramp no longer raises the order on {model.value} at width "
                f"{width}: {ramped.order:.3f}"
            )
