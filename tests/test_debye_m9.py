"""M9: the strided box face, and the two invariants that used to ride along on it.

The boundary sum was `O(face nodes x atoms)` — the only superlinear stage debye
had. Striding the face removes it. What makes this worth its own module is not
the speed but the three things that go quiet if it is done carelessly:

* the exact path has to stay **bit-identical**, or every small recording moves
  for nothing;
* the six faces have to agree on the nodes they **share**, or an edge node gets
  two values and whichever face is written last wins;
* the guard that refuses an atom on the box face was *implemented by* the
  `O(nodes x atoms)` distance block, so removing the block removes the guard —
  silently, and the damage is not the singularity the old comment described.

ROADMAP.md section 12, "M9 — a boundary that does not cost `O(nodes x atoms)`".
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from sashimi.artifacts import content_address
from sashimi.debye import sources
from sashimi.debye.backend import DebyeSolver
from sashimi.debye.grid import size_grid
from sashimi.debye.options import DebyeOptions
from sashimi.debye.sources import (
    DEFAULT_BOUNDARY_PITCH_A,
    EXACT_FACE_PAIRS,
    PITCH_CLEARANCE_FRACTION,
    _exact_nodes,
    _sampled_nodes,
    boundary_mask,
    debye_huckel_boundaries,
    plan_face_sampling,
    solute_clearance,
)
from sashimi.errors import InputError
from sashimi.pqr import parse_pqr
from sashimi.protocol import (
    FiniteDifferenceRequest,
    GridSpec,
    PQRData,
    SolventModel,
    SurfaceModel,
)

FAS2 = Path("tests/data/apbs-examples/fas2.pqr")
ALA_GLY = Path("tests/data/ala-gly.pqr")

SALT = SolventModel(surface_model=SurfaceModel.MOLECULAR, ionic_strength=0.15)
STATES = [(SALT, False), (SALT, True)]


def _load(path: Path) -> PQRData:
    raw = path.read_bytes()
    return parse_pqr(gzip.decompress(raw).decode() if path.suffix == ".gz" else raw.decode())


def _grid(structure: PQRData, resolution: float = 1.0):
    return size_grid(structure, GridSpec(resolution=resolution, padding=10.0))


# --- bit-identity -----------------------------------------------------------


def test_the_exact_face_uses_the_enumeration_the_recordings_were_made_with():
    """The exact path must *be* `np.argwhere(boundary_mask(...))`, not agree with it.

    **This cannot be tested with a literal energy.** The first version of this
    test pinned -218.62772042354118, taken from `a0862ce` on darwin/arm64, and
    CI failed on linux/amd64 with -218.62772042354138. Bit-identity in this
    solver is a per-platform property, so an absolute anchor tests the platform
    as much as the code. What is portable is the *scheme*: every recording was
    made through this enumeration, and the two below are not interchangeable.
    """
    shape = _grid(_load(FAS2)).shape
    exact = _exact_nodes(shape)
    unique = _sampled_nodes(shape, plan_face_sampling(_grid(_load(FAS2)), _load(FAS2), 0.0))

    assert np.array_equal(exact, unique), "same nodes, in the same order"
    assert not exact.flags["C_CONTIGUOUS"], "argwhere returns a transposed view"
    assert unique.flags["C_CONTIGUOUS"], "and np.unique does not"


def test_the_enumeration_is_load_bearing_and_not_cosmetic(monkeypatch):
    """The layout above changes the answer, which is why the test before it exists.

    Same nodes, same order, C-contiguous instead of transposed — and the face
    values move. Measured on `peptide-molecular` that was one ULP on 13,043 of
    22,530 nodes, enough to move a recorded energy. If this ever stops failing,
    the test above has become decoration.
    """
    structure = _load(FAS2)
    grid = _grid(structure)
    shipped = debye_huckel_boundaries(grid, structure, STATES, pitch_a=0.0)

    monkeypatch.setattr(
        sources, "_exact_nodes", lambda shape: np.ascontiguousarray(_exact_nodes(shape))
    )
    relaid = debye_huckel_boundaries(grid, structure, STATES, pitch_a=0.0)

    assert not any(np.array_equal(a, b) for a, b in zip(shipped, relaid, strict=True)), (
        "a C-contiguous copy of the same indices in the same order produced "
        "bit-identical values, so the enumeration no longer decides the answer "
        "and the test above is guarding nothing"
    )
    for a, b in zip(shipped, relaid, strict=True):
        # Scaled to the field, not to each node: the face crosses zero, and a
        # per-node relative tolerance is unbounded where it does.
        assert np.abs(a - b).max() <= 1e-12 * np.abs(a).max(), (
            "the difference should be rounding, not a different boundary"
        )


def test_a_small_solute_is_held_on_the_exact_face_whatever_the_pitch():
    """Under `EXACT_FACE_PAIRS` the pitch is ignored, so small recordings do not move."""
    structure = _load(ALA_GLY)
    grid = _grid(structure, resolution=0.5)
    faces = int(boundary_mask(grid.shape).sum())
    assert faces * structure.n_atoms <= EXACT_FACE_PAIRS, "ala-gly should be under the line"

    assert plan_face_sampling(grid, structure, DEFAULT_BOUNDARY_PITCH_A).exact
    strided = debye_huckel_boundaries(grid, structure, STATES, pitch_a=DEFAULT_BOUNDARY_PITCH_A)
    exact = debye_huckel_boundaries(grid, structure, STATES, pitch_a=0.0)
    for a, b in zip(strided, exact, strict=True):
        assert np.array_equal(a, b)


def test_a_protein_is_not_held_on_the_exact_face():
    """The guard above must not be so wide that it swallows the case M9 is about."""
    structure = _load(FAS2)
    sampling = plan_face_sampling(_grid(structure), structure, DEFAULT_BOUNDARY_PITCH_A)
    assert not sampling.exact
    assert sampling.n_samples < 2_000, "the whole point is that this is a few hundred nodes"


# --- the six faces have to agree --------------------------------------------


def test_the_sampled_nodes_carry_their_exact_values():
    """Interpolation reproduces its own samples, so the scheme is exact where it looked.

    A weight matrix that is off by a shift would still produce a smooth, plausible
    face — this is the assertion that says it is the *right* smooth face.
    """
    structure = _load(FAS2)
    grid = _grid(structure)
    sampling = plan_face_sampling(grid, structure, DEFAULT_BOUNDARY_PITCH_A)
    exact = debye_huckel_boundaries(grid, structure, STATES, pitch_a=0.0)
    strided = debye_huckel_boundaries(grid, structure, STATES, pitch_a=DEFAULT_BOUNDARY_PITCH_A)

    ix, iy, iz = sampling.indices
    for a, b in zip(exact, strided, strict=True):
        # The x = 0 face, at its own sampled (y, z) nodes.
        assert np.allclose(a[0][np.ix_(iy, iz)], b[0][np.ix_(iy, iz)], rtol=0, atol=1e-12)
        assert np.allclose(a[:, -1][np.ix_(ix, iz)], b[:, -1][np.ix_(ix, iz)], rtol=0, atol=1e-12)


def test_faces_agree_on_the_edges_they_share():
    """A node on two faces gets one value, because all six faces share the index sets.

    Per-face index sets would leave this node interpolated from different samples
    depending on which face wrote it, and the array would carry whichever ran
    last — a difference no energy test would notice.
    """
    structure = _load(FAS2)
    grid = _grid(structure)
    sampling = plan_face_sampling(grid, structure, DEFAULT_BOUNDARY_PITCH_A)
    field = debye_huckel_boundaries(grid, structure, STATES, pitch_a=DEFAULT_BOUNDARY_PITCH_A)[0]
    exact = debye_huckel_boundaries(grid, structure, STATES, pitch_a=0.0)[0]

    # The precondition that makes the seams agree: both endpoints are in every
    # axis's sample set, so the two faces meeting at an edge interpolate along
    # that edge from the *same* sampled points.
    for axis, index in enumerate(sampling.indices):
        assert index[0] == 0
        assert index[-1] == grid.shape[axis] - 1

    # And the consequence, checked rather than argued. The edge (x=0, y=0) lies
    # on both the x face and the y face; whichever wrote it, its value must be
    # the linear lift along z of the exact values at the sampled z nodes. If the
    # two faces disagreed, one of them would have written something else.
    iz = sampling.indices[2]
    for edge in (np.s_[0, 0, :], np.s_[0, -1, :], np.s_[-1, 0, :], np.s_[-1, -1, :]):
        want = np.interp(np.arange(grid.shape[2]), iz, exact[edge][iz])
        assert np.allclose(field[edge], want, rtol=0, atol=1e-12)

    again = debye_huckel_boundaries(grid, structure, STATES, pitch_a=DEFAULT_BOUNDARY_PITCH_A)[0]
    assert np.array_equal(field, again)


def test_the_interior_is_untouched():
    """Dirichlet data is the six faces and nothing else; `_solve_state` relies on it."""
    structure = _load(FAS2)
    grid = _grid(structure)
    field = debye_huckel_boundaries(grid, structure, STATES, pitch_a=DEFAULT_BOUNDARY_PITCH_A)[0]
    assert np.count_nonzero(field[1:-1, 1:-1, 1:-1]) == 0


# --- the guard that used to ride along --------------------------------------


def test_an_atom_on_the_box_face_is_refused_even_though_the_face_is_strided():
    """The case the old guard caught, under the scheme that removed its mechanism.

    The old check was a running minimum over the `O(nodes x atoms)` block. A
    strided face visits a few hundred of tens of thousands of nodes, so that
    check could only have fired by luck. This is the same structure
    `test_a_charge_on_the_box_face_is_refused_rather_than_clipped` uses.
    """
    structure = PQRData(
        coords=np.array([[0.0, 0.0, 0.0], [6.0, 6.0, 6.0]]),
        charges=np.array([1.0, 0.0]),
        radii=np.array([0.0, 3.0]),
    )
    grid = size_grid(structure, GridSpec(resolution=1.0, padding=0.0))
    for pitch in (0.0, DEFAULT_BOUNDARY_PITCH_A):
        with pytest.raises(InputError, match="padding"):
            debye_huckel_boundaries(grid, structure, STATES, pitch_a=pitch)


def test_the_refusal_is_linear_in_atoms_not_quadratic():
    """It must not be reachable only by building the thing M9 deletes.

    Timing would be flaky; what is asserted instead is the mechanism — the check
    fires on a grid whose face is never evaluated at all, which is impossible for
    a check that lives inside the evaluation.
    """
    structure = PQRData(
        coords=np.array([[0.0, 0.0, 0.0], [6.0, 6.0, 6.0]]),
        charges=np.array([1.0, 0.0]),
        radii=np.array([0.0, 3.0]),
    )
    grid = size_grid(structure, GridSpec(resolution=1.0, padding=0.0))
    with pytest.raises(InputError, match=r"loses charge|padding"):
        # No state asked for, so nothing would be summed over the face.
        debye_huckel_boundaries(grid, structure, [], pitch_a=0.0)


# --- provenance -------------------------------------------------------------


def test_the_boundary_scheme_is_named_in_provenance():
    """Two schemes 1.4% apart in energy must not address one file.

    Striding changes the boundary and not the box, so `grid.as_diagnostics()` is
    byte-identical across it and this label is the only field that can tell them
    apart. Before it did, `mdh`, `sdh` and a zeroed boundary all addressed
    `63fb17d4943d` and the second solve silently reused the first's map.
    """
    structure = _load(FAS2)
    grid = _grid(structure)
    exact = plan_face_sampling(grid, structure, 0.0).label
    strided = plan_face_sampling(grid, structure, DEFAULT_BOUNDARY_PITCH_A).label
    coarser = plan_face_sampling(grid, structure, 2 * DEFAULT_BOUNDARY_PITCH_A).label
    assert len({exact, strided, coarser}) == 3, (exact, strided, coarser)


def test_two_pitches_do_not_share_a_saved_map():
    """The label above has to survive into the address, not merely exist."""
    structure = _load(FAS2)
    request = FiniteDifferenceRequest(
        structure=structure,
        grid=GridSpec(resolution=2.0, padding=10.0),
        solvent=SALT,
        want_energy=False,
        want_potential=True,
    )
    addresses = set()
    for pitch in (0.0, DEFAULT_BOUNDARY_PITCH_A):
        result = DebyeSolver(DebyeOptions(boundary_pitch_a=pitch)).solve(request)
        addresses.add(content_address(structure, result.provenance.resolved_parameters))
    assert len(addresses) == 2


# --- the pitch has to survive a smaller box ---------------------------------


def test_the_pitch_is_capped_by_the_solutes_clearance():
    """A fixed distance-pitch scales the wrong way, so it is not left fixed.

    Shrink `padding` and the face moves closer to the solute: the field on it
    varies faster, while a fixed pitch buys *fewer* samples because the box is
    smaller. Measured before the cap existed, a 12 A pitch at `padding = 3` put
    fas2 at r = 0.998798 and +0.7536% energy — outside M9's gate — where the same
    pitch at `padding = 10` was r = 0.999998. `padding` is a caller's knob that
    `protocol.py` bounds only below, so this cannot be left to the default.
    """
    structure = _load(FAS2)
    tight = size_grid(structure, GridSpec(resolution=1.0, padding=3.0))
    roomy = size_grid(structure, GridSpec(resolution=1.0, padding=10.0))

    coarse = 4 * DEFAULT_BOUNDARY_PITCH_A
    assert plan_face_sampling(tight, structure, coarse).pitch_a < coarse, "cap should bind"
    assert plan_face_sampling(tight, structure, coarse).pitch_a == pytest.approx(
        PITCH_CLEARANCE_FRACTION * solute_clearance(tight, structure)
    )

    # The cap is what makes a smaller box sample *more* finely, not less.
    tight_pitch = plan_face_sampling(tight, structure, coarse).pitch_a
    roomy_pitch = plan_face_sampling(roomy, structure, coarse).pitch_a
    assert tight_pitch < roomy_pitch


def test_the_default_pitch_holds_the_gate_on_a_tight_box():
    """The end-to-end check, on the configuration that broke the first default.

    fas2 at `padding = 3` against the exact sum on the same lattice: the energy
    clause M9 gates on, at the condition where a distance-pitch is worst.
    """
    structure = _load(FAS2)
    request = FiniteDifferenceRequest(
        structure=structure,
        grid=GridSpec(resolution=1.0, padding=3.0),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
        want_energy=True,
        want_potential=False,
    )
    exact = DebyeSolver(DebyeOptions(boundary_pitch_a=0.0)).solve(request)
    strided = DebyeSolver(DebyeOptions()).solve(request)
    assert exact.energy_kj_mol is not None and strided.energy_kj_mol is not None
    moved = abs(strided.energy_kj_mol - exact.energy_kj_mol) / abs(exact.energy_kj_mol)
    assert moved < 0.005, f"energy moved {100 * moved:.4f}%, gate is 0.5%"
