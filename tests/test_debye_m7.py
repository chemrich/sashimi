"""M7: the batched rim query, and the two things that could make it a lie.

ROADMAP.md section 12 M7. The change under test replaces 11,380 ball queries
per lattice with one batched query per two thousand rims. It is a performance
change and nothing else, so the tests here are almost entirely about the answer
not moving.

**Bit-identical is the bar, not a tolerance.** `decided` feeds a boolean or, so
a node claimed by *any* rim is the node claimed by *the first*, and nothing
downstream depends on which rim won. That makes the rewrite checkable against
the last digit of an energy — and a tolerance here would pass exactly the bug
the check exists for, which is the same reasoning `tests/test_bench.py` records
for the instrument that measures it.

**`RIM_BATCH` is the thing to be afraid of.** It is a memory bound with no
business in an answer, which is precisely the shape of constant this repo keeps
catching after it has quietly become one: ROADMAP.md section 12 records a
sampled surface where the sample count *was* the answer, and where 256 happened
to match the reference. So the batch size is swept rather than reasoned about,
at values that put one rim, a few rims, and every rim in a single batch.
"""

from __future__ import annotations

import numpy as np
import pytest

from sashimi.debye import surface as surface_module
from sashimi.debye.grid import axis_coordinates, size_grid
from sashimi.debye.surface import ReducedSurface, _Bins, _ragged
from sashimi.pqr import read_pqr
from sashimi.protocol import GridSpec, SolventModel, SurfaceModel

# Twenty atoms, and the smallest structure in the corpus whose rims decide
# anything: a lone sphere never reaches this code at all, since its two
# boundaries coincide and `undecided` comes back empty.
PEPTIDE = "tests/data/ala-gly.pqr"

# One rim per batch, a few, and more than exist. The first and last are the
# interesting ones: at 1 every batch boundary is exercised, and at 100,000 the
# loop runs once and the batching is bypassed entirely.
BATCH_SIZES = (1, 3, 97, 100_000)


def _molecular_mask(batch: int | None = None, monkeypatch=None) -> np.ndarray:
    structure = read_pqr(PEPTIDE)
    solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR)
    axes = axis_coordinates(size_grid(structure, GridSpec()))
    if batch is not None and monkeypatch is not None:
        monkeypatch.setattr(surface_module, "RIM_BATCH", batch)
    return ReducedSurface(structure, solvent).inside(axes)


def test_the_structure_actually_exercises_the_rim_loop():
    """Otherwise every test below passes by never running the code under test.

    The trap this repo keeps recording: a check that cannot fail. A structure
    whose rims decide nothing would make the batch sweep unanimous for reasons
    having nothing to do with batching.
    """
    structure = read_pqr(PEPTIDE)
    surface = ReducedSurface(structure, SolventModel(surface_model=SurfaceModel.MOLECULAR))
    assert len(surface.rims) > 1
    # The sweep is only a sweep if it spans the boundary: some value has to
    # force several batches and some value has to force exactly one.
    assert min(BATCH_SIZES) < len(surface.rims) <= max(BATCH_SIZES)


@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_the_batch_size_does_not_move_a_single_node(batch: int, monkeypatch):
    """`RIM_BATCH` bounds a temporary, so it must be invisible in the answer.

    Compared node by node rather than by a count of solute nodes: two masks can
    hold the same number of nodes and disagree about which, and a count would
    call that identical.
    """
    reference = _molecular_mask()
    assert np.array_equal(_molecular_mask(batch, monkeypatch), reference)


def test_the_batched_query_answers_exactly_what_the_single_query_does(rng_seed: int = 0):
    """`near_many` against `near`, which is the primitive the milestone rests on.

    Random points and random query radii rather than a molecule, because what
    is being checked is the bin arithmetic — the ragged expansion over bins and
    then over the points in them — and a lattice would hide a whole class of
    off-by-one behind its regularity. Radii deliberately span the cell size, so
    queries narrower than one bin and wider than several are both covered.
    """
    rng = np.random.default_rng(rng_seed)
    points = rng.uniform(-8.0, 8.0, size=(4000, 3))
    centres = rng.uniform(-10.0, 10.0, size=(200, 3))
    radii = rng.uniform(0.2, 6.0, size=200)
    bins = _Bins(points, cell=2.0)

    query, found = bins.near_many(centres, radii)
    for index, (centre, radius) in enumerate(zip(centres, radii, strict=True)):
        expected = np.sort(bins.near(centre, radius))
        assert np.array_equal(np.sort(found[query == index]), expected)


def test_a_point_exactly_on_the_query_radius_is_inside_it():
    """The boundary the random sweep above cannot reach.

    Random points land on a query radius with probability zero, so relaxing the
    batched filter from `<=` to `<` passes every other test in this file — which
    is the shape of hole ROADMAP.md section 12 and the `guards that guard
    nothing` history keep recording. A rim node sitting exactly at
    `ring_radius + probe` is not exotic: the lattice is regular and the geometry
    is built from the same radii, so exact hits happen on symmetric fixtures
    every run.

    The distances here are exact in binary, so `<=` and `<` genuinely differ.
    """
    points = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.5, 0.0, 0.0]])
    bins = _Bins(points, cell=1.0)
    centres = np.zeros((1, 3))
    radii = np.array([2.0])

    _, found = bins.near_many(centres, radii)
    assert set(found.tolist()) == {0, 1, 2}
    # And the single-query path agrees, since the rim loop used to take it.
    assert set(bins.near(centres[0], 2.0).tolist()) == {0, 1, 2}


def test_the_batched_query_returns_its_pairs_grouped_by_query():
    """The rim loop slices per rim with `searchsorted`, which needs this to hold.

    Not an implementation detail: if the pairs came back unordered, the slices
    would silently hand each rim another rim's nodes, and every node would still
    be a real node near *some* rim — so the answer would be wrong in a way no
    shape check would catch.
    """
    rng = np.random.default_rng(1)
    bins = _Bins(rng.uniform(-8.0, 8.0, size=(2000, 3)), cell=2.0)
    query, _ = bins.near_many(rng.uniform(-8.0, 8.0, size=(50, 3)), np.full(50, 3.0))
    assert np.all(np.diff(query) >= 0)


def test_an_empty_batch_of_queries_is_answered_rather_than_crashed():
    """The last batch is short whenever the rim count is not a multiple of the size."""
    bins = _Bins(np.zeros((4, 3)), cell=1.0)
    query, found = bins.near_many(np.zeros((0, 3)), np.zeros(0))
    assert len(query) == len(found) == 0


def test_a_query_that_reaches_nothing_contributes_no_pairs():
    bins = _Bins(np.zeros((4, 3)), cell=1.0)
    query, found = bins.near_many(np.array([[100.0, 100.0, 100.0]]), np.array([0.5]))
    assert len(query) == len(found) == 0


def test_ragged_expands_variable_lengths_including_empty_ones():
    """An empty segment is ordinary here — a rim covering no occupied bin, a
    candidate with no blockers — so it must fall out rather than be special-cased.
    """
    segment, position = _ragged(np.array([3, 0, 2], dtype=np.int64))
    assert np.array_equal(segment, [0, 0, 0, 2, 2])
    assert np.array_equal(position, [0, 1, 2, 0, 1])


def test_ragged_of_nothing_is_nothing():
    segment, position = _ragged(np.zeros(0, dtype=np.int64))
    assert len(segment) == len(position) == 0
    segment, position = _ragged(np.array([0, 0], dtype=np.int64))
    assert len(segment) == len(position) == 0
