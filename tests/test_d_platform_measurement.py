"""Grade initiative D's seat kernel on a second platform, against the reference.

**A measurement, not a feature.** Every number this project has for the seat
kernel was taken on one arm64 laptop. `_rim` once differed across platforms on
`x ** 2` versus `x * x`, and bit-identity is what caught it; the kernel in
`tests/prototype_d_seats.py` adds a construct nothing shipped uses — an array
reallocated inside an `njit` body — so it is graded here before it is trusted.

The reference is the numpy `surface._probe_seats`, which defines the answer, not
the shipped kernel. `_Spheres` is built once and handed to both sides, so a
shared error in the neighbour search cannot cancel; that is the trap the
`_masks` docstring in `test_debye_kernel.py` records the file falling into once.
"""

from __future__ import annotations

import importlib.util
import time
from typing import Any

import numpy as np
import pytest

from sashimi import corpus
from sashimi.debye import kernel
from sashimi.debye import surface as surface_module
from sashimi.protocol import SolventModel, SurfaceModel
from tests.prototype_d_seats import probe_seats

needs_numba = pytest.mark.skipif(
    importlib.util.find_spec("numba") is None, reason="the compiled kernel is not installed"
)

# The seven homogeneous rungs (ROADMAP.md:6130-6131), plus ala-gly as the small
# control. `serum-albumin` is the one where any scaling claim is decided and is
# also the one that exercises the grown buffer hardest, so it stays in.
RUNGS: list[tuple[str, str | None]] = [
    ("ala-gly", None),
    ("fas2", "fas2-molecular"),
    ("barstar", "barstar-molecular"),
    ("barnase", "barnase-molecular"),
    ("1a63", "protein-1a63-molecular"),
    ("actin-monomer", "actin-monomer"),
    ("acetylcholinesterase", "acetylcholinesterase"),
    ("serum-albumin", "serum-albumin"),
]

ALA_GLY = "tests/data/ala-gly.pqr"


def _spheres(case_name: str | None) -> Any:
    """The `_Spheres` for a rung, built once on the fast path.

    Building it under the reference is element-identical and ~40x slower, so the
    build is not what is under test here — the seat enumeration is.
    """
    from sashimi.pqr import read_pqr  # noqa: PLC0415

    if case_name is None:
        structure = read_pqr(ALA_GLY)
        radius = SolventModel(surface_model=SurfaceModel.MOLECULAR).surface_radius
    else:
        case = {c.name: c for c in corpus.MANIFEST}[case_name]
        structure = case.structure()
        radius = case.solvent.surface_radius
    return surface_module._Spheres.around(structure, radius)


def _reference(spheres: Any, monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    """The numpy `_probe_seats`, forced, with the forcing asserted both ways."""
    with monkeypatch.context() as patched:
        patched.setenv(kernel.DISABLE, "true")
        assert not kernel.available(), "the reference half is not the reference"
        return surface_module._probe_seats(spheres)


@needs_numba
@pytest.mark.parametrize(("rung", "case_name"), RUNGS, ids=[r for r, _ in RUNGS])
def test_the_single_pass_seat_kernel_is_identical_on_this_platform(
    rung: str, case_name: str | None, monkeypatch: pytest.MonkeyPatch, record_property: Any
) -> None:
    """Exact bits and exact row order, against the numpy reference, on every rung.

    Row order is the bar rather than the seat set: the seats reach an answer
    through `_within`, which reduces with `any`, so an ordered comparison is
    strictly stronger than the answer needs and is therefore the one to make.
    """
    spheres = _spheres(case_name)
    assert kernel.available(), "the compiled half of this comparison is not compiled"

    started = time.perf_counter()
    reference = _reference(spheres, monkeypatch)
    reference_seconds = time.perf_counter() - started
    assert len(reference) > 0, "no probe seats against three atoms in this fixture"

    # Every variant that a shipped version might choose between, plus `seed=1`,
    # which forces the buffer to reallocate on nearly every emitted row instead
    # of once — the construct being graded, exercised rather than merely touched.
    for label, kwargs in (
        ("ascending", {"nearest": False}),
        ("nearest-first", {"nearest": True}),
        ("nearest-first, grown from 1", {"nearest": True, "seed": 1}),
    ):
        got = probe_seats(spheres, **kwargs)
        assert got.shape == reference.shape, f"{label}: {got.shape} against {reference.shape}"
        assert np.array_equal(reference, got), f"{label} diverged from the reference on {rung}"

    # And the shipped kernel on the same inputs, so a failure above can be told
    # apart from a platform defect that main already carries.
    assert np.array_equal(
        reference,
        kernel.probe_seats(
            spheres.coords,
            spheres.inflated,
            spheres.sorted_testable_table,
            surface_module._overlapping_pairs(
                spheres.neighbours, spheres.inflated, len(spheres.coords)
            ),
            surface_module.DEGENERATE,
        ),
    ), f"the SHIPPED kernel diverges from the reference on {rung} — a defect on main"

    record_property("atoms", len(spheres.coords))
    record_property("seat_rows", len(reference))
    record_property("reference_seconds", round(reference_seconds, 3))


@needs_numba
def test_the_row_order_comparison_can_actually_fail() -> None:
    """The mutation that reddens it, so the gate above is not a guard that guards nothing.

    One cursor emitting both mirrors of a triple adjacently gives the identical
    seat *set* and a different row order. It passes every set-wise check and
    every downstream use, and it is exactly what an ordered comparison exists to
    catch — so if this test ever goes green the comparison has stopped working.
    """
    spheres = _spheres("fas2-molecular")
    good = probe_seats(spheres, nearest=True)
    trapped = probe_seats(spheres, nearest=True, trap=True)

    assert trapped.shape == good.shape, "the trap changed the seat count, not just the order"
    assert not np.array_equal(good, trapped), "the ordered comparison no longer detects reordering"
    assert np.array_equal(good[np.lexsort(good.T)], trapped[np.lexsort(trapped.T)]), (
        "the trap changed the seat set, so it is testing the wrong thing"
    )
    differing = int((good != trapped).any(axis=1).sum())
    assert differing > len(good) // 4, f"only {differing} of {len(good)} rows moved"
