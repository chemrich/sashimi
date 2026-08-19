"""The optional compiled kernel, and the one thing it is not allowed to be: different.

`sashimi.debye.kernel` is a second implementation of the hottest loop in
`surface.py`, selected when `sashimi-electro[fast]` is installed. The numpy path
is the reference — it defines the answer and it is what the corpus is recorded
against — so every test here is about the two agreeing, not about either being
fast.

**Bit-identical, not close.** `decided` feeds a boolean or, so a node claimed by
any rim is the node claimed by the first and nothing downstream depends on which.
That makes exact equality the right bar, and a tolerance here would pass exactly
the bug the check exists for — the same reasoning `tests/test_bench.py` and
`tests/test_debye_m7.py` both record.

**Skipping is the danger, so it is made loud.** numba is absent from the default
development environment by design, which means most runs of this file exercise
nothing. That is the "a skipped tier and a passing tier look identical" trap this
project keeps hitting — CI therefore installs the extra on its `full` leg and
asserts the compiled path was actually taken, and
`test_the_environment_agrees_with_itself` below fails rather than skips if the
two ever disagree about whether the kernel is present.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from sashimi.debye import kernel
from sashimi.debye import surface as surface_module
from sashimi.debye.grid import axis_coordinates, size_grid
from sashimi.debye.surface import ReducedSurface
from sashimi.pqr import read_pqr
from sashimi.protocol import GridSpec, SolventModel, SurfaceModel

# Small enough to run on every push, and it genuinely reaches the rim loop: a
# lone sphere never does, because its two boundaries coincide.
PEPTIDE = "tests/data/ala-gly.pqr"
HAVE_NUMBA = importlib.util.find_spec("numba") is not None
needs_numba = pytest.mark.skipif(not HAVE_NUMBA, reason="numba is an optional extra")


def _masks(structure: str, resolution: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """The molecular mask down both paths, on the same lattice."""
    pqr = read_pqr(structure)
    solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR)
    axes = axis_coordinates(size_grid(pqr, GridSpec(resolution=resolution)))
    surface = ReducedSurface(pqr, solvent)

    os.environ[kernel.DISABLE] = "true"
    try:
        reference = surface.inside(axes)
    finally:
        del os.environ[kernel.DISABLE]
    return reference, ReducedSurface(pqr, solvent).inside(axes)


def test_the_environment_agrees_with_itself(monkeypatch):
    """`available()` must match reality, or every skip below is meaningless.

    It answers by `find_spec` rather than by importing numba, since importing it
    costs about a second on every `import sashimi`. That is a real optimisation
    and therefore a real thing to get wrong.
    """
    monkeypatch.delenv(kernel.DISABLE, raising=False)
    assert kernel.available() == HAVE_NUMBA


@pytest.mark.parametrize("spelling", ["TRUE", "True", "true", "1", "yes", "on", "ture"])
def test_the_disable_switch_accepts_how_people_write_it(spelling: str, monkeypatch):
    """Readable spellings, and a typo, all mean disable.

    The asymmetry is deliberate and documented in `kernel._disabled`: the
    variable is named `NO_NUMBA`, so setting it at all expresses intent to turn
    the kernel off, and off is the safe direction — the answer is identical
    either way and only the wait changes. `ture` is in this list on purpose.
    """
    monkeypatch.setenv(kernel.DISABLE, spelling)
    assert not kernel.available()
    reason = kernel.why_unavailable()
    assert reason is not None
    assert kernel.DISABLE in reason


@pytest.mark.parametrize("spelling", ["0", "false", "False", "FALSE", "no", "off", ""])
def test_a_false_spelling_does_not_disable_it(spelling: str, monkeypatch):
    """The trap the first version of this shipped with.

    `os.environ.get(DISABLE)` is truthy for any non-empty string, so
    `SASHIMI_NO_NUMBA=false` and `=0` both *disabled* the kernel — the opposite
    of what they say, and silently, because a correct-but-slow answer is
    indistinguishable from a correct one.
    """
    monkeypatch.setenv(kernel.DISABLE, spelling)
    assert kernel.available() == HAVE_NUMBA


def test_disabling_it_yields_the_reference_path(monkeypatch):
    """Otherwise the reference half of every comparison below is not the reference."""
    monkeypatch.setenv(kernel.DISABLE, "true")
    assert not kernel.available()
    assert kernel.DISABLE in str(kernel.why_unavailable())


def test_an_absent_kernel_explains_itself_and_a_present_one_says_nothing():
    if HAVE_NUMBA:
        assert kernel.why_unavailable() is None
    else:
        reason = kernel.why_unavailable()
        assert reason is not None
        assert "numba" in reason
        # The size of the ask is part of the message: a caller deciding whether
        # to install it needs the cost, not just the benefit.
        assert "145 MB" in reason


@needs_numba
def test_the_compiled_mask_is_identical_on_a_peptide():
    reference, compiled = _masks(PEPTIDE)
    assert reference.any(), "the fixture decided nothing, so this compares two empties"
    assert np.array_equal(reference, compiled)


@needs_numba
def test_the_compiled_mask_is_identical_on_a_protein():
    """Peptide geometry is not protein geometry: 62 rims against 11,380.

    The trap M4 records is a construction that is right on small solutes and
    wrong where three atoms meet, which is most of a real surface.
    """
    reference, compiled = _masks("tests/data/apbs-examples/fas2.pqr", resolution=1.0)
    assert reference.any()
    assert np.array_equal(reference, compiled)


@needs_numba
def test_the_compiled_energy_is_identical_to_the_last_digit():
    """The claim the corpus recordings rest on, stated end to end."""
    from sashimi.debye import DebyeSolver  # noqa: PLC0415 — keeps import cost local
    from sashimi.protocol import FiniteDifferenceRequest  # noqa: PLC0415

    request = FiniteDifferenceRequest(
        structure=read_pqr(PEPTIDE),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
        grid=GridSpec(resolution=0.5),
        want_potential=False,
    )
    os.environ[kernel.DISABLE] = "true"
    try:
        reference = DebyeSolver().solve(request).energy_kj_mol
    finally:
        del os.environ[kernel.DISABLE]
    compiled = DebyeSolver().solve(request).energy_kj_mol
    assert reference is not None
    assert repr(compiled) == repr(reference)


@needs_numba
def test_the_blocker_table_round_trips_every_rim():
    """The CSR both paths index. An off-by-one here is a wrong surface, silently."""
    pqr = read_pqr(PEPTIDE)
    surface = ReducedSurface(pqr, SolventModel(surface_model=SurfaceModel.MOLECULAR))
    flat, offset, count = surface_module._blocker_table(surface.rims)
    assert len(offset) == len(count) == len(surface.rims)
    assert int(count.sum()) == len(flat)
    for index, (*_, blockers) in enumerate(surface.rims):
        start = int(offset[index])
        assert np.array_equal(flat[start : start + int(count[index])], blockers)
