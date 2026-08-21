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


def _masks(structure: str, monkeypatch, resolution: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """The molecular mask down both paths, on the same lattice.

    `monkeypatch` rather than mutating `os.environ` directly: a developer who
    has `SASHIMI_NO_NUMBA` exported — which is how the README says to turn the
    kernel off — would otherwise lose it for the rest of the pytest process the
    first time this ran, and every later test would silently switch paths
    against their explicit instruction.

    **Both halves set the variable, and the compiled half checks it took.** The
    first version only forced the reference half and let the other inherit the
    environment — so for exactly that developer, with numba installed and
    `SASHIMI_NO_NUMBA` exported, both halves ran the numpy path and every
    comparison in this file passed by comparing a run to itself. A check that
    cannot fail, in the file whose entire purpose is one check.
    """
    pqr = read_pqr(structure)
    solvent = SolventModel(surface_model=SurfaceModel.MOLECULAR)
    axes = axis_coordinates(size_grid(pqr, GridSpec(resolution=resolution)))

    with monkeypatch.context() as patched:
        patched.setenv(kernel.DISABLE, "true")
        reference = ReducedSurface(pqr, solvent).inside(axes)
    with monkeypatch.context() as patched:
        patched.delenv(kernel.DISABLE, raising=False)
        assert kernel.available(), "the compiled half of this comparison is not compiled"
        compiled = ReducedSurface(pqr, solvent).inside(axes)
    return reference, compiled


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


def test_an_absent_kernel_explains_itself_and_a_present_one_says_nothing(monkeypatch):
    """`delenv` first: the README documents exporting `SASHIMI_NO_NUMBA`, and a
    developer who has taken that advice would otherwise fail this assertion for
    doing what they were told."""
    monkeypatch.delenv(kernel.DISABLE, raising=False)
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
def test_a_numba_that_will_not_import_falls_back_rather_than_crashing(monkeypatch):
    """The regression an optional accelerator must never introduce.

    Needs numba present to be meaningful: without it `find_spec` short-circuits
    first and the branch under test is unreachable, so this asserts nothing on
    a machine that has no numba at all.

    numba raises `ImportError` from its own `__init__` when numpy is outside the
    window it supports, and this package pins no numpy ceiling — so `find_spec`
    saying yes is not proof the import succeeds. Before the guard, that
    environment died inside `_toroidally_reachable` on every `molecular` solve,
    on a machine that had worked before the extra existed. Slower is acceptable;
    broken is not.
    """
    monkeypatch.delenv(kernel.DISABLE, raising=False)
    kernel._importable.cache_clear()
    monkeypatch.setattr(kernel, "_importable", lambda: False)
    assert not kernel.available()
    reason = kernel.why_unavailable()
    assert reason is not None
    assert "will not import" in reason


@needs_numba
def test_the_compiled_mask_is_identical_on_a_peptide(monkeypatch):
    reference, compiled = _masks(PEPTIDE, monkeypatch)
    assert reference.any(), "the fixture decided nothing, so this compares two empties"
    assert np.array_equal(reference, compiled)


@needs_numba
def test_the_compiled_mask_is_identical_on_a_protein(monkeypatch):
    """Peptide geometry is not protein geometry: 62 rims against 11,380.

    The trap M4 records is a construction that is right on small solutes and
    wrong where three atoms meet, which is most of a real surface.
    """
    reference, compiled = _masks("tests/data/apbs-examples/fas2.pqr", monkeypatch, resolution=1.0)
    assert reference.any()
    assert np.array_equal(reference, compiled)


@needs_numba
def test_the_compiled_energy_is_identical_to_the_last_digit(monkeypatch):
    """The claim the corpus recordings rest on, stated end to end."""
    from sashimi.debye import DebyeSolver  # noqa: PLC0415 — keeps import cost local
    from sashimi.protocol import FiniteDifferenceRequest  # noqa: PLC0415

    request = FiniteDifferenceRequest(
        structure=read_pqr(PEPTIDE),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
        grid=GridSpec(resolution=0.5),
        want_potential=False,
    )
    with monkeypatch.context() as patched:
        patched.setenv(kernel.DISABLE, "true")
        reference = DebyeSolver().solve(request).energy_kj_mol
    with monkeypatch.context() as patched:
        patched.delenv(kernel.DISABLE, raising=False)
        assert kernel.available(), "the compiled half of this comparison is not compiled"
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


def _per_family(
    structure: str, monkeypatch, resolution: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each family's mask down both paths, on one lattice and one geometry.

    The families are chained rather than run independently, because that is how
    `inside()` runs them: each is asked only about the nodes its predecessors
    left undecided. Grading one on the whole shell would grade it on a node set
    the solver never hands it — and the vertex family in particular exists for
    the junctions the other two cannot reach, so on the full shell it would be
    asked mostly about nodes that are already decided.
    """
    pqr = read_pqr(structure)
    surface = ReducedSurface(pqr, SolventModel(surface_model=SurfaceModel.MOLECULAR))
    axes = axis_coordinates(size_grid(pqr, GridSpec(resolution=resolution)))
    union = surface_module.inside_union_of_spheres
    undecided = union(axes, pqr.coords, pqr.radii + surface.probe) & ~union(
        axes, pqr.coords, pqr.radii
    )

    families = {
        "radial": lambda still: surface_module._radially_reachable(axes, surface.spheres, still),
        "toroidal": lambda still: surface_module._toroidally_reachable(axes, surface, still),
        "vertex": lambda still: surface_module._vertex_reachable(axes, surface, still),
    }
    out = {}
    still = undecided
    for name, family in families.items():
        with monkeypatch.context() as patched:
            patched.setenv(kernel.DISABLE, "true")
            reference = family(still)
        with monkeypatch.context() as patched:
            patched.delenv(kernel.DISABLE, raising=False)
            assert kernel.available(), "the compiled half of this comparison is not compiled"
            out[name] = (reference, family(still))
        still = still & ~reference
    return out


@needs_numba
@pytest.mark.parametrize("family", ["radial", "toroidal", "vertex"])
def test_every_compiled_family_is_identical_on_a_protein(family: str, monkeypatch):
    """All three families are compiled now, so all three are held to the same bar.

    The count assertion is not decoration. Two of these families decide a few
    hundred nodes where the first decides tens of thousands, and a comparison of
    two empty masks passes for the wrong reason — which is this repository's most
    frequent defect and the reason the vertex family is chained onto the other
    two above rather than run on the whole shell.
    """
    reference, compiled = _per_family(
        "tests/data/apbs-examples/fas2.pqr", monkeypatch, resolution=1.0
    )[family]
    assert reference.sum() > 0, f"the {family} family decided nothing, so this compares two empties"
    assert np.array_equal(reference, compiled)


@needs_numba
def test_the_compiled_energy_is_identical_on_a_protein(monkeypatch):
    """The peptide reaches every family; only a protein reaches them at scale.

    `fas2` has 906 atoms against the dipeptide's 20, and the families divide its
    shell 4,732 / 1,928 / 446 — so all three carry weight here and a kernel that
    agreed on the peptide by deciding nothing would show up.
    """
    from sashimi.debye import DebyeSolver  # noqa: PLC0415 — keeps import cost local
    from sashimi.protocol import FiniteDifferenceRequest  # noqa: PLC0415

    request = FiniteDifferenceRequest(
        structure=read_pqr("tests/data/apbs-examples/fas2.pqr"),
        solvent=SolventModel(surface_model=SurfaceModel.MOLECULAR),
        grid=GridSpec(resolution=1.0),
        want_potential=False,
    )
    with monkeypatch.context() as patched:
        patched.setenv(kernel.DISABLE, "true")
        reference = DebyeSolver().solve(request).energy_kj_mol
    with monkeypatch.context() as patched:
        patched.delenv(kernel.DISABLE, raising=False)
        assert kernel.available(), "the compiled half of this comparison is not compiled"
        compiled = DebyeSolver().solve(request).energy_kj_mol
    assert reference is not None
    assert repr(compiled) == repr(reference)


def test_the_ragged_table_round_trips_every_atoms_neighbours():
    """The CSR the radial kernel indexes, which is `_blocker_table`'s generalisation.

    Same failure mode as the rim table and a wider blast radius: an off-by-one
    here gives an atom somebody else's blockers, and the surface comes out
    subtly wrong everywhere rather than visibly wrong somewhere.
    """
    pqr = read_pqr(PEPTIDE)
    surface = ReducedSurface(pqr, SolventModel(surface_model=SurfaceModel.MOLECULAR))
    flat, offset, count = surface_module._ragged_table(surface.spheres.testable)
    assert len(offset) == len(count) == len(surface.spheres.testable)
    assert int(count.sum()) == len(flat)
    for index, testable in enumerate(surface.spheres.testable):
        start = int(offset[index])
        assert np.array_equal(flat[start : start + int(count[index])], testable)
