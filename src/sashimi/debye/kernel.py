"""The optional compiled kernel for the rim loop, and the reason it is optional.

`surface.py` classifies points against the solvent-excluded surface, and that is
86-92% of a debye solve. The numpy implementation there is the reference: it
defines the answer, it runs everywhere, and it is what the corpus is recorded
against. This module is a second implementation of its hottest loop, compiled by
numba, selected at run time when numba is installed.

**Measured worth, on the structures ROADMAP.md section 12 grades at.** Against
the numpy rim loop, masks bit-identical at every multigrid level:

    actin-monomer,  382 residues    8.2x on the finest level, 6.8x overall
    serum-albumin, 1,156 residues   9.5x on the finest level, 7.0x overall

`parallel=True` was measured and is not used: it bought 7.0x against 6.8x
single-threaded, which is nothing, and it would have made a library take four
cores without being asked. The gain here is compiled code, not concurrency —
the same conclusion the threading experiment in section 12 reached from the
other direction.

**Why it is an extra and not a dependency.** numba brings llvmlite, and the two
are ~145 MB installed. sashimi's whole proposition is that it installs anywhere
with nothing to fetch by hand, and quadrupling the install to speed up one
backend is not a trade to make on a caller's behalf. Callers who solve one small
structure should not pay it; callers doing real electrostatics on proteins
should, and `sashimi_capabilities` and the README both say so.

**The reference implementation stays authoritative.** This kernel is required to
be *bit*-identical, not close: `decided` feeds a boolean or, so a node claimed by
any rim is the node claimed by the first, and nothing downstream depends on
which. `tests/test_debye_kernel.py` asserts that on real geometry, and CI runs
the numpy path on two of its three legs and this one on the third. A kernel that
disagreed would be a bug in this file, never a new answer.
"""

from __future__ import annotations

import importlib.util
import os
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sashimi.protocol import FloatArray

__all__ = ["available", "decide_rims", "why_unavailable"]

# An escape hatch for measurement and for a caller who has numba installed for
# something else and does not want it used here. Checked every call rather than
# cached, so a test can flip it without reaching into module state.
DISABLE = "SASHIMI_NO_NUMBA"

# Spellings that mean "no, do not disable it". Everything else that is set at
# all means "disable", including a typo — see `_disabled`.
_FALSE = frozenset({"", "0", "false", "no", "off"})


def _disabled() -> bool:
    """Whether the environment asks for the reference path.

    **Unset is not the same as set to a false value, and neither is the same as
    set to anything else.** The first version of this read
    `os.environ.get(DISABLE)` for truthiness, which quietly made
    `SASHIMI_NO_NUMBA=false` and `SASHIMI_NO_NUMBA=0` *disable* the kernel — the
    exact opposite of what someone writing them means, and silently, since a
    correct-but-slow answer looks like a correct answer.

    So false spellings are honoured as false, and **anything else that is set is
    honoured as true, including `ture`**. That asymmetry is deliberate: the
    variable is named `NO_NUMBA`, so setting it at all expresses intent to turn
    the thing off, and turning it off is the safe direction — the answer is
    identical either way and only the wait changes. A typo that disables an
    optimisation costs seconds; a typo that silently ignores an operator's
    instruction is the kind of thing that is found much later.
    """
    value = os.environ.get(DISABLE)
    if value is None:
        return False
    return value.strip().lower() not in _FALSE


def available() -> bool:
    """Whether the compiled path will be used.

    Two-stage, and the second stage is the point. `find_spec` is cheap and
    answers "is numba installed" without paying the ~1 s import on every
    `import sashimi` — but **it is not proof that importing numba succeeds.**
    numba raises `ImportError` from its own `__init__` when numpy is outside its
    supported window (`numpy_version > (2, 5)` today), and this package declares
    `numpy>=2.0` with no upper bound. A caller with numba installed for
    something else and a newer numpy would have had `find_spec` say yes, the
    import fail deep inside a solve, and every `molecular` request die with an
    exception outside this project's error taxonomy — on a machine that solved
    fine before the accelerator existed.

    **An optional accelerator must never turn "slower" into "broken."** So the
    import is attempted once, behind a cache, and any failure is a permanent
    fall-through to the reference path.
    """
    if _disabled():
        return False
    if importlib.util.find_spec("numba") is None:
        return False
    return _importable()


@cache
def _importable() -> bool:
    """Whether `import numba` actually works, tried once and remembered.

    Deliberately catches `Exception` rather than `ImportError` alone: an llvmlite
    ABI mismatch in a conda environment surfaces as neither reliably, and there
    is no failure here worth propagating when a correct answer is one branch
    away.
    """
    try:
        import numba  # noqa: F401, PLC0415 — probing, not using
    except Exception:
        return False
    return True


def why_unavailable() -> str | None:
    """A sentence for a caller wondering why a solve is slow, or None if it is not."""
    if _disabled():
        return (
            f"the compiled kernel is disabled by {DISABLE}="
            f"{os.environ.get(DISABLE)!r} in the environment"
        )
    if importlib.util.find_spec("numba") is not None and not _importable():
        return (
            "numba is installed but will not import — most often numpy is "
            "outside the window numba supports. debye is solving on the "
            "pure-numpy surface path, which is correct and slower"
        )
    if importlib.util.find_spec("numba") is None:
        return (
            "numba is not installed, so debye is solving on the pure-numpy "
            "surface path. Installing `sashimi-electro[fast]` makes the surface "
            "classification about 7x faster, which is most of a solve — at the "
            "cost of roughly 145 MB, since numba brings llvmlite with it"
        )
    return None


# PLR0915 on both of these: the kernel is one flat numeric loop, and splitting it
# into helpers is exactly what must not happen — numba inlines across a compiled
# unit, and a Python-level call per rim is the cost this module exists to remove.
@cache
def _compiled() -> Any:  # noqa: PLR0915
    """Build the kernel once. Imported here so `import sashimi` never pays for numba."""
    from numba import njit  # noqa: PLC0415 — the whole point is that this is lazy

    @njit(cache=True, fastmath=False, nogil=True)
    def kernel(  # type: ignore[no-untyped-def]  # noqa: PLR0917, PLR0915, PLR0912
        points,
        order,
        starts,
        stops,
        codes,
        low_key,
        high_key,
        strides,
        bin_origin,
        cell,
        origins,
        normals,
        ring_radii,
        probe,
        blocker_flat,
        blocker_offset,
        blocker_count,
        coords,
        inflated,
        decided,
    ):  # pragma: no cover - compiled; exercised through `decide_rims`
        """One rim per iteration, with no array ever materialised.

        This is the whole reason the kernel wins. The numpy path has to build a
        `(rim, node)` pair list and then a `(node, blocker)` one, and the second
        is 69 million entries on a protein — every one of them a gathered
        coordinate. Here the same arithmetic happens in registers and the pair
        lists never exist, which is also why it needs no batch-size constant.
        """
        degenerate = 1e-12
        squared_probe = probe * probe
        for r in range(origins.shape[0]):
            ox, oy, oz = origins[r, 0], origins[r, 1], origins[r, 2]
            nx, ny, nz = normals[r, 0], normals[r, 1], normals[r, 2]
            ring = ring_radii[r]
            reach = ring + probe
            lo0 = max(int(np.floor((ox - reach - bin_origin[0]) / cell)), low_key[0])
            lo1 = max(int(np.floor((oy - reach - bin_origin[1]) / cell)), low_key[1])
            lo2 = max(int(np.floor((oz - reach - bin_origin[2]) / cell)), low_key[2])
            hi0 = min(int(np.floor((ox + reach - bin_origin[0]) / cell)), high_key[0])
            hi1 = min(int(np.floor((oy + reach - bin_origin[1]) / cell)), high_key[1])
            hi2 = min(int(np.floor((oz + reach - bin_origin[2]) / cell)), high_key[2])
            if lo0 > hi0 or lo1 > hi1 or lo2 > hi2:
                continue
            first_blocker = blocker_offset[r]
            last_blocker = first_blocker + blocker_count[r]

            for i in range(lo0, hi0 + 1):
                for j in range(lo1, hi1 + 1):
                    for k in range(lo2, hi2 + 1):
                        code = (
                            (i - low_key[0]) * strides[0]
                            + (j - low_key[1]) * strides[1]
                            + (k - low_key[2]) * strides[2]
                        )
                        at = np.searchsorted(codes, code)
                        if at >= codes.shape[0] or codes[at] != code:
                            continue
                        for slot in range(starts[at], stops[at]):
                            node = order[slot]
                            if decided[node]:
                                continue
                            dx = points[node, 0] - ox
                            dy = points[node, 1] - oy
                            dz = points[node, 2] - oz
                            if dx * dx + dy * dy + dz * dz > reach * reach:
                                continue
                            axial = dx * nx + dy * ny + dz * nz
                            rx = dx - axial * nx
                            ry = dy - axial * ny
                            rz = dz - axial * nz
                            length = np.sqrt(rx * rx + ry * ry + rz * rz)
                            if length <= degenerate:
                                continue
                            gap = length - ring
                            if gap * gap + axial * axial > squared_probe:
                                continue
                            # `ring * r / length`, associated exactly as the
                            # reference does it in `surface.py` — which computes
                            # `(ring_radii * radial) / length`. Hoisting
                            # `ring / length` into a scale factor changes the
                            # association and so the last bit, and a projected
                            # centre landing within an ulp of a blocker's radius
                            # would flip one node of the mask. Bit-identical is
                            # this module's contract, so it is arranged rather
                            # than hoped for.
                            px = ox + ring * rx / length
                            py = oy + ring * ry / length
                            pz = oz + ring * rz / length
                            legal = True
                            for b in range(first_blocker, last_blocker):
                                atom = blocker_flat[b]
                                ax = px - coords[atom, 0]
                                ay = py - coords[atom, 1]
                                az = pz - coords[atom, 2]
                                if ax * ax + ay * ay + az * az < inflated[atom] * inflated[atom]:
                                    legal = False
                                    break
                            if legal:
                                decided[node] = True

    return kernel


def decide_rims(  # noqa: PLR0917 — mirrors the reference loop's inputs
    points: FloatArray,
    bins: Any,
    origins: FloatArray,
    normals: FloatArray,
    ring_radii: FloatArray,
    blockers: tuple[np.ndarray, np.ndarray, np.ndarray],
    coords: FloatArray,
    inflated: FloatArray,
    probe: float,
    decided: np.ndarray,
) -> None:
    """Mark every node some rim's legal probe centre reaches. Mutates `decided`.

    Contiguity is forced rather than assumed: numba specialises on layout, so a
    non-contiguous view would either recompile or refuse, and both are worse
    than one copy of an array this size.
    """
    blocker_flat, blocker_offset, blocker_count = blockers
    _compiled()(
        np.ascontiguousarray(points),
        bins.order,
        bins.starts,
        bins.stops,
        bins.codes,
        bins.low_key,
        bins.high_key,
        bins.strides,
        bins.bin_origin,
        bins.cell,
        np.ascontiguousarray(origins),
        np.ascontiguousarray(normals),
        np.ascontiguousarray(ring_radii),
        float(probe),
        blocker_flat,
        blocker_offset,
        blocker_count,
        np.ascontiguousarray(coords),
        np.ascontiguousarray(inflated),
        decided,
    )
