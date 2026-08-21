"""The optional compiled kernels for the three surface families, and why they are optional.

`surface.py` classifies points against the solvent-excluded surface, and it does
it in three families — a probe on one atom, on two, on three. This module is a
second implementation of all three, compiled by numba, selected at run time when
numba is installed. The numpy implementations there stay the reference: they
define the answer, they run everywhere, and they are what the corpus is recorded
against.

**Measured worth per family**, on the finest lattice at 1.0 A, masks
bit-identical at every multigrid level:

    family      fas2, 59 aa    actin-monomer, 382 aa   serum-albumin, 1,156 aa
    radial          25.4x              28.1x                   28.4x
    toroidal         9.7x               9.6x                    9.0x
    vertex          17.9x              16.2x                   17.2x

**And what that is worth on a whole solve, which is much less**: 1.92x on fas2,
1.81x on actin-monomer, 1.73x on serum-albumin (149.33 s to 86.09 s of CPU), with
the energies identical to the last digit. The gap between the two tables is the
finding M7's port ended on, and ROADMAP.md section 12 records it: by the time
these three loops are compiled, they are 13% of a solve. The **one-time**
reduced-surface construction they read from — `_neighbours`, `_rims`,
`_probe_seats`, none of them compiled — is 55%, and it is built once per solve
rather than once per lattice, so no per-lattice speed-up touches it.

`parallel=True` was measured on the rim loop and is not used: it bought 7.0x
against 6.8x single-threaded, which is nothing, and it would have made a library
take four cores without being asked. The gain here is compiled code, not
concurrency — the same conclusion the threading experiment in section 12 reached
from the other direction.

**Why it is an extra and not a dependency.** numba brings llvmlite, and the two
are ~145 MB installed. sashimi's whole proposition is that it installs anywhere
with nothing to fetch by hand, and quadrupling the install to speed up one
backend is not a trade to make on a caller's behalf. Callers who solve one small
structure should not pay it; callers doing real electrostatics on proteins
should, and `sashimi_capabilities` and the README both say so.

**The reference implementation stays authoritative.** These kernels are required
to be *bit*-identical, not close: every family writes into a boolean or, so a
node claimed by any feature is the node claimed by the first, and nothing
downstream depends on which. `tests/test_debye_kernel.py` asserts that per family
on real geometry, and CI runs the numpy path on two of its three legs and these
on the third. A kernel that disagreed would be a bug in this file, never a new
answer.

**One honest limit on that claim.** The bar is exact equality, and the fixtures
reach it: a control mutation that stops the radial family marking anything
reddens five tests. But three *subtle* mutations — the two boundary-equality
flips, and hoisting `radius / length` out of the radial projection, which is
exactly the association trap the rim kernel below documents — move not one node
across 614,476 undecided nodes on two proteins at two resolutions. So the
floating-point discipline in these loops is a precaution that no fixture here
demonstrates the need for. It is kept because the cost is a comment and the
failure it prevents is silent, not because it has been caught failing.
"""

from __future__ import annotations

import importlib.util
import os
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sashimi.protocol import FloatArray

__all__ = [
    "available",
    "decide_radial",
    "decide_rims",
    "decide_seated",
    "why_unavailable",
]

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
            "classification 9-28x faster, which is about 1.8x on a whole solve "
            "— at the cost of roughly 145 MB, since numba brings llvmlite with it"
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


@cache
def _compiled_radial() -> Any:
    """Build the radial-family kernel once. Same laziness as `_compiled`."""
    from numba import njit  # noqa: PLC0415 — the whole point is that this is lazy

    @njit(cache=True, fastmath=False, nogil=True)
    def kernel(  # type: ignore[no-untyped-def]  # noqa: PLR0917, PLR0912
        first_axis,
        second_axis,
        third_axis,
        undecided,
        reachable,
        coords,
        inflated,
        test_flat,
        test_offset,
        test_count,
        degenerate,
    ):  # pragma: no cover - compiled; exercised through `decide_radial`
        """One atom per iteration, over the nodes inside its accessible sphere.

        The reference walks the same atoms and windows the lattice per atom with
        two `searchsorted` calls an axis, then builds a candidate point cloud,
        pushes it out and broadcasts it against the atom's neighbours. Every one
        of those is an array the size of the window, and the window on a protein
        at 1.0 Å holds a few hundred nodes — so the arithmetic is small and the
        allocation is not. Here the window is three index ranges and the pushed
        point never exists outside three registers.
        """
        atoms = coords.shape[0]
        for atom in range(atoms):
            radius = inflated[atom]
            if radius <= 0.0:
                continue
            cx, cy, cz = coords[atom, 0], coords[atom, 1], coords[atom, 2]
            lo0 = np.searchsorted(first_axis, cx - radius, "left")
            hi0 = np.searchsorted(first_axis, cx + radius, "right")
            if lo0 >= hi0:
                continue
            lo1 = np.searchsorted(second_axis, cy - radius, "left")
            hi1 = np.searchsorted(second_axis, cy + radius, "right")
            if lo1 >= hi1:
                continue
            lo2 = np.searchsorted(third_axis, cz - radius, "left")
            hi2 = np.searchsorted(third_axis, cz + radius, "right")
            if lo2 >= hi2:
                continue

            first_blocker = test_offset[atom]
            last_blocker = first_blocker + test_count[atom]
            for i in range(lo0, hi0):
                ox = first_axis[i] - cx
                for j in range(lo1, hi1):
                    oy = second_axis[j] - cy
                    for k in range(lo2, hi2):
                        if not undecided[i, j, k] or reachable[i, j, k]:
                            continue
                        oz = third_axis[k] - cz
                        length = np.sqrt(ox * ox + oy * oy + oz * oz)
                        # A node on the atom's own centre has no radial
                        # direction. It cannot be a shell node unless the atom
                        # has zero radius, and it is skipped rather than
                        # guarded against, exactly as the reference does.
                        if length <= degenerate:
                            continue
                        # `centre + radius * offset / distance`, associated as
                        # numpy associates it in `_radially_reachable`: the
                        # multiply happens before the divide. The rim kernel
                        # above records why that is arranged rather than hoped
                        # for — a candidate landing within an ulp of a
                        # blocker's radius decides a node either way.
                        px = cx + radius * ox / length
                        py = cy + radius * oy / length
                        pz = cz + radius * oz / length
                        legal = True
                        for b in range(first_blocker, last_blocker):
                            other = test_flat[b]
                            ax = px - coords[other, 0]
                            ay = py - coords[other, 1]
                            az = pz - coords[other, 2]
                            if ax * ax + ay * ay + az * az < inflated[other] * inflated[other]:
                                legal = False
                                break
                        if legal:
                            reachable[i, j, k] = True

    return kernel


@cache
def _compiled_seated() -> Any:
    """Build the vertex-family kernel once. Same laziness as `_compiled`."""
    from numba import njit  # noqa: PLC0415 — the whole point is that this is lazy

    @njit(cache=True, fastmath=False, nogil=True)
    def kernel(  # type: ignore[no-untyped-def]  # noqa: PLR0917
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
        seats,
        radius,
        hit,
    ):  # pragma: no cover - compiled; exercised through `decide_seated`
        """One node per iteration, against the seats binned around it.

        The mirror image of the rim kernel's walk: there the nodes were binned
        and each rim swept them, here the seats are binned and each node asks
        its own bin and the twenty-six around it. That is the reference's
        arrangement too — `_within` bins the centres — and it is kept because
        the seats are the smaller set on a protein and are built once for every
        lattice that asks.
        """
        squared = radius * radius
        for node in range(points.shape[0]):
            px, py, pz = points[node, 0], points[node, 1], points[node, 2]
            key0 = int(np.floor((px - bin_origin[0]) / cell))
            key1 = int(np.floor((py - bin_origin[1]) / cell))
            key2 = int(np.floor((pz - bin_origin[2]) / cell))
            lo0 = max(key0 - 1, low_key[0])
            hi0 = min(key0 + 1, high_key[0])
            lo1 = max(key1 - 1, low_key[1])
            hi1 = min(key1 + 1, high_key[1])
            lo2 = max(key2 - 1, low_key[2])
            hi2 = min(key2 + 1, high_key[2])
            if lo0 > hi0 or lo1 > hi1 or lo2 > hi2:
                continue
            found = False
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
                            seat = order[slot]
                            dx = px - seats[seat, 0]
                            dy = py - seats[seat, 1]
                            dz = pz - seats[seat, 2]
                            if dx * dx + dy * dy + dz * dz <= squared:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                hit[node] = True

    return kernel


def decide_radial(  # noqa: PLR0917 — mirrors the reference loop's inputs
    axes: list[FloatArray],
    coords: FloatArray,
    inflated: FloatArray,
    testable: tuple[np.ndarray, np.ndarray, np.ndarray],
    undecided: np.ndarray,
    reachable: np.ndarray,
    degenerate: float,
) -> None:
    """Mark every node some atom's own sphere reaches. Mutates `reachable`."""
    test_flat, test_offset, test_count = testable
    _compiled_radial()(
        np.ascontiguousarray(axes[0]),
        np.ascontiguousarray(axes[1]),
        np.ascontiguousarray(axes[2]),
        undecided,
        reachable,
        np.ascontiguousarray(coords),
        np.ascontiguousarray(inflated),
        test_flat,
        test_offset,
        test_count,
        float(degenerate),
    )


def decide_seated(
    points: FloatArray,
    bins: Any,
    seats: FloatArray,
    radius: float,
    hit: np.ndarray,
) -> None:
    """Mark every node within `radius` of a seat. Mutates `hit`."""
    _compiled_seated()(
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
        np.ascontiguousarray(seats),
        float(radius),
        hit,
    )
