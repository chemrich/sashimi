"""The optional compiled kernels for debye's surface geometry, and why they are optional.

`surface.py` builds the solvent-excluded surface and then classifies grid points
against it, and that is nearly all of a debye solve. This module is a second
implementation of the six loops it does that in, compiled by numba, selected at
run time when numba is installed. The numpy implementations there stay the
reference: they define the answer, they run everywhere, and they are what the
corpus is recorded against.

**Measured worth, loop by loop**, at 1.0 A, interleaved and best of three:

    built once per solve            fas2, 59 aa   actin, 382 aa   albumin, 1,156 aa
      _neighbours                       87x           116x              94x
      _rims                              12x           --                --
      _probe_seats                        6.0x         10.2x             --

    per lattice, sixteen a solve
      _radially_reachable                25.4x         28.1x             28.4x
      _toroidally_reachable               9.7x          9.6x              9.0x
      _vertex_reachable                  17.9x         16.2x             17.2x

**And what that is worth on a whole solve**, CPU seconds, energies identical to
the last digit:

    fas2, 59 aa            5.82 -> 1.39 s     4.19x
    actin-monomer, 382 aa 62.52 -> 18.86 s    3.31x
    serum albumin, 1,156  155.21 -> 45.99 s   3.37x

**The two halves of that table were compiled in the wrong order, and the record
says so.** M7 compiled the classification families first, on a profile that
charged the one-time builders to whichever family read them through a
`cached_property` — so `_probe_seats`, the largest single item in a solve, was
reported as part of a family that is 0.2% of one. Compiling all three families
was worth 1.73x; compiling the three builders they read from took the same solve
from 86 s to 46 s. ROADMAP.md section 12 has the exclusive re-measurement.

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
to be *bit*-identical, not close, and the bar means two different things for the
two halves. A classification family writes into a boolean or, so a node claimed
by any feature is the node claimed by the first and the floating-point
association only has to be *arranged* to match. A builder returns **geometry** —
rim circles, seat coordinates — that later stages compare against radii, so one
ulp there is a different surface and the association has to be *proved*.
`tests/test_debye_kernel.py` asserts both loop by loop on real geometry, and CI
runs the numpy path on two of its three legs and these on the third.

**That distinction found a defect in the reference, not in a kernel.** `x ** 2`
on a scalar is a call to the platform's `pow`, which here is not correctly
rounded — so `_rim` was computing a square that `x * x` gets right and that no
compiled kernel could reproduce. See `surface._rim`. Every recorded corpus
energy passing through a rim depended on whose libm ran it; all 58 cases
reproduce with the multiplication.

**One honest limit on the classification half of that claim.** The bar is exact
equality, and the fixtures reach it: a control mutation that stops the radial
family marking anything reddens five tests. But three *subtle* mutations — the
two boundary-equality
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

from sashimi.protocol import DIMENSIONS

if TYPE_CHECKING:
    from sashimi.protocol import FloatArray

__all__ = [
    "available",
    "decide_radial",
    "decide_rims",
    "decide_seated",
    "distance_rims",
    "enumerate_rims",
    "mark_union",
    "neighbour_lists",
    "probe_seats",
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


@cache
def _compiled_distance() -> Any:  # noqa: PLR0915
    """The distance twin of `_compiled`. Built once, lazily, for the same reason."""
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
        origins,
        normals,
        ring_radii,
        probe,
        blocker_flat,
        blocker_offset,
        blocker_count,
        coords,
        inflated,
        best,
    ):  # pragma: no cover - compiled; exercised through `distance_rims`
        """One rim per iteration, keeping the distance the boolean twin discards.

        **Three lines of `_compiled` are deliberately absent, and each would be a
        wrong answer rather than a slow one.**

        - *No `if decided[node]: continue`.* A boolean or does not care which rim
          claimed a node; a minimum does, so every rim that reaches a node has to
          be offered. `surface._toroidal_distance` says the same thing about why
          it has no `decided` array at all.
        - *No probe cull.* `_compiled` skips a pair once
          `gap*gap + axial*axial > probe*probe`, which is exactly the boolean
          question. The distance is consumed out to `2 * probe`
          (`surface.ReducedSurface.signed_gap`), so those pairs reach the answer.
          Transcribing that line moves 1,571 nodes on `fas2` at 1.0 A, 733 of
          them inside the ramp band — which is why it is the control mutation
          `tests/test_debye_kernel.py` reinstates to prove the gate can fail.
        - *No `squared_probe` at all*, so the omission cannot be undone by
          accident later.

        Everything else is transcribed operation for operation, because the
        contract here is bit-identity and this loop returns **geometry**: `best`
        is compared against a ramp width downstream, so an ulp is a different
        dielectric map rather than a rounding difference. `ring * rx / length`
        keeps the reference's association, three-term sums stay left to right,
        and every square is a multiplication — see this module's header for the
        `x ** 2` defect that rule comes from.
        """
        degenerate = 1e-12
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
                                gap = length - ring
                                apart = np.sqrt(gap * gap + axial * axial)
                                # Sequential, not a scatter-reduce, and that is
                                # measured rather than assumed: the reference's
                                # `np.minimum.at` has unique indices within every
                                # call, and `apart` is `sqrt(square + square)` so
                                # it can be neither NaN nor -0.0 — the two values
                                # that would make a minimum order-dependent.
                                best[node] = min(best[node], apart)

    return kernel


def distance_rims(  # noqa: PLR0917 — mirrors the reference loop's inputs
    points: FloatArray,
    bins: Any,
    origins: FloatArray,
    normals: FloatArray,
    ring_radii: FloatArray,
    blockers: tuple[np.ndarray, np.ndarray, np.ndarray],
    coords: FloatArray,
    inflated: FloatArray,
    probe: float,
    best: FloatArray,
) -> None:
    """Nearest legal rim projection per node, as a distance. Mutates `best`.

    `decide_rims` with the verdict replaced by the quantity the verdict was read
    from. `best` arrives pre-filled with `inf` and is lowered in place, so the
    caller's `np.minimum` against the accumulated `nearest` is unchanged.
    """
    blocker_flat, blocker_offset, blocker_count = blockers
    _compiled_distance()(
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
        best,
    )


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


@cache
def _compiled_neighbours() -> Any:
    """Build the neighbour-search kernel once. Same laziness as `_compiled`."""
    from numba import njit  # noqa: PLC0415 — the whole point is that this is lazy

    @njit(cache=True, fastmath=False, nogil=True)
    def kernel(  # type: ignore[no-untyped-def]  # noqa: PLR0917, PLR0912
        coords,
        inflated,
        keys,
        order,
        starts,
        stops,
        codes,
        low_key,
        high_key,
        strides,
        counting,
        offset,
        flat,
    ):  # pragma: no cover - compiled; exercised through `neighbour_lists`
        """Which atoms overlap which, in the reference's own order.

        **Two passes over one loop body, and the flag is why.** The output is
        ragged and its size is the answer — 1.3 million entries at 18,242 atoms —
        so it cannot be preallocated from anything cheaper than the search
        itself. Counting first and filling second runs the distance test twice;
        a worst-case buffer would be the atoms in twenty-seven bins for every
        atom, which is 114 MB against the 10 MB the answer needs. The test is a
        subtract, a square and a compare, and the alternative was measured in
        `PAIR_BATCH`'s comment: memory is the axis that bites here, not
        arithmetic.

        **The twenty-seven offsets are walked in the reference's order and each
        bin in ascending atom index**, which makes the two paths comparable
        element by element rather than as sets. Nothing downstream depends on
        the order — every consumer either sorts or reduces with `all`/`any` —
        but a test that can say `array_equal` is a better test than one that has
        to say "same set", and the cost of arranging it is a stable sort that
        was happening anyway.
        """
        total = 0
        for atom in range(coords.shape[0]):
            if inflated[atom] <= 0.0:
                continue
            cx, cy, cz = coords[atom, 0], coords[atom, 1], coords[atom, 2]
            reach = inflated[atom]
            found = 0
            at_atom = 0 if counting else offset[atom]
            for di in range(-1, 2):
                first = keys[atom, 0] + di
                if first < low_key[0] or first > high_key[0]:
                    continue
                for dj in range(-1, 2):
                    second = keys[atom, 1] + dj
                    if second < low_key[1] or second > high_key[1]:
                        continue
                    for dk in range(-1, 2):
                        third = keys[atom, 2] + dk
                        # Clipped to the occupied extent before encoding, not
                        # after. The encoding is injective only inside it: a key
                        # one past the top of the second axis encodes to exactly
                        # the code of `(first + 1, low, low)`, so an unclipped
                        # walk would find a bin that is nowhere near the atom and
                        # hand it somebody else's neighbours.
                        if third < low_key[2] or third > high_key[2]:
                            continue
                        code = (
                            (first - low_key[0]) * strides[0]
                            + (second - low_key[1]) * strides[1]
                            + (third - low_key[2]) * strides[2]
                        )
                        at = np.searchsorted(codes, code)
                        if at >= codes.shape[0] or codes[at] != code:
                            continue
                        for slot in range(starts[at], stops[at]):
                            other = order[slot]
                            if other == atom:
                                continue
                            dx = coords[other, 0] - cx
                            dy = coords[other, 1] - cy
                            dz = coords[other, 2] - cz
                            # `np.linalg.norm` of a three-vector, compared
                            # against a sum of radii — kept as a square root
                            # rather than folded into a squared comparison,
                            # because the two disagree in the last bit and the
                            # reference is the one with the root in it.
                            if np.sqrt(dx * dx + dy * dy + dz * dz) < reach + inflated[other]:
                                if not counting:
                                    flat[at_atom + found] = other
                                found += 1
            if counting:
                offset[atom] = found
            total += found
        return total

    return kernel


def neighbour_lists(
    coords: FloatArray,
    inflated: FloatArray,
    keys: np.ndarray,
    bins: Any,
    order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every atom's overlapping neighbours, as flat values, offsets and counts.

    `order` is the bin index's slots mapped back to atom numbers: the bins are
    built over the live atoms alone, exactly as the reference builds its dict,
    so a slot addresses that subset and not the structure.
    """
    compiled = _compiled_neighbours()

    def pass_over(counting: bool, out: np.ndarray, flat: np.ndarray) -> int:
        return int(
            compiled(
                np.ascontiguousarray(coords),
                np.ascontiguousarray(inflated),
                keys,
                order,
                bins.starts,
                bins.stops,
                bins.codes,
                bins.low_key,
                bins.high_key,
                bins.strides,
                counting,
                out,
                flat,
            )
        )

    count = np.zeros(len(coords), dtype=np.int64)
    total = pass_over(True, count, np.zeros(0, dtype=np.int64))
    offset = np.cumsum(count) - count
    flat = np.zeros(total, dtype=np.int64)
    pass_over(False, offset, flat)
    return flat, offset, count


@cache
def _compiled_rims() -> Any:  # noqa: PLR0915
    """Build the rim-enumeration kernel once. Same laziness as `_compiled`."""
    from numba import njit  # noqa: PLC0415 — the whole point is that this is lazy

    @njit(cache=True, fastmath=False, nogil=True)
    def kernel(  # type: ignore[no-untyped-def]  # noqa: PLR0917, PLR0915, PLR0912
        coords,
        inflated,
        neighbour_flat,
        neighbour_offset,
        neighbour_count,
        test_flat,
        test_offset,
        test_count,
        degenerate,
        counting,
        blocker_count,
        origins,
        normals,
        ring_radii,
        blocker_offset,
        blocker_flat,
        scratch,
    ):  # pragma: no cover - compiled; exercised through `enumerate_rims`
        """Where two accessible spheres meet, and which third atoms reach it.

        **The first kernel here whose output is geometry rather than a verdict.**
        Every loop compiled before this one wrote into a boolean or, where a
        node claimed by any feature is the node claimed by the first — so the
        floating-point association only had to be *arranged* to match, never
        proved. A rim circle is compared against radii by three later stages, so
        a last-bit difference in one is a different surface. Every expression
        below is therefore written in the reference's own order, and
        `tests/test_debye_kernel.py` compares the arrays with `array_equal`
        rather than `allclose`.

        Two passes for the same reason `neighbour_lists` takes two: the blocker
        set per rim is the answer and cannot be preallocated. A worst-case
        buffer is 704 MB at 18,242 atoms against the 2 MB the answer needs.
        """
        atoms = coords.shape[0]
        kept = 0
        blockers_total = 0
        for i in range(atoms):
            if inflated[i] <= 0.0:
                continue
            first_neighbour = neighbour_offset[i]
            first_test = test_offset[i]
            last_test = first_test + test_count[i]
            for slot in range(first_neighbour, first_neighbour + neighbour_count[i]):
                j = neighbour_flat[slot]
                if j <= i or inflated[j] <= 0.0:
                    continue

                # `_rim`: the circle where the two accessible spheres meet.
                vx = coords[j, 0] - coords[i, 0]
                vy = coords[j, 1] - coords[i, 1]
                vz = coords[j, 2] - coords[i, 2]
                separation = np.sqrt(vx * vx + vy * vy + vz * vz)
                if separation >= inflated[i] + inflated[j] or separation <= degenerate:
                    continue
                if separation <= abs(inflated[i] - inflated[j]):
                    continue  # one sphere swallows the other; there is no rim
                nx, ny, nz = vx / separation, vy / separation, vz / separation
                along = (
                    separation * separation + inflated[i] * inflated[i] - inflated[j] * inflated[j]
                ) / (2.0 * separation)
                squared = inflated[i] * inflated[i] - along * along
                if squared <= 0.0:
                    continue
                ring = np.sqrt(squared)
                ox = coords[i, 0] + along * nx
                oy = coords[i, 1] + along * ny
                oz = coords[i, 2] + along * nz

                # `_blockers`: which third atoms can cover part of this rim, and
                # whether one covers all of it. Both tests are the closed form
                # for a circle's nearest and farthest point from a sphere centre.
                swallowed = False
                found = 0
                for probe_slot in range(first_test, last_test):
                    other = test_flat[probe_slot]
                    if other == j:
                        continue
                    gx = coords[other, 0] - ox
                    gy = coords[other, 1] - oy
                    gz = coords[other, 2] - oz
                    axial = gx * nx + gy * ny + gz * nz
                    rx = gx - axial * nx
                    ry = gy - axial * ny
                    rz = gz - axial * nz
                    radial = np.sqrt(rx * rx + ry * ry + rz * rz)
                    limit = inflated[other] * inflated[other]
                    axial_squared = axial * axial
                    far = radial + ring
                    if axial_squared + far * far <= limit:
                        swallowed = True
                        break
                    near = radial - ring
                    if axial_squared + near * near < limit:
                        # Into scratch, never straight into the output. A rim
                        # can still be found swallowed *after* some of its
                        # blockers are known, and writing those at
                        # `blocker_offset[kept]` reads that array one past its
                        # end when the swallowed pair happens to follow the last
                        # rim that is kept — an out-of-bounds write in compiled
                        # code with bounds checking off, which is the failure
                        # mode this file can least afford.
                        scratch[found] = other
                        found += 1
                if swallowed:
                    continue

                if counting:
                    blocker_count[kept] = found
                else:
                    base = blocker_offset[kept]
                    for entry in range(found):
                        blocker_flat[base + entry] = scratch[entry]
                    origins[kept, 0] = ox
                    origins[kept, 1] = oy
                    origins[kept, 2] = oz
                    normals[kept, 0] = nx
                    normals[kept, 1] = ny
                    normals[kept, 2] = nz
                    ring_radii[kept] = ring
                kept += 1
                blockers_total += found
        return kept if counting else blockers_total

    return kernel


def enumerate_rims(
    coords: FloatArray,
    inflated: FloatArray,
    neighbours: tuple[np.ndarray, np.ndarray, np.ndarray],
    testable: tuple[np.ndarray, np.ndarray, np.ndarray],
    degenerate: float,
) -> tuple[FloatArray, FloatArray, FloatArray, np.ndarray, np.ndarray, np.ndarray]:
    """Every reachable rim: origins, normals, radii, and each one's blocking atoms."""
    neighbour_flat, neighbour_offset, neighbour_count = neighbours
    test_flat, test_offset, test_count = testable
    compiled = _compiled_rims()
    pairs = int(neighbour_count.sum())
    scratch = np.zeros(int(test_count.max()) + 1 if len(test_count) else 1, dtype=np.int64)
    nothing = np.zeros(0, dtype=np.float64)
    nothing_2d = np.zeros((0, DIMENSIONS), dtype=np.float64)

    blocker_count = np.zeros(pairs, dtype=np.int64)
    rims = compiled(
        coords, inflated,
        neighbour_flat, neighbour_offset, neighbour_count,
        test_flat, test_offset, test_count,
        degenerate, True,
        blocker_count,
        nothing_2d, nothing_2d, nothing, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
        scratch,
    )  # fmt: skip

    blocker_count = blocker_count[:rims]
    blocker_offset = np.cumsum(blocker_count) - blocker_count
    origins = np.zeros((rims, DIMENSIONS), dtype=np.float64)
    normals = np.zeros((rims, DIMENSIONS), dtype=np.float64)
    ring_radii = np.zeros(rims, dtype=np.float64)
    blocker_flat = np.zeros(int(blocker_count.sum()), dtype=np.int64)
    compiled(
        coords, inflated,
        neighbour_flat, neighbour_offset, neighbour_count,
        test_flat, test_offset, test_count,
        degenerate, False,
        blocker_count,
        origins, normals, ring_radii, blocker_offset, blocker_flat,
        scratch,
    )  # fmt: skip
    return origins, normals, ring_radii, blocker_offset, blocker_count, blocker_flat


@cache
def _compiled_seats() -> Any:  # noqa: PLR0915
    """Build the probe-seat kernel once. Same laziness as `_compiled`."""
    from numba import njit  # noqa: PLC0415 — the whole point is that this is lazy

    @njit(cache=True, fastmath=False, nogil=True)
    def kernel(  # type: ignore[no-untyped-def]  # noqa: PLR0917, PLR0915, PLR0912
        coords,
        inflated,
        test_flat,
        test_offset,
        test_count,
        overlapping,
        atom_count,
        degenerate,
        counting,
        per_atom,
        offset,
        seats,
    ):  # pragma: no cover - compiled; exercised through `probe_seats`
        """A probe seated against three atoms and overlapping none.

        The reference raises each triple under its *smallest* member and asks
        every one of that atom's higher-numbered neighbour pairs, so the loop
        here is the same: atom, then the upper triangle of its sorted
        neighbours, then the closed-form trilateration.

        **Emission order matches the reference's, which took a second look.**
        `_tangency_points` returns both mirror images as one array — every
        `+z` seat and then every `-z` seat — and `points[legal]` preserves that,
        so an atom contributes its `+z` seats in triple order followed by its
        `-z` seats. A kernel emitting the two seats of a triple together would
        be answer-identical, because `_within` reduces with `any`, and would
        make the arrays impossible to compare with `array_equal`. Two sweeps of
        the triple loop, one per mirror, is the cost of a comparable test.
        """
        total = 0
        for atom in range(coords.shape[0]):
            if inflated[atom] <= 0.0:
                continue
            first_test = test_offset[atom]
            last_test = first_test + test_count[atom]
            r1 = inflated[atom]
            p1x, p1y, p1z = coords[atom, 0], coords[atom, 1], coords[atom, 2]
            found = 0
            base = 0 if counting else offset[atom]

            for mirror in range(2):
                for first in range(first_test, last_test):
                    j = test_flat[first]
                    if j <= atom:
                        continue
                    for second in range(first + 1, last_test):
                        k = test_flat[second]
                        if k <= atom:
                            continue
                        # The third pair has to overlap too, or the three
                        # spheres leave no seat. Binary search of the same
                        # sorted key array the reference builds, whose keys are
                        # `low * count + high` — `k > j` here because
                        # `test_flat` arrives sorted per atom, which is also why
                        # the trilateration frame below is built from the same
                        # atom the reference builds it from.
                        wanted = j * atom_count + k
                        at = np.searchsorted(overlapping, wanted)
                        if at >= overlapping.shape[0] or overlapping[at] != wanted:
                            continue

                        # `_tangency_points`, one triple at a time.
                        ex0 = coords[j, 0] - p1x
                        ex1 = coords[j, 1] - p1y
                        ex2 = coords[j, 2] - p1z
                        span = np.sqrt(ex0 * ex0 + ex1 * ex1 + ex2 * ex2)
                        if span <= degenerate:
                            continue
                        ex0, ex1, ex2 = ex0 / span, ex1 / span, ex2 / span
                        t0 = coords[k, 0] - p1x
                        t1 = coords[k, 1] - p1y
                        t2 = coords[k, 2] - p1z
                        along = ex0 * t0 + ex1 * t1 + ex2 * t2
                        ey0 = t0 - along * ex0
                        ey1 = t1 - along * ex1
                        ey2 = t2 - along * ex2
                        height = np.sqrt(ey0 * ey0 + ey1 * ey1 + ey2 * ey2)
                        if height <= degenerate:
                            continue
                        ey0, ey1, ey2 = ey0 / height, ey1 / height, ey2 / height
                        ez0 = ex1 * ey2 - ex2 * ey1
                        ez1 = ex2 * ey0 - ex0 * ey2
                        ez2 = ex0 * ey1 - ex1 * ey0
                        r2, r3 = inflated[j], inflated[k]
                        x = (r1 * r1 - r2 * r2 + span * span) / (2.0 * span)
                        y = (
                            r1 * r1 - r3 * r3 + along * along + height * height - 2.0 * along * x
                        ) / (2.0 * height)
                        squared = r1 * r1 - x * x - y * y
                        if squared <= 0.0:
                            continue
                        z = np.sqrt(squared)
                        if mirror == 1:
                            z = -z
                        sx = p1x + x * ex0 + y * ey0 + z * ez0
                        sy = p1y + x * ex1 + y * ey1 + z * ez1
                        sz = p1z + x * ex2 + y * ey2 + z * ez2

                        # `_legal`, with `j` and `k` exempt: a seat lies at
                        # exactly `R` from the atoms it was built from, and a
                        # comparison against the very sphere a point lies on
                        # rejects it about half the time.
                        allowed = True
                        for slot in range(first_test, last_test):
                            other = test_flat[slot]
                            if other == j or other == k:  # noqa: PLR1714, SIM109
                                continue
                            gx = sx - coords[other, 0]
                            gy = sy - coords[other, 1]
                            gz = sz - coords[other, 2]
                            if gx * gx + gy * gy + gz * gz < inflated[other] * inflated[other]:
                                allowed = False
                                break
                        if not allowed:
                            continue
                        if not counting:
                            seats[base + found, 0] = sx
                            seats[base + found, 1] = sy
                            seats[base + found, 2] = sz
                        found += 1
            if counting:
                per_atom[atom] = found
            total += found
        return total

    return kernel


def probe_seats(
    coords: FloatArray,
    inflated: FloatArray,
    testable: tuple[np.ndarray, np.ndarray, np.ndarray],
    overlapping: np.ndarray,
    degenerate: float,
) -> FloatArray:
    """Every legal probe seat, in the reference's order.

    `testable` must be sorted within each atom, which is what the reference's
    own `np.sort` gives it. The order is not cosmetic: it decides which of the
    three atoms the trilateration frame is built from, and a frame built from
    the other one produces the same two points to within a rounding rather than
    to the last bit.
    """
    test_flat, test_offset, test_count = testable
    compiled = _compiled_seats()
    atoms = len(coords)
    nothing = np.zeros((0, DIMENSIONS), dtype=np.float64)
    per_atom = np.zeros(atoms, dtype=np.int64)
    counted = compiled(
        coords, inflated, test_flat, test_offset, test_count,
        overlapping, atoms, degenerate, True, per_atom, np.zeros(atoms, dtype=np.int64), nothing,
    )  # fmt: skip
    total = int(counted)
    offset = np.cumsum(per_atom) - per_atom
    seats = np.zeros((total, DIMENSIONS), dtype=np.float64)
    compiled(
        coords, inflated, test_flat, test_offset, test_count,
        overlapping, atoms, degenerate, False, per_atom, offset, seats,
    )  # fmt: skip
    return seats


@cache
def _compiled_union() -> Any:
    """Build the union-of-spheres kernel once. Same laziness as `_compiled`."""
    from numba import njit  # noqa: PLC0415 — the whole point is that this is lazy

    @njit(cache=True, fastmath=False, nogil=True)
    def kernel(  # type: ignore[no-untyped-def]  # noqa: PLR0917
        first_axis,
        second_axis,
        third_axis,
        coords,
        radii,
        mask,
    ):  # pragma: no cover - compiled; exercised through `mark_union`
        """Mark every lattice node inside any sphere. Mutates `mask`.

        The last uncompiled loop in the geometry, and the cheapest to reason
        about: the output is a boolean or, so a node claimed by any sphere is
        the node claimed by the first and the association cannot leak into an
        answer. The reference sums `(dx**2 + dy**2) + dz**2` — an *array*
        `** 2`, which numpy fast-paths to `np.square` — so the same
        left-to-right order here is the same number.

        The window is the sphere's own index range per axis, which is what
        keeps this proportional to the volume the spheres occupy rather than to
        atoms times points. ROADMAP.md section 7 records what the whole-grid
        version cost before that: 64 s on a 1,960-atom protein.
        """
        for atom in range(coords.shape[0]):
            radius = radii[atom]
            if radius <= 0.0:
                continue  # a zero-radius atom bounds no volume; Kirkwood's has one
            cx, cy, cz = coords[atom, 0], coords[atom, 1], coords[atom, 2]
            lo0 = np.searchsorted(first_axis, cx - radius, "left")
            hi0 = np.searchsorted(first_axis, cx + radius, "right")
            lo1 = np.searchsorted(second_axis, cy - radius, "left")
            hi1 = np.searchsorted(second_axis, cy + radius, "right")
            lo2 = np.searchsorted(third_axis, cz - radius, "left")
            hi2 = np.searchsorted(third_axis, cz + radius, "right")
            if lo0 >= hi0 or lo1 >= hi1 or lo2 >= hi2:
                continue  # the sphere falls between nodes, or outside the box
            squared_radius = radius * radius
            for i in range(lo0, hi0):
                dx = first_axis[i] - cx
                dx2 = dx * dx
                for j in range(lo1, hi1):
                    dy = second_axis[j] - cy
                    plane = dx2 + dy * dy
                    for k in range(lo2, hi2):
                        dz = third_axis[k] - cz
                        if plane + dz * dz <= squared_radius:
                            mask[i, j, k] = True

    return kernel


def mark_union(
    axes: list[FloatArray], coords: FloatArray, radii: FloatArray, mask: np.ndarray
) -> None:
    """Mark every node inside any sphere. Mutates `mask`."""
    _compiled_union()(
        np.ascontiguousarray(axes[0]),
        np.ascontiguousarray(axes[1]),
        np.ascontiguousarray(axes[2]),
        np.ascontiguousarray(coords),
        np.ascontiguousarray(radii),
        mask,
    )
