"""Initiative D's seat kernel, carried here to be graded on a second platform.

**This is a measurement, not a proposal.** Nothing under `src/` calls it. It
exists so CI's `ubuntu-latest` — native amd64, where the shipped kernel is only
ever exercised as a whole — can answer one question before M7 is un-parked:
does the restructure stay bit-identical off this project's one arm64 laptop?

The question is not idle. `surface._rim` once differed between platforms on
`x ** 2` against `x * x`, and only a bit-identity comparison caught it. This
kernel adds a construct nothing in `src/sashimi/debye/kernel.py` uses — an array
**reallocated inside** an `njit` body — which numba supports but does not
document as a contract.

Three redundancies come out of `_compiled_seats` at once:

* the two mirror sweeps (`kernel.py:1110`), replaced by one sweep that appends
  `+z` seats straight to the buffer and holds each atom's `-z` seats in a small
  scratch until the atom is done. That reproduces the reference's "every `+z` in
  triple order, then every `-z`" without knowing the split point in advance,
  which is the only reason the counting pass was needed to make one sweep work;
* the counting pass itself (`kernel.py:1218-1228`), replaced by a buffer grown
  geometrically from `natoms` — seats come out at roughly one per atom, so the
  seed costs at most one growth;
* the index-ascending legality scan (`kernel.py:1173-1182`), optionally ordered
  nearest-first per atom, in-kernel, into a scratch. The scan AND-reduces with an
  early `break`, so no permutation of it can change a verdict — the numpy
  reference already scans in a *different* order and is still bit-identical.

`trap=True` builds the wrong version deliberately: one cursor emitting both
mirrors of a triple adjacently. It produces the identical seat *set* and the
wrong row order, and it is here so the comparison can be shown capable of
failing. A bit-identity test that no mutation reddens is a guard that guards
nothing.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np

DIMENSIONS = 3


@cache
def compiled(nearest: bool, trap: bool) -> Any:  # noqa: PLR0915
    """Build one variant. `nearest` and `trap` are compile-time constants."""
    from numba import njit  # noqa: PLC0415 — lazy, exactly as kernel.py is

    @njit(cache=True, fastmath=False, nogil=True)
    def kernel(  # noqa: PLR0917, PLR0915, PLR0912
        coords,
        inflated,
        test_flat,
        test_offset,
        test_count,
        overlapping,
        atom_count,
        degenerate,
        seats,
        minus,
        order,
        keys,
    ):
        cap = seats.shape[0]
        written = 0
        for atom in range(coords.shape[0]):
            if inflated[atom] <= 0.0:
                continue
            first_test = test_offset[atom]
            last_test = first_test + test_count[atom]
            r1 = inflated[atom]
            p1x, p1y, p1z = coords[atom, 0], coords[atom, 1], coords[atom, 2]
            held = 0

            # The blocker order for this atom. Nearest-first is a stable
            # insertion sort over at most a few hundred entries, once per atom,
            # amortised over the atom's ~n^2/2 candidate pairs — which is why it
            # needs no persistent table and costs no bytes that outlive the call.
            width = last_test - first_test
            if nearest:
                for t in range(width):
                    other = test_flat[first_test + t]
                    dx = coords[other, 0] - p1x
                    dy = coords[other, 1] - p1y
                    dz = coords[other, 2] - p1z
                    keys[t] = dx * dx + dy * dy + dz * dz
                    order[t] = other
                for a in range(1, width):
                    kk, ii, b = keys[a], order[a], a - 1
                    while b >= 0 and keys[b] > kk:
                        keys[b + 1] = keys[b]
                        order[b + 1] = order[b]
                        b -= 1
                    keys[b + 1] = kk
                    order[b + 1] = ii

            for first in range(first_test, last_test):
                j = test_flat[first]
                if j <= atom:
                    continue
                for second in range(first + 1, last_test):
                    k = test_flat[second]
                    if k <= atom:
                        continue
                    wanted = j * atom_count + k
                    at = np.searchsorted(overlapping, wanted)
                    if at >= overlapping.shape[0] or overlapping[at] != wanted:
                        continue

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
                    y = (r1 * r1 - r3 * r3 + along * along + height * height - 2.0 * along * x) / (
                        2.0 * height
                    )
                    squared = r1 * r1 - x * x - y * y
                    if squared <= 0.0:
                        continue
                    z = np.sqrt(squared)

                    for half in range(2):
                        zz = z if half == 0 else -z
                        sx = p1x + x * ex0 + y * ey0 + zz * ez0
                        sy = p1y + x * ex1 + y * ey1 + zz * ez1
                        sz = p1z + x * ex2 + y * ey2 + zz * ez2

                        allowed = True
                        for slot in range(width):
                            other = order[slot] if nearest else test_flat[first_test + slot]
                            if other in (j, k):
                                continue
                            gx = sx - coords[other, 0]
                            gy = sy - coords[other, 1]
                            gz = sz - coords[other, 2]
                            if gx * gx + gy * gy + gz * gz < inflated[other] * inflated[other]:
                                allowed = False
                                break
                        if not allowed:
                            continue

                        if half == 0 or trap:
                            if written == cap:
                                bigger = np.empty((cap * 2 + 1, DIMENSIONS), dtype=np.float64)
                                bigger[:written] = seats[:written]
                                seats = bigger
                                cap = seats.shape[0]
                            seats[written, 0] = sx
                            seats[written, 1] = sy
                            seats[written, 2] = sz
                            written += 1
                        else:
                            minus[held, 0] = sx
                            minus[held, 1] = sy
                            minus[held, 2] = sz
                            held += 1

            if held > 0:
                need = written + held
                if need > cap:
                    while cap < need:
                        cap = cap * 2 + 1
                    bigger = np.empty((cap, DIMENSIONS), dtype=np.float64)
                    bigger[:written] = seats[:written]
                    seats = bigger
                    cap = seats.shape[0]
                for t in range(held):
                    seats[written + t, 0] = minus[t, 0]
                    seats[written + t, 1] = minus[t, 1]
                    seats[written + t, 2] = minus[t, 2]
                written += held
        return seats, written

    return kernel


def probe_seats(
    spheres: Any, *, nearest: bool = True, trap: bool = False, seed: int | None = None
) -> np.ndarray:
    """Drive one variant over a `_Spheres`, returning its seats.

    `seed` forces the initial buffer capacity. Passing 1 makes the buffer grow on
    nearly every emitted row, which is how the reallocation gets exercised hard
    on a platform rather than merely once.
    """
    from sashimi.debye import surface as surface_module  # noqa: PLC0415

    coords, inflated = spheres.coords, spheres.inflated
    atoms = len(coords)
    overlapping = surface_module._overlapping_pairs(spheres.neighbours, inflated, atoms)
    test_flat, test_offset, test_count = spheres.sorted_testable_table

    widest = int(test_count.max()) if atoms else 0
    kernel = compiled(nearest, trap)
    seats, written = kernel(
        coords,
        inflated,
        test_flat,
        test_offset,
        test_count,
        overlapping,
        atoms,
        surface_module.DEGENERATE,
        np.empty((max(1, seed if seed is not None else atoms), DIMENSIONS), dtype=np.float64),
        np.empty((max(1, widest * widest), DIMENSIONS), dtype=np.float64),
        np.empty(max(1, widest), dtype=np.int64),
        np.empty(max(1, widest), dtype=np.float64),
    )
    return np.asarray(seats[:written])
