"""Two independent exact-ish signed distances to the boundary of a union of balls.

Used only to referee `sashimi.debye.surface._union_gap`, which is exact outside
the union and an upper bound inside it.

`analytic_sdf` enumerates the boundary features: exposed spherical patches,
exposed pieces of the pairwise intersection circles, and exposed triple points.
`ray_sdf` marches rays out of the point and takes the shortest exit. They share
no code path, which is the point of having both.
"""

from __future__ import annotations

import itertools

import numpy as np

EPS = 1e-9


def _exposed(p, coords, radii, skip=()):
    """True when p lies on no ball's strict interior other than the ones it defines."""
    for k, (c, r) in enumerate(zip(coords, radii)):
        if k in skip:
            continue
        if np.linalg.norm(p - c) < r - EPS:
            return False
    return True


def analytic_sdf(x, coords, radii):
    x = np.asarray(x, dtype=float)
    coords = np.asarray(coords, dtype=float)
    radii = np.asarray(radii, dtype=float)
    best = np.inf

    # (a) exposed spherical patches
    for i, (c, r) in enumerate(zip(coords, radii)):
        d = x - c
        n = np.linalg.norm(d)
        if n < EPS:
            # every point of the sphere is equidistant; test one and trust symmetry
            # only if the sphere is wholly exposed, otherwise fall through to (b)/(c)
            for u in np.eye(3):
                p = c + r * u
                if _exposed(p, coords, radii, skip=(i,)):
                    best = min(best, r)
            continue
        p = c + r * d / n
        if _exposed(p, coords, radii, skip=(i,)):
            best = min(best, abs(n - r))

    # (b) exposed points of the pairwise intersection circles
    for i, j in itertools.combinations(range(len(radii)), 2):
        ci, cj, ri, rj = coords[i], coords[j], radii[i], radii[j]
        sep = np.linalg.norm(cj - ci)
        if sep < EPS or sep >= ri + rj or sep <= abs(ri - rj):
            continue
        u = (cj - ci) / sep
        a = (sep * sep + ri * ri - rj * rj) / (2.0 * sep)
        rho2 = ri * ri - a * a
        if rho2 <= 0:
            continue
        rho = np.sqrt(rho2)
        o = ci + a * u
        d = x - o
        axial = float(d @ u)
        radial = d - axial * u
        nr = np.linalg.norm(radial)
        if nr < EPS:
            # equidistant from the whole circle: sample it
            basis = np.eye(3)[np.argmin(np.abs(u))]
            e1 = np.cross(u, basis)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(u, e1)
            for t in np.linspace(0.0, 2.0 * np.pi, 361)[:-1]:
                p = o + rho * (np.cos(t) * e1 + np.sin(t) * e2)
                if _exposed(p, coords, radii, skip=(i, j)):
                    best = min(best, np.linalg.norm(x - p))
        else:
            p = o + rho * radial / nr
            if _exposed(p, coords, radii, skip=(i, j)):
                best = min(best, np.linalg.norm(x - p))

    # (c) exposed triple points
    for i, j, k in itertools.combinations(range(len(radii)), 3):
        for p in _triple_points(coords[[i, j, k]], radii[[i, j, k]]):
            if _exposed(p, coords, radii, skip=(i, j, k)):
                best = min(best, np.linalg.norm(x - p))

    inside = any(np.linalg.norm(x - c) <= r for c, r in zip(coords, radii))
    return -best if inside else best


def _triple_points(coords, radii):
    """The 0, 1 or 2 points equidistant-on-surface of three spheres."""
    c0, c1, c2 = coords
    r0, r1, r2 = radii
    ex = c1 - c0
    d = np.linalg.norm(ex)
    if d < EPS:
        return []
    ex = ex / d
    t = c2 - c0
    i = float(ex @ t)
    ey = t - i * ex
    ny = np.linalg.norm(ey)
    if ny < EPS:
        return []
    ey = ey / ny
    ez = np.cross(ex, ey)
    j = float(ey @ t)
    xx = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    yy = (r0 * r0 - r2 * r2 + i * i + j * j - 2.0 * i * xx) / (2.0 * j)
    zz2 = r0 * r0 - xx * xx - yy * yy
    if zz2 < 0:
        return []
    zz = np.sqrt(zz2)
    base = c0 + xx * ex + yy * ey
    return [base + zz * ez, base - zz * ez] if zz > EPS else [base]


def _fibonacci(n):
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    theta = np.pi * (1.0 + 5.0**0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def ray_sdf(x, coords, radii, directions=20000):
    """Shortest exit from the union along `directions` rays. Inside only."""
    x = np.asarray(x, dtype=float)
    coords = np.asarray(coords, dtype=float)
    radii = np.asarray(radii, dtype=float)
    best = np.inf
    for u in _fibonacci(directions):
        t = 0.0
        for _ in range(4 * len(radii) + 4):
            p = x + t * u
            d = p[None, :] - coords
            n = np.linalg.norm(d, axis=1)
            covering = np.nonzero(n <= radii + 1e-12)[0]
            if covering.size == 0:
                break
            # farthest exit among the balls covering p
            b = -(d[covering] @ u)
            disc = b * b - (n[covering] ** 2 - radii[covering] ** 2)
            disc = np.maximum(disc, 0.0)
            t = t + float(np.max(b + np.sqrt(disc))) + 1e-12
        best = min(best, t)
    return -best
