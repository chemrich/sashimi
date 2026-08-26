import numpy as np


def two_ball_gap(points, first, second):
    (c1, r1), (c2, r2) = first, second
    axis = c2 - c1
    sep = float(np.linalg.norm(axis))
    unit = axis / sep
    along = (sep * sep + r1 * r1 - r2 * r2) / (2.0 * sep)
    ring = float(np.sqrt(r1 * r1 - along * along))
    origin = c1 + along * unit

    best = np.full(len(points), np.inf)
    for centre, radius, other, other_radius in ((c1, r1, c2, r2), (c2, r2, c1, r1)):
        offset = points - centre
        span = np.maximum(np.linalg.norm(offset, axis=1), 1e-12)
        foot = centre + radius * offset / span[:, None]
        exposed = np.linalg.norm(foot - other, axis=1) >= other_radius
        best = np.where(exposed, np.minimum(best, np.abs(span - radius)), best)
    offset = points - origin
    axial = offset @ unit
    radial = np.linalg.norm(offset - axial[:, None] * unit, axis=1)
    best = np.minimum(best, np.sqrt((radial - ring) ** 2 + axial**2))
    inside = (np.linalg.norm(points - c1, axis=1) <= r1) | (
        np.linalg.norm(points - c2, axis=1) <= r2
    )
    return np.where(inside, -best, best)
