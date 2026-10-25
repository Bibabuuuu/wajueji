"""Conservative support connectivity for static terrain observations only.

All rocks remain in the physical simulator. AABB overlap overestimates support:
it can retain a detached rock, but cannot remove a rock because of a mesh hole.
"""
import numpy as np


def static_rock_ids(grounded_soil, rock_points, margin):
    soil = np.asarray(grounded_soil, float)
    if not len(soil) or not np.isfinite(soil).all() or not 0 < margin <= .2:
        raise ValueError('invalid_static_support_observation')
    boxes = {key: (points.min(0), points.max(0))
             for key, points in rock_points.items()}
    static = set()
    distances = {}
    for key, (low, high) in boxes.items():
        gap = np.maximum(np.maximum(low-soil, soil-high), 0.)
        distance = float(np.linalg.norm(gap, axis=1).min())
        distances[key] = distance
        if low[2] <= margin or distance <= margin:
            static.add(key)
    while True:
        old = static.copy()
        for key, (low, high) in boxes.items():
            for other in old:
                lower, upper = boxes[other]
                gap = np.maximum(np.maximum(low-upper, lower-high), 0.)
                if np.linalg.norm(gap) <= margin:
                    static.add(key)
                    break
        if old == static:
            break
    return static, distances
