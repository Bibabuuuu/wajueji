"""Exact sampled bucket/truck surface refinement; no mesh simplification.

Does not qualify signed solid clearance or continuous collision detection.
All triangle pairs pruned here have AABB distance greater than the margin.
"""
import numpy as np
from audit_mesh_distance import paired_tri_distance


def bucket_clearance(a, b, margin):
    alo, ahi = a.min(1), a.max(1)
    first, second = [], []
    for j, triangle in enumerate(b):
        gap = np.maximum(np.maximum(alo-triangle.max(0), triangle.min(0)-ahi), 0)
        ii = np.flatnonzero(np.linalg.norm(gap, axis=1) <= margin)
        first.extend(ii.tolist()); second.extend([j]*len(ii))
    for start in range(0, len(first), 2048):
        distance2 = paired_tri_distance(a[first[start:start+2048]], b[second[start:start+2048]])
        if not np.isfinite(distance2).all():
            raise ValueError('nonfinite_triangle_distance')
        if np.any(distance2 <= margin**2):
            raise ValueError('bucket_truck_surface_clearance')
