"""Conservative sampled CAD clearance, shared by kinematic/dynamic audits."""
import numpy as np


class SceneClearance:
    def __init__(self, geometries, truck_triangles, bank_bounds, margin=.02, *, bucket_narrow_phase=None):
        self.geometries = geometries
        self.tri = np.asarray(truck_triangles, float)
        self.lo, self.hi = self.tri.min(1), self.tri.max(1)
        self.bank_lo, self.bank_hi = np.asarray(bank_bounds, float)
        self.margin = margin
        # Optional exact mesh refinement: callback must raise on a violation.
        # Default remains the conservative broad-phase rejection.
        self.bucket_narrow_phase = bucket_narrow_phase
        self.local_bounds = []
        for g in geometries:
            t = g['body_local_vertices'][g['faces']]
            self.local_bounds.append((t.min(1), t.max(1)))

    def check(self, bodies, stage):
        lowest = None
        for g, (rlo, rhi) in zip(self.geometries, self.local_bounds):
            pose = bodies[g['body']]
            v = g['body_local_vertices'] @ pose[:3, :3].T + pose[:3, 3]
            if not np.isfinite(v).all():
                raise ValueError('nonfinite_scene_pose')
            lo, hi = v.min(0), v.max(0)
            if g['mesh'] == 'bucket':
                lowest = float(lo[2])
                if lowest < -1e-6:
                    raise ValueError('bucket_ground')
                overlap = np.all(hi[:2] >= self.bank_lo[:2]) and np.all(lo[:2] <= self.bank_hi[:2])
                if stage in ('APPROACH', 'RETURN_SWING', 'RETURN') and overlap and lowest < self.bank_hi[2]+.10:
                    raise ValueError('bucket_bank_clearance')
            overlap = np.all((self.hi >= lo-self.margin) & (self.lo <= hi+self.margin), axis=1)
            if not overlap.any():
                continue
            if g['mesh'] == 'bucket':
                if self.bucket_narrow_phase is None:
                    raise ValueError('bucket_truck_broad_overlap')
                self.bucket_narrow_phase(v[g['faces']], self.tri[overlap], self.margin)
                continue
            local_truck = (self.tri[overlap]-pose[:3, 3]) @ pose[:3, :3]
            for triangle in local_truck:
                if np.all((rhi >= triangle.min(0)-self.margin) &
                          (rlo <= triangle.max(0)+self.margin), axis=1).any():
                    raise ValueError('mechanism_truck_triangle_bounds_overlap:'+g['mesh'])
        if lowest is None:
            raise ValueError('missing_bucket_geometry')
        return {'bucket_min_z_m': lowest}
