"""Stop BC probes near the fixed ground plane using actual bucket CAD vertices.

This is a sampled stop condition, not a replacement for ground collision
physics or a proof of continuous swept clearance between simulation frames.
"""
import hashlib
import numpy as np


class BucketGroundGuard:
    def __init__(self,assembly,initial_qpos,*,ground_z_m=0.,margin_m=.005):
        if not np.isfinite([ground_z_m,margin_m]).all() or margin_m<0:
            raise ValueError('invalid_ground_guard_parameters')
        self.assembly=assembly;self.ground_z=float(ground_z_m);self.margin=float(margin_m)
        geometries,_=assembly.assemble(np.asarray(initial_qpos))
        parts=[np.asarray(g['body_local_vertices'],dtype=float) for g in geometries if g['body']=='bucket']
        if not parts:raise ValueError('bucket_ground_guard_missing_geometry')
        self.vertices=np.concatenate(parts)
        if self.vertices.ndim!=2 or self.vertices.shape[1]!=3 or not self.vertices.size or not np.isfinite(self.vertices).all():
            raise ValueError('bucket_ground_guard_invalid_geometry')
        self.geometry_sha256=hashlib.sha256(self.vertices.tobytes()).hexdigest()

    def measure(self,qpos):
        q=np.asarray(qpos,dtype=float)
        if not np.isfinite(q).all():raise ValueError('bucket_ground_guard_nonfinite_pose')
        pose=np.asarray(self.assembly.assemble(q,include_meshes=False)[1]['bucket'])
        if pose.shape!=(4,4) or not np.isfinite(pose).all():raise ValueError('bucket_ground_guard_invalid_transform')
        minimum=float(np.min(self.vertices@pose[2,:3]+pose[2,3]));clearance=minimum-self.ground_z
        return dict(minimum_bucket_z_m=minimum,ground_z_m=self.ground_z,margin_m=self.margin,
                    clearance_m=clearance,safe=bool(clearance>self.margin))
