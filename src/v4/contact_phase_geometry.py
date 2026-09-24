"""CPU ground-connected soil/solid-proxy gap observer, without physics changes.

Distances refer to particle centers and existing convex proxy solids, not the
rendered surface, MPM support radii, or rock clearance. Connectivity remains the
existing 4cm six-neighbor heuristic. Missing ground support is unknown.
"""
import numpy as np
from scipy.ndimage import label
from mixed_material_r0 import quaternion_matrix_wxyz


def ground_connected_mask(points, *, cell_m=.04, ground_z_m=.06):
    q = np.asarray(points, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 3 or not len(q) or not np.isfinite(q).all():
        raise ValueError('invalid_soil_points')
    if not np.isfinite(cell_m) or cell_m <= 0 or not np.isfinite(ground_z_m):
        raise ValueError('invalid_connectivity_config')
    origin = q.min(0)-2*cell_m
    ijk = np.floor((q-origin)/cell_m).astype(np.int64)
    shape = ijk.max(0)+3
    if any(int(x) > 12000000 for x in shape) or np.prod(shape.astype(object)) > 12000000:
        raise ValueError('soil_connectivity_grid_budget')
    grid = np.zeros(tuple(shape), dtype=bool)
    grid[tuple(ijk.T)] = True
    components, _ = label(grid)
    ids = components[tuple(ijk.T)]
    ground = np.unique(ids[q[:, 2] < ground_z_m])
    return np.isin(ids, ground)


def triangle_distances(points, triangle):
    """Exact unsigned point-to-triangle distance for a nondegenerate face."""
    a,b,c = triangle
    u=b-a; v=c-a
    normal=np.cross(u,v); norm=np.linalg.norm(normal)
    if norm < 1e-12:
        raise ValueError('degenerate_proxy_triangle')
    normal /= norm
    signed=(points-a)@normal
    projected=points-signed[:,None]*normal
    w=projected-a
    uu=u@u;uv=u@v;vv=v@v;den=uu*vv-uv*uv
    s=(vv*(w@u)-uv*(w@v))/den
    t=(uu*(w@v)-uv*(w@u))/den
    inside=(s>=0)&(t>=0)&(s+t<=1)
    result=np.full(len(points), np.inf)
    result[inside]=np.abs(signed[inside])
    for first,second in ((a,b),(b,c),(c,a)):
        edge=second-first
        fraction=np.clip(((points-first)@edge)/(edge@edge),0,1)
        result=np.minimum(result,np.linalg.norm(points-first-fraction[:,None]*edge,axis=1))
    return result


class SolidBucketGap:
    def __init__(self, proxy):
        self.boxes=[];self.sides=[]
        for box in proxy['shell_boxes']+proxy['tooth_boxes']:
            h=np.asarray(box['half_size_m'],dtype=float)
            center=np.asarray(box['position_bucket_m'],dtype=float)
            rotation=quaternion_matrix_wxyz(box['quaternion_wxyz'])
            if h.shape!=(3,) or center.shape!=(3,) or not np.isfinite(h).all() or not (h>0).all() or not np.isfinite(center).all():
                raise ValueError('invalid_proxy_box')
            self.boxes.append((center,rotation,h))
        for side in proxy['side_plate_meshes']:
            vertices=np.asarray(side['vertices_bucket_m'],dtype=float)
            faces=np.asarray(side['faces'],dtype=int)
            center=vertices.mean(0);planes=[];triangles=vertices[faces]
            for tri in triangles:
                a,b,c=tri;n=np.cross(b-a,c-a);norm=np.linalg.norm(n)
                if norm < 1e-12:raise ValueError('degenerate_proxy_triangle')
                n/=norm
                if (center-a)@n>0:n=-n
                d=a@n
                if np.max(vertices@n-d)>2e-6:raise ValueError('proxy_side_not_convex')
                planes.append((n,d))
            self.sides.append((vertices.min(0),vertices.max(0),triangles,planes))
        if not self.boxes or not self.sides:raise ValueError('incomplete_bucket_proxy')

    def minimum_gap(self, world_points, pose):
        """Zero for a particle inside a solid; otherwise exact exterior gap.

Broad-phase rejection uses distance to an enclosing AABB, a lower bound.
No downsampling of ground-connected particles or bucket faces is performed.
"""
        q=np.asarray(world_points,dtype=float);pose=np.asarray(pose,dtype=float)
        if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all():raise ValueError('invalid_gap_points')
        if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1],atol=1e-9,rtol=0) or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-8,rtol=0) or not np.isclose(np.linalg.det(pose[:3,:3]),1,atol=1e-8,rtol=0):
            raise ValueError('invalid_bucket_rigid_pose')
        if not len(q):return None
        points=(q-pose[:3,3])@pose[:3,:3]
        best=float('inf')
        for center,rotation,h in self.boxes:
            outside=np.maximum(np.abs((points-center)@rotation)-h,0)
            best=min(best,float(np.linalg.norm(outside,axis=1).min()))
            if best==0:return 0.
        for lo,hi,triangles,planes in self.sides:
            bound=np.linalg.norm(np.maximum(np.maximum(lo-points,points-hi),0),axis=1)
            selected=points[bound<=best]
            if not len(selected):continue
            inside=np.ones(len(selected),dtype=bool)
            for normal,d in planes:inside &= selected@normal-d <= 0
            if inside.any():return 0.
            for tri in triangles:
                best=min(best,float(triangle_distances(selected,tri).min()))
        return best

    def observe(self, particle_q, pose):
        mask=ground_connected_mask(particle_q)
        count=int(mask.sum())
        # Same minimum observed stock count as the existing terrain observer.
        gap=self.minimum_gap(np.asarray(particle_q)[mask],pose) if count>=1000 else None
        return dict(remaining_soil_gap_m=gap,ground_connected_particles=count,
                    connectivity_cell_m=.04,ground_anchor_below_z_m=.06,
                    distance_scope='particle_centers_to_solid_proxy_no_rock_clearance')
