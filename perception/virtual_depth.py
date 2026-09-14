"""Virtual first-return depth from CAD triangles and material spheres.

Only this sensor backend receives particle state. No particle labels, occupancy
or mass are returned to the reconstruction code. CAD-only robot depth is a
proprioceptive self-filter prediction, not material segmentation ground truth.
"""
import numpy as np
import warp as wp
from .core import rigid_transform


@wp.kernel
def _mesh_depth(meshes: wp.array(dtype=wp.uint64), inverse_poses: wp.array(dtype=wp.transform),
                robot_count: int, camera_pose: wp.transform, intr: wp.vec4, width: int,
                near: float, far: float, ground: int, depth: wp.array(dtype=float), robot_depth: wp.array(dtype=float)):
    i = wp.tid(); u = i % width; v = i // width
    optical = wp.normalize(wp.vec3((float(u)-intr[2])/intr[0], (float(v)-intr[3])/intr[1], 1.0))
    origin = wp.transform_get_translation(camera_pose)
    direction = wp.transform_vector(camera_pose, optical)
    best = far; robot = far
    if ground == 1 and wp.abs(direction[2]) > 1.0e-8:
        zdepth = -origin[2]/direction[2]*optical[2]
        if zdepth >= near and zdepth < best:
            best = zdepth
    for j in range(meshes.shape[0]):
        ro = wp.transform_point(inverse_poses[j], origin)
        rd = wp.transform_vector(inverse_poses[j], direction)
        hit = wp.mesh_query_ray(meshes[j], ro, rd, far/optical[2])
        if hit.result:
            zdepth = hit.t*optical[2]
            if zdepth >= near:
                best = wp.min(best, zdepth)
                if j < robot_count:
                    robot = wp.min(robot, zdepth)
    depth[i] = best; robot_depth[i] = robot


@wp.kernel
def _sphere_depth(points: wp.array(dtype=wp.vec3), camera_inverse: wp.transform, radius: float,
                  intr: wp.vec4, width: int, height: int, near: float, far: float, depth: wp.array(dtype=float)):
    i = wp.tid(); c = wp.transform_point(camera_inverse, points[i])
    if c[2] > radius and c[2]+radius >= near and c[2]-radius < far:
        denom = c[2]*c[2]-radius*radius
        dx = radius*wp.sqrt(c[0]*c[0]+denom)
        dy = radius*wp.sqrt(c[1]*c[1]+denom)
        u0 = wp.max(0, int(wp.floor(intr[0]*(c[0]*c[2]-dx)/denom+intr[2])))
        u1 = wp.min(width-1, int(wp.ceil(intr[0]*(c[0]*c[2]+dx)/denom+intr[2])))
        v0 = wp.max(0, int(wp.floor(intr[1]*(c[1]*c[2]-dy)/denom+intr[3])))
        v1 = wp.min(height-1, int(wp.ceil(intr[1]*(c[1]*c[2]+dy)/denom+intr[3])))
        for y in range(v0, v1+1):
            for x in range(u0, u1+1):
                ray = wp.vec3((float(x)-intr[2])/intr[0], (float(y)-intr[3])/intr[1], 1.0)
                a = wp.dot(ray, ray); b = wp.dot(ray, c)
                discriminant = b*b-a*(wp.dot(c, c)-radius*radius)
                if discriminant >= 0.0:
                    t = (b-wp.sqrt(discriminant))/a
                    if t >= near and t < far:
                        wp.atomic_min(depth, y*width+x, t)


def _transform(pose):
    p = rigid_transform(pose)
    # Warp extracts XYZW quaternion from the rotation matrix.
    q = wp.quat_from_matrix(wp.mat33(*p[:3, :3].ravel().tolist()))
    return wp.transform(wp.vec3(*p[:3, 3]), q)


class VirtualDepth:
    def __init__(self, camera, meshes=(), *, robot_count=0, radius_m=.0125, device='cuda:0', ground=True):
        """meshes is a sequence of (body-local vertices, triangle indices)."""
        if not 0 <= robot_count <= len(meshes) or not np.isfinite(radius_m) or radius_m <= 0:
            raise ValueError('invalid_sensor_scene')
        self.camera, self.device, self.radius, self.ground = camera, device, radius_m, ground
        self.meshes = []
        for vertices, faces in meshes:
            vertices, faces = np.asarray(vertices, np.float32), np.asarray(faces, np.int32)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(vertices)):
                raise ValueError('invalid_sensor_mesh')
            self.meshes.append(wp.Mesh(points=wp.array(vertices, dtype=wp.vec3, device=device), indices=wp.array(faces.ravel(), dtype=wp.int32, device=device)))
        self.ids = wp.array([m.id for m in self.meshes], dtype=wp.uint64, device=device)
        self.robot_count = robot_count
        self.depth = wp.empty(camera.width*camera.height, dtype=float, device=device)
        self.robot_depth = wp.empty_like(self.depth)

    def capture(self, particle_positions, world_from_camera, mesh_poses=()):
        if len(mesh_poses) != len(self.meshes):
            raise ValueError('sensor_pose_count')
        p = rigid_transform(world_from_camera)
        inv = wp.array([wp.transform_inverse(_transform(t)) for t in mesh_poses], dtype=wp.transform, device=self.device)
        camera = self.camera
        intr = wp.vec4(camera.fx, camera.fy, camera.cx, camera.cy)
        points = np.asarray(particle_positions, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('invalid_sensor_particles')
        q = wp.array(points, dtype=wp.vec3, device=self.device)
        wp.launch(_mesh_depth, dim=camera.width*camera.height,
                  inputs=[self.ids, inv, self.robot_count, _transform(p), intr, camera.width,
                          camera.near, camera.far, int(self.ground), self.depth, self.robot_depth], device=self.device)
        if len(points):
            wp.launch(_sphere_depth, dim=len(points), inputs=[q, wp.transform_inverse(_transform(p)), self.radius, intr,
                      camera.width, camera.height, camera.near, camera.far, self.depth], device=self.device)
        depth = self.depth.numpy().reshape(camera.height, camera.width).copy()
        robot = self.robot_depth.numpy().reshape(camera.height, camera.width).copy()
        depth[depth >= camera.far] = np.nan; robot[robot >= camera.far] = np.nan
        return depth, robot
