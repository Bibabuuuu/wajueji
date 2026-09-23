"""Opt-in moving-bucket point sweep; no default runtime installation.

Uses the existing open proxy convex pieces, preserves the entering side, and
recovers shallow initial penetrations on the old thin-wall side. Rotation
uses a local endpoint chord, not an exact continuous rigid-motion solution.
Only the additional velocity impulse is returned for explicit rigid routing;
position corrections and pre-existing native projections remain diagnostic.
"""
import numpy as np
import warp as wp


@wp.kernel
def _sweep(offsets: wp.array[int], normals: wp.array[wp.vec3], distances: wp.array[float],
           primary: wp.array[wp.vec2i], old_pose: wp.transform, new_pose: wp.transform,
           linear: wp.vec3, angular: wp.vec3, friction: float, mass: float,
           lower: wp.vec3, upper: wp.vec3, skin: float,
           previous: wp.array[wp.vec3], q: wp.array[wp.vec3],
           v: wp.array[wp.vec3], grad: wp.array[wp.mat33],
           summary: wp.array[wp.float64], flags: wp.array[int]):
    i = wp.tid()
    inverse = wp.transform_inverse(new_pose)
    start = wp.transform_point(wp.transform_inverse(old_pose), previous[i])
    target = wp.transform_point(inverse, q[i])
    flags[i] = 0
    if (wp.max(start[0], target[0]) < lower[0] or wp.min(start[0], target[0]) > upper[0]
            or wp.max(start[1], target[1]) < lower[1] or wp.min(start[1], target[1]) > upper[1]
            or wp.max(start[2], target[2]) < lower[2] or wp.min(start[2], target[2]) > upper[2]):
        return
    original_q = q[i]
    original_v = v[i]
    surface = linear + wp.cross(angular, q[i] - wp.transform_get_translation(new_pose))
    relative = wp.transform_vector(inverse, v[i] - surface)
    count = int(0)
    recovered = int(0)
    for iteration in range(8):
        delta = target - start
        best_t = float(2.0)
        best_normal = wp.vec3(0.0)
        best_contact = wp.vec3(0.0)
        was_inside = int(0)
        for piece in range(primary.shape[0]):
            enter = float(0.0)
            leave = float(1.0)
            face = int(-1)
            valid = bool(True)
            inside = bool(True)
            for plane in range(offsets[piece], offsets[piece + 1]):
                signed = wp.dot(normals[plane], start) - distances[plane]
                speed = wp.dot(normals[plane], delta)
                if signed > 0.0:
                    inside = False
                if wp.abs(speed) < 1.e-10:
                    if signed > 0.0:
                        valid = False
                else:
                    t = -signed / speed
                    if speed < 0.0:
                        if t >= enter:
                            enter = t
                            face = plane
                    else:
                        leave = wp.min(leave, t)
            if inside:
                pair = primary[piece]
                first = pair[0]
                second = pair[1]
                s0 = wp.dot(normals[first], start) - distances[first]
                s1 = wp.dot(normals[second], start) - distances[second]
                face = wp.where(s0 >= s1, first, second)
                signed = wp.dot(normals[face], start) - distances[face]
                if best_t > 0.0:
                    best_t = 0.0
                    best_normal = normals[face]
                    best_contact = start - signed * best_normal
                    was_inside = 1
            elif valid and face >= 0 and enter <= leave and enter >= 0.0 and enter <= 1.0 and enter < best_t:
                best_t = enter
                best_normal = normals[face]
                best_contact = start + delta * enter
                was_inside = 0
        if best_t > 1.0:
            break
        normal = best_normal
        contact = best_contact
        rest = target - contact
        normal_rest = wp.min(wp.dot(rest, normal), 0.0)
        tangent_rest = rest - normal * normal_rest
        vn = wp.min(wp.dot(relative, normal), 0.0)
        vt = relative - normal * vn
        tangent_scale = wp.max(0.0, 1.0 + friction * normal_rest / wp.max(wp.length(tangent_rest), 1.e-12))
        velocity_scale = wp.max(0.0, 1.0 + friction * vn / wp.max(wp.length(vt), 1.e-12))
        relative = vt * velocity_scale
        start = contact + normal * skin
        target = start + tangent_rest * tangent_scale
        count += 1
        recovered += was_inside
    if count > 0:
        corrected = wp.transform_point(new_pose, target)
        corrected_v = (wp.transform_vector(new_pose, relative) + linear
                       + wp.cross(angular, corrected - wp.transform_get_translation(new_pose)))
        q[i] = corrected
        v[i] = corrected_v
        gradient = grad[i]
        grad[i] = 0.5 * (gradient - wp.transpose(gradient))
        flags[i] = count
        # Opposite of the additional particle impulse, not total native contact.
        reaction = -mass * (corrected_v - original_v)
        torque = wp.cross(corrected - wp.transform_get_translation(new_pose), reaction)
        for axis in range(3):
            wp.atomic_add(summary, axis, wp.float64(reaction[axis]))
            wp.atomic_add(summary, axis + 3, wp.float64(torque[axis]))
        wp.atomic_add(summary, 6, wp.float64(1.0))
        wp.atomic_add(summary, 7, wp.float64(recovered))
        wp.atomic_max(summary, 8, wp.float64(wp.length(corrected - original_q)))
        wp.atomic_add(summary, 9, wp.float64(count))


class MovingBucketSweep:
    """Correction primitive; the caller owns timing and reaction routing."""
    def __init__(self, mesh, state, particle_mass, friction, *, proxy=None):
        self.mesh = mesh  # keep native mesh storage alive
        self.mass = float(particle_mass)
        self.friction = float(friction)
        self.device = state.particle_q.device
        points = mesh.points.numpy()
        self.lower = wp.vec3(*(points.min(axis=0) - 1.e-4))
        self.upper = wp.vec3(*(points.max(axis=0) + 1.e-4))
        offsets, normals, distances, primary = convex_planes(mesh, proxy)
        self.offsets = wp.array(offsets, dtype=int, device=self.device)
        self.normals = wp.array(normals, dtype=wp.vec3, device=self.device)
        self.distances = wp.array(distances, dtype=float, device=self.device)
        self.primary = wp.array(primary, dtype=wp.vec2i, device=self.device)
        self.summary = wp.zeros(10, dtype=wp.float64, device=self.device)
        self.flags = wp.zeros(state.particle_q.shape[0], dtype=int, device=self.device)

    def apply(self, state, previous, old_pose, new_pose, linear, angular):
        self.summary.zero_()
        wp.launch(_sweep, dim=state.particle_q.shape[0], inputs=[
            self.offsets, self.normals, self.distances, self.primary, old_pose, new_pose, wp.vec3(*linear), wp.vec3(*angular),
            self.friction, self.mass, self.lower, self.upper, 1.e-5,
            previous, state.particle_q, state.particle_qd,
            state.mpm.particle_qd_grad, self.summary, self.flags], device=self.device)
        values = self.summary.numpy()
        return {
            "reaction_impulse_ns": values[:3].tolist(),
            "reaction_angular_impulse_nms": values[3:6].tolist(),
            "corrected_particles": int(values[6]),
            "initial_overlap_recoveries": int(values[7]),
            "maximum_position_correction_m": float(values[8]),
            "hits": int(values[9]),
            "position_correction_reaction_included": False,
            "exact_rotational_ccd": False,
        }


def convex_planes(mesh, proxy):
    """Exact half-spaces of the existing boxes and convex side plates; no cap."""
    from mixed_material_r0 import quaternion_matrix_wxyz
    pieces = []
    if proxy is None:
        points = mesh.points.numpy().astype(np.float64)
        faces = mesh.indices.numpy().reshape(-1, 3)
        pieces.append((points, faces, None))
    else:
        for box in proxy['shell_boxes'] + proxy['tooth_boxes']:
            h = np.asarray(box['half_size_m'])
            rotation = quaternion_matrix_wxyz(box['quaternion_wxyz'])
            center = np.asarray(box['position_bucket_m'])
            ns = np.vstack([rotation.T, -rotation.T])
            ds = ns @ center + np.tile(h, 2)
            axis = int(np.argmin(h))
            pieces.append((ns, ds, (axis, axis + 3)))
        for side in proxy['side_plate_meshes']:
            pieces.append((np.asarray(side['vertices_bucket_m']),
                           np.asarray(side['faces']), None))
    offsets = [0]; normals = []; distances = []; primary = []
    for points, faces, pair in pieces:
        if pair is not None:
            ns, ds = points, faces
        else:
            center = points.mean(axis=0)
            planes = {}
            for tri in faces:
                a, b, c = points[tri]
                n = np.cross(b-a, c-a); length = np.linalg.norm(n)
                if length < 1.e-12:
                    continue
                n /= length
                if np.dot(n, center-a) > 0:
                    n = -n
                d = np.dot(n, a)
                if np.max(points @ n - d) > 2.e-6:
                    raise ValueError('sweep_piece_not_convex')
                planes[tuple(np.round(np.r_[n,d], 6))] = (n,d)
            ns = np.array([p[0] for p in planes.values()])
            ds = np.array([p[1] for p in planes.values()])
            # Opposite plane pair with minimum slab width.
            choices = [(ds[i]+ds[j],i,j) for i in range(len(ns))
                       for j in range(i+1,len(ns)) if np.dot(ns[i],ns[j]) < -.999999]
            if not choices:
                raise ValueError('sweep_piece_missing_thin_face_pair')
            _, first, second = min(choices)
            pair = (first,second)
        base = len(normals)
        primary.append([base+pair[0],base+pair[1]])
        normals.extend(ns); distances.extend(ds); offsets.append(len(normals))
    return offsets, normals, distances, primary
