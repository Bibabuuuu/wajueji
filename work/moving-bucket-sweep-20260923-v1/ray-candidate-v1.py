"""Opt-in moving-bucket point sweep; no default runtime installation.

Uses the existing open proxy triangles, preserves the entering side, and
recovers shallow initial penetrations from the closest outward face. Rotation
uses a local endpoint chord, not an exact continuous rigid-motion solution.
Only the additional velocity impulse is returned for explicit rigid routing;
position corrections and pre-existing native projections remain diagnostic.
"""
import numpy as np
import warp as wp


@wp.kernel
def _sweep(mesh: wp.uint64, old_pose: wp.transform, new_pose: wp.transform,
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
    # A restored / just-moved boundary can already contain a shallow point.
    # The old endpoint selects the side, never the leaked target endpoint.
    nearest = wp.mesh_query_point_no_sign(mesh, start, 0.06)
    if nearest.result:
        cp = wp.mesh_eval_position(mesh, nearest.face, nearest.u, nearest.v)
        n = wp.mesh_eval_face_normal(mesh, nearest.face)
        if wp.dot(start - cp, n) < -1.e-7:
            start = cp + n * skin
            recovered = 1
    for iteration in range(8):
        displacement = target - start
        length = wp.length(displacement)
        if length < 1.e-7:
            break
        direction = displacement / length
        hit = wp.mesh_query_ray(mesh, start, direction, length)
        if not hit.result:
            break
        normal = hit.normal
        if wp.dot(direction, normal) >= -1.e-6:
            start += direction * (hit.t + skin)
            continue
        contact = start + direction * hit.t
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
    def __init__(self, mesh, state, particle_mass, friction):
        self.mesh = mesh  # keep native mesh storage alive
        self.mass = float(particle_mass)
        self.friction = float(friction)
        self.device = state.particle_q.device
        points = mesh.points.numpy()
        self.lower = wp.vec3(*(points.min(axis=0) - 1.e-4))
        self.upper = wp.vec3(*(points.max(axis=0) + 1.e-4))
        self.summary = wp.zeros(10, dtype=wp.float64, device=self.device)
        self.flags = wp.zeros(state.particle_q.shape[0], dtype=int, device=self.device)

    def apply(self, state, previous, old_pose, new_pose, linear, angular):
        self.summary.zero_()
        wp.launch(_sweep, dim=state.particle_q.shape[0], inputs=[
            self.mesh.id, old_pose, new_pose, wp.vec3(*linear), wp.vec3(*angular),
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
