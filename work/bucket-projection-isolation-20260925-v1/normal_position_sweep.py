"""Diagnostic normal-only positional repair; velocity friction is unchanged."""
import warp as wp
@wp.kernel
def normal_position_sweep(offsets: wp.array[int], normals: wp.array[wp.vec3], distances: wp.array[float],
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
        target = start + tangent_rest
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
