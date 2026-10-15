# Derived from Newton rasterized_collisions.py (Apache-2.0).
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Isolated bucket-only velocity-extension diagnostic, not a deployed fix."""
from contextlib import contextmanager
import warp as wp
from newton._src.solvers.implicit_mpm.rasterized_collisions import (
    Collider, _INFINITY, _NULL_COLLIDER_ID, _ALL_COLLIDER_WORLDS,
    _SMALL_ANGLE_EPS, _query_collider_sdf)
@wp.func
def collision_sdf_query_velocity(
    x: wp.vec3,
    environment_index: int,
    collider: Collider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    dt: float,
):
    min_sdf = float(_INFINITY)
    sdf_grad = wp.vec3(0.0)
    sdf_vel = wp.vec3(0.0)
    closest_point = wp.vec3(0.0)
    collider_id = int(_NULL_COLLIDER_ID)
    material_id = int(0)  # default material, always valid

    shared_worlds = environment_index == _ALL_COLLIDER_WORLDS
    query_begin = int(0)
    query_count = collider.collider_mesh.shape[0]

    if not shared_worlds:
        query_begin = collider.world_collider_offsets[environment_index]
        query_count = collider.world_collider_offsets[environment_index + 1] - query_begin

    for query_offset in range(query_count):
        stable_collider_id = query_offset
        if not shared_worlds:
            stable_collider_id = collider.world_collider_ids[query_begin + query_offset]

        query_result, sdf, query_grad, query_vel, query_closest_point, query_material_id = _query_collider_sdf(
            x, collider, body_q, stable_collider_id
        )
        if query_result and sdf < min_sdf:
            min_sdf = sdf
            sdf_grad = query_grad
            sdf_vel = query_vel
            closest_point = query_closest_point
            collider_id = stable_collider_id
            material_id = query_material_id

    # If closest collider has rigid motion, transform back to world frame
    # Do that as a second step to avoid requiring more registers inside bvh query loop
    if collider_id >= 0:
        body_id = collider.collider_body_index[collider_id]
        if body_id >= 0:
            b_xform = body_q[body_id]
            b_rot = wp.transform_get_rotation(b_xform)

            sdf_vel = wp.quat_rotate(b_rot, sdf_vel)
            sdf_grad = wp.normalize(wp.quat_rotate(b_rot, sdf_grad))

            # Compute rigid body velocity at the contact point
            if body_q_prev:
                # backward-differenced velocity from position change
                b_xform_prev = body_q_prev[body_id]
                closest_point_world = wp.transform_point(b_xform, closest_point)
                closest_point_world_prev = wp.transform_point(b_xform_prev, closest_point)
                sdf_vel += (closest_point_world - closest_point_world_prev) / dt

            if body_qd:
                b_v = wp.spatial_top(body_qd[body_id])
                b_w = wp.spatial_bottom(body_qd[body_id])
                b_com = collider.body_com[body_id]
                com_offset_cur = wp.quat_rotate(b_rot, closest_point - b_com)
                if body_id == 47:
                    com_offset_cur = x - wp.transform_point(b_xform, b_com)
                ang_vel = wp.length(b_w)
                angle_delta = ang_vel * dt
                if angle_delta > _SMALL_ANGLE_EPS:
                    # forward-differenced velocity from current velocity
                    # (using exponential map)
                    b_rot_delta = wp.quat_from_axis_angle(b_w / ang_vel, angle_delta)
                    com_offset_next = wp.quat_rotate(b_rot_delta, com_offset_cur)
                    sdf_vel += b_v + (com_offset_next - com_offset_cur) / dt
                else:
                    # Instantaneous rigid body velocity (v + omega x r)
                    sdf_vel += b_v + wp.cross(b_w, com_offset_cur)

    return min_sdf, sdf_grad, sdf_vel, collider_id, material_id

@contextmanager
def installed():
    from newton._src.solvers.implicit_mpm import rasterized_collisions as module
    original=module.collision_sdf
    module.collision_sdf=collision_sdf_query_velocity
    try:
        yield
    finally:
        module.collision_sdf=original
