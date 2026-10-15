# Derived from Newton rasterized_collisions.py (Apache-2.0).
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic ablation only; D133 sweep must remain enabled."""
import newton
import warp as wp
from newton._src.solvers.implicit_mpm.rasterized_collisions import Collider, collision_sdf, _ALL_COLLIDER_WORLDS
from newton._src.solvers.implicit_mpm.contact_solver_kernels import solve_coulomb_isotropic

@wp.kernel
def project_normal_bucket(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    velocity_gradients: wp.array[wp.mat33],
    particle_flags: wp.array[wp.int32],
    particle_mass: wp.array[float],
    particle_environment: wp.array[int],
    collider: Collider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    dt: float,
    bucket_id: int,
    positions_out: wp.array[wp.vec3],
    velocities_out: wp.array[wp.vec3],
    velocity_gradients_out: wp.array[wp.mat33],
):
    """Project particles outside colliders and apply Coulomb response.

    For active particles, queries the nearest collider surface, computes the
    penetration at the end of the step, applies a Coulomb friction response
    against the collider velocity, projects positions outside by the required
    signed distance, and rigidifies the particle velocity gradient. Inactive
    and kinematic (zero-mass) particles are passed through unchanged.

    Args:
        positions: Current particle positions.
        velocities: Current particle velocities.
        velocity_gradients: Current particle velocity gradients.
        particle_flags: Per-particle flags; particles without :attr:`ACTIVE` are skipped.
        particle_mass: Per-particle mass; zero-mass (kinematic) particles are skipped.
        particle_environment: Per-particle world IDs, or null to query every collider.
        collider: Collider description and geometry.
        body_q: Rigid body transforms.
        body_qd: Rigid body velocities.
        body_q_prev: Previous rigid body transforms (for finite-difference velocity).
        dt: Timestep length.
        positions_out: Output particle positions.
        velocities_out: Output particle velocities.
        velocity_gradients_out: Output particle velocity gradients.
    """
    i = wp.tid()

    pos_adv = positions[i]
    p_vel = velocities[i]
    vel_grad = velocity_gradients[i]

    if (~particle_flags[i] & newton.ParticleFlags.ACTIVE) or particle_mass[i] == 0.0:
        positions_out[i] = positions[i]
        velocities_out[i] = p_vel
        velocity_gradients_out[i] = vel_grad
        return

    environment_index = int(_ALL_COLLIDER_WORLDS)
    if particle_environment:
        environment_index = particle_environment[i]

    # project outside of collider
    sdf, sdf_gradient, sdf_vel, _collider_id, material_id = collision_sdf(
        pos_adv, environment_index, collider, body_q, body_qd, body_q_prev, dt
    )

    sdf_end = sdf - wp.dot(sdf_vel, sdf_gradient) * dt + collider.material_projection_threshold[material_id]
    if sdf_end < 0:
        # remove normal vel
        friction = collider.material_friction[material_id]
        delta_vel = solve_coulomb_isotropic(friction, sdf_gradient, p_vel - sdf_vel) + sdf_vel - p_vel

        p_vel += delta_vel
        if _collider_id == bucket_id:
            pos_adv += wp.dot(delta_vel, sdf_gradient) * sdf_gradient * dt
        else:
            pos_adv += delta_vel * dt

        # project out
        pos_adv -= wp.min(0.0, sdf_end + dt * wp.dot(delta_vel, sdf_gradient)) * sdf_gradient  # delta_vel * dt

        # make velocity gradient rigid
        vel_grad = 0.5 * (vel_grad - wp.transpose(vel_grad))

    positions_out[i] = pos_adv
    velocities_out[i] = p_vel
    velocity_gradients_out[i] = vel_grad


def install(solver, bucket_id):
    def project(state_in, state_out, dt, gap=None):
        collider = solver._mpm_model.collider
        old_gap = collider.query_max_dist
        if gap is not None:
            collider.query_max_dist = gap
        try:
            solver._last_step_data.require_collider_previous_position(state_in.body_q)
            wp.launch(project_normal_bucket, dim=state_in.particle_count,
                inputs=[state_in.particle_q, state_in.particle_qd,
                    state_in.mpm.particle_qd_grad, solver._mpm_model.particle_flags,
                    solver.model.particle_mass, solver._particle_environment,
                    collider, state_in.body_q,
                    state_in.body_qd if solver.collider_velocity_mode == 'forward' else None,
                    solver._last_step_data.body_q_prev if solver.collider_velocity_mode == 'backward' else None,
                    dt, int(bucket_id)],
                outputs=[state_out.particle_q, state_out.particle_qd, state_out.mpm.particle_qd_grad],
                device=state_in.particle_q.device)
        finally:
            collider.query_max_dist = old_gap
    solver.project_outside = project
