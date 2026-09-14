# Derived from Newton's Apache-2.0 update_particle_strains integrand.
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Experimental finite rotation update; no material, force, position or bounds change.
from newton._src.solvers.implicit_mpm.implicit_mpm_solver_kernels import *

@wp.func
def objective_increment(F: wp.mat33, E: wp.mat33, W: wp.mat33):
    spin=wp.vec3(W[2,1],W[0,2],W[1,0])
    angle=wp.length(spin)
    stretched=F+E@F
    if angle>0.0:
        rotation=wp.quat_to_matrix(wp.quat_from_axis_angle(spin/angle,angle))
        return rotation@stretched
    return stretched

@fem.integrand
def update_particle_strains_objective(
    s: fem.Sample,
    domain: fem.Domain,
    grid_vel: fem.Field,
    plastic_strain_delta: fem.Field,
    elastic_strain_delta: fem.Field,
    stress: fem.Field,
    dt: float,
    particle_flags: wp.array[wp.int32],
    particle_density: wp.array[float],
    particle_volume: wp.array[float],
    material_parameters: MaterialParameters,
    elastic_strain_prev: wp.array[wp.mat33],
    particle_Jp_prev: wp.array[float],
    elastic_strain: wp.array[wp.mat33],
    particle_Jp: wp.array[float],
    particle_stress: wp.array[wp.mat33],
):
    if ~particle_flags[s.qp_index] & newton.ParticleFlags.ACTIVE:
        elastic_strain[s.qp_index] = elastic_strain_prev[s.qp_index]
        particle_Jp[s.qp_index] = particle_Jp_prev[s.qp_index]
        return
    if particle_density[s.qp_index] == 0.0:
        elastic_strain[s.qp_index] = elastic_strain_prev[s.qp_index]
        particle_Jp[s.qp_index] = particle_Jp_prev[s.qp_index]
        return

    # plastic strain
    p_strain_delta = plastic_strain_delta(s)
    p_rate = wp.trace(p_strain_delta)

    delta_Jp = wp.exp(
        p_rate
        * wp.where(
            p_rate < 0.0, material_parameters.hardening_rate[s.qp_index], material_parameters.softening_rate[s.qp_index]
        )
    )
    particle_Jp_new = particle_Jp_prev[s.qp_index] * wp.clamp(delta_Jp, MIN_JP_DELTA, MAX_JP_DELTA)

    elastic_parameters_vec = get_elastic_parameters(s.qp_index, material_parameters)
    compliance, _poisson, _damping = extract_elastic_parameters(elastic_parameters_vec)

    yield_parameters_vec = get_yield_parameters(s.qp_index, material_parameters, particle_Jp_new, dt)
    stress_0 = fem.SymmetricTensorMapper.value_to_dof_3d(stress(s))
    particle_stress_new = fem.SymmetricTensorMapper.dof_to_value_3d(project_stress(stress_0, yield_parameters_vec))

    # elastic strain
    prev_strain = elastic_strain_prev[s.qp_index]
    vel_grad = fem.grad(grid_vel, s)
    skew = 0.5 * dt * (vel_grad - wp.transpose(vel_grad))
    strain_delta = elastic_strain_delta(s) + skew

    # The skew-symmetric part of the velocity gradient is used as a linearized
    # approximation of the finite rotation increment (matches standard deformation gradient update).
    elastic_strain_new = objective_increment(prev_strain, elastic_strain_delta(s), skew)
    elastic_strain_new = project_particle_strain(elastic_strain_new, prev_strain, compliance)

    gimp_weight = s.qp_weight * fem.measure(domain, s) / particle_volume[s.qp_index]
    wp.atomic_add(particle_Jp, s.qp_index, gimp_weight * particle_Jp_new)
    wp.atomic_add(particle_stress, s.qp_index, gimp_weight * particle_stress_new)
    wp.atomic_add(elastic_strain, s.qp_index, gimp_weight * elastic_strain_new)


def install():
    from newton._src.solvers.implicit_mpm import solver_implicit_mpm as solver_module
    solver_module.update_particle_strains=update_particle_strains_objective
