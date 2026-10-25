"""Opt-in D-127 experiment: spatial Hencky reconstruction and finite rotation.

No default entry point imports this module. The context manager substitutes one
integrand in this process only; the installed Newton package remains unchanged.
Particle update integrand derived from Newton (Apache-2.0).
SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
SPDX-License-Identifier: Apache-2.0
"""
from contextlib import contextmanager

import newton
import warp as wp
import warp.fem as fem
from warp.fem.linalg import symmetric_eigenvalues_qr
from newton._src.solvers.implicit_mpm.implicit_mpm_solver_kernels import (
    MaterialParameters, MIN_JP_DELTA, MAX_JP_DELTA, MIN_PRINCIPAL_STRAIN,
    MAX_PRINCIPAL_STRAIN, EPSILON, get_elastic_parameters,
    extract_elastic_parameters, get_yield_parameters, project_particle_strain,
    project_stress,
)


@wp.func
def hencky_proposal(F: wp.mat33, E: wp.mat33, W: wp.mat33):
    """Return proposal and rejection code; never clip or reset a bad state.

    1: invalid previous stretch; 2: target outside original stretch bounds or
    nonfinite; 3: reconstructed matrix fails original SVD guard.
    """
    U, xi, V = wp.svd3(F)
    if not (wp.min(xi) > 0.0 and wp.max(xi) <= MAX_PRINCIPAL_STRAIN):
        return F, int(1)
    H = U @ wp.diag(wp.vec3(wp.log(xi[0]), wp.log(xi[1]), wp.log(xi[2]))) @ wp.transpose(U)
    target = H + 0.5 * (E + wp.transpose(E))
    eigen, rows = symmetric_eigenvalues_qr(target, 1.0e-7)
    if not (wp.min(eigen) >= wp.log(MIN_PRINCIPAL_STRAIN) and wp.max(eigen) <= wp.log(MAX_PRINCIPAL_STRAIN)):
        return F, int(2)
    stretch = wp.transpose(rows) @ wp.diag(wp.vec3(wp.exp(eigen[0]), wp.exp(eigen[1]), wp.exp(eigen[2]))) @ rows
    spin = wp.vec3(W[2, 1], W[0, 2], W[1, 0])
    angle = wp.length(spin)
    Q = wp.identity(n=3, dtype=float)
    if angle > 0.0:
        Q = wp.quat_to_matrix(wp.quat_from_axis_angle(spin / angle, angle))
    proposed = Q @ stretch @ U @ wp.transpose(V)
    _u, stretches, _v = wp.svd3(proposed)
    if not (wp.min(stretches) >= MIN_PRINCIPAL_STRAIN and wp.max(stretches) <= MAX_PRINCIPAL_STRAIN):
        return F, int(3)
    return proposed, int(0)


@fem.integrand
def update_particle_strains_hencky(
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
    # Preserve the original plastic history and stress update verbatim.
    p_rate = wp.trace(plastic_strain_delta(s))
    delta_Jp = wp.exp(p_rate * wp.where(
        p_rate < 0.0, material_parameters.hardening_rate[s.qp_index], material_parameters.softening_rate[s.qp_index]
    ))
    particle_Jp_new = particle_Jp_prev[s.qp_index] * wp.clamp(delta_Jp, MIN_JP_DELTA, MAX_JP_DELTA)
    elastic_parameters_vec = get_elastic_parameters(s.qp_index, material_parameters)
    compliance, _poisson, _damping = extract_elastic_parameters(elastic_parameters_vec)
    yield_parameters_vec = get_yield_parameters(s.qp_index, material_parameters, particle_Jp_new, dt)
    stress_0 = fem.SymmetricTensorMapper.value_to_dof_3d(stress(s))
    particle_stress_new = fem.SymmetricTensorMapper.dof_to_value_3d(project_stress(stress_0, yield_parameters_vec))
    prev_strain = elastic_strain_prev[s.qp_index]
    vel_grad = fem.grad(grid_vel, s)
    skew = 0.5 * dt * (vel_grad - wp.transpose(vel_grad))
    elastic_strain_new = wp.identity(n=3, dtype=float)
    if compliance > EPSILON:
        elastic_strain_new, _rejected = hencky_proposal(prev_strain, elastic_strain_delta(s), skew)
        elastic_strain_new = project_particle_strain(elastic_strain_new, prev_strain, compliance)
    gimp_weight = s.qp_weight * fem.measure(domain, s) / particle_volume[s.qp_index]
    wp.atomic_add(particle_Jp, s.qp_index, gimp_weight * particle_Jp_new)
    wp.atomic_add(particle_stress, s.qp_index, gimp_weight * particle_stress_new)
    wp.atomic_add(elastic_strain, s.qp_index, gimp_weight * elastic_strain_new)


@contextmanager
def installed_candidate():
    """Explicit process-scoped opt-in; always restore the original binding."""
    from newton._src.solvers.implicit_mpm import solver_implicit_mpm as module
    from newton._src.solvers.implicit_mpm.implicit_mpm_solver_kernels import update_particle_strains
    if module.update_particle_strains is not update_particle_strains:
        raise RuntimeError('another_particle_update_override_is_active')
    module.update_particle_strains = update_particle_strains_hencky
    try:
        yield
    finally:
        module.update_particle_strains = update_particle_strains
