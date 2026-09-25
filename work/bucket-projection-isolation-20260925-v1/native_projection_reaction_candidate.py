"""Opt-in attribution and routing of Newton's postsolve velocity impulse.

Only additional velocity impulses from project_outside are routed. Its position
corrections remain diagnostic, and this module does not change the projection
kernel, MPM constitutive law, or D133 sweep. Never imported by default paths.
"""
import numpy as np
import warp as wp

from trace_native_projection_owners import nearest_penetrating_collider


class NativeProjectionReaction:
    def __init__(self, fixture):
        self.fixture = fixture
        self.solver = fixture.solver
        self.original = self.solver.project_outside
        self.owners_wp = wp.empty(fixture.state.particle_count, dtype=int,
                                  device=fixture.state.particle_q.device)
        self.mass = fixture.model.particle_mass.numpy().astype(np.float64)
        self.name_by_id = {}
        for name, ids in {**fixture.collider_ids, 'bucket': fixture.bucket_collider_ids}.items():
            for collider_id in ids:
                if int(collider_id) in self.name_by_id:
                    raise ValueError('duplicate_dynamic_collider_owner')
                self.name_by_id[int(collider_id)] = name
        self.static_ids = set(int(x) for x in fixture.static_collider_ids)
        self.pending = None
        self.calls = 0
        self.routed = 0
        self.max_linear_residual_ns = 0.0
        self.dynamic_reaction_ns = np.zeros(3)
        self.static_reaction_ns = np.zeros(3)
        self.particle_velocity_impulse_ns = np.zeros(3)
        self.routed_reaction_ns = np.zeros(3)
        self.corrected_particles = 0

    def install(self):
        if getattr(self.solver, '_native_projection_reaction_candidate', None) is not None:
            raise ValueError('native_projection_reaction_already_installed')
        self.solver._native_projection_reaction_candidate = self
        self.solver.project_outside = self.project

    def uninstall(self):
        self.solver.project_outside = self.original
        self.solver._native_projection_reaction_candidate = None

    def project(self, state_in, state_out, dt, gap=None):
        if self.pending is not None:
            raise ValueError('previous_native_reaction_not_routed')
        collider = self.solver._mpm_model.collider
        old_gap = collider.query_max_dist
        if gap is not None:
            collider.query_max_dist = gap
        try:
            self.solver._last_step_data.require_collider_previous_position(state_in.body_q)
            wp.launch(nearest_penetrating_collider, dim=state_in.particle_count,
                      inputs=[state_in.particle_q, self.solver._mpm_model.particle_flags,
                              self.fixture.model.particle_mass, self.solver._particle_environment,
                              collider, state_in.body_q,
                              state_in.body_qd if self.solver.collider_velocity_mode == 'forward' else None,
                              self.solver._last_step_data.body_q_prev if self.solver.collider_velocity_mode == 'backward' else None,
                              dt], outputs=[self.owners_wp], device=state_in.particle_q.device)
            owners = self.owners_wp.numpy().copy()
        finally:
            collider.query_max_dist = old_gap
        before_q = state_in.particle_q.numpy().astype(np.float64)
        before_v = state_in.particle_qd.numpy().astype(np.float64)
        result = self.original(state_in, state_out, dt, gap=gap)
        after_q = state_out.particle_q.numpy().astype(np.float64)
        after_v = state_out.particle_qd.numpy().astype(np.float64)
        delta_p = self.mass[:, None] * (after_v - before_v)
        if not np.isfinite(delta_p).all():
            raise ValueError('native_reaction_nonfinite')
        if np.max(np.abs(delta_p[owners < 0])) > 1e-12:
            raise ValueError('unmatched_native_velocity_impulse')
        reactions = {}
        static = np.zeros(3)
        dynamic = np.zeros(3)
        corrected_count = 0
        for collider_id in np.unique(owners[owners >= 0]):
            cid = int(collider_id)
            mask = owners == cid
            impulse = -np.sum(delta_p[mask], axis=0, dtype=np.float64)
            midpoint = 0.5 * (before_q[mask] + after_q[mask])
            torque_world = np.sum(np.cross(midpoint, -delta_p[mask]), axis=0,
                                  dtype=np.float64)
            if cid in self.name_by_id:
                name = self.name_by_id[cid]
                if name not in reactions:
                    reactions[name] = dict(impulse=np.zeros(3), torque_world=np.zeros(3))
                reactions[name]['impulse'] += impulse
                reactions[name]['torque_world'] += torque_world
                dynamic += impulse
            elif cid in self.static_ids:
                static += impulse
            else:
                raise ValueError('unowned_native_collider')
            corrected_count += int(mask.sum())
        particle_impulse = np.sum(delta_p, axis=0, dtype=np.float64)
        residual = particle_impulse + dynamic + static
        maximum = float(np.max(np.abs(residual)))
        if maximum > 1e-8:
            raise ValueError('native_reaction_linear_balance')
        self.max_linear_residual_ns = max(self.max_linear_residual_ns, maximum)
        self.dynamic_reaction_ns += dynamic
        self.static_reaction_ns += static
        self.particle_velocity_impulse_ns += particle_impulse
        self.corrected_particles += corrected_count
        self.pending = dict(reactions=reactions, dynamic=dynamic, static=static,
                            particle=particle_impulse, residual=residual,
                            corrected_count=corrected_count)
        self.calls += 1
        return result

    def route(self, wrench, origins, dt):
        if self.pending is None:
            raise ValueError('native_reaction_missing')
        for name, reaction in self.pending['reactions'].items():
            impulse = reaction['impulse']
            torque = reaction['torque_world'] - np.cross(np.asarray(origins[name]), impulse)
            wrench[name]['force'] += impulse / dt
            wrench[name]['torque'] += torque / dt
            self.routed_reaction_ns += impulse
        self.pending = None
        self.routed += 1

    def summary(self):
        return dict(calls=self.calls, routed_steps=self.routed,
                    maximum_linear_balance_residual_ns=self.max_linear_residual_ns,
                    dynamic_reaction_ns=self.dynamic_reaction_ns.tolist(),
                    static_reaction_ns=self.static_reaction_ns.tolist(),
                    particle_velocity_impulse_ns=self.particle_velocity_impulse_ns.tolist(),
                    routed_dynamic_reaction_ns=self.routed_reaction_ns.tolist(),
                    dynamic_route_residual_ns=(self.dynamic_reaction_ns-self.routed_reaction_ns).tolist(),
                    corrected_particles_total=self.corrected_particles,
                    static_reaction_not_routed_to_mujoco=True,
                    position_correction_impulse_included=False)
