"""Read-only native projection reaction ledger at saved frame 1680->1681.

The original Newton projection executes unchanged. This audit attributes its
additional velocity impulse by collider and checks the opposite reaction sum.
It does not apply any new force to the rigid bodies or alter production physics.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from layered_47_cycle_runtime import CycleRuntime, REPO
from trace_native_projection_owners import nearest_penetrating_collider


WORK = Path(__file__).resolve().parent
SOURCE = WORK / 'manual-prefix-continuation-1620/state-001680.npz'
ORIGINAL = REPO / 'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275/input.ndjson'
REFERENCE = WORK / 'manual-prefix-continuation-1620/display/particles.npy'
OUT = WORK / 'native-reaction-audit-1680-1681'
FRAME = 1681


def main():
    started = time.monotonic()
    command = json.loads(ORIGINAL.read_text().splitlines()[FRAME - 1])['applied_command_si']
    report = dict(status='initializing', frame=FRAME,
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  input_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                  forces_routed_to_rigid=False, position_correction_impulse_included=False)
    with CycleRuntime(SOURCE, OUT, enable_truck=True, freeze_passive_jp=True,
                      bucket_motion_consistency=True) as runtime:
        fixture = runtime.fixture
        solver = fixture.solver
        model = fixture.model
        owners_wp = wp.empty(fixture.state.particle_count, dtype=int,
                             device=fixture.state.particle_q.device)
        old_project = solver.project_outside
        observed = []

        def project(state_in, state_out, dt, gap=None):
            collider = solver._mpm_model.collider
            old_gap = collider.query_max_dist
            if gap is not None:
                collider.query_max_dist = gap
            try:
                solver._last_step_data.require_collider_previous_position(state_in.body_q)
                wp.launch(nearest_penetrating_collider, dim=state_in.particle_count,
                          inputs=[state_in.particle_q, solver._mpm_model.particle_flags,
                                  model.particle_mass, solver._particle_environment,
                                  collider, state_in.body_q,
                                  state_in.body_qd if solver.collider_velocity_mode == 'forward' else None,
                                  solver._last_step_data.body_q_prev if solver.collider_velocity_mode == 'backward' else None,
                                  dt], outputs=[owners_wp], device=state_in.particle_q.device)
                owners = owners_wp.numpy().copy()
            finally:
                collider.query_max_dist = old_gap
            before_q = state_in.particle_q.numpy().copy()
            before_v = state_in.particle_qd.numpy().copy()
            result = old_project(state_in, state_out, dt, gap=gap)
            after_q = state_out.particle_q.numpy().copy()
            after_v = state_out.particle_qd.numpy().copy()
            mass = model.particle_mass.numpy().astype(np.float64)
            delta_momentum = mass[:, None] * (after_v.astype(np.float64) - before_v)
            active_ids = sorted(int(x) for x in np.unique(owners) if x >= 0)
            per_collider = {}
            for collider_id in active_ids:
                mask = owners == collider_id
                reaction = -np.sum(delta_momentum[mask], axis=0, dtype=np.float64)
                position = 0.5 * (before_q[mask].astype(np.float64) + after_q[mask])
                torque_world_origin = np.sum(np.cross(position, -delta_momentum[mask]),
                                             axis=0, dtype=np.float64)
                per_collider[str(collider_id)] = dict(
                    particle_count=int(mask.sum()),
                    particle_velocity_impulse_ns=np.sum(delta_momentum[mask], axis=0).tolist(),
                    reaction_impulse_ns=reaction.tolist(),
                    reaction_angular_impulse_about_world_origin_nms=torque_world_origin.tolist())
            total_particle = np.sum(delta_momentum, axis=0, dtype=np.float64)
            total_reaction = np.sum([x['reaction_impulse_ns'] for x in per_collider.values()],
                                    axis=0, dtype=np.float64)
            unmatched = owners < 0
            if np.max(np.abs(delta_momentum[unmatched])) > 1e-12:
                raise ValueError('unmatched_native_velocity_impulse')
            residual = total_particle + total_reaction
            observed.append(dict(per_collider=per_collider,
                total_particle_velocity_impulse_ns=total_particle.tolist(),
                total_opposite_reaction_ns=total_reaction.tolist(),
                linear_balance_residual_ns=residual.tolist(),
                maximum_abs_linear_balance_residual_ns=float(np.max(np.abs(residual))),
                position_corrected_count=int(np.sum(np.linalg.norm(after_q - before_q, axis=1) > 1e-6)),
                velocity_changed_count=int(np.sum(np.linalg.norm(after_v - before_v, axis=1) > 1e-9))))
            return result

        solver.project_outside = project
        requested = np.asarray(command)
        applied = runtime.projector.project(runtime.bridge.state['mechanism_qpos'], requested)
        if not np.array_equal(requested, applied):
            raise ValueError('command_changed')
        row = runtime.step(applied)
        if not (row['hard_guard'] and row['hencky_material_guard']['safe']
                and row['moving_bucket_sweep']['safe']):
            raise ValueError('one_step_guard_failed')
        if len(observed) != 1:
            raise ValueError('native_projection_call_count')
        reference = np.load(REFERENCE, mmap_mode='r')
        diff = float(np.max(np.abs(fixture.state.particle_q.numpy() - reference[FRAME - 180])))
        if diff != 0 or observed[0]['maximum_abs_linear_balance_residual_ns'] > 1e-8:
            raise ValueError('native_reaction_audit_contract')
        report.update(status='one_step_audit_passed',
                      baseline_particle_positions_exact=True,
                      original_command_exact=True,
                      hard_guard_safe=True, material_guard_safe=True, sweep_safe=True,
                      native_projection=observed[0],
                      collider_groups=dict(bucket=[int(x) for x in fixture.bucket_collider_ids],
                          static=[int(x) for x in fixture.static_collider_ids],
                          rock=[int(x) for ids in fixture.collider_ids.values() for x in ids]),
                      static_group_includes_ground_and_truck=True)
    report['wall_seconds'] = time.monotonic() - started
    OUT.mkdir(exist_ok=True)
    (OUT / 'reaction-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print('NATIVE_REACTION_AUDIT', report['status'],
          report['native_projection']['maximum_abs_linear_balance_residual_ns'], flush=True)


if __name__ == '__main__':
    main()
