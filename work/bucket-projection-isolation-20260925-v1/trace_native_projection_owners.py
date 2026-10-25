"""Read-only collider attribution for native postsolve projection at three frames."""
import hashlib
import json
import time
import traceback
from pathlib import Path

import newton
import numpy as np
import warp as wp
from scipy.spatial import cKDTree
from newton._src.solvers.implicit_mpm.rasterized_collisions import (
    Collider, collision_sdf, _ALL_COLLIDER_WORLDS)

from audit_layered_47_loaded_handoff import cavity_mask
from layered_47_cycle_runtime import CycleRuntime, KINEMATICS, REPO
from layered_excavation_approach_planner import _load_kinematics


WORK = Path(__file__).resolve().parent
SOURCE = WORK / 'manual-prefix-continuation-1620/state-001080.npz'
ORIGINAL = REPO / 'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275/input.ndjson'
REFERENCE = WORK / 'manual-prefix-continuation-1620/display/particles.npy'
OUT = WORK / 'native-projection-owners-1080-1530'
START, END, WALL_LIMIT = 1080, 1530, 180
SAMPLES = (1490, 1510, 1530)


@wp.kernel
def nearest_penetrating_collider(
    positions: wp.array[wp.vec3], particle_flags: wp.array[wp.int32],
    particle_mass: wp.array[float], particle_environment: wp.array[int],
    collider: Collider, body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector], body_q_prev: wp.array[wp.transform],
    dt: float, owners: wp.array[int],
):
    i = wp.tid()
    owners[i] = -2
    if (~particle_flags[i] & newton.ParticleFlags.ACTIVE) or particle_mass[i] == 0.0:
        return
    environment = int(_ALL_COLLIDER_WORLDS)
    if particle_environment:
        environment = particle_environment[i]
    sdf, gradient, velocity, collider_id, material_id = collision_sdf(
        positions[i], environment, collider, body_q, body_qd, body_q_prev, dt)
    sdf_end = sdf - wp.dot(velocity, gradient) * dt + collider.material_projection_threshold[material_id]
    owners[i] = wp.where(sdf_end < 0.0, collider_id, -1)


def nearest_distance(q):
    return cKDTree(q).query(q, k=2, workers=1)[0][:, 1]


def main():
    started = time.monotonic()
    entries = [json.loads(line) for line in ORIGINAL.read_text().splitlines()][START:END]
    assert len(entries) == END - START
    reference = np.load(REFERENCE, mmap_mode='r')
    assembly, _ = _load_kinematics(str(KINEMATICS), str(KINEMATICS / 'scene/input/model/A.xml'),
                                   str(KINEMATICS / 'scene/input/mesh'))
    contract = json.loads(Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json').read_text())
    polygon = np.asarray(contract['bucket']['polygon_xz_m'])
    y_limits = np.asarray(contract['bucket']['y_limits_m'])
    front = np.load('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/front-rim.npz')['front']
    report = dict(status='initializing', source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  input_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                  sample_frames=SAMPLES, rows=[], anchors=[], guards_failed=[], changed_commands=[])
    runtime = None
    try:
        with CycleRuntime(SOURCE, OUT, enable_truck=True, freeze_passive_jp=True,
                          bucket_motion_consistency=True) as runtime:
            q0 = runtime.fixture.state.particle_q.numpy()
            pose0 = assembly.assemble(np.asarray(runtime.bridge.state['mechanism_qpos']),
                                      include_meshes=False)[1]['bucket']
            local0 = (q0 - pose0[:3, 3]) @ pose0[:3, :3]
            ids = np.flatnonzero(cavity_mask(local0, polygon, y_limits, front))
            if len(ids) != 12086:
                raise ValueError('source_cohort_changed')
            fixture = runtime.fixture
            solver = fixture.solver
            owner_buffer = wp.empty(fixture.state.particle_count, dtype=int,
                                    device=fixture.state.particle_q.device)
            original_projection = solver.project_outside
            bucket_ids = set(int(x) for x in fixture.bucket_collider_ids)
            static_ids = set(int(x) for x in fixture.static_collider_ids)
            rock_ids = set(int(x) for v in fixture.collider_ids.values() for x in v)
            report['collider_groups'] = dict(bucket=sorted(bucket_ids),
                                             static=sorted(static_ids), rock=sorted(rock_ids))

            def name(owner):
                if int(owner) in bucket_ids:
                    return 'bucket'
                if int(owner) in static_ids:
                    return 'static'
                if int(owner) in rock_ids:
                    return 'rock'
                if int(owner) == -1:
                    return 'no_penetrating_collider'
                return 'other'

            def traced_projection(state_in, state_out, dt, gap=None):
                frame = START + runtime.frames + 1
                if frame not in SAMPLES:
                    return original_projection(state_in, state_out, dt, gap=gap)
                collider = solver._mpm_model.collider
                old_gap = collider.query_max_dist
                if gap is not None:
                    collider.query_max_dist = gap
                try:
                    solver._last_step_data.require_collider_previous_position(state_in.body_q)
                    wp.launch(nearest_penetrating_collider, dim=state_in.particle_count,
                              inputs=[state_in.particle_q, solver._mpm_model.particle_flags,
                                      solver.model.particle_mass, solver._particle_environment,
                                      collider, state_in.body_q,
                                      state_in.body_qd if solver.collider_velocity_mode == 'forward' else None,
                                      solver._last_step_data.body_q_prev if solver.collider_velocity_mode == 'backward' else None,
                                      dt], outputs=[owner_buffer], device=state_in.particle_q.device)
                    owners = owner_buffer.numpy()[ids].copy()
                finally:
                    collider.query_max_dist = old_gap
                before = state_in.particle_q.numpy()[ids].copy()
                result = original_projection(state_in, state_out, dt, gap=gap)
                after = state_out.particle_q.numpy()[ids].copy()
                changed = np.linalg.norm(after - before, axis=1) > 1e-6
                near_before = nearest_distance(before)
                near_after = nearest_distance(after)
                newly_close = (near_before >= .008) & (near_after < .008)
                old_close = (near_before < .008) & (near_after >= .008)
                counts = {}
                for owner_name in ('bucket', 'static', 'rock', 'no_penetrating_collider', 'other'):
                    mask = np.asarray([name(x) == owner_name for x in owners])
                    counts[owner_name] = dict(penetrating=int(mask.sum()),
                                              position_corrected=int(np.sum(mask & changed)),
                                              newly_close=int(np.sum(mask & newly_close)),
                                              no_longer_close=int(np.sum(mask & old_close)))
                report['rows'].append(dict(frame=frame, owners=counts,
                    corrected_cohort=int(changed.sum()),
                    newly_close=int(newly_close.sum()),
                    no_longer_close=int(old_close.sum()),
                    nearest_below_8mm_before=int(np.sum(near_before < .008)),
                    nearest_below_8mm_after=int(np.sum(near_after < .008))))
                print('NATIVE_OWNER', frame, counts, flush=True)
                return result

            solver.project_outside = traced_projection
            report['status'] = 'running'
            for entry in entries:
                frame = START + runtime.frames + 1
                if time.monotonic() - started > WALL_LIMIT:
                    raise TimeoutError('native_owner_wall_budget')
                requested = np.asarray(entry['applied_command_si'])
                applied = runtime.projector.project(runtime.bridge.state['mechanism_qpos'], requested)
                if not np.array_equal(requested, applied):
                    report['changed_commands'].append(frame)
                row = runtime.step(applied)
                if not (row['hard_guard'] and row['hencky_material_guard']['safe']
                        and row['moving_bucket_sweep']['safe']):
                    report['guards_failed'].append(frame)
                    raise RuntimeError('guard_failed')
                if frame in (1081, 1200, 1400, 1490, 1500, 1510, 1520, 1530):
                    diff = float(np.max(np.abs(runtime.fixture.state.particle_q.numpy()
                                               - reference[frame - 180])))
                    report['anchors'].append(dict(frame=frame, max_abs_vs_prior_m=diff))
                    if diff != 0:
                        raise RuntimeError('baseline_replay_changed')
            report['status'] = 'bounded_attribution_passed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['frames_completed'] = runtime.frames if runtime is not None else 0
        report['wall_seconds'] = time.monotonic() - started
        OUT.mkdir(exist_ok=True)
        (OUT / 'native-owner-report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
