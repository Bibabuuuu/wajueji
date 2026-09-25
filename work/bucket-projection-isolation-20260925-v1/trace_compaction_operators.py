"""Attribute first-scoop particle crowding to individual existing operators.

This is one bounded replay of saved input. It changes no solver, material,
contact, guard, or production setting and writes only a diagnostic report.
"""
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from audit_layered_47_loaded_handoff import cavity_mask
from layered_47_cycle_runtime import CycleRuntime, KINEMATICS, REPO
from layered_excavation_approach_planner import _load_kinematics


WORK = Path(__file__).resolve().parent
SOURCE = WORK / 'manual-prefix-continuation-1620/state-001080.npz'
ORIGINAL = REPO / 'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275/input.ndjson'
REFERENCE = WORK / 'manual-prefix-continuation-1620/display/particles.npy'
OUT = WORK / 'compaction-operators-1080-1530'
START, END, LIMIT_SECONDS = 1080, 1530, 180
SAMPLE_START = 1490
STAGES = ('start', 'solver', 'native', 'sweep_predicted', 'sweep_actual')


def metrics(world, pose, side_limit):
    local = (world - pose[:3, 3]) @ pose[:3, :3]
    _, counts = np.unique(np.floor(local / .02).astype(np.int16), axis=0,
                          return_counts=True)
    nearest = cKDTree(world).query(world, k=2, workers=1)[0][:, 1]
    return dict(occupied_2cm_cells=int(len(counts)),
                maximum_particles_per_cell=int(counts.max()),
                nearest_p10_m=float(np.quantile(nearest, .1)),
                nearest_p50_m=float(np.median(nearest)),
                nearest_below_8mm=int(np.sum(nearest < .008)),
                negative_side_2cm_count=int(np.sum(local[:, 1] < side_limit + .02)))


def main():
    started = time.monotonic()
    entries = [json.loads(line) for line in ORIGINAL.read_text().splitlines()][START:END]
    assert len(entries) == END - START
    reference = np.load(REFERENCE, mmap_mode='r')
    assembly, _ = _load_kinematics(str(KINEMATICS), str(KINEMATICS / 'scene/input/model/A.xml'),
                                   str(KINEMATICS / 'scene/input/mesh'))
    contract = json.loads(Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json').read_text())
    front = np.load('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/front-rim.npz')['front']
    polygon = np.asarray(contract['bucket']['polygon_xz_m'])
    y_limits = np.asarray(contract['bucket']['y_limits_m'])
    report = dict(status='initializing', source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  input_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                  start_frame=START, end_frame_limit=END, wall_limit_seconds=LIMIT_SECONDS,
                  sampling_frames=[SAMPLE_START, END], rows=[], anchor_checks=[],
                  guard_failures=[], command_changes=[])
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
                raise ValueError('tracked_cohort_changed')
            report['tracked_particles'] = len(ids)
            snapshots = {}
            solver = runtime.fixture.solver
            original_update = solver._update_particles
            original_projection = solver.project_outside
            original_sweep = runtime.sweep._correct
            sweep_calls = 0

            def sample(label):
                if START + runtime.frames + 1 >= SAMPLE_START:
                    snapshots[label] = runtime.fixture.state.particle_q.numpy()[ids].copy()

            def traced_update(*args, **kwargs):
                result = original_update(*args, **kwargs)
                sample('solver')
                return result

            def traced_projection(*args, **kwargs):
                result = original_projection(*args, **kwargs)
                sample('native')
                return result

            def traced_sweep(*args, **kwargs):
                nonlocal sweep_calls
                label = 'sweep_predicted' if sweep_calls == 0 else 'sweep_actual'
                sweep_calls += 1
                result = original_sweep(*args, **kwargs)
                sample(label)
                return result

            solver._update_particles = traced_update
            solver.project_outside = traced_projection
            runtime.sweep._correct = traced_sweep
            report['status'] = 'running'
            for entry in entries:
                sweep_calls = 0
                frame = START + runtime.frames + 1
                if time.monotonic() - started > LIMIT_SECONDS:
                    raise TimeoutError('compaction_operator_budget')
                sample('start')
                requested = np.asarray(entry['applied_command_si'])
                command = runtime.projector.project(runtime.bridge.state['mechanism_qpos'], requested)
                if not np.array_equal(requested, command):
                    report['command_changes'].append(frame)
                row = runtime.step(command)
                if not (row['hard_guard'] and row['hencky_material_guard']['safe']
                        and row['moving_bucket_sweep']['safe']):
                    report['guard_failures'].append(frame)
                    raise RuntimeError('protection_failed')
                if frame in (1081, 1200, 1400, 1490, 1500, 1510, 1520, 1530):
                    q = runtime.fixture.state.particle_q.numpy()
                    difference = float(np.max(np.abs(q - reference[frame - 180])))
                    report['anchor_checks'].append(dict(frame=frame, max_abs_vs_prior_m=difference))
                    if difference != 0:
                        raise RuntimeError('baseline_replay_changed')
                if frame >= SAMPLE_START:
                    if any(label not in snapshots for label in STAGES):
                        raise RuntimeError('stage_snapshot_missing')
                    pose = assembly.assemble(np.asarray(runtime.bridge.state['mechanism_qpos']),
                                             include_meshes=False)[1]['bucket']
                    result = {label: metrics(snapshots[label], pose, y_limits[0]) for label in STAGES}
                    result['frame'] = frame
                    report['rows'].append(result)
                    if frame in (1490, 1510, 1530):
                        print('COMPACTION', frame,
                              [(label, result[label]['occupied_2cm_cells'],
                                result[label]['nearest_below_8mm']) for label in STAGES],
                              flush=True)
                snapshots.clear()
            report['status'] = 'bounded_trace_passed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['frames_completed'] = runtime.frames if runtime is not None else 0
        report['wall_seconds'] = time.monotonic() - started
        OUT.mkdir(exist_ok=True)
        (OUT / 'compaction-operator-report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
