"""Bounded diagnostic replay of saved input; no production physics changes."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from audit_layered_47_loaded_handoff import cavity_mask
from layered_47_cycle_runtime import CycleRuntime, KINEMATICS, REPO
from layered_excavation_approach_planner import _load_kinematics
import run_layered_47_sparse_manual_contact_episode as manual


WORK = Path(__file__).resolve().parent
SOURCE = WORK / 'manual-prefix-continuation-1620/state-001080.npz'
OUT = WORK / 'onset-stage-trace-1080-1530'
ORIGINAL = REPO / 'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275/input.ndjson'
START, END = 1080, 1530


def main():
    entries = [json.loads(line) for line in ORIGINAL.read_text().splitlines()][START:END]
    assert len(entries) == END - START
    assembly, _ = _load_kinematics(str(KINEMATICS), str(KINEMATICS / 'scene/input/model/A.xml'),
                                   str(KINEMATICS / 'scene/input/mesh'))
    contract = json.loads(Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json').read_text())
    front = np.load('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/front-rim.npz')['front']
    polygon = np.asarray(contract['bucket']['polygon_xz_m'])
    y_limits = np.asarray(contract['bucket']['y_limits_m'])
    report = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  original_input_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                  start=START, end=END, stages=[], frame_checks=[], status='running')
    begin = time.monotonic()
    with CycleRuntime(SOURCE, OUT, enable_truck=True, freeze_passive_jp=True,
                      bucket_motion_consistency=True) as runtime:
        q0 = runtime.fixture.state.particle_q.numpy()
        pose0 = assembly.assemble(np.asarray(runtime.bridge.state['mechanism_qpos']),
                                  include_meshes=False)[1]['bucket']
        local0 = (q0 - pose0[:3, 3]) @ pose0[:3, :3]
        ids = np.flatnonzero(cavity_mask(local0, polygon, y_limits, front))
        assert len(ids) == 12086
        report['tracked_particles'] = len(ids)
        snapshots = {}
        solver = runtime.fixture.solver
        update = solver._update_particles
        projection = solver.project_outside
        sweep = runtime.sweep._correct
        sweep_calls = 0

        def sample(label):
            if START + runtime.frames + 1 >= 1490:
                snapshots[label] = runtime.fixture.state.particle_q.numpy()[ids].copy()

        def traced_update(*args, **kwargs):
            result = update(*args, **kwargs)
            sample('solver')
            return result

        def traced_projection(*args, **kwargs):
            sample('before_native')
            result = projection(*args, **kwargs)
            sample('native')
            return result

        def traced_sweep(*args, **kwargs):
            nonlocal sweep_calls
            label = 'sweep_predicted' if sweep_calls == 0 else 'sweep_actual'
            sweep_calls += 1
            sample('before_' + label)
            result = sweep(*args, **kwargs)
            sample(label)
            return result

        solver._update_particles = traced_update
        solver.project_outside = traced_projection
        runtime.sweep._correct = traced_sweep
        prior = WORK / 'manual-prefix-continuation-1620/display/particles.npy'
        reference = np.load(prior, mmap_mode='r')
        try:
            for entry in entries:
                sweep_calls = 0
                frame = START + runtime.frames + 1
                if time.monotonic() - begin > 180:
                    raise TimeoutError('onset_stage_180s_budget')
                sample('start')
                requested = np.asarray(entry['applied_command_si'])
                command = runtime.projector.project(runtime.bridge.state['mechanism_qpos'], requested)
                row = runtime.step(command)
                if frame in (1081, 1200, 1400, 1490, 1500, 1510, 1520, 1530):
                    current = runtime.fixture.state.particle_q.numpy()
                    report['frame_checks'].append(dict(frame=frame,
                        max_abs_vs_saved_display_m=float(np.max(np.abs(current - reference[frame - 180]))),
                        changed_command=bool(not np.array_equal(requested, command)),
                        hard_guard=bool(row['hard_guard'])))
                if frame >= 1490:
                    pose = assembly.assemble(np.asarray(runtime.bridge.state['mechanism_qpos']),
                                             include_meshes=False)[1]['bucket']
                    local = (runtime.fixture.state.particle_q.numpy()[ids] - pose[:3, 3]) @ pose[:3, :3]
                    cells, counts = np.unique(np.floor(local / .02).astype('int16'), axis=0,
                                               return_counts=True)
                    def delta(before, after):
                        if before not in snapshots or after not in snapshots:
                            return None
                        vectors = snapshots[after] - snapshots[before]
                        lengths = np.linalg.norm(vectors, axis=1)
                        return dict(mean_world_m=np.mean(vectors, axis=0).tolist(),
                                    p50_m=float(np.median(lengths)),
                                    p99_m=float(np.quantile(lengths, .99)),
                                    maximum_m=float(lengths.max()))
                    report['stages'].append(dict(frame=frame, occupied_2cm_cells=len(cells),
                        max_particles_per_cell=int(counts.max()),
                        near_negative_side_2cm=int(np.sum(local[:, 1] < y_limits[0] + .02)),
                        solver=delta('start', 'solver'), native=delta('before_native', 'native'),
                        sweep_predicted=delta('before_sweep_predicted', 'sweep_predicted'),
                        sweep_actual=delta('before_sweep_actual', 'sweep_actual'),
                        native_report=row['postsolve_projection'],
                        moving_bucket_sweep=row['moving_bucket_sweep']))
                snapshots.clear()
            report['status'] = 'bounded_trace_passed'
        except BaseException as exc:
            report['status'] = 'failed'
            report['error'] = repr(exc)
            raise
        finally:
            report['wall_seconds'] = time.monotonic() - begin
            (OUT / 'onset-stage-trace.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
