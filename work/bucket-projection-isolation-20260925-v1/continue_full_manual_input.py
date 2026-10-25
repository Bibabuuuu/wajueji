"""Replay saved manual commands after the verified corrected 1800-frame prefix.

This is a diagnostic continuation, not a claim of stable bucket retention.
All original hard, material, and moving-sweep guards remain active. The
original recording and physics defaults are not modified.
"""
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np

from layered_47_cycle_runtime import CycleRuntime, REPO
from live_bucket_fill import MixedFillEstimator, load_config
from view_contact_bc_probe import ProbeDisplayCache
import run_layered_47_sparse_manual_contact_episode as manual


WORK = Path(__file__).resolve().parent
PREFIX = WORK / 'manual-prefix-continuation-1620'
SOURCE = PREFIX / 'state-001800.npz'
ORIGINAL = REPO / 'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275'
OUTPUT = WORK / 'manual-full-tail-1800-8122'
START, END = 1800, 8122
WALL_LIMIT_SECONDS = 1800
EXPECTED_INPUT_SHA = 'dd479e6addc20a3b09479dbd3c4798929707af40ccabbf890bbfcd560dcb35e6'


def main():
    prefix = json.loads((PREFIX / 'report.json').read_text())
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    input_sha = hashlib.sha256((ORIGINAL / 'input.ndjson').read_bytes()).hexdigest()
    if (prefix['status'] != 'bounded_manual_prefix_passed'
            or prefix['frames'] != 1620
            or source_sha != prefix['checkpoints']['1800']['sha256']
            or input_sha != EXPECTED_INPUT_SHA
            or input_sha != prefix['input_sha256']):
        raise ValueError('saved_manual_prefix_or_input_changed')
    inputs = [json.loads(x) for x in (ORIGINAL / 'input.ndjson').read_text().splitlines()][START:END]
    if len(inputs) != END - START:
        raise ValueError('saved_manual_input_length')
    started = time.perf_counter()
    report = dict(status='initializing', source_global_frame=START, end_global_frame=END,
                  source_sha256=source_sha, input_sha256=input_sha,
                  wall_limit_seconds=WALL_LIMIT_SECONDS,
                  purpose='diagnostic replay of recorded manual commands with corrected motion interface',
                  not_stable_retention_qualification=True, original_recording_unchanged=True,
                  measurements=[], projection_changes=[], checkpoints={}, guard_failures=[])
    runtime = None
    cache = None
    try:
        with CycleRuntime(SOURCE, OUTPUT, enable_truck=True, freeze_passive_jp=True,
                          bucket_motion_consistency=True) as runtime:
            report['restore'] = runtime.restored
            report['bindings'] = runtime.bindings
            vertices = manual._cache_rock_vertices(runtime.root, runtime.records)
            estimator = MixedFillEstimator(
                load_config(REPO / 'configs/v4/live-bucket-fill-local-v1.json'),
                [x['volume_m3'] for x in runtime.records], vertices)
            cache = ProbeDisplayCache(
                OUTPUT / 'display', END - START, runtime.arrays['particle_q'].shape,
                dict(root=str(runtime.root), records=runtime.records,
                     selected_cut=runtime.metadata['selected_cut'],
                     particle_spacing_m=runtime.base['mpm']['particle_spacing_m'],
                     showcase_continuous=True))
            cache.capture(0, manual.compact_state(runtime.bridge.state),
                          runtime.fixture.state.particle_q.numpy())
            report['status'] = 'running'
            for local_frame, entry in enumerate(inputs, 1):
                global_frame = START + local_frame
                if time.perf_counter() - started > WALL_LIMIT_SECONDS:
                    raise TimeoutError('full_manual_tail_wall_budget')
                requested = np.asarray(entry['applied_command_si'])
                applied = runtime.projector.project(runtime.bridge.state['mechanism_qpos'], requested)
                if not np.array_equal(requested, applied):
                    report['projection_changes'].append(dict(frame=global_frame,
                        reason=runtime.projector.reason, requested=requested.tolist(),
                        applied=applied.tolist()))
                row = runtime.step(applied)
                if not (row['hard_guard'] and row['hencky_material_guard']['safe']
                        and row['moving_bucket_sweep']['safe']):
                    report['guard_failures'].append(global_frame)
                    raise RuntimeError('manual_tail_guard_failed')
                q = runtime.fixture.state.particle_q.numpy()
                cache.capture(local_frame, manual.compact_state(runtime.bridge.state), q)
                if global_frame % 60 == 0 or global_frame == END:
                    pose = runtime.assembly.assemble(
                        np.asarray(runtime.bridge.state['mechanism_qpos']),
                        include_meshes=False)[1]['bucket']
                    fill = estimator.measure_snapshot(dict(
                        frame=global_frame, soil=q, pose=pose,
                        rocks=np.asarray([x['qpos'] for x in runtime.bridge.state['rocks']])))
                    report['measurements'].append(fill)
                    cache.positions.flush()
                    print('MANUAL_TAIL', global_frame, 'elapsed',
                          round(time.perf_counter() - started, 1), 'fill',
                          fill['fill_percent'], 'valid', fill['valid_estimate'],
                          'projected_commands', len(report['projection_changes']), flush=True)
                if global_frame % 600 == 0 or global_frame == END:
                    report['checkpoints'][str(global_frame)] = runtime.save(
                        OUTPUT / f'state-{global_frame:06d}.npz')
                    (OUTPUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            report['status'] = 'full_saved_input_diagnostic_passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['frames_completed'] = runtime.frames if runtime is not None else 0
        report['last_global_frame'] = START + report['frames_completed']
        report['wall_seconds'] = time.perf_counter() - started
        OUTPUT.mkdir(exist_ok=True)
        (OUTPUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        if cache is not None:
            report['display_cache'] = cache.finish(report['status'])
            (OUTPUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
