"""Isolated zero-residual manual replay gate. No policy, training or geometry driver."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src/v4')]
import numpy as np
from manual_trajectory_residual import compose_action, compare_arrays

INPUT = ROOT/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275/input.ndjson'
SOURCE = ROOT/'runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz'
INPUT_SHA = 'dd479e6addc20a3b09479dbd3c4798929707af40ccabbf890bbfcd560dcb35e6'
SOURCE_SHA = '700f76be970c5754bde905837faf642cca0317fef097e31d17cf48f008ba0545'
OLD = ROOT/'work/bucket-projection-isolation-20260925-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, choices=[0, 120, 1170], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--short-receipt', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('immutable_output_already_exists')
    if sha(INPUT) != INPUT_SHA or sha(SOURCE) != SOURCE_SHA:
        raise ValueError('manual_source_identity_changed')
    inputs = [json.loads(x) for x in INPUT.read_text().splitlines()]
    if len(inputs) != 8122 or [x['frame'] for x in inputs] != list(range(1, 8123)):
        raise ValueError('manual_source_frames')
    source, offset = SOURCE, 0
    receipt = None
    if args.frames == 1170:
        if args.short_receipt is None:
            raise ValueError('1170_requires_passed_120_receipt')
        receipt = json.loads(args.short_receipt.read_text())
        if receipt['status'] != 'passed' or receipt['end_global_frame'] != 120:
            raise ValueError('120_receipt_not_passed')
        if receipt['input_sha256'] != INPUT_SHA or receipt['original_source_sha256'] != SOURCE_SHA:
            raise ValueError('120_receipt_source_changed')
        for path, expected in receipt['new_code_bindings'].items():
            if sha(path) != expected:
                raise ValueError('120_receipt_new_code_changed')
        source = Path(receipt['final_checkpoint']['path'])
        if sha(source) != receipt['final_checkpoint']['sha256']:
            raise ValueError('120_receipt_checkpoint_changed')
        offset = 120
    bindings = {str(Path(__file__)): sha(__file__),
                str(ROOT/'src/v4/manual_trajectory_residual.py'): sha(ROOT/'src/v4/manual_trajectory_residual.py')}
    original_rows = {}
    if args.frames:
        for name, start in [('manual-prefix-1800', 0), ('manual-prefix-continuation-1620', 180)]:
            report = json.loads((OLD/name/'report.json').read_text())
            changes = {x['frame']: x['applied'] for x in report.get('projection_changes', [])}
            for local, line in enumerate((OLD/name/'frames.ndjson').open(), 1):
                frame = start+local
                if offset < frame <= args.frames:
                    original_rows[frame] = (json.loads(line), changes.get(frame, inputs[frame-1]['applied_command_si']))
    args.output.mkdir(parents=True)
    wall_limit = 600 if args.frames == 1170 else 180
    report = dict(status='initializing', frames_requested=args.frames, start_global_frame=offset,
                  end_global_frame=offset, new_physics_frames=0, training_started=False,
                  baseline='saved_applied_manual_actions', residual_caps_si=[0]*4,
                  input_sha256=INPUT_SHA, original_source_sha256=SOURCE_SHA,
                  runtime_source=str(source), runtime_source_sha256=sha(source),
                  reused_short_receipt=None if args.short_receipt is None else str(args.short_receipt),
                  new_code_bindings=bindings, wall_limit_seconds=wall_limit,
                  physics_domain='CycleRuntime_D165_motion_consistency_not_D152_full_response',
                  source_unchanged=None, checks={}, score=None, gap_window=[])
    started = time.perf_counter()
    runtime = None
    stage = 'import_runtime'
    try:
        from layered_47_cycle_runtime import CycleRuntime
        from layered_47_cycle_bridge import controller_snapshot
        stage = 'initialize_runtime'
        with CycleRuntime(source, args.output/'physics', enable_truck=True,
                          freeze_passive_jp=True, bucket_motion_consistency=True) as runtime:
            report['restore'] = runtime.restored
            report['runtime_bindings'] = runtime.bindings
            if receipt is not None and runtime.bindings != receipt['runtime_bindings']:
                raise ValueError('120_receipt_runtime_changed')
            current = runtime.capture()
            with np.load(source, allow_pickle=False) as saved:
                exact, errors = compare_arrays(current, {k:saved[k] for k in saved.files})
            report['zero_step_arrays_exact'] = exact
            report['zero_step_errors'] = errors
            physical_equal = all(v for k,v in exact.items() if k != 'simulation_time_s')
            report['checks']['zero_step_physical_arrays_exact'] = physical_equal
            report['checks']['zero_step_clock_tolerance_1e_12_s'] = errors['simulation_time_s'] <= 1e-12
            if not all(report['checks'].values()):
                raise ValueError('zero_step_reconstruction_changed')
            comparison = dict(commands_exact=True, particles_exact=True, compact_state_exact=True,
                              hard_guards=True, material_guards=True, sweep_guards=True)
            caches = {}
            if args.frames:
                for name, start in [('manual-prefix-1800', 0), ('manual-prefix-continuation-1620', 180)]:
                    directory = OLD/name/'display'
                    caches[start] = (np.load(directory/'particles.npy', mmap_mode='r'),
                                     json.loads((directory/'states.json').read_text()))
            from contact_phase_geometry import SolidBucketGap
            geometry = SolidBucketGap(runtime.proxy)
            for frame in range(offset+1, args.frames+1):
                stage = 'physical_step'
                if time.perf_counter()-started > wall_limit:
                    raise TimeoutError('manual_zero_residual_wall_budget')
                base = np.asarray(inputs[frame-1]['applied_command_si'], dtype=float)
                requested, delta = compose_action(base, [0]*4, [0]*4)
                applied = runtime.projector.project(runtime.bridge.state['mechanism_qpos'], requested)
                with (args.output/'actions.ndjson').open('a') as log:
                    log.write(json.dumps(dict(frame=frame, baseline_si=base.tolist(), residual_si=delta.tolist(),
                         combined_request_si=requested.tolist(), applied_si=applied.tolist(),
                         projection_reason=runtime.projector.reason))+'\n')
                row = runtime.step(applied)
                report['end_global_frame'] = frame
                report['new_physics_frames'] = runtime.frames
                oldrow, expected_action = original_rows[frame]
                origin = 0 if frame <= 180 else 180
                positions, states = caches[origin]
                expected_state = states[frame-origin]
                per_frame = dict(commands_exact=bool(np.array_equal(applied, expected_action)),
                                 particles_exact=bool(np.array_equal(runtime.fixture.state.particle_q.numpy(), positions[frame-origin])),
                                 compact_state_exact=row['after_state'] == expected_state,
                                 hard_guards=bool(row['hard_guard']), material_guards=bool(row['hencky_material_guard']['safe']),
                                 sweep_guards=bool(row['moving_bucket_sweep']['safe']))
                for k,v in per_frame.items():
                    comparison[k] &= v
                if not all(per_frame.values()):
                    report['first_divergence'] = dict(frame=frame, checks=per_frame)
                    raise ValueError('manual_zero_residual_replay_diverged')
                if frame >= 1166:
                    pose = runtime.assembly.assemble(np.asarray(runtime.bridge.state['mechanism_qpos']), include_meshes=False)[1]['bucket']
                    report['gap_window'].append(dict(frame=frame, **geometry.observe(runtime.fixture.state.particle_q.numpy(), pose)))
                if frame % 60 == 0:
                    write(args.output/'report.json', report)
                    print('MANUAL_ZERO_RESIDUAL', frame, 'wall', round(time.perf_counter()-started, 2), flush=True)
            stage = 'save_recovery'
            report['post_physics_checkpoint'] = runtime.save(args.output/'post-physics.npz')
            report['checks'].update(comparison)
            if args.frames == 1170:
                stage = 'endpoint_score'
                from live_bucket_fill import MixedFillEstimator, load_config
                from run_layered_47_sparse_manual_contact_episode import _cache_rock_vertices
                vertices = _cache_rock_vertices(runtime.root, runtime.records)
                estimator = MixedFillEstimator(load_config(ROOT/'configs/v4/live-bucket-fill-local-v1.json'),
                            [x['volume_m3'] for x in runtime.records], vertices)
                pose = runtime.assembly.assemble(np.asarray(runtime.bridge.state['mechanism_qpos']), include_meshes=False)[1]['bucket']
                report['score'] = estimator.measure_snapshot(dict(frame=1170, soil=runtime.fixture.state.particle_q.numpy(), pose=pose,
                                     rocks=np.asarray([x['qpos'] for x in runtime.bridge.state['rocks']])))
                report['checks']['five_gap_frames_over_3mm'] = len(report['gap_window']) == 5 and all(
                    x['remaining_soil_gap_m'] is not None and x['remaining_soil_gap_m'] >= .003 for x in report['gap_window'])
                report['checks']['endpoint_display_score_valid'] = report['score']['valid_estimate']
                report['checks']['cached_endpoint_score_matches_1e_8_pp'] = abs(report['score']['fill_percent']-146.71951799587762) <= 1e-8 if report['score']['valid_estimate'] else False
            report['source_unchanged'] = sha(INPUT) == INPUT_SHA and sha(SOURCE) == SOURCE_SHA and sha(source) == report['runtime_source_sha256']
            report['checks']['source_unchanged'] = report['source_unchanged']
            if not all(report['checks'].values()):
                raise ValueError('manual_zero_residual_final_checks_failed')
            report['final_checkpoint'] = report['post_physics_checkpoint']
            report['final_controller'] = controller_snapshot(runtime.bridge.state)
            report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', failure_stage=stage, exception_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['source_unchanged'] = sha(INPUT) == INPUT_SHA and sha(SOURCE) == SOURCE_SHA and sha(source) == report['runtime_source_sha256']
        report['wall_seconds'] = time.perf_counter()-started
        report['scope'] = 'zero-residual manual replay only; not RL training, calibrated volume or full truck delivery'
        write(args.output/'report.json', report)


if __name__ == '__main__':
    main()
