#!/usr/bin/env python3
"""Bounded four-axis noncontact motion from an independently audited v3 60Hz point."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import traceback

import newton
import numpy as np
import warp as wp

import run_layered_sparse_a30_probe as probe
import run_layered_sparse_resume_audit as audit
from layered_frequency_bridge import HoldBridge, subcycled_step
from layered_excavation_approach_planner import COMMAND_LIMITS_M_S, _read_obj
from run_layered_47_excavation_approach import AXES

HZ = 60
NAMES = ('swing', 'boom', 'arm', 'bucket')
RATES = np.array([.03, .003, .003, .006])
ENVELOPE = np.array([.03, .006, .006, .006])
STOP_SPEEDS = np.array([.002, .001, .001, .001])


def schedule():
    phases = [('hold', 30, np.zeros(4), None)]
    for axis, name in enumerate(NAMES):
        command = np.zeros(4)
        command[axis] = RATES[axis]
        phases += [(name, 30, command, axis), (name + '_stop', 30, np.zeros(4), axis)]
    return phases


class CommandBridge(HoldBridge):
    def __init__(self, bridge):
        super().__init__(bridge)
        self.command = np.zeros(4)

    def step(self, dt, substeps, wrenches):
        if (self.command.shape != (4,) or not np.isfinite(self.command).all()
                or np.any(np.abs(self.command) > COMMAND_LIMITS_M_S)):
            raise ValueError('motion_command_limit')
        return subcycled_step(self.bridge, dt, substeps, self.command, wrenches)


def motion_response(delta, expected):
    return bool(np.isfinite(delta) and .25 * expected <= delta <= 1.5 * expected + 1e-5)


def stopped(row):
    return bool(np.all(np.abs(row['axis_velocity']) <= STOP_SPEEDS)
        and row['bucket_linear_speed_m_s'] <= .02 and row['bucket_angular_speed_rad_s'] <= .02)


def qualify(row):
    probe.classify_frame(row, 'measured', 0)
    row['motion_guard'] = bool(row['strict']
        and row['bucket_vertical_clearance_m'] >= .15
        and np.all(np.abs(row['axis_offset']) <= ENVELOPE))
    return row['motion_guard']


def terrain_top(fixture, bridge, radii, particle_spacing):
    # Circumspheres enclose every rotated prototype vertex; no per-frame OBJ IO.
    soil_top = float(fixture.state.particle_q.numpy()[:, 2].max()) + .45 * particle_spacing
    rocks_top = max(float(r['qpos'][2]) + radius for r, radius in zip(bridge.state['rocks'], radii, strict=True))
    return max(soil_top, rocks_top)


def run(args):
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump({'status': 'initializing'}, stream)
    journal = out.with_suffix('.progress.ndjson')
    started = time.perf_counter()
    bridge = fixture = None
    rows, phase_results = [], []
    report = dict(schema='layered_47_sparse_motion_v1', status='initializing', rocks=47, hz=HZ,
        scope=dict(training_eligible=False, manual_collection_eligible=False,
                   digging_executed=False, gamepad_connected=False, formal_configuration_changed=False),
        protocol=dict(seconds=4.5, frames=270, rates_si=RATES.tolist(),
            phase_seconds=.5, initial_hold_seconds=.5, stop_tail_seconds=.25,
            minimum_clearance_m=.15, axis_envelope_si=ENVELOPE.tolist(),
            axis_stop_speed_si=STOP_SPEEDS.tolist(), bucket_stop_linear_m_s=.02,
            bucket_stop_angular_rad_s=.02, continuous_collision_proven=False),
        runner_sha256=probe.bench.sha256(Path(__file__)))

    def save_snapshot(label):
        if fixture is None or bridge is None:
            return
        values = probe.snapshots.capture_arrays(fixture, bridge.state,
            float(arrays['simulation_time_s'][0]) + len(rows)/HZ)
        info = probe.snapshots.write_checkpoint_atomic(out.with_name(out.stem + '-' + label + '.diagnostic.npz'), values)
        meta = dict(status='sparse_motion_diagnostic_only_not_resume_qualified', hz=HZ,
            source=before, original_source=source['source_checkpoint'], scope=report['scope'],
            solver_profile=source['solver_profile'], frames=len(rows), **info)
        probe.bench._write_json(Path(info['path'] + '.json'), meta)
        info['metadata_sha256'] = probe.bench.sha256(Path(info['path'] + '.json'))
        report.setdefault('diagnostic_snapshots', []).append(info)
        probe.bench._append_journal(journal, dict(event='snapshot_saved', label=label, **info))

    try:
        root = Path(args.repository_root).resolve()
        loader, identity = probe.select(47)
        base = loader(args.config, root)
        candidate_path = Path(args.candidate_report).resolve()
        audit_path = Path(args.resume_report).resolve()
        arrays, source, before = audit.load_candidate(candidate_path, base, 47, HZ)
        audit.validate_seed_audit(json.loads(audit_path.read_text()), candidate_path, source, before, 47, HZ)
        if before['sha256'] != args.checkpoint_sha256:
            raise ValueError('motion_input_checkpoint_sha256')
        if source['config_sha256'] != probe.bench.sha256(Path(args.config)):
            raise ValueError('motion_input_config_changed')
        original = source['source_checkpoint']
        if probe.bench.checkpoint_identity(original['path']) != original:
            raise ValueError('motion_original_source_changed')
        original_meta = Path(original['path'] + '.json')
        if probe.bench.sha256(original_meta) != source['metadata_sha256']:
            raise ValueError('motion_original_metadata_changed')
        report.update(source_checkpoint=before, original_source=original,
            candidate_report=str(candidate_path), resume_report=str(audit_path),
            candidate_report_sha256=probe.bench.sha256(candidate_path),
            resume_report_sha256=probe.bench.sha256(audit_path), solver_profile=source['solver_profile'])
        manifest = json.loads((root / base['physical_geometry']['manifest_path']).read_text())
        records = [r for r in manifest['rocks'] if r['layer_id'] in identity['layers']]
        if [r['rock_id'] for r in records] != identity['expected_ids']:
            raise ValueError('motion_manifest_order')
        radii = []
        for r in records:
            vertices, _ = _read_obj(root / 'assets/v4/mixed-material-r0' / (r['prototype_id'] + '.obj'))
            radii.append(float(np.linalg.norm(vertices, axis=1).max()) * r['equivalent_diameter_m'])
        _, proxy = probe.p47._load_bound_json(root, base['open_bucket_proxy'], 'proxy')
        base['probe'] = {**base['probe'], 'output_dt_s': 1/HZ, 'mujoco_internal_substeps': 80}
        probe.bench._append_journal(journal, dict(event='started', **report))
        newton.use_coord_layout_targets = True
        wp.init()
        wp.set_device(args.device)
        with tempfile.TemporaryDirectory(prefix='sparse-motion-') as raw:
            fixture, held, report['restore'] = audit.restore_candidate_runtime(
                args, base, root, records, arrays, proxy, source['solver_profile'], Path(raw))
            bridge = CommandBridge(held.bridge)
            initial_axes = arrays['mechanism_qpos'][AXES]
            def clearance():
                return float(probe.p47._bucket_world_vertices(proxy, bridge.state)[:, 2].min()) - terrain_top(
                    fixture, bridge, radii, base['mpm']['particle_spacing_m'])
            report['initial_clearance_m'] = clearance()
            if report['initial_clearance_m'] < .15:
                raise ValueError('motion_initial_clearance')
            for phase, frames, command, axis in schedule():
                phase_start = np.asarray(bridge.state['mechanism_qpos'])[AXES].copy()
                start_index = len(rows)
                bridge.command = command.copy()
                for _ in range(frames):
                    tick = time.perf_counter()
                    row, _ = probe.bench._advance_frame(fixture, bridge, base, len(rows)+1)
                    row['finite'] = bool(row['finite'] and probe.bench._finite_tree(bridge.state))
                    positions = np.asarray([r['qpos'][:3] for r in bridge.state['rocks']])
                    q = np.asarray(bridge.state['mechanism_qpos'])[AXES]
                    row.update(phase=phase, command_si=command.tolist(), axis_positions=q.tolist(),
                        axis_offset=(q-initial_axes).tolist(),
                        axis_velocity=np.asarray(bridge.state['mechanism_qvel'])[AXES-1].tolist(),
                        bucket_vertical_clearance_m=clearance(),
                        bucket_linear_speed_m_s=float(np.linalg.norm(bridge.state['bucket_linear_velocity_world_m_s'])),
                        bucket_angular_speed_rad_s=float(np.linalg.norm(bridge.state['bucket_angular_velocity_world_rad_s'])),
                        rock_displacement_from_source_m=float(np.linalg.norm(positions-arrays['rigid_qpos'][:, :3], axis=1).max()),
                        elapsed_simulation_seconds=(len(rows)+1)/HZ,
                        total_frame_wall_seconds=time.perf_counter()-tick)
                    qualify(row)
                    row['stopped'] = stopped(row)
                    rows.append(row)
                    probe.bench._append_journal(journal, dict(event='frame', row=row))
                    if len(rows) % 15 == 0 or not row['motion_guard']:
                        print(f"SPARSE_MOTION frame={len(rows)}/270 phase={phase} strict={row['strict']} clearance={row['bucket_vertical_clearance_m']:.6f}", flush=True)
                    if not row['motion_guard']:
                        report['rejection'] = dict(reason='frame_guard', row=row)
                        break
                done = len(rows)-start_index == frames
                result = dict(phase=phase, frames=len(rows)-start_index, complete=done,
                    all_frames_passed=all(r['motion_guard'] for r in rows[start_index:]))
                if done and axis is not None:
                    if phase.endswith('_stop'):
                        result['stopped_tail'] = all(r['stopped'] for r in rows[-15:])
                    else:
                        delta = rows[-1]['axis_positions'][axis] - phase_start[axis]
                        result['actual_axis_delta'] = float(delta)
                        result['requested_axis_delta'] = float(command[axis] * frames/HZ)
                        result['motion_response_passed'] = motion_response(delta, result['requested_axis_delta'])
                result['passed'] = all(result.get(k, True) for k in ('complete', 'all_frames_passed', 'stopped_tail', 'motion_response_passed'))
                phase_results.append(result)
                if rows[-1]['finite']:
                    save_snapshot(phase)
                probe.bench._append_journal(journal, dict(event='phase_complete', **result))
                if not result['passed']:
                    report.setdefault('rejection', dict(reason='phase_acceptance', phase=result, last_row=rows[-1]))
                    break
            report['checks'] = dict(restore_exact=report['restore']['exact'], all_frames_executed=len(rows)==270,
                all_frames_strict=all(r['strict'] for r in rows), all_frames_noncontact_safe=all(r['motion_guard'] for r in rows),
                all_nine_phases_passed=len(phase_results)==9 and all(r['passed'] for r in phase_results),
                first_step_continuous=rows[0]['maximum_rock_position_step_m'] <= .01/HZ*1.01,
                source_unchanged=probe.bench.checkpoint_identity(before['path']) == before,
                metadata_unchanged=probe.bench.sha256(Path(before['path']+'.json')) == source['diagnostic_snapshots'][-1]['metadata_sha256'],
                original_source_unchanged=probe.bench.checkpoint_identity(original['path']) == original,
                original_metadata_unchanged=probe.bench.sha256(original_meta) == source['metadata_sha256'],
                candidate_report_unchanged=probe.bench.sha256(candidate_path)==report['candidate_report_sha256'],
                resume_report_unchanged=probe.bench.sha256(audit_path)==report['resume_report_sha256'],
                lineage_unchanged=probe.seed_unchanged(source['seed']))
            report['status'] = 'sparse_motion_passed' if all(report['checks'].values()) else 'sparse_motion_rejected'
    except (Exception, KeyboardInterrupt) as exc:
        report['status'] = 'sparse_motion_failed'
        report['failure'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        if rows and rows[-1]['finite']:
            try:
                save_snapshot('interrupted')
            except Exception as error:
                report['snapshot_error'] = str(error)
    finally:
        if bridge is not None:
            try:
                bridge.close(kill=report['status']=='sparse_motion_failed')
            except Exception as exc:
                report['cleanup_error'] = str(exc)
        report.update(trace=rows, phases=phase_results,
            execution=dict(frames=len(rows), wall_seconds=time.perf_counter()-started, journal=str(journal)))
        physics_wall = sum(r['wall_seconds'] for r in rows)
        report['timing'] = dict(simulation_seconds=len(rows)/HZ, physics_wall_seconds=physics_wall,
            measured_frame_wall_seconds=sum(r['total_frame_wall_seconds'] for r in rows),
            simulation_seconds_per_physics_wall_second=len(rows)/HZ/physics_wall if physics_wall else None)
        def clean(value):
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, list):
                return [clean(v) for v in value]
            return None if isinstance(value, float) and not np.isfinite(value) else value
        probe.bench._write_json(out, clean(report))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('config', 'repository-root', 'output', 'candidate-report', 'resume-report',
                 'checkpoint-sha256', 'device', 'mujoco-include', 'mujoco-library-dir',
                 'mechanism-header-dir', 'controller-header-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--cxx', default='g++')
    args = parser.parse_args()
    result = run(args)
    print('COPY_RESULT_BEGIN')
    print(json.dumps({k: result.get(k) for k in ('status', 'rocks', 'hz', 'checks', 'execution', 'timing',
        'phases', 'failure', 'rejection', 'source_checkpoint', 'diagnostic_snapshots')}
        | dict(report=str(Path(args.output).resolve()), report_exists=Path(args.output).is_file(),
               runner_exit_code=0 if result['status']=='sparse_motion_passed' else 1,
               next_gate='review_before_manual_noncontact_practice_no_training'), indent=2))
    print('COPY_RESULT_END')
    raise SystemExit(0 if result['status']=='sparse_motion_passed' else 1)
