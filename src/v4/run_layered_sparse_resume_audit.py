#!/usr/bin/env python3
"""New-process, zero-warmup continuation audit of a bound sparse candidate."""
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
from layered_frequency_bridge import HoldBridge


AUDIT_CHECKS = ('restore_exact', 'all_frames_executed', 'all_frames_strict',
    'first_step_continuous', 'candidate_unchanged', 'original_source_unchanged',
    'candidate_report_unchanged', 'metadata_unchanged')


def validate_seed_audit(audit, report_path, source, before, rocks, hz,
                        *, auditor_source_path=None):
    """No missing checks or audit of a different checkpoint may unlock a transition."""
    # Historical evidence binds its actual archived producer, never a silently
    # rewritten report hash. Current runtime restoration is checked separately.
    auditor_source = (Path(auditor_source_path) if auditor_source_path is not None
                      else Path(__file__))
    if (audit.get('schema') != 'layered_sparse_resume_audit_v1'
            or audit.get('status') != 'sparse_resume_audit_passed'
            or audit.get('rocks') != rocks or audit.get('hz') != hz
            or any(audit.get('checks', {}).get(k) is not True for k in AUDIT_CHECKS)
            or audit.get('runner_sha256') != probe.bench.sha256(auditor_source)
            or audit.get('candidate_report') != str(report_path.resolve())
            or audit.get('candidate_report_sha256') != probe.bench.sha256(report_path)
            or audit.get('candidate_checkpoint') != before
            or audit.get('execution', {}).get('frames') != 2 * hz
            or len(audit.get('trace', [])) != 2 * hz
            or not all(r.get('strict') is True for r in audit['trace'])
            or source['policy']['hz'] != hz):
        raise ValueError('seed_independent_audit_binding')


def restore_candidate_runtime(args, base, root, records, arrays, proxy,
                              solver_profile, work, controller_state=None,
                              static_fixtures=None,
                              bucket_collision_geometry=None):
    model, binary, mm, bm, command = probe.bench.build_model_and_bridge(args, root,
        root / base['capacity_bridge']['config_path'],
        probe.bench.records_at_qpos(records, arrays['rigid_qpos']), work, 'audit')
    bridge = HoldBridge(probe.bench.RestoredRuntimeBridge(binary, model, len(records),
        arrays['mechanism_qpos'].tolist(), arrays['mechanism_qvel'].tolist(),
        arrays['rigid_qpos'], arrays['rigid_qvel'], controller_state))
    try:
        fixture = probe.p47.ExcavationFixture(base, root, records, arrays['grid_reference_particle_q'],
            proxy, bridge.state, args.device, solver_profile=solver_profile,
            static_fixtures=static_fixtures,
            bucket_collision_geometry=bucket_collision_geometry)
        fields = probe.bench.restore_fixture_prefix(fixture, arrays)
        fixture.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        fixture.assign(bridge.state)
        solver = probe.snapshots.restore_solver_continuation(fixture, arrays)
        errors = {}
        for name, actual in [('rigid_qpos', [r['qpos'] for r in bridge.state['rocks']]),
                             ('rigid_qvel', [r['qvel'] for r in bridge.state['rocks']]),
                             ('mechanism_qpos', bridge.state['mechanism_qpos']),
                             ('mechanism_qvel', bridge.state['mechanism_qvel'])]:
            errors[name] = float(np.max(np.abs(arrays[name] - np.asarray(actual))))
        if controller_state is not None:
            for name in ('controller_target', 'controller_target_velocity',
                         'bucket_integral_effort'):
                errors[name] = float(np.max(np.abs(
                    np.asarray(controller_state[name], dtype=np.float64)
                    - np.asarray(bridge.state[name], dtype=np.float64))))
        exact = all(fields['field_readback_exact'].values()) and all(solver['grid_readback_exact'].values())
        exact = exact and all(solver['solver_field_readback_exact'].values()) and max(errors.values()) <= 1e-12
        restored = dict(particles=fields, solver=solver, errors=errors, exact=exact,
                        runtime_build=dict(model=mm, bridge=bm, command=command))
        if not exact:
            raise ValueError('candidate_restore_inexact')
        return fixture, bridge, restored
    except BaseException:
        bridge.close(kill=True)
        raise


def load_candidate(report_path, base, rocks, hz):
    report = json.loads(report_path.read_text())
    if (report.get('schema') != 'layered_sparse_a30_probe_v2'
            or report.get('status') != 'candidate_short_hold_passed'
            or any(report.get('checks', {}).get(k) is not True for k in probe.REQUIRED_CHECKS)
            or not all(report['checks'].values())
            or report['rocks'] != rocks or report['policy'] != probe.policy(hz, report.get('transition', 'grid'))
            or report['solver'] != 'sparse10'
            or report['source_checkpoint']['sha256'] != base['input_checkpoint']['sha256']
            or report['solver_profile'] != probe.profile(base, 'sparse10')
            or report['runner_sha256'] != probe.bench.sha256(Path(probe.__file__))):
        raise ValueError('candidate_report_binding')
    selected = report['diagnostic_snapshots'][-1]
    path = Path(selected['path'])
    if not path.name.endswith('-measured.diagnostic.npz'):
        raise ValueError('candidate_must_be_complete_measured_endpoint')
    identity = probe.bench.checkpoint_identity(path)
    if any(identity[k] != selected[k] for k in ('sha256', 'size_bytes', 'mtime_ns')):
        raise ValueError('candidate_checkpoint_identity')
    metadata_path = Path(str(path) + '.json')
    if probe.bench.sha256(metadata_path) != selected['metadata_sha256']:
        raise ValueError('candidate_metadata_sha256')
    metadata = json.loads(metadata_path.read_text())
    if (metadata['status'] != 'grid_candidate_diagnostic_only' or metadata['hz'] != hz
            or metadata['solver_profile'] != report['solver_profile']
            or metadata['source'] != report['source_checkpoint']
            or metadata.get('transition', 'grid') != report.get('transition', 'grid')
            or metadata.get('seed') != report.get('seed')
            or any(metadata['scope'].values())):
        raise ValueError('candidate_metadata_contract')
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    probe.snapshots._validate(arrays, base['input_checkpoint']['particle_count'], rocks)
    if report.get('transition') == 'frequency':
        if (report['checks'].get('seed_restore_exact') is not True
                or report['checks'].get('audited_seed_unchanged') is not True
                or not probe.seed_unchanged(report['seed'])):
            raise ValueError('candidate_frequency_seed_binding')
    return arrays, report, identity


def run(args):
    started = time.perf_counter()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump({'status': 'initializing'}, stream)
    report = dict(schema='layered_sparse_resume_audit_v1', status='initializing', rocks=args.rocks,
                  hz=args.hz, scope=dict(training_eligible=False, manual_collection_eligible=False,
                  bucket_contact_qualified=False, formal_a30_equivalent=False), checkpoint_written=False)
    rows, bridge = [], None
    journal = output.with_suffix('.progress.ndjson')
    try:
        root = Path(args.repository_root).resolve()
        loader, identity = probe.select(args.rocks)
        base = loader(args.config, root)
        candidate_report = Path(args.candidate_report).resolve()
        report_sha = probe.bench.sha256(candidate_report)
        arrays, source, before = load_candidate(candidate_report, base, args.rocks, args.hz)
        if source['config_sha256'] != probe.bench.sha256(Path(args.config)):
            raise ValueError('candidate_config_changed')
        source_path = Path(source['source_checkpoint']['path'])
        if probe.bench.checkpoint_identity(source_path) != source['source_checkpoint']:
            raise ValueError('original_source_changed')
        report.update(candidate_checkpoint=before, candidate_report_sha256=report_sha,
                      candidate_report=str(candidate_report), runner_sha256=probe.bench.sha256(Path(__file__)))
        manifest = json.loads((root / base['physical_geometry']['manifest_path']).read_text())
        records = [r for r in manifest['rocks'] if r['layer_id'] in identity['layers']]
        if [r['rock_id'] for r in records] != identity['expected_ids']:
            raise ValueError('audit_manifest_order')
        _, proxy = probe.p47._load_bound_json(root, base['open_bucket_proxy'], 'proxy')
        base['probe'] = {**base['probe'], 'output_dt_s': 1/args.hz, 'mujoco_internal_substeps': 4800//args.hz}
        newton.use_coord_layout_targets = True
        wp.init()
        wp.set_device(args.device)
        with tempfile.TemporaryDirectory(prefix='sparse-resume-') as raw:
            fixture, bridge, restored = restore_candidate_runtime(args, base, root, records,
                arrays, proxy, source['solver_profile'], Path(raw))
            exact = restored['exact']
            report['restore'] = restored
            probe.bench._append_journal(journal, dict(event='restored', **report))
            for frame in range(2 * args.hz):
                row, _ = probe.bench._advance_frame(fixture, bridge, base, frame + 1)
                row['finite'] = bool(row['finite'] and probe.bench._finite_tree(bridge.state))
                positions = np.asarray([r['qpos'][:3] for r in bridge.state['rocks']])
                row['rock_displacement_from_source_m'] = float(np.linalg.norm(positions - arrays['rigid_qpos'][:, :3], axis=1).max())
                probe.classify_frame(row, 'measured', 0)
                rows.append(row)
                probe.bench._append_journal(journal, dict(event='audit_frame', row=row))
                if frame == 0 or (frame + 1) % 15 == 0 or not row['strict']:
                    print(f"SPARSE_RESUME hz={args.hz} frame={frame+1}/{2*args.hz} strict={row['strict']}", flush=True)
                if not row['strict']:
                    report['rejection'] = dict(phase='independent_resume', row=row)
                    break
            report['checks'] = dict(restore_exact=exact, all_frames_executed=len(rows) == 2*args.hz,
                all_frames_strict=all(r['strict'] for r in rows),
                first_step_continuous=rows[0]['maximum_rock_position_step_m'] <= .01 / args.hz * 1.01,
                candidate_unchanged=probe.bench.checkpoint_identity(before['path']) == before,
                original_source_unchanged=probe.bench.checkpoint_identity(source_path) == source['source_checkpoint'],
                candidate_report_unchanged=probe.bench.sha256(candidate_report) == report_sha,
                metadata_unchanged=probe.bench.sha256(Path(before['path'] + '.json')) == source['diagnostic_snapshots'][-1]['metadata_sha256'])
            report['status'] = 'sparse_resume_audit_passed' if all(report['checks'].values()) else 'sparse_resume_audit_rejected'
    except (Exception, KeyboardInterrupt) as exc:
        report['status'] = 'sparse_resume_audit_failed'
        report['failure'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        if bridge is not None:
            try:
                bridge.close(kill=report['status'] == 'sparse_resume_audit_failed')
            except Exception as exc:
                report['cleanup_error'] = str(exc)
        seconds = sum(r['wall_seconds'] for r in rows)
        report.update(trace=rows, execution=dict(frames=len(rows), wall_seconds=time.perf_counter()-started,
            journal=str(journal)), timing=dict(simulation_seconds=len(rows)/args.hz,
            physics_wall_seconds=seconds, simulation_seconds_per_wall_second=len(rows)/args.hz/seconds if seconds else None))
        def clean(v):
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if isinstance(v, list):
                return [clean(x) for x in v]
            return None if isinstance(v, float) and not np.isfinite(v) else v
        probe.bench._write_json(output, clean(report))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('config', 'repository-root', 'output', 'candidate-report', 'mujoco-include',
                 'mujoco-library-dir', 'mechanism-header-dir', 'controller-header-dir', 'device'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--rocks', type=int, choices=(47, 65), required=True)
    parser.add_argument('--hz', type=int, choices=(60, 120), required=True)
    args = parser.parse_args()
    result = run(args)
    print('COPY_RESULT_BEGIN')
    print(json.dumps({k: v for k, v in result.items() if k in ('status', 'rocks', 'hz', 'checks', 'execution', 'timing', 'failure', 'rejection', 'candidate_checkpoint')}
        | dict(report=str(Path(args.output).resolve()), report_exists=Path(args.output).is_file(),
               runner_exit_code=0 if result['status'] == 'sparse_resume_audit_passed' else 1), indent=2))
    print('COPY_RESULT_END')
    raise SystemExit(0 if result['status'] == 'sparse_resume_audit_passed' else 1)
