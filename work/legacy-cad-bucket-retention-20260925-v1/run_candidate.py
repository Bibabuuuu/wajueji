"""One bounded 10 s legacy-CAD bucket retention candidate from a saved load."""

from contextlib import ExitStack
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import traceback

import newton
import numpy as np
import warp as wp

import audit_layered_47_truck_physics_gate as gate
from layered_excavation_approach_planner import _load_kinematics
from legacy_cad_bucket_collision_candidate import build_geometry, public_summary
from live_bucket_fill import load_config, MixedFillEstimator
from soil_hencky_update_guard import installed_material_guard
import soil_hencky_update_candidate as hencky


ROOT = Path('/home/qunwang/wajueji')
WORK = Path(__file__).resolve().parent
BUNDLE = Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1')
KIN = Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
LOADED = ROOT/'runs/layered-47-loaded-handoff-v1/d129-loaded-handoff-20260923-v2'
SOURCE_RUN = ROOT/'runs/layered-47-loaded-transport-v1/loaded-transport-20260923-215454-962382'
WALL_LIMIT_S = 420.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-name', default='candidate-600')
    parser.add_argument('--frames', type=int, default=600)
    parser.add_argument('--check-only', action='store_true')
    cli = parser.parse_args()
    if cli.frames < 1 or cli.frames > 600:
        raise ValueError('legacy_cad_candidate_frames_range')
    frames_limit = cli.frames
    out = WORK/cli.output_name
    out.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(
        repository_root=str(BUNDLE/'repo'), truck_repository_root=str(ROOT),
        config=str(BUNDLE/'repo/configs/v4/layered-47-excavation-preflight-v1.json'),
        local_source_config=str(ROOT/'configs/v4/layered-47-interactive-proxy-benchmark-v1.json'),
        checkpoint=str(LOADED/'loaded-handoff.npz'),
        metadata=str(LOADED/'loaded-handoff.npz.json'),
        approach_report=str(ROOT/'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089/report.json'),
        truck_config=str(ROOT/'configs/v4/layered-47-truck-physics-v1.json'),
        device='cuda:0', cxx='g++',
        mujoco_include=str(BUNDLE/'mujoco/include'),
        mujoco_library_dir=str(BUNDLE/'mujoco/lib'),
        mechanism_header_dir=str(BUNDLE/'repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8'),
        controller_header_dir=str(BUNDLE/'repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator'),
        kinematics_python_dir=str(KIN),
        kinematics_model=str(KIN/'scene/input/model/A.xml'),
        kinematics_mesh_dir=str(KIN/'scene/input/mesh'))
    trace = [json.loads(line) for line in
             (SOURCE_RUN/'physics.ndjson').read_text().splitlines()][:frames_limit]
    if len(trace) != frames_limit:
        raise ValueError('legacy_cad_candidate_source_trace_length')
    arrays, metadata, source_identity = gate.handoff.load_checkpoint_readonly(
        Path(args.checkpoint), Path(args.metadata))
    base = gate.motion.probe.p47.load_config(args.config, BUNDLE/'repo')
    base['input_checkpoint'] = json.loads(
        Path(args.local_source_config).read_text())['input_checkpoint']
    base['probe'] = {**base['probe'], 'output_dt_s': 1/60,
                     'mujoco_internal_substeps': 80}
    manifest_path = BUNDLE/'repo'/base['physical_geometry']['manifest_path']
    manifest = json.loads(manifest_path.read_text())
    _, profile = gate.motion.probe.select(47)
    records = [record for record in manifest['rocks']
               if record['layer_id'] in profile['layers']]
    _, proxy = gate.motion.probe.p47._load_bound_json(
        BUNDLE/'repo', base['open_bucket_proxy'], 'proxy')
    approach = json.loads(Path(args.approach_report).read_text())
    assembly, chain = _load_kinematics(
        args.kinematics_python_dir, args.kinematics_model,
        args.kinematics_mesh_dir)
    projector = gate.manual.ContactCommandProjector(chain)
    cad = build_geometry(assembly, arrays['mechanism_qpos'])
    cached_rocks = gate.manual._cache_rock_vertices(BUNDLE/'repo', records)
    estimator = MixedFillEstimator(
        load_config(ROOT/'configs/v4/live-bucket-fill-local-v1.json'),
        [record['volume_m3'] for record in records], cached_rocks)
    contract = json.loads(Path(
        '/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json').read_text())
    with np.load('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/front-rim.npz') as data:
        front = data['front'].copy()
    polygon = np.asarray(contract['bucket']['polygon_xz_m'])
    y_limits = np.asarray(contract['bucket']['y_limits_m'])
    initial_pose = assembly.assemble(
        arrays['mechanism_qpos'], include_meshes=False)[1]['bucket']
    initial_local = ((arrays['particle_q']-initial_pose[:3, 3])
                     @ initial_pose[:3, :3])
    initial_cavity = gate.loaded.cavity_mask(
        initial_local, polygon, y_limits, front)

    bindings_paths = [Path(args.checkpoint), Path(args.metadata),
                      Path(args.config), Path(args.truck_config),
                      SOURCE_RUN/'physics.ndjson', Path(__file__),
                      ROOT/'src/v4/run_layered_47_excavation_preflight.py',
                      ROOT/'src/v4/run_layered_sparse_resume_audit.py',
                      ROOT/'src/v4/legacy_cad_bucket_collision_candidate.py',
                      Path(hencky.__file__), Path(cad['source_mesh_path'])]
    bindings = {str(path): sha256(path) for path in bindings_paths}
    protocol = {
        'schema': 'legacy_cad_bucket_retention_protocol_v1',
        'frames': frames_limit, 'simulation_seconds': frames_limit/60,
        'wall_limit_seconds': WALL_LIMIT_S,
        'source_checkpoint': source_identity,
        'source_trace': str(SOURCE_RUN/'physics.ndjson'),
        'collision_candidate': public_summary(cad),
        'moving_bucket_sweep_enabled': False,
        'soil_material_changed': False, 'rocks_changed': False,
        'commands_changed': False, 'truck_changed': False,
        'default_runtime_changed': False, 'resume_qualified': False,
        'training_eligible': False, 'bindings': bindings,
        'acceptance': {'minimum_strict_volume_retention_fraction': .95,
                       'maximum_fill_loss_percentage_points': 5.0,
                       'all_hard_guards': True}}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')

    report = {'schema': 'legacy_cad_bucket_retention_report_v1',
              'status': 'initializing', 'protocol': protocol}
    bridge = fixture = None
    rows = []
    started = time.perf_counter()
    completed = 0
    captured = {}

    def measure_fill(frame):
        pose = assembly.assemble(
            np.asarray(bridge.state['mechanism_qpos']),
            include_meshes=False)[1]['bucket']
        result = estimator.measure_snapshot({
            'frame': frame, 'soil': fixture.state.particle_q.numpy(),
            'pose': pose,
            'rocks': np.asarray([rock['qpos'] for rock in bridge.state['rocks']])})
        local = ((fixture.state.particle_q.numpy()-pose[:3, 3]) @ pose[:3, :3])
        cavity = gate.loaded.cavity_mask(local, polygon, y_limits, front)
        result['initial_cavity_particles_retained'] = int(
            np.count_nonzero(initial_cavity & cavity))
        result['initial_cavity_particle_count'] = int(initial_cavity.sum())
        return result

    try:
        wp.init(); wp.set_device(args.device); newton.use_coord_layout_targets = True
        with ExitStack() as contexts:
            contexts.enter_context(gate.optimized_bridge(
                args.mechanism_header_dir, out/'optimized-build'))
            contexts.enter_context(gate.bucket_integral_bridge(
                True, out/'bucket-integral-build'))
            fixtures = contexts.enter_context(gate.truck.installed_candidate(
                args.truck_config, ROOT))
            contexts.enter_context(hencky.installed_candidate())
            with gate.handoff.restored_sparse_grid_topology(arrays) as grid_restore:
                fixture, held, restored = gate.motion.audit.restore_candidate_runtime(
                    args, base, BUNDLE/'repo', records, arrays, proxy,
                    approach['solver_profile'], out/'runtime-build',
                    metadata['controller_state'], fixtures,
                    bucket_collision_geometry=cad)
            bridge = gate.rigid_frequency.RigidFrequencyCommandBridge(
                held.bridge, 4800)
            guard = contexts.enter_context(installed_material_guard(fixture.solver))
            if not restored['exact'] or grid_restore['count'] != 1:
                raise ValueError('legacy_cad_candidate_restore_inexact')
            if (fixture.bucket_collision_mode != 'legacy_cad_mesh_native_contact'
                    or fixture.bucket_collision_geom_count != 1
                    or fixture.bucket_collider_ids.size != 1):
                raise ValueError('legacy_cad_candidate_not_active')
            original_step = fixture.step
            def measured_step(dt, origins):
                wrenches, particles = original_step(dt, origins)
                captured['bucket_wrench'] = wrenches['bucket']
                return wrenches, particles
            fixture.step = measured_step
            report['restore'] = restored
            report['collision_runtime'] = {
                'mode': fixture.bucket_collision_mode,
                'geom_count': fixture.bucket_collision_geom_count,
                'solver_bucket_collider_count': int(fixture.bucket_collider_ids.size),
                'moving_bucket_sweep_enabled': False}
            report['fill_samples'] = [measure_fill(0)]
            if cli.check_only:
                report.update(
                    status='candidate_preflight_passed', frames=0,
                    checks={
                        'restore_exact': bool(restored['exact']),
                        'legacy_cad_collision_active': True,
                        'single_solver_bucket_collider':
                            fixture.bucket_collider_ids.size == 1,
                        'initial_fill_valid':
                            report['fill_samples'][0]['valid_estimate'],
                        'source_checkpoint_unchanged':
                            gate.motion.probe.bench.checkpoint_identity(
                                args.checkpoint) == source_identity,
                        'bindings_unchanged': all(
                            sha256(path) == digest
                            for path, digest in bindings.items())})
                if not all(report['checks'].values()):
                    raise RuntimeError('legacy_cad_candidate_preflight_checks')
                return
            with (out/'trace.ndjson').open('w') as stream:
                for frame, source in enumerate(trace, 1):
                    if time.perf_counter()-started > WALL_LIMIT_S:
                        raise TimeoutError('legacy_cad_candidate_wall_budget')
                    frame_started = time.perf_counter()
                    bridge.command = np.asarray(source['applied_command_si'])
                    guard.prepare_frame()
                    row, _ = gate.motion.probe.bench._advance_frame(
                        fixture, bridge, base, frame)
                    completed = frame
                    material = guard.summarize(fixture.state)
                    q = fixture.state.particle_q.numpy()
                    qd = fixture.state.particle_qd.numpy()
                    speed = np.linalg.norm(qd, axis=1)
                    row.update(
                        finite=bool(row['finite'] and np.isfinite(q).all()
                                    and np.isfinite(qd).all()),
                        particle_max_speed_m_s=float(speed.max()),
                        particle_p99_speed_m_s=float(np.quantile(speed, .99)),
                        particle_minimum_z_m=float(q[:, 2].min()),
                        applied_command_si=bridge.command.tolist(),
                        hencky_material_guard=material,
                        joint_bounds_ok=projector.physical_bounds_ok(
                            np.asarray(bridge.state['mechanism_qpos'])),
                        bucket_mpm_wrench_finite=all(
                            np.isfinite(value).all() for value in
                            captured['bucket_wrench'].values()),
                        rigid_contact_substeps=bridge.state['contact_substeps'])
                    hard = gate.manual.hard_guard_checks(row, allow_truck=True)
                    record = {
                        'frame': frame, 'hard_guard_checks': hard,
                        'material_safe': material['safe'],
                        'particle_max_speed_m_s': row['particle_max_speed_m_s'],
                        'bucket_mpm_contact_samples': row['bucket_mpm_contact_samples'],
                        'postsolve_projection': row['postsolve_projection'],
                        'frame_wall_seconds': time.perf_counter()-frame_started}
                    rows.append(record)
                    stream.write(json.dumps(record)+'\n'); stream.flush()
                    if not all(hard.values()) or not material['safe']:
                        raise RuntimeError('legacy_cad_candidate_hard_guard')
                    if frame % 60 == 0:
                        fill = measure_fill(frame)
                        report['fill_samples'].append(fill)
                        print('CAD_RETENTION', frame, 'fill', fill['fill_percent'],
                              'wall', round(time.perf_counter()-started, 1), flush=True)
            final = gate.motion.probe.snapshots.capture_arrays(
                fixture, bridge.state,
                float(arrays['simulation_time_s'][0])+completed/60)
            report['terminal'] = gate.motion.probe.snapshots.write_rejected_diagnostic_atomic(
                out/f'terminal-{completed:06d}.npz', final,
                'legacy CAD collision candidate only; not a qualified resume point')
            initial_fill = report['fill_samples'][0]
            final_fill = report['fill_samples'][-1]
            retention = final_fill['liters']/initial_fill['liters']
            loss_points = initial_fill['fill_percent']-final_fill['fill_percent']
            checks = {
                'all_requested_frames_executed': completed == frames_limit,
                'all_hard_guards_passed': all(
                    all(row['hard_guard_checks'].values()) for row in rows),
                'all_material_guards_passed': all(
                    row['material_safe'] for row in rows),
                'all_fill_samples_valid': all(
                    item['valid_estimate'] for item in report['fill_samples']),
                'strict_volume_retention_at_least_95_percent': retention >= .95,
                'fill_loss_no_more_than_5_percentage_points': loss_points <= 5.0,
                'source_checkpoint_unchanged':
                    gate.motion.probe.bench.checkpoint_identity(args.checkpoint)
                    == source_identity,
                'bindings_unchanged': all(
                    sha256(path) == digest for path, digest in bindings.items())}
            report.update(
                strict_volume_retention_fraction=retention,
                fill_loss_percentage_points=loss_points,
                checks=checks,
                status=('candidate_passed' if all(checks.values())
                        else 'candidate_rejected'))
    except BaseException as error:
        report.update(status='candidate_failed', frames=completed,
                      error=repr(error), traceback=traceback.format_exc())
        if fixture is not None and bridge is not None:
            try:
                final = gate.motion.probe.snapshots.capture_arrays(
                    fixture, bridge.state,
                    float(arrays['simulation_time_s'][0])+completed/60)
                report['failure_snapshot'] = (
                    gate.motion.probe.snapshots.write_rejected_diagnostic_atomic(
                        out/f'failure-{completed:06d}.npz', final,
                        'failed legacy CAD candidate; not resumable'))
            except Exception as save_error:
                report['failure_snapshot_error'] = repr(save_error)
        raise
    finally:
        report['frames'] = completed
        report['wall_seconds'] = time.perf_counter()-started
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        if bridge is not None:
            bridge.close(kill=report['status'] == 'candidate_failed')
    print(json.dumps({'status': report['status'], 'frames': completed,
                      'retention': report.get('strict_volume_retention_fraction'),
                      'loss_points': report.get('fill_loss_percentage_points'),
                      'wall_seconds': report['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
