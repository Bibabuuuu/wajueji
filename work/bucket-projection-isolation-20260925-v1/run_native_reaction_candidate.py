"""Bounded, opt-in native projection velocity-reaction continuation.

This does not alter Newton's projection or the repository default. It routes
only the measured postsolve velocity impulse to dynamic collider wrenches.
"""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np

from audit_layered_47_loaded_handoff import cavity_mask
from layered_47_cycle_runtime import CycleRuntime, KINEMATICS, REPO
from layered_excavation_approach_planner import _load_kinematics
from native_projection_reaction_candidate import NativeProjectionReaction


WORK = Path(__file__).resolve().parent
ORIGINAL = REPO / 'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275/input.ndjson'
REFERENCE = WORK / 'manual-prefix-continuation-1620/display/particles.npy'
PHYSICAL_FIELDS = ('particle_q', 'particle_qd', 'particle_f', 'particle_elastic_strain',
                   'particle_Jp', 'particle_stress', 'particle_qd_grad', 'particle_transform')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, choices=(1080, 1680), default=1080)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()
    start = args.start
    end = args.end or (1530 if start == 1080 else 1681)
    if not start < end <= 1800:
        raise ValueError('bounded_frame_range')
    wall_limit = 180 if end - start > 1 else 45
    source = WORK / f'manual-prefix-continuation-1620/state-{start:06d}.npz'
    out = WORK / f'native-reaction-{start}-{end}'
    started = time.monotonic()
    entries = [json.loads(line) for line in ORIGINAL.read_text().splitlines()][start:end]
    assert len(entries) == end - start
    reference = np.load(REFERENCE, mmap_mode='r')
    assembly, _ = _load_kinematics(str(KINEMATICS), str(KINEMATICS / 'scene/input/model/A.xml'),
                                   str(KINEMATICS / 'scene/input/mesh'))
    contract = json.loads(Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json').read_text())
    polygon = np.asarray(contract['bucket']['polygon_xz_m'])
    y_limits = np.asarray(contract['bucket']['y_limits_m'])
    front = np.load('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/front-rim.npz')['front']
    report = dict(status='initializing', candidate='route_native_projection_velocity_reaction',
                  start_frame=start, end_frame_limit=end, wall_limit_seconds=wall_limit,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  input_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                  original_default_unchanged=True, training_eligible=False,
                  position_correction_impulse_included=False,
                  rows=[], command_changes=[], guards_failed=[])
    runtime = None
    reaction = None
    try:
        with CycleRuntime(source, out, enable_truck=True, freeze_passive_jp=True,
                          bucket_motion_consistency=True) as runtime:
            fixture = runtime.fixture
            before = {name: getattr(fixture.state.mpm, name).numpy().copy()
                      if hasattr(fixture.state.mpm, name) else getattr(fixture.state, name).numpy().copy()
                      for name in PHYSICAL_FIELDS}
            reaction = NativeProjectionReaction(fixture)
            reaction.install()
            prior_step = fixture.step

            def step_with_reaction(dt, origins):
                wrenches, particles = prior_step(dt, origins)
                reaction.route(wrenches, origins, dt)
                runtime.captured['wrenches'] = wrenches
                return wrenches, particles

            fixture.step = step_with_reaction
            unchanged = {name: bool(np.array_equal(
                before[name], getattr(fixture.state.mpm, name).numpy()
                if hasattr(fixture.state.mpm, name) else getattr(fixture.state, name).numpy()))
                for name in PHYSICAL_FIELDS}
            if not all(unchanged.values()):
                raise ValueError('native_reaction_zero_step_state_changed')
            report['zero_step_physical_arrays_unchanged'] = unchanged
            q0 = fixture.state.particle_q.numpy()
            pose0 = assembly.assemble(np.asarray(runtime.bridge.state['mechanism_qpos']),
                                      include_meshes=False)[1]['bucket']
            local0 = (q0 - pose0[:3, 3]) @ pose0[:3, :3]
            ids = np.flatnonzero(cavity_mask(local0, polygon, y_limits, front))
            report['tracked_particle_count'] = len(ids)
            if start == 1080 and len(ids) != 12086:
                raise ValueError('native_reaction_source_cohort_changed')
            report['status'] = 'running'
            for entry in entries:
                frame = start + runtime.frames + 1
                if time.monotonic() - started > wall_limit:
                    raise TimeoutError('native_reaction_wall_budget')
                requested = np.asarray(entry['applied_command_si'])
                applied = runtime.projector.project(runtime.bridge.state['mechanism_qpos'], requested)
                if not np.array_equal(requested, applied):
                    report['command_changes'].append(dict(frame=frame, reason=runtime.projector.reason))
                row = runtime.step(applied)
                if not (row['hard_guard'] and row['hencky_material_guard']['safe']
                        and row['moving_bucket_sweep']['safe']):
                    report['guards_failed'].append(frame)
                    raise ValueError('native_reaction_guard_failed')
                if frame in (1081, 1200, 1400, 1490, 1500, 1510, 1520, 1530, 1681):
                    q = fixture.state.particle_q.numpy()
                    pose = assembly.assemble(np.asarray(runtime.bridge.state['mechanism_qpos']),
                                             include_meshes=False)[1]['bucket']
                    local = (q[ids] - pose[:3, 3]) @ pose[:3, :3]
                    _, counts = np.unique(np.floor(local / .02).astype('int16'), axis=0,
                                          return_counts=True)
                    item = dict(frame=frame, occupied_2cm_cells=len(counts),
                                maximum_particles_per_cell=int(counts.max()),
                                source_particles_still_in_cavity=int(np.sum(cavity_mask(local, polygon, y_limits, front))),
                                max_abs_particle_difference_from_P0_m=float(np.max(np.abs(q - reference[frame - 180]))),
                                hard_guard=True, material_guard_safe=True, sweep_safe=True)
                    report['rows'].append(item)
                    print('NATIVE_REACTION', frame, item['occupied_2cm_cells'],
                          item['maximum_particles_per_cell'], flush=True)
            if reaction.calls != end - start or reaction.routed != end - start or reaction.pending is not None:
                raise ValueError('native_reaction_call_count')
            if report['command_changes']:
                raise ValueError('native_reaction_changed_input')
            report['status'] = 'bounded_candidate_passed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['frames_completed'] = runtime.frames if runtime is not None else 0
        report['wall_seconds'] = time.monotonic() - started
        if reaction is not None:
            report['reaction'] = reaction.summary()
        out.mkdir(exist_ok=True)
        (out / 'native-reaction-report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
