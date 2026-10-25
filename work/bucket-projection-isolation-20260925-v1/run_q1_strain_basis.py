"""One isolated Q1 strain-basis candidate from the saved first-scoop state.

The saved P0 particle/material/native arrays are preserved. The incompatible
P0 solver warmstart is deliberately rebuilt and never called an exact solver
continuation. This file does not change the formal/default solver profile.
"""
from contextlib import contextmanager
import hashlib
import json
import time
import traceback
from pathlib import Path

import newton
import numpy as np

from audit_layered_47_loaded_handoff import cavity_mask
from layered_47_cycle_runtime import CycleRuntime, KINEMATICS, REPO
from layered_excavation_approach_planner import _load_kinematics
import run_layered_47_excavation_preflight as preflight
import run_layered_47_sparse_motion as motion


WORK = Path(__file__).resolve().parent
SOURCE = WORK / 'manual-prefix-continuation-1620/state-001080.npz'
ORIGINAL = REPO / 'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275/input.ndjson'
OUT = WORK / 'q1-strain-1080-1530'
START, END, WALL_LIMIT = 1080, 1530, 180


@contextmanager
def allow_only_q1():
    original = preflight.mpm_solver_profile

    def isolated_profile(config, override=None):
        if override is None:
            return original(config, override)
        candidate = dict(override)
        if candidate.get('strain_basis') != 'Q1':
            raise ValueError('q1_candidate_expected')
        original(config, {**candidate, 'strain_basis': 'P0'})
        return candidate

    preflight.mpm_solver_profile = isolated_profile
    try:
        yield
    finally:
        preflight.mpm_solver_profile = original


class Q1CycleRuntime(CycleRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.solver_profile['strain_basis'] != 'P0':
            raise ValueError('q1_candidate_source_not_p0')
        self.solver_profile = {**self.solver_profile, 'strain_basis': 'Q1'}

    def _initialize_fixture(self, work, fixtures):
        model, binary, model_meta, bridge_meta, command = motion.probe.bench.build_model_and_bridge(
            self.args, self.root, self.root / self.base['capacity_bridge']['config_path'],
            motion.probe.bench.records_at_qpos(self.records, self.arrays['rigid_qpos']),
            work, 'q1-strain')
        held = motion.audit.HoldBridge(motion.probe.bench.RestoredRuntimeBridge(
            binary, model, len(self.records), self.arrays['mechanism_qpos'].tolist(),
            self.arrays['mechanism_qvel'].tolist(), self.arrays['rigid_qpos'],
            self.arrays['rigid_qvel'], self.metadata['controller_state']))
        try:
            with allow_only_q1():
                fixture = motion.probe.p47.ExcavationFixture(
                    self.base, self.root, self.records, self.arrays['grid_reference_particle_q'],
                    self.proxy, held.state, self.args.device, solver_profile=self.solver_profile,
                    static_fixtures=fixtures)
            restored = motion.probe.bench.restore_fixture_prefix(fixture, self.arrays)
            fixture.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
            fixture.assign(held.state)
            native_errors = {}
            for name, actual in (
                ('mechanism_qpos', held.state['mechanism_qpos']),
                ('mechanism_qvel', held.state['mechanism_qvel']),
                ('rigid_qpos', [item['qpos'] for item in held.state['rocks']]),
                ('rigid_qvel', [item['qvel'] for item in held.state['rocks']]),
            ):
                native_errors[name] = float(np.max(np.abs(np.asarray(actual) - self.arrays[name])))
            if not all(restored['field_readback_exact'].values()) or max(native_errors.values()) > 1e-12:
                raise ValueError('q1_physical_zero_step_restore_inexact')
            return fixture, held, dict(
                physical_arrays_exact=True, native_errors=native_errors,
                solver_profile=self.solver_profile, solver_warmstart='rebuilt_for_Q1',
                exact_old_solver_continuation=False,
                particle_material_history_preserved=True,
                runtime_build=dict(model=model_meta, bridge=bridge_meta, command=command))
        except BaseException:
            held.close(kill=True)
            raise


def main():
    start_time = time.monotonic()
    entries = [json.loads(line) for line in ORIGINAL.read_text().splitlines()][START:END]
    assert len(entries) == END - START
    assembly, _ = _load_kinematics(str(KINEMATICS), str(KINEMATICS / 'scene/input/model/A.xml'),
                                   str(KINEMATICS / 'scene/input/mesh'))
    contract = json.loads(Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json').read_text())
    front = np.load('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/front-rim.npz')['front']
    polygon = np.asarray(contract['bucket']['polygon_xz_m'])
    y_limits = np.asarray(contract['bucket']['y_limits_m'])
    prior = np.load(WORK / 'manual-prefix-continuation-1620/display/particles.npy', mmap_mode='r')
    report = dict(status='initializing', candidate='Q1_strain_basis_only', source=str(SOURCE),
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  input_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                  start_frame=START, end_frame_limit=END, wall_limit_seconds=WALL_LIMIT,
                  baseline='P0_origin_and_timing_fixed', numerical_warmstart_rebuilt=True,
                  default_profile_changed=False, training_eligible=False,
                  rows=[], command_changes=[])
    runtime = None
    try:
        with Q1CycleRuntime(SOURCE, OUT, enable_truck=True, freeze_passive_jp=True,
                            bucket_motion_consistency=True) as runtime:
            report['restore'] = runtime.restored
            initial_q = runtime.fixture.state.particle_q.numpy()
            pose = assembly.assemble(np.asarray(runtime.bridge.state['mechanism_qpos']),
                                     include_meshes=False)[1]['bucket']
            local = (initial_q - pose[:3, 3]) @ pose[:3, :3]
            ids = np.flatnonzero(cavity_mask(local, polygon, y_limits, front))
            if len(ids) != 12086:
                raise ValueError('q1_original_cavity_identity_changed')
            report['tracked_particle_count'] = len(ids)
            report['status'] = 'running'
            for entry in entries:
                frame = START + runtime.frames + 1
                if time.monotonic() - start_time > WALL_LIMIT:
                    raise TimeoutError('q1_candidate_wall_budget')
                requested = np.asarray(entry['applied_command_si'])
                applied = runtime.projector.project(runtime.bridge.state['mechanism_qpos'], requested)
                if not np.array_equal(requested, applied):
                    report['command_changes'].append(dict(frame=frame, reason=runtime.projector.reason))
                row = runtime.step(applied)
                if frame in (1081, 1200, 1400, 1490, 1500, 1510, 1520, 1530):
                    q = runtime.fixture.state.particle_q.numpy()
                    pose = assembly.assemble(np.asarray(runtime.bridge.state['mechanism_qpos']),
                                             include_meshes=False)[1]['bucket']
                    local = (q[ids] - pose[:3, 3]) @ pose[:3, :3]
                    _, counts = np.unique(np.floor(local / .02).astype('int16'), axis=0,
                                          return_counts=True)
                    report['rows'].append(dict(frame=frame, occupied_2cm_cells=len(counts),
                        maximum_particles_per_cell=int(counts.max()),
                        negative_side_2cm_count=int(np.sum(local[:, 1] < y_limits[0] + .02)),
                        source_particles_still_in_cavity=int(np.sum(cavity_mask(local, polygon, y_limits, front))),
                        max_abs_particle_difference_from_P0_m=float(np.max(np.abs(q - prior[frame - 180]))),
                        hard_guard=bool(row['hard_guard']),
                        material_guard_safe=bool(row['hencky_material_guard']['safe']),
                        native_projection_corrected=row['postsolve_projection']['corrected_particle_count'],
                        sweep_safe=bool(row['moving_bucket_sweep']['safe'])))
                    print('Q1', frame, report['rows'][-1]['occupied_2cm_cells'],
                          report['rows'][-1]['negative_side_2cm_count'], flush=True)
            report['status'] = 'bounded_q1_candidate_passed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['frames_completed'] = runtime.frames if runtime is not None else 0
        report['wall_seconds'] = time.monotonic() - start_time
        OUT.mkdir(exist_ok=True)
        (OUT / 'q1-candidate-report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
