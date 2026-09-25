"""One bounded radius/volume consistency candidate on saved first-scoop input.

The repository's material particle mass uses spacing**3 while Newton uses
(2*radius)**3 for material volume. This opt-in process restores the source,
sets radius to spacing/2, recomputes Newton material properties, and clears
only numerical warmstarts. The formal/default fixture is unchanged.
"""
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


WORK = Path(__file__).resolve().parent
SOURCE = WORK / 'manual-prefix-continuation-1620/state-001080.npz'
ORIGINAL = REPO / 'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275/input.ndjson'
OUT = WORK / 'radius-volume-1080-1530'
START, END, WALL_LIMIT = 1080, 1530, 180
PHYSICAL_FIELDS = ('particle_q', 'particle_qd', 'particle_f', 'particle_elastic_strain',
                   'particle_Jp', 'particle_stress', 'particle_qd_grad', 'particle_transform')


def correct_radius(runtime):
    fixture = runtime.fixture
    model = fixture.model
    mpm = fixture.solver._mpm_model
    spacing = float(runtime.base['mpm']['particle_spacing_m'])
    intended_density = float(runtime.base['mpm']['particle_density_kg_m3'])
    before = {name: getattr(fixture.state.mpm, name).numpy().copy()
              if hasattr(fixture.state.mpm, name) else getattr(fixture.state, name).numpy().copy()
              for name in PHYSICAL_FIELDS}
    before_radius = float(mpm.particle_radius.numpy()[0])
    before_volume = float(mpm.particle_volume.numpy()[0])
    before_density = float(mpm.particle_density.numpy()[0])
    model.particle_radius.fill_(spacing / 2.0)
    fixture.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    # Preserve all particle/material arrays and clear only solver history tied
    # to the old support volume. This is an explicit numerical restart.
    fixture.solver.reset(fixture.state, flags=0)
    after_radius = float(mpm.particle_radius.numpy()[0])
    after_volume = float(mpm.particle_volume.numpy()[0])
    after_density = float(mpm.particle_density.numpy()[0])
    unchanged = {name: bool(np.array_equal(
        before[name], getattr(fixture.state.mpm, name).numpy()
        if hasattr(fixture.state.mpm, name) else getattr(fixture.state, name).numpy()))
        for name in PHYSICAL_FIELDS}
    if (not all(unchanged.values()) or abs(after_radius - spacing / 2) > 1e-8
            or abs(after_volume - spacing**3) > 1e-10
            or abs(after_density - intended_density) > 1e-3):
        raise ValueError('radius_volume_zero_step_contract')
    return dict(physical_arrays_unchanged=unchanged,
                old_radius_m=before_radius, new_radius_m=after_radius,
                old_solver_volume_m3=before_volume, new_solver_volume_m3=after_volume,
                old_effective_density_kg_m3=before_density,
                new_effective_density_kg_m3=after_density,
                intended_density_kg_m3=intended_density,
                solver_warmstart='cleared_after_radius_change',
                exact_old_solver_continuation=False)


def main():
    started = time.monotonic()
    entries = [json.loads(line) for line in ORIGINAL.read_text().splitlines()][START:END]
    assert len(entries) == END - START
    assembly, _ = _load_kinematics(str(KINEMATICS), str(KINEMATICS / 'scene/input/model/A.xml'),
                                   str(KINEMATICS / 'scene/input/mesh'))
    contract = json.loads(Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json').read_text())
    front = np.load('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/front-rim.npz')['front']
    polygon = np.asarray(contract['bucket']['polygon_xz_m'])
    y_limits = np.asarray(contract['bucket']['y_limits_m'])
    prior = np.load(WORK / 'manual-prefix-continuation-1620/display/particles.npy', mmap_mode='r')
    report = dict(status='initializing', candidate='radius_half_spacing_volume_match',
                  source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  input_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                  start_frame=START, end_frame_limit=END, wall_limit_seconds=WALL_LIMIT,
                  old_default_unchanged=True, training_eligible=False,
                  rows=[], command_changes=[])
    runtime = None
    try:
        with CycleRuntime(SOURCE, OUT, enable_truck=True, freeze_passive_jp=True,
                          bucket_motion_consistency=True) as runtime:
            report['zero_step'] = correct_radius(runtime)
            q0 = runtime.fixture.state.particle_q.numpy()
            pose = assembly.assemble(np.asarray(runtime.bridge.state['mechanism_qpos']),
                                     include_meshes=False)[1]['bucket']
            local = (q0 - pose[:3, 3]) @ pose[:3, :3]
            ids = np.flatnonzero(cavity_mask(local, polygon, y_limits, front))
            if len(ids) != 12086:
                raise ValueError('radius_candidate_source_identity')
            report['tracked_particle_count'] = len(ids)
            report['status'] = 'running'
            for entry in entries:
                frame = START + runtime.frames + 1
                if time.monotonic() - started > WALL_LIMIT:
                    raise TimeoutError('radius_candidate_wall_budget')
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
                        sweep_safe=bool(row['moving_bucket_sweep']['safe'])))
                    print('RADIUS', frame, report['rows'][-1]['occupied_2cm_cells'],
                          report['rows'][-1]['negative_side_2cm_count'], flush=True)
            report['status'] = 'bounded_radius_candidate_passed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['frames_completed'] = runtime.frames if runtime is not None else 0
        report['wall_seconds'] = time.monotonic() - started
        OUT.mkdir(exist_ok=True)
        (OUT / 'radius-candidate-report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
