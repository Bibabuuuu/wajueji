"""One bounded finite-radius rigid-contact candidate on saved scoop input.

Use the existing particle radius as the solid contact SDF thickness for rocks,
ground and bucket, preserving native projection and all reaction/guard paths.
This changes only the isolated process and is not a production profile.
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
OUT = WORK / 'finite-contact-1080-1530'
START, END, WALL_LIMIT = 1080, 1530, 180
PHYSICAL_FIELDS = ('particle_q', 'particle_qd', 'particle_f', 'particle_elastic_strain',
                   'particle_Jp', 'particle_stress', 'particle_qd_grad', 'particle_transform')


def apply_finite_contact(runtime):
    fixture = runtime.fixture
    solver = fixture.solver
    mpm = solver._mpm_model
    collider = mpm.collider
    before = {name: getattr(fixture.state.mpm, name).numpy().copy()
              if hasattr(fixture.state.mpm, name) else getattr(fixture.state, name).numpy().copy()
              for name in PHYSICAL_FIELDS}
    radius = float(mpm.particle_radius.numpy()[0])
    old_thickness = np.unique(collider.material_thickness.numpy()).tolist()
    old_max_thickness = np.unique(collider.collider_max_thickness.numpy()).tolist()
    old_projection_threshold = collider.material_projection_threshold.numpy().copy()
    old_particle_volume = mpm.particle_volume.numpy().copy()
    old_particle_density = mpm.particle_density.numpy().copy()
    collider.material_thickness.fill_(radius)
    collider.collider_max_thickness.fill_(radius)
    # The old nodal contact warmstart was built from point-collider SDFs.
    # Keep physical arrays and explicitly rebuild only numerical history.
    solver.reset(fixture.state, flags=0)
    unchanged = {name: bool(np.array_equal(
        before[name], getattr(fixture.state.mpm, name).numpy()
        if hasattr(fixture.state.mpm, name) else getattr(fixture.state, name).numpy()))
        for name in PHYSICAL_FIELDS}
    if (not all(unchanged.values())
            or not np.allclose(collider.material_thickness.numpy(), radius, atol=1e-8, rtol=0)
            or not np.allclose(collider.collider_max_thickness.numpy(), radius, atol=1e-8, rtol=0)
            or not np.array_equal(collider.material_projection_threshold.numpy(), old_projection_threshold)
            or not np.array_equal(mpm.particle_volume.numpy(), old_particle_volume)
            or not np.array_equal(mpm.particle_density.numpy(), old_particle_density)
            or abs(float(mpm.particle_radius.numpy()[0]) - radius) > 1e-8):
        raise ValueError('finite_contact_zero_step_contract')
    return dict(physical_arrays_unchanged=unchanged,
                particle_radius_m=radius, new_contact_thickness_m=radius,
                old_material_thickness_m=old_thickness,
                old_collider_max_thickness_m=old_max_thickness,
                projection_threshold_unchanged=True,
                particle_mass_volume_density_unchanged=True,
                solver_warmstart='cleared_after_contact_geometry_change',
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
    report = dict(status='initializing', candidate='finite_radius_rigid_contact',
                  source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  input_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                  start_frame=START, end_frame_limit=END, wall_limit_seconds=WALL_LIMIT,
                  old_default_unchanged=True, training_eligible=False,
                  rows=[], command_changes=[])
    runtime = None
    try:
        with CycleRuntime(SOURCE, OUT, enable_truck=True, freeze_passive_jp=True,
                          bucket_motion_consistency=True) as runtime:
            report['zero_step'] = apply_finite_contact(runtime)
            q0 = runtime.fixture.state.particle_q.numpy()
            pose = assembly.assemble(np.asarray(runtime.bridge.state['mechanism_qpos']),
                                     include_meshes=False)[1]['bucket']
            local = (q0 - pose[:3, 3]) @ pose[:3, :3]
            ids = np.flatnonzero(cavity_mask(local, polygon, y_limits, front))
            if len(ids) != 12086:
                raise ValueError('finite_contact_source_identity')
            report['tracked_particle_count'] = len(ids)
            report['status'] = 'running'
            for entry in entries:
                frame = START + runtime.frames + 1
                if time.monotonic() - started > WALL_LIMIT:
                    raise TimeoutError('finite_contact_wall_budget')
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
                    print('FINITE_CONTACT', frame, report['rows'][-1]['occupied_2cm_cells'],
                          report['rows'][-1]['negative_side_2cm_count'], flush=True)
            report['status'] = 'bounded_finite_contact_candidate_passed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['frames_completed'] = runtime.frames if runtime is not None else 0
        report['wall_seconds'] = time.monotonic() - started
        OUT.mkdir(exist_ok=True)
        (OUT / 'finite-contact-report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
