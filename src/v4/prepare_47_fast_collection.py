"""Prepare an explicitly discontinuous training prepoint, without transport."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from layered_47_training_reset import TrainingResetRuntime, MODE
from layered_47_cycle_geometry import bucket_mask
from layered_47_cycle_checkpoint import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    original, _ = load(args.source, allow_training_reset=True)
    runtime = TrainingResetRuntime(args.source, args.output)
    out = Path(args.output)
    report = dict(schema=MODE, status='planned', source=runtime.metadata['source_identity'],
                  plan={k:v for k,v in runtime.plan.items() if k != 'qpos'},
                  withdrawal=runtime.withdrawal, observation=runtime.observation,
                  continuous_physics=False, training_eligible=False)
    soil = np.ones(len(original['particle_q']), bool)
    soil[runtime.withdrawal['soil_indices']] = False
    rocks = np.ones(len(original['rigid_qpos']), bool)
    rocks[runtime.withdrawal['rock_indices']] = False
    checks = {}
    for name, before in original.items():
        if name.startswith('particle_'):
            checks[name] = bool(np.array_equal(before[soil], runtime.arrays[name][soil]))
        elif name.startswith('rigid_'):
            checks[name] = bool(np.array_equal(before[rocks], runtime.arrays[name][rocks]))
    report['unwithdrawn_state_exact'] = checks
    if not all(checks.values()):
        raise ValueError('training_reset_changed_unwithdrawn_state')
    pose = runtime.assembly.assemble(runtime.arrays['mechanism_qpos'], include_meshes=False)[1]['bucket']
    if bucket_mask(runtime.arrays['particle_q'], pose).any():
        raise ValueError('training_initial_bucket_not_empty')
    started = time.monotonic()
    try:
        if args.execute:
            with runtime as r:
                report['initialization'] = r.restored
                initial = r.capture()
                report['initial_state_readback_exact'] = {
                    name:bool(np.array_equal(r.arrays[name], initial[name]))
                    for name in r.arrays if name.startswith(('particle_', 'rigid_'))}
                if not all(report['initial_state_readback_exact'].values()):
                    raise ValueError('training_reset_native_or_material_readback_inexact')
                if not r.restored['reset_state_particle_readback_exact']:
                    raise ValueError('training_reset_particle_readback_inexact')
                for _ in range(30):
                    if time.monotonic()-started > 180:
                        raise RuntimeError('training_reset_hold_budget')
                    row = r.step([0]*4)
                    if row['bucket_mpm_contact_samples'] or row['rock_bucket_contact_substeps']:
                        from scipy.spatial import cKDTree
                        impulses, positions, ids = r.fixture.solver.collect_collider_impulses(r.fixture.state)
                        impulses, positions, ids = impulses.numpy(), positions.numpy(), ids.numpy()
                        mask = np.isin(ids,r.fixture.bucket_collider_ids) & (np.linalg.norm(impulses,axis=1)>1e-12)
                        distance, index = cKDTree(r.fixture.state.particle_q.numpy()).query(positions[mask])
                        report['unexpected_contact'] = dict(positions=positions[mask].tolist(),
                            impulses=impulses[mask].tolist(),nearest_distance=distance.tolist(),nearest_ids=index.tolist())
                        raise ValueError('training_prepoint_bucket_contact')
                report.update(status='hold_passed', frames=r.frames,
                              checkpoint=r.save(out/'prepoint.npz', stage='prepoint'))
    except BaseException as error:
        report.update(status='failed', failure=repr(error), frames=runtime.frames)
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-started
        (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('withdrawal','initialization')}, indent=2))


if __name__ == '__main__':
    main()
