"""Read-only onset diagnostics; no phase labels or physics are changed.

Run with PYTHONPATH=src/v4:. in the existing physics Python environment.
Thresholds are a sensitivity grid, not accepted production parameters.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from layered_excavation_approach_planner import _load_kinematics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ASSETS = Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
RUNS = ('010900-1088617', '113555-1443279', '114908-1452582', '115812-1458627')
THRESHOLDS = (0.001, 1., 10., 25., 50., 100., 500.)
DWELLS = (1, 3, 6)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readrows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def first_window(values, threshold, dwell):
    """Return onset and confirmation separately; streaming confirmation is causal."""
    count = 0
    for i, value in enumerate(values):
        count = count + 1 if value > threshold else 0
        if count == dwell:
            return dict(onset_frame_end=i - dwell + 2,
                        confirmation_frame_end=i + 1,
                        first_available_decision_frame=i + 2)
    return None


def main():
    assembly, _ = _load_kinematics(
        ASSETS, ASSETS/'scene/input/model/A.xml', ASSETS/'scene/input/mesh')
    episodes = []
    inputs = []
    for suffix in RUNS:
        run = ROOT/'runs/layered-47-sparse-manual-contact-v1'/('manual-contact-20260924-' + suffix)
        paths = [run/name for name in ('physics.ndjson', 'observation.ndjson', 'annotations.ndjson')]
        sources = [dict(path=str(path), sha256=sha(path)) for path in paths]
        inputs.extend(sources)
        rows = readrows(paths[0])
        observations = readrows(paths[1])
        assert [r['frame'] for r in rows] == list(range(1, len(rows) + 1))
        force = np.array([r['bucket_mpm_force_norm_n'] for r in rows])
        assert np.isfinite(force).all() and (force >= 0).all()
        vectors = np.array([r['bucket_mpm_force_world_n'] for r in rows])
        assert np.allclose(np.linalg.norm(vectors, axis=1), force, rtol=1.e-12, atol=1.e-9)
        prefix = []
        for row in rows[:30]:
            if any(abs(x) > 1.e-12 for x in row['applied_command_si']):
                break
            prefix.append(row)
        baseline = dict(zero_command_prefix_frames=len(prefix), prefix_cap_frames=30,
                        maximum_force_n=max((r['bucket_mpm_force_norm_n'] for r in prefix), default=None),
                        qualified_no_contact_baseline=False,
                        note='Zero command alone does not establish absence of contact.')
        trials = []
        for threshold in THRESHOLDS:
            for dwell in DWELLS:
                hit = first_window(force, threshold, dwell)
                trials.append(dict(threshold_n=threshold, dwell_frames=dwell, event=hit))
        band = [t['event']['onset_frame_end'] for t in trials
                if 10 <= t['threshold_n'] <= 100 and t['dwell_frames'] in (3, 6) and t['event']]
        events = {}
        for name, test in (
            ('raw_soil_sample', lambda r: r['bucket_mpm_contact_samples'] > 0),
            ('rock_bucket', lambda r: r['rock_bucket_contact_substeps'] > 0),
            ('bucket_motion_sweep', lambda r: r['moving_bucket_sweep']['actual_motion_pending_next_frame']['hits'] > 0),
            ('mpm_motion_sweep', lambda r: r['moving_bucket_sweep']['mpm_correction']['hits'] > 0),
        ):
            row = next((r for r in rows if test(r)), None)
            events[name] = None if row is None else dict(
                frame_end=row['frame'], force_n=row['bucket_mpm_force_norm_n'],
                available_from_decision_frame=row['frame'] + 1)
        # Sample recorded full-width CAD vertices against the latest same/past
        # observation at observation frames. This vertical offset is NOT an SDF
        # or an exact particle contact test. Fused grid includes both soil/rocks.
        geometries, _ = assembly.assemble(np.asarray(rows[0]['before_state']['mechanism_qpos']))
        vertices = np.concatenate([np.asarray(g['body_local_vertices']) for g in geometries if g['body'] == 'bucket'])
        cutoff = max(band + [events['bucket_motion_sweep']['frame_end'] if events['bucket_motion_sweep'] else 0]) + 30
        geometry = []
        for obs in observations:
            frame = obs['frame']
            if not 0 < frame <= min(cutoff, len(rows)) or 'height_m' not in obs['terrain']:
                continue
            terrain = obs['terrain']
            assert obs['state'] == rows[frame-1]['after_state']
            assert terrain['resolution_m'] == 0.1
            grid = np.asarray(terrain['height_m'], dtype=float)
            pose = assembly.assemble(np.asarray(rows[frame-1]['after_state']['mechanism_qpos']), include_meshes=False)[1]['bucket']
            world = vertices @ pose[:3, :3].T + pose[:3, 3]
            cell = np.floor((world[:, :2] - terrain['origin_xy_m']) / terrain['resolution_m']).astype(int)
            inside = (cell[:, 0] >= 0) & (cell[:, 0] < grid.shape[1]) & (cell[:, 1] >= 0) & (cell[:, 1] < grid.shape[0])
            z = grid[cell[inside, 1], cell[inside, 0]]
            known = np.isfinite(z)
            offset = world[inside, 2][known] - z[known]
            geometry.append(dict(frame_end=frame, observation_frame=frame,
                                 first_available_decision_frame=frame+1,
                                 minimum_vertex_vertical_offset_m=float(offset.min()) if offset.size else None,
                                 known_vertex_count=int(known.sum()),
                                 unknown_or_outside_vertex_count=int(len(vertices)-known.sum()),
                                 force_n=float(force[frame-1])))
        first_overlap = next((g for g in geometry if g['minimum_vertex_vertical_offset_m'] is not None
                              and g['minimum_vertex_vertical_offset_m'] <= 0), None)
        episodes.append(dict(run=run.name, source_inputs=sources, baseline=baseline,
                             first_frame_force_n=float(force[0]), first_frame_samples=rows[0]['bucket_mpm_contact_samples'],
                             force_sensitivity=trials,
                             illustrative_10_to_100n_3_to_6frame_onset_range=[min(band), max(band)] if band else None,
                             interval_is_confidence_interval=False,
                             independent_recorded_events=events,
                             first_sampled_fused_grid_overlap=first_overlap,
                             cad_vertex_count=len(vertices), cad_vertices_sha256=hashlib.sha256(vertices.tobytes()).hexdigest(),
                             geometry_samples=geometry,
                             original_phase_events=[dict(label=e['label'], next_frame=e['next_frame'])
                                                    for e in readrows(paths[2]) if e.get('kind') == 'phase'],
                             confirmed_contact_label=None))
        print(run.name, 'sensitivity', episodes[-1]['illustrative_10_to_100n_3_to_6frame_onset_range'],
              'grid overlap', None if first_overlap is None else first_overlap['frame_end'], flush=True)
    assert all(sha(Path(s['path'])) == s['sha256'] for s in inputs)
    report = dict(schema='contact_onset_diagnostic_v1', status='offline_candidates_not_labels',
                  script_sha256=sha(Path(__file__)), physics_advanced=False, production_parameters_changed=False,
                  geometry_model=dict(path=str(ASSETS/'scene/input/model/A.xml'), sha256=sha(ASSETS/'scene/input/model/A.xml')),
                  implementation_inputs=[dict(path=str(ROOT/name), sha256=sha(ROOT/name)) for name in (
                      'src/v4/layered_excavation_approach_planner.py',
                      'src/v4/run_layered_47_excavation_preflight.py',
                      'src/v4/run_layered_47_sparse_manual_contact_episode.py',
                      'src/v4/moving_bucket_sweep_runtime.py')],
                  frequency_hz=60, threshold_grid_n=THRESHOLDS, persistence_frames=DWELLS,
                  geometry_scope='CAD vertex vertical offset against fused sampled 10cm soil/rock height grid; not particle distance, no exact touch certification; unknown cells remain unknown.',
                  force_scope='Logged resultant soil wrench includes routed moving-bucket corrections; net cancellation and absent native projection reactions limit interpretation.',
                  online_scope='Confirm after dwell; onset is retrospective. End-of-frame evidence only usable at following decision.',
                  deployment_qualified=False, episodes=episodes)
    (OUT/'contact-onset-audit-v1.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
