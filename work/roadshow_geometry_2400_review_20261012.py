"""Read-only audit and compact control export for the bounded geometry run."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import numpy as np

ROOT = Path('/home/qunwang/wajueji')
RUN = ROOT / 'runs/roadshow-measurements-20261011/geometry-independent-20261012-v4'
OLD = ROOT / 'runs/roadshow-measurements-20261011/paired-driver580178-20261012/geometry'
OUT = RUN / 'offline-analysis'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    assert not OUT.exists(), 'immutable_analysis_already_exists'
    report = json.loads((RUN / 'physics/report.json').read_text())
    binding = json.loads((RUN / 'metrics/comparison-binding.json').read_text())
    environment = json.loads((RUN / 'metrics/environment.json').read_text())
    actions = read_lines(RUN / 'physics/actions.ndjson')
    phases = read_lines(RUN / 'physics/phase-trace.ndjson')
    old = read_lines(OLD / 'physics/actions.ndjson')
    display = json.loads((RUN / 'physics/display/reconstruction.json').read_text())
    display_states = json.loads((RUN / 'physics/display/states.json').read_text())
    assert report['status'] == 'bounded_probe_completed' and report['frames_completed'] == 2400
    assert report['source_unchanged'] and report['all_executed_frames_safe']
    assert not report['policy_used_for_control'] and not binding['learned_weights_used']
    assert report['worker_ready']['neural_inference'] is False
    assert len(report['zero_step_arrays_exact']) == 24 and all(report['zero_step_arrays_exact'].values())
    assert all(report['same_instance_gates'][str(frame)]['passed'] for frame in (120, 1200))
    assert [row['frame'] for row in actions] == list(range(1, 2401))
    assert len(phases) == 2400 and len(display_states) == 2401 and display['frames'] == 2400
    assert display['status'] == 'captured_probe' and display['label'] == 'GEOMETRY CONTACT WINDOW REPLAY'
    assert len(old) == 900 and actions[0]['before_state'] == old[0]['before_state']
    assert all(row['phase'] == prior['phase'] for row, prior in zip(actions[:900], old))
    assert all(row['source'] == 'geometric_feedback_without_policy' for row in actions)
    assert all(np.isfinite(row['applied_command_si']).all() for row in actions)
    assert all(row['latest_allowed_phase_evidence_frame'] == row['frame'] - 1 for row in actions)
    assert all(row['before_state'] == display_states[row['frame'] - 1] for row in actions)
    assert np.allclose(display_states[-1]['bucket_origin_world_m'], report['final_bucket_origin_world_m'], atol=0, rtol=0)
    assert sha(RUN / 'physics/final.npz') == report['final_checkpoint']['sha256']
    with np.load(RUN / 'physics/final.npz') as archive:
        assert len(archive.files) == 24 and all(np.isfinite(archive[name]).all() for name in archive.files if np.issubdtype(archive[name].dtype, np.number))

    OUT.mkdir()
    with (OUT / 'control.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        axes = ('swing', 'boom', 'arm', 'bucket')
        writer.writerow(['frame', 'state_frame', 'sim_elapsed_s', 'phase', 'control_mode', 'closure_fraction',
                         'preview_scale', *[f'{axis}_{field}' for axis in axes for field in
                                            ('position', 'controller_target', 'requested_velocity', 'applied_velocity')]])
        start = actions[0]['before_state']['time_s']
        for action in actions:
            state = action['before_state']
            elapsed = state['time_s'] - start
            assert abs(elapsed - (action['frame'] - 1) / 60) < 1e-8
            writer.writerow([action['frame'], action['frame'] - 1, elapsed, action['phase'],
                             action['feedback']['mode'], action['feedback']['closure_fraction'],
                             action['feedback']['preview_scale'],
                             *[value for j, index in enumerate((7, 9, 12, 15)) for value in
                               (state['mechanism_qpos'][index], state['controller_target'][j],
                                action['requested_command_si'][j], action['applied_command_si'][j])]])

    old_applied = np.asarray([row['applied_command_si'] for row in old])
    new_applied = np.asarray([row['applied_command_si'] for row in actions[:900]])
    delta = np.abs(new_applied - old_applied)
    lift = next((row for row in actions if row['feedback']['lift_intent']), None)
    sources = [RUN / 'physics/report.json', RUN / 'physics/actions.ndjson', RUN / 'physics/phase-trace.ndjson',
               RUN / 'physics/final.npz', RUN / 'physics/gate-1200.json', RUN / 'physics/display/reconstruction.json',
               RUN / 'metrics/comparison-binding.json', RUN / 'metrics/environment.json',
               RUN / 'metrics/latency.csv', RUN / 'metrics/telemetry.csv']
    review = dict(status='passed', scope='one independent full-initial-state geometry run; 2400-frame bound',
                  frames=2400, wall_seconds=report['wall_seconds'], run=str(RUN),
                  controller='geometry_feedback_without_neural_weights',
                  initial_state_sha256=report['source']['sha256'], code_binding=binding,
                  environment=environment, phase_counts=dict(collections.Counter(row['phase'] for row in actions)),
                  control_modes=dict(collections.Counter(row['feedback']['mode'] for row in actions)),
                  phase_events=report['events'], final_phase=report['final_phase'],
                  closure_fraction_last_before_state=actions[-1]['feedback']['closure_fraction'],
                  first_lift_intent_frame=None if lift is None else lift['frame'],
                  origin_rise_since_lift_intent_m=None if lift is None else
                      report['final_bucket_origin_world_m'][2] - lift['before_state']['bucket_origin_world_m'][2],
                  gate120_passed=True, gate1200_passed=True, all_executed_frames_safe=True,
                  first_900_phase_equal=True, first_900_mean_abs_applied_velocity_difference_si=delta.mean(axis=0).tolist(),
                  first_900_max_abs_applied_velocity_difference_si=delta.max(axis=0).tolist(),
                  exact_900_replay=False, full_contact_complete=False, formal_volume_valid=False,
                  policy_resume_qualified=False, new_physics_steps=0,
                  display_cache_frames=display['frames'], display_cache_hashes=display['cache_sha256'],
                  bindings=[dict(path=str(path), sha256=sha(path)) for path in sources],
                  control_csv_sha256=sha(OUT / 'control.csv'))
    (OUT / 'review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: review[key] for key in ('status', 'frames', 'final_phase', 'closure_fraction_last_before_state',
                   'first_lift_intent_frame', 'origin_rise_since_lift_intent_m', 'full_contact_complete')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
