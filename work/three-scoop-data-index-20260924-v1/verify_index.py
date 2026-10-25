"""Independently verify emitted references, filtering, timing, and sequence gaps."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
manifest = json.loads((OUT / 'report.json').read_text())
checks = {}
total = 0
for episode in manifest['episodes']:
    run = Path('/home/qunwang/wajueji/runs/layered-47-sparse-manual-contact-v1') / episode['run']
    def lines(path):
        return [json.loads(s) for s in path.read_text().splitlines()]
    rows = lines(Path(episode['index_path']))
    actions = lines(run / 'input.ndjson')
    physics = lines(run / 'physics.ndjson')
    observations = {o['frame']: o for o in lines(run / 'observation.ndjson')}
    original_report = json.loads((run / 'report.json').read_text())
    assert original_report['initial_observation']['terrain'] is not None
    assert hashlib.sha256(Path(episode['index_path']).read_bytes()).hexdigest() == episode['index_sha256']
    assert len(rows) == episode['candidate_frames']
    expected = [a['frame'] for a in actions if a['input_device'] in ('gamepad', 'keyboard') and a['armed'] and a['connected'] and a['window_focused']]
    assert [r['frame'] for r in rows] == expected
    previous = None
    sequence = 0
    for row in rows:
        frame = row['frame']
        action = actions[frame - 1]
        reset = previous is None or frame != previous + 1
        sequence += int(reset)
        assert row['reset_learning_history'] == reset and row['sequence_id'] == sequence
        assert row['split_group'] == manifest['split_group']
        assert not row['phase_feature_allowed'] and not row['training_eligible']
        assert row['target_applied_action_si'] == action['applied_command_si'] == physics[frame - 1]['applied_command_si']
        assert row['previous_action_si'] == (actions[frame - 2]['applied_command_si'] if frame > 1 else [0., 0., 0., 0.])
        anchor = row['terrain_anchor_frame']
        assert anchor == max(f for f in [0, *observations] if f < frame)
        assert 0 <= row['terrain_age_frames'] == frame - 1 - anchor <= 2
        assert row['proprioception_reference'] == f'physics.ndjson:frame={frame}:before_state'
        assert row['load_history_reference'] == (None if frame == 1 else f'physics.ndjson:frames=1..{frame-1}')
        previous = frame
    assert sequence == episode['sequence_count']
    phases = json.loads(Path(episode['phase_path']).read_text())
    assert phases['original_operator_events'] == original_report['phase_events']
    covered = [f for p in phases['intervals'] for f in range(p['start_frame'], p['end_frame'] + 1)]
    assert covered == list(range(1, len(actions) + 1))
    assert phases['posthoc_only'] and phases['not_a_runtime_router'] and phases['manual_review_required']
    for filename, digest in episode['source_bindings'].items():
        assert hashlib.sha256((run / filename).read_bytes()).hexdigest() == digest
    checks[episode['run']] = dict(passed=True, candidate_frames=len(rows), sequences=sequence)
    total += len(rows)
assert total == manifest['candidate_frames'] == 4126
result = dict(passed=True, total_candidate_frames=total, episodes=checks,
              scope='reference audit only; not training qualification or phase ground truth')
(OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
