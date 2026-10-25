"""Read-only causal and guard-label audit of the historical first 1800 frames."""

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path('/home/qunwang/wajueji')
RUN = ROOT / 'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275'
OUT = ROOT / 'runs/roadshow-measurements-20261011/historical-first-scoop-label-audit-20261012-v1/review.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    if OUT.exists():
        raise ValueError('immutable_review_already_exists')
    actions = rows(RUN / 'input.ndjson')
    physics = rows(RUN / 'physics.ndjson')
    observations = rows(RUN / 'observation.ndjson')
    first_actions = actions[:1800]
    first_observations = [o for o in observations if o['frame'] < 1800]
    projection = Counter(a['command_projection_reason'] for a in first_actions)
    anchor_projection = Counter(actions[o['frame']]['command_projection_reason']
                                for o in first_observations)
    checks = {
        'full_input_physics_counts': len(actions) == len(physics) == 8122,
        'first_1800_observation_count': len(first_observations) == 599,
        'observations_are_actual_post_action_states': all(
            o['state'] == physics[o['frame'] - 1]['after_state']
            for o in first_observations),
        'next_action_is_causal_frame': all(
            actions[o['frame']]['frame'] == o['frame'] + 1
            for o in first_observations),
        'all_phases_unmarked': all(a['phase'] == 'unmarked' for a in first_actions)
            and all(o['phase'] == 'unmarked' for o in first_observations),
        'all_training_ineligible': all(a['training_eligible'] is False
                                       for a in first_actions)
            and all(o['training_eligible'] is False for o in first_observations),
        'projection_counts': projection[None] == 786
            and projection['bucket_internal_upper_target'] == 1014,
    }
    result = {
        'schema': 'roadshow_historical_first_scoop_label_review_v1',
        'status': 'causal_anchors_available_labels_not_admitted'
            if all(checks.values()) else 'audit_failed',
        'checks': checks,
        'frames_considered': 1800,
        'causal_observation_anchors': len(first_observations),
        'anchor_frame_range': [first_observations[0]['frame'],
                               first_observations[-1]['frame']],
        'first_1800_projection_reasons': {str(k): v for k, v in projection.items()},
        'next_action_at_anchor_projection_reasons':
            {str(k): v for k, v in anchor_projection.items()},
        'qualification_boundary': 'Post-action observations can be paired with the next action, but phases are unmarked and 1014/1800 human requests were projected at the bucket upper limit. Reconstruct phase and same-state geometry intent, then review or exclude projected labels before fitting a residual.',
        'training_eligible': False,
        'sha256': {str(path): sha(path) for path in (
            RUN / 'report.json', RUN / 'input.ndjson',
            RUN / 'physics.ndjson', RUN / 'observation.ndjson')},
    }
    OUT.parent.mkdir(exist_ok=False)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                              allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'checks': checks,
                      'review': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
