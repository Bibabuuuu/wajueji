"""Read-only inventory of historical high-fill trajectories for D152 reuse."""

import hashlib
import json
from pathlib import Path


ROOT = Path('/home/qunwang/wajueji')
OLD = ROOT / 'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275'
CORRECTED = ROOT / 'work/bucket-projection-isolation-20260925-v1'
THIRD = ROOT / 'runs/contact-derived-demonstration-v1/third-scoop-observe-v2'
CURRENT = ROOT / 'runs/roadshow-measurements-20261011/step-keyboard-d152-20261012-214337-390485'
OUT = ROOT / 'runs/roadshow-measurements-20261011/historical-high-fill-reuse-audit-20261012-v1/review.json'
D152_SHA = '700f76be970c5754bde905837faf642cca0317fef097e31d17cf48f008ba0545'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    if OUT.exists():
        raise ValueError('immutable_review_already_exists')
    old_report = read(OLD / 'report.json')
    old_input = [json.loads(line) for line in (OLD / 'input.ndjson').read_text().splitlines()]
    prefix = read(CORRECTED / 'manual-prefix-continuation-1620/report.json')
    tail = read(CORRECTED / 'manual-full-tail-1800-8122/report.json')
    third = read(THIRD / 'report.json')
    current_binding = read(CURRENT / 'binding.json')
    source_sha = old_report['handoff_checkpoint']['sha256']
    input_sha = sha(OLD / 'input.ndjson')
    measurements = {m['frame']: m for m in prefix['measurements'] + tail['measurements']}
    first_corrected = json.loads((CORRECTED / 'manual-prefix-continuation-1620/frames.ndjson').open().readline())
    checks = {
        'historical_and_d152_initial_checkpoint_same': source_sha == current_binding['source_sha256'] == D152_SHA,
        'old_input_complete_and_not_admitted': len(old_input) == 8122
            and old_input[0]['frame'] == 1 and old_input[-1]['frame'] == 8122
            and all(row['training_eligible'] is False for row in old_input),
        'corrected_replay_uses_saved_input': prefix['input_sha256'] == tail['input_sha256'] == input_sha,
        'corrected_replay_complete': prefix['frames'] == 1620
            and tail['frames_completed'] == 6322 and tail['last_global_frame'] == 8122,
        'high_values_are_display_estimates': all(
            measurements[frame]['valid_estimate']
            and measurements[frame]['display_only']
            and measurements[frame]['calibration_status'] == 'user_selected_threshold_not_recalibrated'
            for frame in (1080, 1680, 4200, 4860)),
        'corrected_frame_record_lacks_training_observation':
            'after_state' in first_corrected
            and 'before_state' not in first_corrected
            and 'terrain_anchor_frame' not in first_corrected
            and 'geometry_same_state_intent' not in first_corrected,
        'high_third_is_other_source_and_held_out':
            third['source']['sha256'] != D152_SHA
            and third['split'] == 'validation_runs'
            and third['training_eligible'] is False,
    }
    result = {
        'schema': 'roadshow_historical_high_fill_reuse_review_v1',
        'status': 'historical_actions_candidate_only' if all(checks.values()) else 'audit_failed',
        'checks': checks,
        'first_choice': {
            'kind': 'old human first scoop under corrected diagnostic replay',
            'initial_checkpoint_sha256': source_sha,
            'input_rows': len(old_input),
            'input_sha256': input_sha,
            'display_occupancy_percent': {str(frame): measurements[frame]['fill_percent']
                for frame in (1080, 1680)},
            'measurement_scope': 'Valid at the recorded frame under the chosen occupancy threshold, including connected heap and rocks; display-only, not calibrated stable truck volume.',
            'control_and_physics_scope': 'Saved gamepad commands replayed under optional motion interface corrections; 150 saved commands were reprojected in the 8122-frame replay. Current D152 adaptive feedback and rollout differ.',
            'training_eligible': False,
        },
        'other_evidence': {
            'corrected_second_scoop_display_occupancy_percent':
                {str(frame): measurements[frame]['fill_percent'] for frame in (4200, 4860)},
            'third_derived_contact_rows': third['teacher_tensor']['rows'],
            'third_derived_source_sha256': third['source']['sha256'],
            'third_derived_split': third['split'],
            'third_derived_training_eligible': third['training_eligible'],
        },
        'next_gate': [
            'Reconstruct causal observations and same-state geometry intent for the selected first-scoop actions, then review expert intent versus protected executed action.',
            'Keep independent lineage for evaluation; do not train on the held-out third-scoop line without resplitting and a new untouched holdout.',
            'Only an independently controlled same-initial-state D152 rollout under the same physics and guards can support BC-versus-geometry claims.',
        ],
        'sha256': {str(path): sha(path) for path in (
            OLD / 'report.json', OLD / 'input.ndjson',
            CORRECTED / 'manual-prefix-continuation-1620/report.json',
            CORRECTED / 'manual-full-tail-1800-8122/report.json',
            THIRD / 'report.json', CURRENT / 'binding.json')},
    }
    OUT.parent.mkdir(exist_ok=False)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                              allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'checks': checks,
                      'review': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
