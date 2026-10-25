"""Read-only audit of the second D152 operator pilot; diagnostic only."""

import hashlib
import json
from collections import Counter
from pathlib import Path


RUN = Path('/home/qunwang/wajueji/runs/roadshow-measurements-20261011/step-keyboard-d152-20261012-214337-390485')
PHYSICS = RUN / 'physics-run'
OUT = RUN / 'offline-analysis/review.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    if OUT.exists():
        raise ValueError('immutable_review_already_exists')
    binding = json.loads((RUN / 'binding.json').read_text())
    report = json.loads((PHYSICS / 'report.json').read_text())
    actions = rows(PHYSICS / 'actions.ndjson')
    physical = rows(PHYSICS / 'physics/frames.ndjson')
    timing = rows(PHYSICS / 'frame-timing.ndjson')
    steps = {step_id: [a for a in actions if a['operator_step_id'] == step_id]
             for step_id in (1, 2)}
    checks = {
        'bound_d152_source': binding['source_sha256'] ==
            '700f76be970c5754bde905837faf642cca0317fef097e31d17cf48f008ba0545',
        'step_keyboard_not_neural': binding['manual_input'] == 'step_keyboard'
            and not binding['reference_policy_used']
            and not report['reference_policy_used_for_control'],
        'user_finished_at_454': report['status'] == 'operator_finished_diagnostic'
            and report['early_stop_reason'] == 'user_finished'
            and report['frames_completed'] == 454,
        'frame_alignment': len(actions) == len(physical) == len(timing) == 454
            and all(a['frame'] == p['frame'] == t['frame'] for a, p, t in
                    zip(actions, physical, timing, strict=True)),
        'applied_commands_match_physics': all(
            a['applied_command_si'] == p['applied_command_si'] for a, p in
            zip(actions, physical, strict=True)),
        'all_executed_hard_guards': report['all_executed_frames_safe']
            and all(p['hard_guard'] for p in physical),
        'gate_120_passed': report['same_instance_gates']['120']['passed'],
        'no_projection_rejections': report['projection_frames'] == 0,
        'all_training_ineligible': not binding['training_eligible']
            and all(a['training_eligible'] is False for a in actions),
        'source_unchanged': report['source_unchanged'],
        'two_30_frame_steps': [len(steps[i]) for i in (1, 2)] == [30, 30]
            and [steps[i][0]['frame'] for i in (1, 2)] == [395, 425],
        'first_step_neutral_second_arm_retract': all(
            a['operator_action_normalized_before_gate'] == [0.0] * 4
            for a in steps[1]) and all(
            a['operator_action_normalized_before_gate'] == [0.0, 0.0, 1.0, 0.0]
            for a in steps[2]),
        'finished_only_curl': report['final_phase'] == 'curl_load'
            and not any(a['phase'] in ('lift', 'off_soil') for a in actions),
    }
    result = {
        'schema': 'roadshow_d152_step_keyboard_diagnostic_review_v1',
        'status': 'diagnostic_only' if all(checks.values()) else 'audit_failed',
        'checks': checks,
        'scope': '454-frame partial operator pilot, not expert data or a BC result',
        'source_run': str(RUN),
        'frames_requested': report['frames_requested'],
        'frames_completed': report['frames_completed'],
        'phase_counts': dict(Counter(a['phase'] for a in actions)),
        'steps': [
            {'id': i, 'first_frame': steps[i][0]['frame'],
             'last_frame': steps[i][-1]['frame'],
             'selected_keys': steps[i][0]['raw_controls']['selected_keys'],
             'effective_frames': sum(a['manual_input_effective'] for a in steps[i]),
             'decision_frame_recorded': steps[i][0]['operator_decision_frame']}
            for i in (1, 2)],
        'end_closure_fraction': actions[-1]['feedback']['closure_fraction'],
        'zero_intent_decision_metadata_gap':
            'Step 1 has an ID but its decision frame/time are null because the current logger emits those fields only for a nonzero selected action.',
        'training_eligible': False,
        'sha256': {str(path): sha(path) for path in (
            RUN / 'binding.json', PHYSICS / 'report.json',
            PHYSICS / 'actions.ndjson', PHYSICS / 'physics/frames.ndjson',
            PHYSICS / 'final.npz')},
    }
    OUT.parent.mkdir(exist_ok=False)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                              allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'checks': checks,
                      'review': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
