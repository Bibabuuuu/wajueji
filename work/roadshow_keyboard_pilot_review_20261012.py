"""Read-only audit of the first D152 keyboard pilot; output is diagnostic only."""

import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path('/home/qunwang/wajueji')
RUN = ROOT / 'runs/roadshow-measurements-20261011/keyboard-d152-20261012-211644-371795'
PHYSICS = RUN / 'physics-run'
OUT = RUN / 'offline-analysis'


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
    checks = {
        'bound_d152_source': binding['source_sha256'] ==
            '700f76be970c5754bde905837faf642cca0317fef097e31d17cf48f008ba0545',
        'keyboard_not_neural': binding['manual_input'] == 'keyboard' and
            not binding['reference_policy_used'] and
            not report['reference_policy_used_for_control'],
        'user_finished_at_575': report['early_stop_reason'] == 'user_finished' and
            report['frames_completed'] == 575,
        'frame_alignment': len(actions) == len(physical) == len(timing) == 575 and
            all(a['frame'] == p['frame'] == t['frame'] for a, p, t in
                zip(actions, physical, timing, strict=True)),
        'applied_commands_match_physics': all(
            a['applied_command_si'] == p['applied_command_si'] for a, p in
            zip(actions, physical, strict=True)),
        'all_executed_hard_guards': report['all_executed_frames_safe'] and
            all(p['hard_guard'] for p in physical),
        'gate_120_passed': report['same_instance_gates']['120']['passed'],
        'all_training_ineligible': not binding['training_eligible'] and
            all(a['training_eligible'] is False for a in actions),
        'source_unchanged': report['source_unchanged'],
        'finished_only_curl': report['final_phase'] == 'curl_load' and
            not any(a['phase'] in ('lift', 'off_soil') for a in actions),
    }
    curl = [a for a in actions if a['phase'] == 'curl_load']
    key_rows = {}
    for key in 'JKLMNIOU':
        pressed = [a for a in actions if key in a['raw_controls']['pressed_keys']]
        effective = [a for a in pressed if a['manual_input_effective']]
        key_rows[key] = {'pressed_frames': len(pressed),
            'effective_curl_frames': len(effective)}
    o = [a for a in curl if 'O' in a['raw_controls']['pressed_keys']
        and a['manual_input_effective']]
    u = [a for a in curl if 'U' in a['raw_controls']['pressed_keys']
        and a['manual_input_effective']]
    o_change = (physical[o[-1]['frame'] - 1]['after_state']['mechanism_qpos'][12]
        - actions[o[0]['frame'] - 1]['before_state']['mechanism_qpos'][12])
    u_change = (physical[u[-1]['frame'] - 1]['after_state']['mechanism_qpos'][12]
        - actions[u[0]['frame'] - 1]['before_state']['mechanism_qpos'][12])
    typical = [t['wall_seconds'] for t in timing if t['wall_seconds'] < 2]
    result = {'schema': 'roadshow_d152_keyboard_pilot_diagnostic_review_v1',
        'status': 'diagnostic_only' if all(checks.values()) else 'audit_failed',
        'checks': checks,
        'scope': '575-frame partial keyboard pilot; not expert data or a BC result',
        'source_run': str(RUN),
        'frames': {'requested': report['frames_requested'],
            'completed': report['frames_completed'],
            'phase_counts': {phase: sum(a['phase'] == phase for a in actions)
                for phase in ('approach_contact', 'penetrate', 'curl_load')},
            'manual_effective': sum(a['manual_input_effective'] for a in actions),
            'key_rows': key_rows,
            'curl_exit_closure_fraction': actions[-1]['feedback']['closure_fraction']},
        'arm_response': {
            'U_effective_frames': len(u), 'U_first_last_frame': [u[0]['frame'], u[-1]['frame']],
            'U_actual_cylinder_delta_m': u_change,
            'O_effective_frames': len(o), 'O_first_last_frame': [o[0]['frame'], o[-1]['frame']],
            'O_first_positive_applied_arm_frame': next(
                a['frame'] for a in o if a['applied_command_si'][2] > 0),
            'O_applied_arm_negative_frames': sum(a['applied_command_si'][2] < 0 for a in o),
            'O_actual_cylinder_delta_m': o_change,
            'O_final_applied_arm_m_s': o[-1]['applied_command_si'][2],
            'O_preview_all_full_scale': all(a['feedback']['preview_scale'] == 1 for a in o),
            'O_projector_rejections': sum(a['command_projection_reason'] is not None for a in o),
            'interpretation': 'Earlier U motion and 2/s normalized slew delayed reversal; net O cylinder motion remained small under coupled feedback and physical load. No single causal factor is isolated.'},
        'interaction': {'typical_frame_wall_seconds_median': statistics.median(typical),
            'typical_frame_wall_seconds_mean': statistics.mean(typical),
            'typical_frame_count': len(typical),
            'simulated_seconds': report['timing']['simulated_seconds'],
            'operator_waits_included_in_total_wall_seconds': True},
        'training_eligible': False,
        'missing_for_expert_admission': ['deliberate M curl input',
            'sustained loaded off-soil outcome', 'operator-controllable response review',
            'independent training admission'],
        'sha256': {str(p): sha(p) for p in (RUN / 'binding.json',
            PHYSICS / 'report.json', PHYSICS / 'actions.ndjson',
            PHYSICS / 'physics/frames.ndjson', PHYSICS / 'final.npz')}}
    OUT.mkdir()
    (OUT / 'review.json').write_text(json.dumps(result, ensure_ascii=False,
        indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'checks': checks,
        'review': str(OUT / 'review.json')}, indent=2))


if __name__ == '__main__':
    main()
