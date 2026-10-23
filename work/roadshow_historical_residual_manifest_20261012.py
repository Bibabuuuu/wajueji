"""Screen historical first-scoop actions for an offline residual study.

Read-only with respect to the source run. No training admission, fitting, or
physics step is performed. Phase labels are an offline causal candidate.
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path('/home/qunwang/wajueji')
RUN = ROOT / 'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275'
OUT = ROOT / 'runs/roadshow-measurements-20261011/historical-residual-screen-20261012-v1'
KIN = Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
CAVITY = Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json')
PHASE = ROOT / 'work/contact-phase-labels-20260924-v1/phase-candidate-v1.json'
COMPLETION = ROOT / 'work/contact-phase-labels-20260924-v1/phase-completion-v1.json'
PRIOR = ROOT / 'runs/roadshow-measurements-20261011/historical-first-scoop-label-audit-20261012-v1/review.json'

sys.path[:0] = [str(ROOT), str(ROOT / 'src/v4')]
from contact_bc_inference import LIMITS
from contact_phase_candidate import ContactPhaseConfig
from contact_phase_completion import CompletionConfig, FivePhaseCandidate
from contact_phase_live import mouth_angle_deg
from layered_excavation_approach_planner import _load_kinematics
from roadshow_geometry_baseline import joint_target_intent
from run_layered_47_sparse_manual_contact_episode import (
    BUCKET_INTERNAL_UPPER_TARGET_M, KINEMATIC_JOINT_MARGIN_M)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read_lines(path, limit=None):
    with path.open() as stream:
        return [json.loads(line) for _, line in zip(range(limit or 10**9), stream)]


def main():
    if OUT.exists():
        raise ValueError('immutable_output_already_exists')
    prior = json.loads(PRIOR.read_text())
    if prior['status'] != 'causal_anchors_available_labels_not_admitted':
        raise ValueError('prior_causal_audit_not_passed')
    for path_string, digest in prior['sha256'].items():
        if sha(Path(path_string)) != digest:
            raise ValueError('source_changed:' + path_string)
    actions = read_lines(RUN / 'input.ndjson', 1800)
    physical = read_lines(RUN / 'physics.ndjson', 1800)
    anchors = [row for row in read_lines(RUN / 'observation.ndjson')
               if row['frame'] < 1800]
    if (len(actions), len(physical), len(anchors)) != (1800, 1800, 599):
        raise ValueError('unexpected_source_counts')
    report = json.loads((RUN / 'report.json').read_text())
    if (report['source_checkpoint']['sha256'] !=
            '700f76be970c5754bde905837faf642cca0317fef097e31d17cf48f008ba0545'):
        raise ValueError('source_domain_changed')
    assembly, chain = _load_kinematics(KIN, KIN / 'scene/input/model/A.xml',
                                       KIN / 'scene/input/mesh')
    polygon = json.loads(CAVITY.read_text())['bucket']['polygon_xz_m']
    initial = np.asarray(physical[0]['before_state']['mechanism_qpos'], float)
    initial_pose = assembly.assemble(initial, include_meshes=False)[1]['bucket']
    router = FivePhaseCandidate(
        ContactPhaseConfig(**json.loads(PHASE.read_text())['parameters']),
        CompletionConfig(**json.loads(COMPLETION.read_text())['parameters']),
        initial_mouth_angle_deg=mouth_angle_deg(initial_pose, polygon))
    goal = report['selected_cut']
    if not all(k in goal for k in ('entry_xy_m', 'surface_z_m')):
        raise ValueError('missing_goal')

    # The geometry controller freezes its reference at the first curl decision.
    # The original manual run has no geometry controller; this is a same-state
    # counterfactual normalized intent, not a replayed safe command.
    geometry_reference = None
    next_phase = {}
    events = []
    for row in physical:
        frame = row['frame']
        if frame != router.last_frame_end + 1 or actions[frame-1]['frame'] != frame:
            raise ValueError('noncontiguous_physical_or_action_log')
        phase_before = router.phase_for_decision(frame)
        if phase_before == 'curl_load' and geometry_reference is None:
            geometry_reference = np.asarray(
                row['before_state']['mechanism_qpos'], float)[chain.axes].copy()
        q = np.asarray(row['after_state']['mechanism_qpos'], float)
        pose = assembly.assemble(q, include_meshes=False)[1]['bucket']
        stage = router.observe_end(
            frame_end=frame,
            soil_force_n=row['bucket_mpm_force_norm_n'],
            rock_contact_substeps=row['rock_bucket_contact_substeps'],
            mouth_angle_deg=mouth_angle_deg(pose, polygon),
            actual_boom_cylinder_m=q[chain.axes[1]],
            bucket_origin_z_m=float(pose[2, 3]),
            # No exact per-frame ground-connected soil gap was saved here.
            remaining_soil_gap_m=None)
        if stage['curl_boundary_unresolved']:
            raise ValueError('curl_boundary_unresolved')
        if stage['event']:
            events.append(stage['event'])
        next_phase[frame+1] = router.phase
    if geometry_reference is None:
        raise ValueError('no_curl_reference')

    lower, high = chain.limits[chain.axes[3]]
    upper = min(BUCKET_INTERNAL_UPPER_TARGET_M,
                high - KINEMATIC_JOINT_MARGIN_M)
    manifest = []
    counts = Counter()
    for anchor in anchors:
        frame = anchor['frame']
        decision = frame + 1
        action = actions[decision-1]
        before = physical[decision-1]['before_state']
        if (anchor['state'] != physical[frame-1]['after_state']
                or anchor['state'] != before):
            raise ValueError('anchor_not_same_causal_decision_state')
        if action['training_eligible'] is not False or anchor['training_eligible'] is not False:
            raise ValueError('unexpected_training_admission')
        human = np.asarray(action['normalized_before_projection'], float)
        request = np.asarray(action['requested_command_si'], float)
        applied = np.asarray(action['applied_command_si'], float)
        if (human.shape != (4,) or not np.isfinite(human).all()
                or np.any(np.abs(human) > 1+1e-8)):
            raise ValueError('invalid_human_intent')
        request_matches = bool(np.allclose(request, human*LIMITS,
                                           atol=1e-12, rtol=0))
        phase = next_phase[decision]
        phase_certain = phase != 'lift'
        reasons = []
        if phase != 'curl_load':
            reasons.append('outside_curl_load' if phase_certain else
                           'lift_or_off_soil_unresolved')
        if action['command_projection_reason'] is not None:
            reasons.append('projected_action')
        if not request_matches:
            reasons.append('request_intent_mismatch')
        if not (action['armed'] and action['connected'] and action['window_focused']
                and not action['emergency_stop_latched']):
            reasons.append('input_gate_inactive')
        if not physical[decision-1]['hard_guard']:
            reasons.append('physical_hard_guard_failed')
        if np.max(np.abs(human)) <= 1e-8:
            reasons.append('zero_operator_intent')
        if action['command_projection_reason'] is None and not np.allclose(
                request, applied, atol=1e-12, rtol=0):
            reasons.append('unexplained_request_application_difference')

        geometry = residual = target = None
        if phase == 'curl_load':
            q = np.asarray(before['mechanism_qpos'], float)[chain.axes]
            g, t = joint_target_intent(q, geometry_reference, lower, upper,
                                       LIMITS, 1.5)
            geometry, target = g.tolist(), t.tolist()
            if not reasons:
                residual = (human-g).tolist()
        accepted = not reasons
        counts['offline_screen_pass' if accepted else 'offline_screen_exclude'] += 1
        for reason in reasons:
            counts['reason:' + reason] += 1
        manifest.append({
            'observation_frame_end': frame,
            'decision_frame': decision,
            'observation_sha256': canonical_sha(anchor),
            'phase_offline_candidate': phase if phase_certain else
                'lift_or_off_soil_unresolved',
            'phase_evidence_latest_frame_end': frame,
            'human_normalized_intent': human.tolist(),
            'geometry_normalized_same_state_intent': geometry,
            'geometry_joint_target': target,
            'human_minus_geometry_normalized_residual': residual,
            'requested_command_si': request.tolist(),
            'applied_command_si': applied.tolist(),
            'command_projection_reason': action['command_projection_reason'],
            'offline_screen_pass': accepted,
            'exclusion_reasons': reasons,
            'training_eligible': False,
        })
    transitions = [(e['transition'], e['confirmation_frame_end']) for e in events]
    if transitions != [('approach_contact_to_penetrate', 46),
                       ('penetrate_to_curl_load', 206),
                       ('curl_load_to_lift', 639)]:
        raise ValueError('unexpected_phase_transitions:' + repr(transitions))
    if counts['offline_screen_pass'] != 130:
        raise ValueError('unexpected_screen_count:' + str(counts['offline_screen_pass']))
    summary = {
        'schema': 'roadshow_historical_residual_screen_v1',
        'status': 'offline_candidates_for_review_not_training_admitted',
        'source_domain': 'historical_gamepad_original_physics',
        'frames_read': 1800,
        'causal_anchors': len(anchors),
        'phase_events': events,
        'geometry_reference_first_curl_decision_frame': 207,
        'geometry_reference_q': geometry_reference.tolist(),
        'geometry_internal_bucket_upper_m': float(upper),
        'counts': dict(sorted(counts.items())),
        'candidate_definition': 'Causal observation at frame f, operator intent at f+1, offline curl phase, nonzero input, active input gate, hard guard, no command projection, and exact requested/applied agreement.',
        'limits': [
            'Offline phase thresholds were calibrated on earlier runs and are not formal human annotations for this episode.',
            'Exact ground-connected soil gap was not logged per frame; after lift starts, off-soil status remains unresolved.',
            'Geometry is a same-state unprojected intent computed on a human trajectory. It is not a geometry-controlled counterfactual rollout.',
            'Residual targets have not been approved for fitting; no learned weights or BC-vs-geometry result exists.',
        ],
        'training_eligible': False,
        'sha256': {str(p): sha(p) for p in (
            RUN/'report.json', RUN/'input.ndjson', RUN/'physics.ndjson',
            RUN/'observation.ndjson', PRIOR, PHASE, COMPLETION, CAVITY,
            KIN/'scene/input/model/A.xml',
            ROOT/'src/v4/roadshow_geometry_baseline.py', Path(__file__))},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    with (OUT/'samples.ndjson').open('w') as stream:
        for row in manifest:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False,
                                    separators=(',', ':')) + '\n')
    summary['samples_ndjson_sha256'] = sha(OUT/'samples.ndjson')
    (OUT/'review.json').write_text(json.dumps(summary, ensure_ascii=False,
                                        allow_nan=False, indent=2) + '\n')
    print(json.dumps({'status': summary['status'], 'counts': summary['counts'],
                      'events': transitions, 'output': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
