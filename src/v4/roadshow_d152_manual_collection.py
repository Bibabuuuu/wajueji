"""Isolated D152 manual curl collection; no training admission by default.

The SHA-bound D154 loop still owns restoration, physics, phase detection,
projection, previews, and hard guards. Only the policy input is replaced by
local operator input in the curl phase; approach and penetration remain
geometric. Every result requires post-run review before training use.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path('/home/qunwang/wajueji')
WORK = ROOT/'work/contact-multisource-lift-20260924-v1'
BASE = WORK/'run_candidate.py'
BASE_SHA = '63a84592718a68d60bad7ee6d518193e67b8c0c78e48bb9775f19c00469c949a'
ADAPTIVE = ROOT/'src/v4/roadshow_adaptive_lift_feedback_v2.py'
ADAPTIVE_SHA = '5d7e960311d719c6915cae69f0fe8cd699744c40879350c323aa85202fc2c430'
SOURCE = ROOT/'runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz'
POLICY_REFERENCE = ROOT/'runs/contact-phase-bc-pilot-v1/pilot.pt'
TRAIN = '/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_source(original, manual_input, frames, ui_smoke=False,
                   step_frames=30):
    edits = [
        ('from contact_phase_bc_inference import PhaseInferenceWorker as InferenceWorker,OnlinePhaseFeatures as OnlineFeatures,with_phase',
         'from contact_phase_bc_inference import OnlinePhaseFeatures as OnlineFeatures,with_phase\nfrom roadshow_d152_manual_worker import ManualWorker as InferenceWorker'),
        ('from candidate_multisource import LiftFeedback as ContactGeometryFeedback,save_control',
         'from candidate_multisource import save_control\nfrom roadshow_adaptive_lift_feedback_v2 import LiftFeedback as ContactGeometryFeedback'),
        ('from gates import behavior_gate',
         '''from gates import behavior_gate as original_behavior_gate
    def behavior_gate(actions, physical, feedback, require_curl):
        result=original_behavior_gate(actions,physical,feedback,require_curl)
        if require_curl:
            checks=result['checks']
            checks['manual_input_used_after_curl']=checks.pop('bc_controlled_after_curl') and any(
                e['manual_input_effective'] and e['operator_gate_armed']
                for e in actions if e['phase']=='curl_load')
            checks['root_reached_soil_before_curl']=checks.pop('root_reached_soil_before_bc')
            result['passed']=all(checks.values())
        return result'''),
        ('if args.frames not in (0,3000):',
         'if args.frames not in (0,120,1800):'),
        ('if time.monotonic()-started>wall_budget:raise TimeoutError(\'bc_probe_wall_budget\')',
         "if worker.active_wall_seconds()>wall_budget:raise TimeoutError('manual_active_wall_budget')"),
        ('                report[\'feedback_parameters\']=contract[\'feedback_parameters\']',
         "                worker.bind(r,feedback,goal,mode="+repr(manual_input)+
         ",device='cuda:0',ui_smoke="+repr(ui_smoke)+
         ",step_frames="+repr(step_frames)+")\n"
         "                report['manual_input']=dict(device="+repr(manual_input)+
         ",neural_weights_used=False,training_eligible=False)\n"
         "                report['feedback_parameters']=contract['feedback_parameters']"),
        ('                    x,digest=features.update(structured);action,timing=worker.predict(x,digest)',
         "                    worker.phase=phase_before\n"
         "                    x,digest=features.update(structured);action,timing=worker.predict(x,digest)\n"
         "                    if worker.stop_reason is not None:\n"
         "                        report['early_stop_reason']=worker.stop_reason;break"),
        ("source='geometry_feedback_then_guarded_phase_bc'",
         "source='geometry_feedback_then_guarded_manual_input'"),
        ('                               previous_applied_action_si=previous_action,previous_drive_load=previous_load)',
         "                               previous_applied_action_si=previous_action,previous_drive_load=previous_load)\n"
         "                    event['geometry_same_state_intent']=worker.geometry_intent(before['mechanism_qpos'],phase_before)\n"
         "                    event.update(worker.last_record)\n"
         "                    event['manual_input_effective']=(phase_before in ('curl_load','lift','off_soil') and control['mode']=='bc_with_lateral_depth_feedback' and worker.input_gate.armed and bool(any(abs(v)>1e-8 for v in action)))"),
    ]
    for old, new in edits:
        if original.count(old) != 1:
            raise ValueError('manual_source_anchor_changed:'+old)
        original = original.replace(old, new)
    compile(original, str(BASE), 'exec')
    return original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--frames', type=int, choices=(0, 120, 1800), default=0)
    parser.add_argument('--manual-input', choices=('zero','keyboard','step_keyboard'), default='step_keyboard')
    parser.add_argument('--step-frames', type=int, choices=(15,30,60), default=30)
    parser.add_argument('--ui-smoke', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.manual_input == 'zero' and args.frames == 1800:
        parser.error('zero input is limited to the 0/120-frame preparation gate')
    if args.ui_smoke and (args.manual_input not in ('keyboard','step_keyboard') or args.frames != 0):
        parser.error('UI smoke requires keyboard input and zero physical frames')
    sys.path[:0] = [str(ROOT), str(ROOT/'src/v4'), str(WORK),
                    str(ROOT/'work/night-dual-start-preparation-20260925-v2')]
    if sha(BASE) != BASE_SHA or sha(ADAPTIVE) != ADAPTIVE_SHA:
        raise ValueError('bound_source_changed')
    protocol = json.loads((WORK/'first-protocol.json').read_text())
    for item in protocol['inputs']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('protocol_input_changed:'+item['path'])
    if sha(SOURCE) != protocol['source_sha256'] or sha(POLICY_REFERENCE) != protocol['policy_sha256']:
        raise ValueError('source_or_reference_changed')
    code = prepare_source(BASE.read_text(), args.manual_input, args.frames,
                          args.ui_smoke, args.step_frames)
    if args.check:
        from roadshow_geometry_baseline import self_check
        self_check()
        print('D152_MANUAL_STATIC_CHECK_PASSED; physics_steps=0')
        return
    if args.output is None or args.output.exists():
        raise ValueError('new_immutable_output_required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    generated = output/'generated-runner.py'
    generated.write_text(code)
    binding = {'schema':'roadshow_d152_manual_collection_binding_v1',
        'status':'preparing', 'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'runner_sha256':sha(__file__), 'base_sha256':sha(BASE),
        'generated_runner_sha256':sha(generated),
        'manual_worker_sha256':sha(ROOT/'src/v4/roadshow_d152_manual_worker.py'),
        'keyboard_input_sha256':sha(ROOT/'src/v4/manual_action_input.py'),
        'input_gate_sha256':sha(ROOT/'src/v4/run_layered_47_optimized_practice.py'),
        'geometry_intent_sha256':sha(ROOT/'src/v4/roadshow_geometry_baseline.py'),
        'adaptive_preview_sha256':sha(ADAPTIVE),
        'source_sha256':sha(SOURCE), 'reference_policy_sha256':sha(POLICY_REFERENCE),
        'reference_policy_used':False, 'manual_input':args.manual_input,
        'frames_requested':args.frames, 'training_eligible':False,
        'ui_smoke':args.ui_smoke, 'step_frames':args.step_frames,
        'control_scope':'geometry approach and penetration; human normalized input into guarded curl; bounded feedback lift',
        'python':sys.version, 'machine':platform.machine(),
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version',
                                      '--format=csv,noheader'],text=True).strip()}
    (output/'binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    sys.argv = [str(BASE), '--source',str(SOURCE),'--policy',str(POLICY_REFERENCE),
        '--policy-sha256',sha(POLICY_REFERENCE),'--output',str(output/'physics-run'),
        '--frames',str(args.frames),'--training-python',TRAIN,'--coupling-hz','60',
        '--phase-contract',str(WORK/'first-protocol.json')]
    try:
        exec(compile(code,str(BASE),'exec'),{'__name__':'__main__',
                                            '__file__':str(BASE)})
        binding['status']='runner_completed'
    except BaseException as error:
        binding['status']='failed'
        binding['error']=repr(error)
        raise
    finally:
        report_path=output/'physics-run/report.json'
        if report_path.exists():
            report=json.loads(report_path.read_text())
            report['schema']='roadshow_d152_manual_collection_report_v1'
            report['candidate_policy_reference']=report.pop('policy')
            report['policy']={'used':False,'controller':'human keyboard input'}
            report['reference_policy_used_for_control']=False
            report['operator_decision_seconds_quantiles']=report.pop('inference_roundtrip_seconds_quantiles')
            if report.get('early_stop_reason') in ('user_finished','window_closed'):
                report['status']='operator_finished_diagnostic'
            report['manual_collection_binding']=binding
            report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            binding['frames_completed']=report.get('frames_completed')
        (output/'binding.json').write_text(json.dumps(binding,indent=2)+'\n')


if __name__ == '__main__':
    main()
