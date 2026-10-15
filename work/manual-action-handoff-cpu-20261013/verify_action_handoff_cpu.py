"""CPU-only unit and frozen-input regression; no runtime construction."""
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types
import unittest
import numpy as np
from manual_residual_sac_env import command_at, control_mode, ACTION_HANDOFF_VERSION

root=Path('/home/qunwang/wajueji')
output=root/'runs/manual-action-handoff-cpu-20261013-v3'
output.mkdir(exist_ok=False)
started=time.monotonic()
sys.path.insert(0,str(root/'tests/v4'))
sources=['src/v4/manual_residual_sac_env.py','tests/v4/test_manual_action_handoff.py',
         'tests/v4/test_manual_trajectory_alignment.py',
         'work/manual-action-handoff-cpu-20261013/verify_action_handoff_cpu.py']
hashes={}
for name in sources:
    path=root/name;hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    target=output/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)

stream=io.StringIO()
suite=unittest.defaultTestLoader.discover(str(root/'tests/v4'),pattern='test_manual*.py')
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
(output/'tests.txt').write_text(stream.getvalue())
checks={'all_cpu_tests_passed':result.wasSuccessful(),'cuda_disabled':os.environ.get('CUDA_VISIBLE_DEVICES')==''}

# Both tests assert actual requests before looking at new logging fields.
old_path=root/'runs/manual-wait-strength-validation-20261013-v1/source-snapshot/manual_residual_sac_env.py'
old=types.ModuleType('old_handoff_regression');old.__file__=str(root/'src/v4/manual_residual_sac_env.py')
exec(compile(old_path.read_text(),str(old_path),'exec'),old.__dict__)
test_module=importlib.import_module('test_manual_action_handoff')
new_class=test_module.ManualResidualEnv;test_module.ManualResidualEnv=old.ManualResidualEnv
red_stream=io.StringIO()
try:
    red_suite=unittest.TestSuite(test_module.HandoffTests(n) for n in
        ['test_follow_to_wait_stops_until_next_decision','test_wait_to_follow_cannot_reinterpret_stop_or_full_or_pace'])
    red=unittest.TextTestRunner(stream=red_stream,verbosity=2).run(red_suite)
finally:test_module.ManualResidualEnv=new_class
(output/'old-code-expected-failures.txt').write_text(red_stream.getvalue())
checks['old_code_fails_both_command_assertions']=len(red.failures)==2 and not red.errors

prior=root/'runs/manual-wait-strength-validation-20261013-v1'
input_path=root/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275/input.ndjson'
commands=np.asarray([json.loads(s)['applied_command_si'] for s in input_path.read_text().splitlines()][:1170])
replay=[]
for label in ['manual_baseline','loaded_half','loaded_full','loaded_stop']:
    directory=prior/label/'episode-0001'
    actions=[json.loads(s) for s in (directory/'actions.ndjson').read_text().splitlines()]
    policies=[json.loads(s) for s in (directory/'policy.ndjson').read_text().splitlines()]
    after={v['frame']:v for v in (json.loads(s) for s in (directory/'alignment.ndjson').read_text().splitlines()) if v['stage']=='after'}
    changed=[];invalid=[];same=True;fallback=True;examined=0;special={}
    for policy in policies:
        if policy['frame_before']==policy['frame_after']:continue
        decision=('hold' if actions[policy['frame_before']]['hold_frame'] is not None else
                  'wait' if policy['mode_before'].startswith('wait') else policy['mode_before'])
        stale=False
        for frame in range(policy['frame_before']+1,policy['frame_after']+1):
            row=actions[frame-1];mode=control_mode(row['cursor_before'],row['alignment'],row['hold_frame'])
            stale |= mode!=decision
            a=np.array(policy['action'],float)
            if stale:
                a=np.zeros(5)
                if mode=='wait':a[:4]=-1
                invalid.append(frame)
            if row['hold_frame'] is not None:
                base=delta=request=np.zeros(4);pace=0.
            else:
                base,delta,request,pace=command_at(commands,row['cursor_before'],a,
                    reference_limit=max(row['cursor_before'],row['alignment']['reference_limit']),
                    wait_target_error=row['alignment']['wait_target_error_si'],waiting_action_mode='error_strength_v1')
            equal=np.array_equal(request,row['request_si']) and pace==row['pace']
            if not equal:changed.append(frame)
            if not stale:same &= equal
            if stale:
                fallback &= not np.any(delta)
                if mode=='wait':fallback &= not np.any(request) and pace==0
                elif mode=='follow':fallback &= np.array_equal(base,request) and pace<=1+1e-12
            if (label=='loaded_stop' and frame in [1307,1452]) or (label=='loaded_full' and frame==1022):
                special[frame]={'old_request':row['request_si'],'new_request_on_frozen_input':request.tolist(),
                                'new_delta':delta.tolist(),'new_pace':pace,'invalidated':bool(stale)}
            # Preserve transient after-state mode changes even if the next before differs.
            next_cursor=min(1170.,row['cursor_before']+row['pace']) if row['hold_frame'] is None else row['cursor_before']
            stale |= control_mode(next_cursor,after[frame],row['hold_frame'])!=decision
            examined+=1
    checks[label+'_unchanged_while_valid']=bool(same)
    checks[label+'_fallback_no_residual']=bool(fallback)
    checks[label+'_all_frames_examined']=examined==len(actions)
    if label=='manual_baseline':checks['baseline_frozen_commands_exact']=not changed
    replay.append({'label':label,'frames':examined,'invalidated_frames':invalid,
                   'changed_command_or_pace_frames':changed,'special_frames':special})

report=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
            cpu_tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),
            audit_v2_note='v2 stopped at a logged zero-frame terminal policy record; v3 excludes records with no executed frames. No physics or production code changes.',
            audit_v1_note='v1 command checks passed but counted hold as invalidated waiting; v2 uses the logged hold flag. Original report preserved.',
            previous_attempt='42 tests: 8 new passed; one legacy guard fixture lacked cursor/hold/mode/last_alignment. Fixture completed without changing assertions.',
            old_code_expected_failures=len(red.failures),old_code_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
            code=hashes,handoff_version=ACTION_HANDOFF_VERSION,replay=replay,
            elapsed_seconds=time.monotonic()-started,new_physics_frames=0,neural_updates=0,
            training_admission=False,physical_validation=False,
            limitation='Frozen-input command replay does not predict the new physical trajectory, hold success or fill.')
(output/'review.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'status':report['status'],'cpu_tests':result.testsRun,'checks':checks,
                  'replay_counts':[{k:v for k,v in r.items() if k in ['label','frames','special_frames']} |
                                   {'invalidated':len(r['invalidated_frames']),'changed':len(r['changed_command_or_pace_frames'])} for r in replay],
                  'elapsed_seconds':report['elapsed_seconds']}))
sys.exit(0 if all(checks.values()) else 1)
