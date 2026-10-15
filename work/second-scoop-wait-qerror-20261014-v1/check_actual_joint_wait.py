"""CPU cached counterfactuals and real env.step with scripted state doubles."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import sys, time, json, unittest, hashlib, io
from unittest.mock import patch
R=Path('/home/qunwang/wajueji')
W=R/'work/second-scoop-wait-qerror-20261014-v1'
FULL=R/'work/second-scoop-full-dig-20261014-v1'
sys.path[:0]=[str(W),str(FULL)]
import run_full_second as m
import manual_residual_sac_env as core
sys.path.insert(0,str(R/'tests/v4'))
from test_manual_action_handoff import HandoffTests
from actual_joint_wait import ActualJointWaitEnv, actual_joint_error, command_alignment, VERSION, AXES
import numpy as np

read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks={}
def check(name,condition):
 checks[name]=bool(condition)
 if not condition:raise AssertionError(name)

def main():
 started=time.monotonic();D=m.OUT
 old=read(D/'report.json');assert old['status']=='failed'
 source_hashes={str(p):sha(p) for p in [D/'report.json',D/'read-only-audit.json',D/'stall-diagnosis.json',FULL/'closure.json',m.MODEL,R/'src/v4/manual_residual_sac_env.py',R/'src/v4/manual_trajectory_alignment.py']}
 env=ActualJointWaitEnv(W/'reference-read',None,collect=False,waiting_action_mode='error_strength_v1',yaw_offset_rad=m.CANDIDATE['yaw_offset_rad'])
 actions=[json.loads(s) for s in (D/'policy/episode-0001/actions.ndjson').read_text().splitlines()]
 states=read(D/'display/states.json');last=actions[-1];cursor=last['cursor_before'];legacy=last['alignment'];q=np.asarray(states[-1]['mechanism_qpos'])
 original_legacy=json.dumps(legacy,sort_keys=True);original_q=q.copy();original_reference=env.reference_q.copy()
 new=command_alignment(legacy,env.reference_q,cursor,q)
 error=actual_joint_error(env.reference_q,cursor,q)
 check('cached_actual_error_sign_and_value',np.array_equal(error,-np.array(read(D/'stall-diagnosis.json')['last_actual_q_minus_reference_q_si'])))
 virtual=[]
 for strength in [0.,.5,1.]:
  action=np.r_[np.full(4,2*strength-1),0.]
  def cmd(al):return core.command_at(env.commands,cursor,action,.1,reference_limit=al['reference_limit'],wait_target_error=al['wait_target_error_si'],waiting_action_mode='error_strength_v1')
  _,_,request,pace=cmd(new);old_request=cmd(legacy)[2]
  check(f'cached_{strength}_pace_frozen',pace==0)
  check(f'cached_{strength}_direction_stop_caps',np.all(request*error>=0) and np.all(abs(request)<=core.LIMITS_SI*.1+1e-15) and np.all(abs(request)<=abs(error)*60+1e-15))
  check(f'cached_{strength}_nonzero_or_stop',np.all(request==0) if strength==0 else np.all(abs(request)>1e-10))
  check(f'cached_{strength}_legacy_remains_dead',np.max(abs(old_request))<1e-10)
  virtual.append(dict(strength=strength,legacy_request_si=old_request.tolist(),actual_joint_request_si=request.tolist(),pace=pace))
 check('cached_no_input_mutation',json.dumps(legacy,sort_keys=True)==original_legacy and np.array_equal(q,original_q) and np.array_equal(env.reference_q,original_reference))
 check('legacy_report_error_distinct_from_command_error',np.max(np.abs(new['legacy_wait_target_error_si']))<1e-12 and np.max(np.abs(new['wait_target_error_si']))>1e-4)
 # Fractional references and rotated heading must operate in joint coordinates.
 ref=np.tile(np.arange(19,dtype=float),(3,1));ref[1,AXES]+=[.01,-.02,.03,-.04];ref[2]=ref[1]
 aq=ref[0].copy()
 check('fractional_interpolation',np.allclose(actual_joint_error(ref,.25,aq),np.array([.01,-.02,.03,-.04])*.25,atol=1e-14))
 check('endpoint_interpolation',np.array_equal(actual_joint_error(ref,2.,ref[2]),np.zeros(4)))
 rotated=ref.copy();rotated[:,7]+=.12755;rq=aq.copy();rq[7]+=.12755
 check('heading_offset_cancels',np.allclose(actual_joint_error(rotated,.25,rq),actual_joint_error(ref,.25,aq),atol=1e-14))
 for name,badref,badcursor,badq in [('negative',ref,-1.,aq),('beyond',ref,3.,aq),('nan_cursor',ref,float('nan'),aq),('short_q',ref,0.,aq[:18]),('nan_q',ref,0.,np.full(19,np.nan)),('bad_reference_shape',ref[:,:18],0.,aq),('nan_reference',np.full((3,19),np.nan),0.,aq)]:
  try:actual_joint_error(badref,badcursor,badq)
  except ValueError:check('reject_'+name,True)
  else:check('reject_'+name,False)
 # Reuse the real original three-physical-frame step, rather than duplicate
 # mode invalidation logic. State doubles do not initialize or step Newton.
 def fixture(initial,modes,**kw):
  test=HandoffTests();e,requests=test.make_env(initial,modes,**kw)
  e.runtime.bridge.state['mechanism_qpos']=np.zeros(19)
  e.reference_q=np.tile(np.array([0.]*19),(1171,1));e.reference_q[:,AXES]=[.01,-.01,.01,-.01]
  old_alignment=e.alignment_state
  e.alignment_state=lambda stage:command_alignment(old_alignment(stage),e.reference_q,e.cursor,e.runtime.bridge.state['mechanism_qpos'])
  return test,e,requests
 cases=[('follow_wait','follow',['follow','wait','wait','wait','wait','wait']),('wait_follow','wait',['wait','follow','follow','follow','follow','follow']),('return_original','follow',['follow','wait','wait','follow','follow','follow']),('preframe_change','follow',['wait']*6)]
 for name,initial,modes in cases:
  test,e,requests=fixture(initial,modes)
  try:
   e.step(np.ones(5));rows=test.rows(e)
   check(name+'_stale_action_invalidated',[x['action_invalidated'] for x in rows]==([True]*3 if name=='preframe_change' else [False,True,True]))
   check(name+'_stale_wait_stops',all(np.all(np.array(x['request_si'])==0) for x in rows if x['action_invalidated'] and x['execution_mode']=='wait'))
   check(name+'_stale_follow_no_residual',all(np.all(np.array(x['residual_si'])==0) and x['pace']==1 for x in rows if x['action_invalidated'] and x['execution_mode']=='follow'))
  finally:test.doCleanups()
 # Multi-frame idealized joint kinematics: the remaining-error cap reaches a
 # tiny target in one frame, then recomputation stops. This is NOT physics.
 test,e,requests=fixture('wait',['wait']*12)
 try:
  e.reference_q[:,AXES]=[.000001,-.000001,.000001,-.000001]
  def advance(request):
   requests.append(request.copy());e.runtime.frames+=1
   e.runtime.bridge.state['mechanism_qpos'][AXES]+=request/60.
  e.runtime.step=advance
  e.step(np.ones(5));e.step(np.ones(5))
  check('multi_frame_recomputes_actual_error',np.array_equal(requests[1:],np.zeros((5,4))))
  check('ideal_kinematic_error_cleared_no_overshoot',np.allclose(e.runtime.bridge.state['mechanism_qpos'][AXES],e.reference_q[0,AXES],rtol=0,atol=1e-17))
  check('multiframe_legacy_observation_error_preserved',e.report['last_alignment']['wait_target_error_si']==[.01,-.01,.01,-.01])
 finally:test.doCleanups()
 # Original observation encoder and safety/progress/hold step are inherited;
 # exercise alignment override while explicitly checking report non-mutation.
 env.runtime=type('Runtime',(),{'bridge':type('Bridge',(),{'state':{'mechanism_qpos':q}})()})()
 env.cursor=cursor;env.report={'last_alignment':dict(legacy)}
 with patch.object(m.YawTransferEnv,'alignment_state',return_value=legacy):
  returned=env.alignment_state('before')
 check('candidate_override_does_not_replace_legacy_model_report',env.report['last_alignment']==legacy and returned['wait_target_error_si']==new['wait_target_error_si'])
 check('same_221_observation_encoder',ActualJointWaitEnv.observation is m.YawTransferEnv.observation)
 check('safety_alignment_gates_unchanged',all(returned[k]==legacy[k] for k in legacy if k!='wait_target_error_si'))
 check('nonwait_mapping_bit_exact',all(np.array_equal(a,b) for a,b in zip(core.command_at(env.commands,76,np.zeros(5),reference_limit=80,wait_target_error=legacy['wait_target_error_si'],waiting_action_mode='error_strength_v1'),core.command_at(env.commands,76,np.zeros(5),reference_limit=80,wait_target_error=command_alignment(dict(legacy,reference_limit=80),env.reference_q,76,q)['wait_target_error_si'],waiting_action_mode='error_strength_v1'))))
 for method in ['step','reset']:
  try:getattr(env,method)(np.zeros(5)) if method=='step' else env.reset()
  except (RuntimeError,ValueError):check('physical_'+method+'_rejected',True)
  else:check('physical_'+method+'_rejected',False)
 check('frozen_source_hashes_unchanged',all(sha(Path(p))==h for p,h in source_hashes.items()))
 suite=unittest.defaultTestLoader.loadTestsFromName('test_manual_action_handoff')
 stream=io.StringIO();regression=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 (W/'original-handoff-regression.log').write_text(stream.getvalue())
 check('original_handoff_regressions',regression.wasSuccessful())
 result=dict(status='cpu_candidate_passed',version=VERSION,check_count=len(checks),checks=checks,original_handoff_tests=regression.testsRun,virtual_actions=virtual,actual_joint_error_si=error.tolist(),reference_cursor=cursor,source_hashes=source_hashes,candidate_sha256=sha(W/'actual_joint_wait.py'),checker_sha256=sha(Path(__file__)),physical_steps=0,native_initializations=0,training_updates=0,physics_recovery_verified=False,observation_contract='same221 encoder; legacy target-error fields unchanged',old_replay_direct_mixing_allowed=False,wall_seconds=time.monotonic()-started)
 (W/'cpu-validation.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
