"""Saved multi-frame and atomic admission checks; no physical runtime."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,signal,shutil,subprocess
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-rise-hold-20261014-v1';P=R/'work/second-scoop-short-minimum-20261014-v1';C=R/'work/second-scoop-minimum-guard-20261014-v1'
sys.path[:0]=[str(W),str(C),str(P),str(R/'work/second-scoop-short-20261014-v1')]
import run_minimum_short as m
from rise_hold_guard import RiseHoldGuard,VERSION
from minimum_guard import MinimumHeightClearance
import numpy as np
from types import SimpleNamespace

def main():
 start=time.monotonic();out=W/'cpu-v1';out.mkdir(exist_ok=False);checks={}
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU180secondlimit')));signal.alarm(180)
 def check(n,v):checks[n]=bool(v);assert v,n
 def rejects(n,fn,reason=None):
  try:fn()
  except ValueError as e:check(n,reason is None or str(e)==reason)
  else:raise AssertionError(n)
 D=m.OUT;p=m.read(D/'report.json');states=m.read(D/'display/states.json');plan=m.read(D/'live-plan.json');steps=plan['segments'][0]['frames'];end=3+steps
 check('old_failure_and_closed_budget_preserved',p['status']=='failed' and p['frames']==2159 and (P/'closure.json').is_file())
 frozen={str(f):m.sha(f) for f in [D/'report.json',D/'read-only-audit.json',D/'stop-diagnosis.json',P/'closure.json',P/'run_minimum_short.py']}
 r=m.CpuRuntime(D/'zero.npz',out/'runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True);r.bridge=SimpleNamespace(state=states[3])
 with np.load(D/'live-terrain.npz') as z:terrain={k:z[k] for k in z.files}
 with np.load(D/'live-plan.npz') as z:goal=z['qpos'][steps].copy()
 g=RiseHoldGuard(r,terrain,goal,steps)
 check('initial_mode_is_rise',g.mode=='rise')
 rejects('premature_transition_rejected',lambda:g.enter_hold(states[end]['mechanism_qpos'],completed_nominal_steps=steps-1),'rise_path_not_completed')
 check('early_rejection_keeps_rise_mode',g.mode=='rise')
 rejects('wrong_endpoint_rejected',lambda:g.enter_hold(states[3]['mechanism_qpos'],completed_nominal_steps=steps),'original_endpoint_guard_before_hold')
 check('bad_endpoint_keeps_rise_mode',g.mode=='rise')
 rejects('boolean_progress_not_proof',lambda:g.enter_hold(goal,completed_nominal_steps=True),'rise_path_not_completed')
 rejects('NaN_state_rejected',lambda:g.check(np.full(19,np.nan)), 'rise_hold_state_contract')
 rejects('bad_goal_rejected',lambda:RiseHoldGuard(r,terrain,np.zeros(18),steps),'rise_hold_goal_contract')
 rejects('bad_steps_rejected',lambda:RiseHoldGuard(r,terrain,goal,0),'rise_hold_steps_contract')
 active_count=0;minimum_step=float('inf')
 for i in range(4,end+1):
  d=g.check(states[i]['mechanism_qpos'],states[i-1]['mechanism_qpos'],physical=True);minimum_step=min(minimum_step,d['minimum_height_step_m']);active_count+=1
 check('all2151_actual_active_rise_frames_pass',active_count==2151 and minimum_step>=-1e-6 and g.mode=='rise')
 # The exact stopped hold motion must still be rejected in active-rise mode.
 rejects('same_descent_still_rejected_in_rise',lambda:g.check(states[-1]['mechanism_qpos'],states[-2]['mechanism_qpos'],physical=True),'return_rise_minimum_bucket_height_descended')
 check('rejected_rise_does_not_switch_mode',g.mode=='rise')
 entry=g.enter_hold(states[end]['mechanism_qpos'],completed_nominal_steps=steps)
 check('hold_admitted_after_path_endpoint_32cm',g.mode=='hold' and entry['bucket_observed_terrain_m']>=.32 and entry['endpoint_axis_error']<=.015)
 hold=[]
 for i in range(end+1,len(states)):hold.append(g.check(states[i]['mechanism_qpos'],states[i-1]['mechanism_qpos'],physical=True))
 check('five_actual_hold_frames_pass_original_guards',len(hold)==5 and all(x['bucket_observed_terrain_m']>=.32 and x['endpoint_axis_error']<=.015 and x['closure_m']<=.001 and x['bucket_truck_m']>=.04 and x['mechanism_truck_m']>=.02 for x in hold))
 rejects('duplicate_switch_rejected',lambda:g.enter_hold(goal,completed_nominal_steps=steps),'rise_hold_repeated_transition')
 bad=np.asarray(states[-1]['mechanism_qpos']).copy();bad[r.chain.axes[2]]=goal[r.chain.axes[2]]+.016
 rejects('hold_drift_not_excused',lambda:g.check(bad,physical=True),'original_endpoint_guard_during_hold')
 # A completed counter plus a nearby goal cannot bypass original soil gap.
 r.bridge.state=states[3];early=RiseHoldGuard(r,terrain,np.asarray(states[3]['mechanism_qpos']),1)
 rejects('completed_counter_cannot_waive32cm',lambda:early.enter_hold(states[3]['mechanism_qpos'],completed_nominal_steps=1))
 check('gap_failure_transition_is_atomic',early.mode=='rise')
 # CPU evaluator remains read-only with respect to all cached physical data.
 check('frozen_reports_code_and_model_unchanged',all(m.sha(Path(n))==h for n,h in frozen.items()) and m.sha(m.MODEL)==m.MODEL_SHA and all(m.sha(n)==h for n,h in m.read(P/'protocol.json')['code_sha256'].items()))
 check('no_native_entry_step_or_reset',not hasattr(g,'step') and not hasattr(g,'__enter__') and not hasattr(g,'reset'))
 check('GPU_idle',not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip())
 snapshot=out/'code-snapshot';snapshot.mkdir()
 for n in ['rise_hold_guard.py','check_rise_hold.py']:shutil.copyfile(W/n,snapshot/n)
 result=dict(status='cpu_phase_candidate_passed',version=VERSION,checks=checks,check_count=len(checks),active_rise_frames=active_count,hold_frames=len(hold),
   actual_transition_frame=end,minimum_active_rise_height_step_m=minimum_step,hold_entry=entry,
   code_sha256={n:m.sha(W/n) for n in ['rise_hold_guard.py','check_rise_hold.py']},frozen_inputs_sha256=frozen,
   physics_steps=0,native_initializations=0,policy_predictions=0,training_updates=0,physical_admission=False,
   limitation='Only five real hold frames are cached. Longer settling, actual mode handoff and later approach/residual are not physically verified. The old2159-frame failed run stays failed.',wall_seconds=time.monotonic()-start)
 m.write(out/'report.json',result);signal.alarm(0);print(m.encode({k:v for k,v in result.items() if k not in ('checks','frozen_inputs_sha256')}),flush=True)
if __name__=='__main__':main()
