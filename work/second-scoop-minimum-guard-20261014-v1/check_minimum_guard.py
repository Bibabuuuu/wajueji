"""CPU-only cached-state and boundary checks. Original failure is never relabeled."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,hashlib,signal,shutil,subprocess
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-minimum-guard-20261014-v1';OLD=R/'work/second-scoop-short-20261014-v1'
sys.path[:0]=[str(W),str(OLD)]
import run_short as m
from minimum_guard import minimum_rise,MinimumHeightClearance,VERSION
import numpy as np
from types import SimpleNamespace

def main():
 start=time.monotonic();checks={};D=m.OUT;out=W/'cpu-v1';out.mkdir(exist_ok=False)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU180secondlimit')));signal.alarm(180)
 def check(n,v):checks[n]=bool(v);assert v,n
 before={str(p):m.sha(p) for p in [D/'report.json',D/'read-only-audit.json',D/'stop-diagnosis.json',OLD/'closure.json',OLD/'run_short.py']}
 def example(n,current,previous,gap=.3,initial=.3,pass_=True,reason=None):
  try:minimum_rise(current,previous,gap,initial);ok=True;error=None
  except ValueError as e:ok=False;error=str(e)
  check(n,ok==pass_ and (reason is None or error==reason))
 base=np.array([[0.,0.,0.],[0.,0.,1.]])
 example('rotation_top_descends_lowest_rises',base+[[0,0,.01],[0,0,-.01]],base)
 example('unchanged_passes',base,base)
 example('all_points_up_passes',base+[0,0,.01],base)
 example('lowest_down_0.9um_allowed',base-[0,0,.9e-6],base)
 example('lowest_down_exact1um_allowed',base-[0,0,1e-6],base)
 example('lowest_down_over1um_rejected',base-[0,0,np.nextafter(1e-6,np.inf)],base,pass_=False,reason='return_rise_minimum_bucket_height_descended')
 example('mean_up_lowest_down_rejected',base+[[0,0,-2e-6],[0,0,.01]],base,pass_=False)
 example('lowest_sample_identity_can_change',np.array([[0,0,1.01],[0,0,.01]]),base)
 example('inherited_gap_exact_passes',base,base,gap=.3)
 example('inherited_gap_tolerance_boundary_passes',base,base,gap=.3-1e-8)
 example('inherited_gap_below_boundary_rejected',base,base,gap=np.nextafter(.3-1e-8,-np.inf),pass_=False,reason='return_rise_reduced_inherited_gap')
 example('height_up_does_not_waive_gap',base+[0,0,.01],base,gap=.299,pass_=False)
 for name,v in [('nan',np.full((2,3),np.nan)),('infinite',np.full((2,3),np.inf)),('empty',np.empty((0,3))),('wrong_columns',np.zeros((2,2)))]:example('reject_'+name,v,base,pass_=False,reason='minimum_rise_input_contract')
 example('reject_nonfinite_gap',base,base,gap=np.nan,pass_=False,reason='minimum_rise_input_contract')
 example('reject_negative_inherited_gap',base,base,initial=-.01,pass_=False,reason='minimum_rise_input_contract')
 states=m.read(D/'display/states.json');p=m.read(D/'report.json');check('old_short_stays_failed',p['status']=='failed' and p['frames']==371)
 check('old_budget_stays_closed',m.read(OLD/'closure.json')['status']=='closed_after_guard_failure')
 r=m.CpuRuntime(D/'zero.npz',out/'runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True);r.bridge=SimpleNamespace(state=states[3])
 with np.load(D/'live-terrain.npz') as f:terrain={k:f[k] for k in f.files}
 new=MinimumHeightClearance(r,terrain,rise=True);old=m.LiveClearance(r,terrain,rise=True)
 comparisons=[];rejections=[];minimum_step=float('inf')
 for i in range(4,len(states)):
  q=np.asarray(states[i]['mechanism_qpos']);prev=np.asarray(states[i-1]['mechanism_qpos'])
  current=new.check(q,prev,physical=True);minimum_step=min(minimum_step,current['minimum_height_step_m'])
  try:
   original=old.check(q,prev,physical=True);check_values=all(current[k]==v for k,v in original.items())
  except ValueError as e:
   rejections.append(dict(frame=i,error=str(e)));check_values=(i==371 and str(e)=='return_rise_bucket_vertex_descended')
  comparisons.append(check_values)
 check('all368_actual_rise_transitions_candidate_pass',len(comparisons)==368 and all(comparisons))
 check('all_unchanged_guards_equal_where_old_passed',all(comparisons))
 check('only_original_frame371_extra_height_gate_differs',rejections==[dict(frame=371,error='return_rise_bucket_vertex_descended')])
 check('actual_lowest_height_always_increased',minimum_step>0)
 with np.load(m.A/'cpu-v4/next-prepoint-plan.npz') as f:qs=f['qpos'].copy()
 a=MinimumHeightClearance(r,terrain,rise=False);b=m.LiveClearance(r,terrain,rise=False);equiv=[]
 for i in sorted(set(range(0,len(qs),120))|{2151,len(qs)-1}):
  def result(c):
   try:return ('pass',c.check(qs[i],physical=True))
   except ValueError as e:return ('reject',str(e))
  equiv.append(result(a)==result(b))
 check('APPROACH_32cm_and_static_rules_identical',all(equiv))
 check('CPU_no_physics_methods',not hasattr(new,'step') and not hasattr(new,'__enter__'))
 check('all_old_results_source_model_core_unchanged',all(m.sha(Path(k))==v for k,v in before.items()) and all(m.sha(k)==v for k,v in m.read(OLD/'protocol.json')['code_sha256'].items()))
 check('GPU_idle',not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip())
 snap=out/'code-snapshot';snap.mkdir()
 for n in ['minimum_guard.py','check_minimum_guard.py']:shutil.copyfile(W/n,snap/n)
 result=dict(status='cpu_candidate_passed',version=VERSION,checks=checks,check_count=len(checks),cached_actual_transitions=368,
  minimum_cached_lowest_height_step_m=minimum_step,old_gate_rejections=rejections,approach_equivalence_samples=len(equiv),
  code_sha256={n:m.sha(W/n) for n in ['minimum_guard.py','check_minimum_guard.py']},frozen_inputs_sha256=before,
  physics_steps=0,native_initializations=0,policy_predictions=0,training_updates=0,physical_admission=False,
  limitation='Retrospective cached CPU predicates do not establish later loaded execution or full second-scoop success. Old failed run stays failed.',wall_seconds=time.monotonic()-start)
 m.write(out/'report.json',result);signal.alarm(0);print(m.encode({k:v for k,v in result.items() if k not in ('checks','frozen_inputs_sha256')}),flush=True)
if __name__=='__main__':main()
