"""Read-only audit of D196 physical logs; never enter a native runtime."""
from pathlib import Path
import importlib.util,json,sys,time,numpy as np
R=Path('/home/qunwang/wajueji');W=R/'work/motion-physical-short-20261015-v2';C=R/'work/motion-review-20261015-v1';D=R/'runs/motion-physical-short-20261015-v2'
sys.path[:0]=[str(C),str(R/'work/continuous-single-scoop-20261014-v1')]
from continuous_scoop import *
from motion_candidate import ReturnClearance
W=R/'work/motion-physical-short-20261015-v2';C=R/'work/motion-review-20261015-v1';D=R/'runs/motion-physical-short-20261015-v2'
def main():
 p=read(D/'report.json');protocol=read(W/'preflight.json');budget=read(W/'budget.json');assert (W/'physical-finished.json').exists()
 checks=[];metrics={};t=time.monotonic()
 def check(name,value):checks.append(dict(check=name,passed=bool(value)))
 check('budget identity',sha(W/'budget.json')==protocol['budget_sha256']);check('finished before original deadline',p['finished_epoch']<=budget['epoch_deadline'])
 check('all input and controller bindings unchanged',all(sha(path)==h for path,h in protocol['bindings_sha256'].items()))
 check('no inference or training',p['inference_calls']==p['training_updates']==0)
 for item in p['cases']:
  folder=D/item['name'];case=next(x for x in protocol['cases'] if x['name']==item['name']);check(item['name']+'frame ceiling',item['frames']<=case['max_frames'])
  check(item['name']+'native zero',all(v for k,v in item.get('zero_arrays_exact',{}).items() if k!='simulation_time_s') and item.get('zero_time_error_s',1)<1e-12 and len(item.get('zero_arrays_exact',{}))==24)
  check(item['name']+'prefix gate',len(item['prefix_checks'])==3 and all(all(x[k] for k in ['command_exact','particles_exact','state_exact']) for x in item['prefix_checks']))
  states=read(folder/'display/states.json');cloud=np.load(folder/'display/particles.npy',mmap_mode='r');expected=read(Path(case['prefix'])/'states.json');expected_cloud=np.load(Path(case['prefix'])/'particles.npy',mmap_mode='r')
  check(item['name']+'source prefix independently exact',states[:4]==expected and np.array_equal(cloud[:4],expected_cloud))
  frames=[json.loads(x) for x in (folder/'physics/frames.ndjson').read_text().splitlines()];dispatch=[json.loads(x) for x in (folder/'dispatch.ndjson').read_text().splitlines()]
  check(item['name']+'all executed physical frames logged',len(frames)==item['frames'] and len(states)==item['frames']+1 and cloud.shape[0]==len(states))
  check(item['name']+'hard material and sweep guards',all(x['hard_guard'] and x['hencky_material_guard']['safe'] and x['moving_bucket_sweep']['safe'] for x in frames))
  check(item['name']+'physical clock continuous',max(abs((s['time_s']-states[0]['time_s'])-i/60) for i,s in enumerate(states))<1e-7)
  check(item['name']+'dispatch matches actual physical commands',all(np.array_equal(row['applied_command_si'],dispatch[i]['command']) for i,row in enumerate(frames)))
  actions=[json.loads(x) for x in (folder/'actions.ndjson').read_text().splitlines()] if (folder/'actions.ndjson').exists() else []
  if actions:
   r=CpuRuntime(folder/'prefix3.npz',W/('audit-'+item['name']),enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
   if item['name']=='loaded':checker=FixedRouteClearance(r)
   else:
    spec=importlib.util.spec_from_file_location('d196_audit_observer',R/'work/contact-fixed-truck-continuation-20260925-v2/observation.py');observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer);terrain,_=observer.stationary_terrain(r,r.arrays);checker=ReturnClearance(r,terrain)
   controller=TransportController(r,checker,case['commands'][-1]);errors=[];bias=[];pitch=[];actual_pitch=[];stage_counts={}
   for x in actions:
    q=np.asarray(x['before_state']['mechanism_qpos']);cmd,detail=controller.command(x['phase'],q,x['before_state']['controller_target'],x['desired'],x['velocity']);errors.append(float(np.max(abs(cmd-x['command']))));bias.append(float(np.max(abs(np.asarray(detail['load_bias'])-x['detail']['load_bias']))))
    _,ref_pitch=tool_pose(r.assembly,np.asarray(x['desired']),MOUTH);angle=float(x['detail']['pitch_diagnostic_rad']);pitch.append([x['phase'],float(np.rad2deg(np.arctan2(np.sin(angle-ref_pitch),np.cos(angle-ref_pitch))))]);stage_counts[x['phase']]=stage_counts.get(x['phase'],0)+1
   check(item['name']+'all controller commands recomputed',max(errors)<1e-12);check(item['name']+'load bias continuity recomputed',max(bias)<1e-12)
   metrics[item['name']]=dict(command_max_error=max(errors),load_bias_max_error=max(bias),phase_max_pitch_error_deg={name:max(abs(e) for phase,e in pitch if phase==name) for name in stage_counts},phase_samples=stage_counts)
   if item['name']=='loaded':check('loaded bucket commands strictly zero',all(x['command'][3]==0 for x in actions))
  if item['status']=='completed':check(item['name']+'checkpoint identity',sha(item['final_checkpoint']['path'])==item['final_checkpoint']['sha256'])
 record=all(x['passed'] for x in checks);result=dict(status='audit_passed' if record else 'audit_failed',record_integrity_passed=record,checks=checks,checks_passed=sum(x['passed'] for x in checks),metrics=metrics,physical_status=p['status'],failure=p.get('failure'),cases_completed=[x['name'] for x in p['cases'] if x['status']=='completed'],physical_frames=p['physical_frames'],native_initializations=p['native_initializations'],new_audit_physics_steps=0,wall_seconds=time.monotonic()-t,limitation='Record audit and task completion are separate. Cavity particle diagnostics are not calibrated fill or a no-spill certificate.')
 write(D/'read-only-audit.json',result);print(encode({k:result[k] for k in ['status','checks_passed','physical_status','failure','metrics','wall_seconds']}),flush=True)
if __name__=='__main__':main()
