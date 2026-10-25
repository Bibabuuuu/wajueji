"""CPU-only minimal truck exit: allow unloaded curl while rising, then approach."""
from pathlib import Path
import importlib.util,json,signal,sys,time
import numpy as np
W=Path('/home/qunwang/wajueji/work/motion-review-20261015-v1');sys.path.insert(0,str(W))
from motion_candidate import *
from planning.cycle_geometry import solve_pose

def main():
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('cpu_v2_240s_ceiling')));signal.alarm(240)
 O=W/'return-candidate-v2';O.mkdir(exist_ok=False);t=time.monotonic();A=R/'runs/edge-first-two-scoop-20261015-continuation';B=R/'runs/edge-adjacent-59um-20261015-v1'
 source=A/'dump.npz';identity=sha(source);r=CpuRuntime(source,O/'runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
 original=read(W/'candidate-v1/report.json');report=dict(status='cpu_return_search_running',physics_steps=0,native_initializations=0,inference_calls=0,training_updates=0,physical_admission=False,source=str(source),source_sha256=identity,prior_failed_candidate_sha256=sha(W/'candidate-v1/report.json'),attempts=[],scope='Unloaded truck exit may curl modestly; the loaded cylinder remains locked. No protection changes.')
 spec=importlib.util.spec_from_file_location('v2_observer',R/'work/contact-fixed-truck-continuation-20260925-v2/observation.py');observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
 terrain,obs=observer.stationary_terrain(r,r.arrays);first=read(A/'scoop-1-plan/actual-cut-trace.json')['trace'][0]['mouth_xy'];ranked,_=rank_adjacent(terrain,first,inward_sign=-1);selected=next(x for x in ranked if abs(x['adjacency']['nominal_entry_boundary_gap_m'])<=float(terrain['resolution_m']))
 assert selected==original['selected'],'fresh_observation_or_selection_drift'
 heading=float(np.arctan2(*np.asarray(selected['entry_xy_m'])[::-1]));prior=read(B/'scoop-2-plan/report.json')['selected'];old_heading=float(np.arctan2(*np.asarray(prior['entry_xy_m'])[::-1]))
 with np.load(B/'scoop-2-plan/align.npz') as z:reference=z['reference_q'].copy()
 reference[:,7]+=heading-old_heading;goal=reference[0].copy();goal[9]+=.001;goal=project(r,goal);start=project(r,r.arrays['mechanism_qpos']);checker=ReturnClearance(r,terrain)
 end_admission=checker.full_roi_admission(goal);poses=[r.assembly.assemble(q,include_meshes=False)[1]['bucket'] for q in reference];alignment=ProgressAlignment(reference,poses).observe(goal,r.assembly.assemble(goal,include_meshes=False)[1]['bucket'],0.);assert alignment['trusted']
 mouth,pitch=tool_pose(r.assembly,start,MOUTH);radial=mouth[:2]/np.linalg.norm(mouth[:2]);solutions=[]
 for rise,retreat,curl in [(.25,0.,.15),(.35,.2,.3),(.5,.35,.3)]:
  label=f'rise{rise}_retreat{retreat}_curl{curl}'
  try:
   target=mouth+np.r_[-retreat*radial,rise];exit_goal=solve_pose(r.assembly,r.chain,start,target,pitch+curl,MOUTH);checker.check(exit_goal)
   first_path=smooth_path(r,start,exit_goal,checker);solutions.append(dict(label=label,target=target.tolist(),exit_q=exit_goal,first_path=first_path))
   print('EXIT_GEOMETRY_PASSED',label,flush=True);break
  except ValueError as e:report['attempts'].append(dict(route=label,status='exit_rejected',failure=str(e)));write(O/'report.json',report)
 accepted=None
 for solution in solutions:
  exit_goal=solution['exit_q'];label=solution['label']
  try:
   rest=smooth_path(r,exit_goal,goal,checker);accepted=[solution['first_path'],rest];report['accepted_route']=label+'_direct_entry';break
  except ValueError as e:report['attempts'].append(dict(route=label+'_direct_entry',status='rejected',failure=str(e)));write(O/'report.json',report)
  try:
   high=solve_pose(r.assembly,r.chain,exit_goal,np.r_[5.2*np.array([np.cos(heading),np.sin(heading)]),2.7],-.32,TIP)
   middle=smooth_path(r,exit_goal,high,checker);last=smooth_path(r,high,goal,checker);accepted=[solution['first_path'],middle,last];report['accepted_route']=label+'_high_entry';break
  except ValueError as e:report['attempts'].append(dict(route=label+'_high_entry',status='rejected',failure=str(e)));write(O/'report.json',report)
 if accepted:
  qs=np.concatenate([p['qpos'] if i==0 else p['qpos'][1:] for i,p in enumerate(accepted)]);vs=np.concatenate([p['velocity'] for p in accepted]);np.savez(O/'return-plan.npz',qpos=qs,velocity=vs,reference_q=reference)
  report.update(status='cpu_return_geometry_passed',selected=selected,fresh_observation=obs,reference_alignment=alignment,full_roi_reference_admission=end_admission,reference_frames=len(vs),nominal_reference_seconds=len(vs)/60,intermediate_waypoints=len(accepted)-1,segments=[{k:v for k,v in p.items() if k not in ['qpos','velocity']} for p in accepted],bucket_cylinder_travel_m=float(np.sum(abs(np.diff(qs[:,15])))),original_after_unload_to_next_dig_s=original['original_after_unload_to_next_dig_s'],guards=original['guard_semantics'],physical_or_fill_improvement_verified=False)
 else:report.update(status='cpu_return_rejected',fallback='Existing verified route unchanged. Do not admit a rejected shortcut.')
 assert r.bridge is None and r.frames==0 and sha(source)==identity
 report['wall_seconds']=time.monotonic()-t;write(O/'report.json',report);print(json.dumps({k:report.get(k) for k in ['status','accepted_route','nominal_reference_seconds','intermediate_waypoints','attempts','wall_seconds']},indent=2))

if __name__=='__main__':main()
