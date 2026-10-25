"""Reuse the qualified outward approach waypoint; remove the old return loop."""
from pathlib import Path
import importlib.util,json,signal,sys,time
import numpy as np
W=Path('/home/qunwang/wajueji/work/motion-review-20261015-v1');sys.path.insert(0,str(W))
from motion_candidate import *
from planning.cycle_geometry import solve_pose

def main():
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('cpu_v3_240s_ceiling')));signal.alarm(240)
 O=W/'return-candidate-v3';O.mkdir(exist_ok=False);t=time.monotonic();A=R/'runs/edge-first-two-scoop-20261015-continuation';B=R/'runs/edge-adjacent-59um-20261015-v1'
 source=A/'dump.npz';identity=sha(source);r=CpuRuntime(source,O/'runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True);prior=read(W/'candidate-v1/report.json')
 report=dict(status='cpu_v3_running',physics_steps=0,native_initializations=0,inference_calls=0,training_updates=0,physical_admission=False,attempts=[],source=str(source),source_sha256=identity)
 write(O/'report.json',report)
 spec=importlib.util.spec_from_file_location('v3_observer',R/'work/contact-fixed-truck-continuation-20260925-v2/observation.py');observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
 terrain,obs=observer.stationary_terrain(r,r.arrays);first=read(A/'scoop-1-plan/actual-cut-trace.json')['trace'][0]['mouth_xy'];ranked,_=rank_adjacent(terrain,first,inward_sign=-1);selected=next(x for x in ranked if abs(x['adjacency']['nominal_entry_boundary_gap_m'])<=float(terrain['resolution_m']))
 assert selected==prior['selected'];heading=float(np.arctan2(*np.asarray(selected['entry_xy_m'])[::-1]));plan=read(B/'scoop-2-plan/report.json');old_heading=float(np.arctan2(*np.asarray(plan['selected']['entry_xy_m'])[::-1]));angle=heading-old_heading
 index=0
 for item in plan['segments']:
  index+=item['frames']
  if item['name']=='OUTWARD_HIGH':break
 else:raise ValueError('qualified_outward_waypoint_missing')
 with np.load(B/'scoop-2-plan/approach.npz') as z:outward=z['qpos'][index].copy()
 with np.load(B/'scoop-2-plan/align.npz') as z:reference=z['reference_q'].copy()
 reference[:,7]+=angle;outward[7]+=angle;outward=project(r,outward);goal=reference[0].copy();goal[9]+=.001;goal=project(r,goal);start=project(r,r.arrays['mechanism_qpos']);checker=ReturnClearance(r,terrain)
 final=smooth_path(r,outward,goal,checker);report['outward_to_reference_geometry_passed']=True;write(O/'report.json',report)
 mouth,pitch=tool_pose(r.assembly,start,MOUTH);radial=mouth[:2]/np.linalg.norm(mouth[:2]);accepted=None
 for rise,retreat,curl in [(.25,0.,.15),(.35,.2,.3),(.5,.35,.3)]:
  name=f'exit_rise{rise}_retreat{retreat}_curl{curl}'
  try:
   exit_goal=solve_pose(r.assembly,r.chain,start,mouth+np.r_[-retreat*radial,rise],pitch+curl,MOUTH);first_path=smooth_path(r,start,exit_goal,checker)
  except ValueError as e:report['attempts'].append(dict(route=name,status='exit_rejected',failure=str(e)));write(O/'report.json',report);continue
  try:
   middle=smooth_path(r,exit_goal,outward,checker);accepted=[first_path,middle,final];report['accepted_route']=name+'_coordinated_outward';break
  except ValueError as e:report['attempts'].append(dict(route=name+'_coordinated_outward',status='rejected',failure=str(e)));write(O/'report.json',report)
  yaw_goal=exit_goal.copy();yaw_goal[7]=heading;yaw_goal=project(r,yaw_goal)
  try:
   yaw_path=smooth_path(r,exit_goal,yaw_goal,checker);middle=smooth_path(r,yaw_goal,outward,checker);accepted=[first_path,yaw_path,middle,final];report['accepted_route']=name+'_clear_yaw_outward';break
  except ValueError as e:report['attempts'].append(dict(route=name+'_clear_yaw_outward',status='rejected',failure=str(e)));write(O/'report.json',report)
 if accepted:
  qs=np.concatenate([p['qpos'] if i==0 else p['qpos'][1:] for i,p in enumerate(accepted)]);vs=np.concatenate([p['velocity'] for p in accepted]);np.savez(O/'return-plan.npz',qpos=qs,velocity=vs,reference_q=reference)
  poses=[r.assembly.assemble(q,include_meshes=False)[1]['bucket'] for q in reference];alignment=ProgressAlignment(reference,poses).observe(goal,r.assembly.assemble(goal,include_meshes=False)[1]['bucket'],0.);assert alignment['trusted']
  report.update(status='cpu_return_geometry_passed',selected=selected,fresh_observation=obs,reference_alignment=alignment,full_roi_reference_admission=checker.full_roi_admission(goal),reference_frames=len(vs),nominal_reference_seconds=len(vs)/60,intermediate_waypoints=len(accepted)-1,segments=[{k:v for k,v in p.items() if k not in ['qpos','velocity']} for p in accepted],segment_labels=['unloaded_exit','coordinated_outward','reference_entry'] if len(accepted)==3 else ['unloaded_exit','clear_truck_yaw','coordinated_outward','reference_entry'],bucket_cylinder_travel_m=float(np.sum(abs(np.diff(qs[:,15])))),original_after_unload_to_next_dig_s=prior['original_after_unload_to_next_dig_s'],guards=prior['guard_semantics'],original_lower_reference_unchanged=True,physical_or_fill_improvement_verified=False,limitations='Nominal geometry only; no physical speed, damping or fill improvement proven. Outward waypoint retained to respect the original32cm margin. No extra per-waypoint hold inserted by this CPU planner.')
 else:report.update(status='cpu_return_rejected',fallback='Existing verified route unchanged; no admitted shortcut.')
 assert r.bridge is None and r.frames==0 and sha(source)==identity
 report['wall_seconds']=time.monotonic()-t;write(O/'report.json',report);print(json.dumps({k:report.get(k) for k in ['status','accepted_route','nominal_reference_seconds','intermediate_waypoints','attempts','wall_seconds']},indent=2))

if __name__=='__main__':main()
