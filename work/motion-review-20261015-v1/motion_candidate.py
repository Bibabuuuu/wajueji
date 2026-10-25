"""D194 CPU-only return/transport candidate; no native entry or policy inference."""
from pathlib import Path
import csv,importlib.util,json,signal,sys,time
import numpy as np
R=Path('/home/qunwang/wajueji');W=R/'work/motion-review-20261015-v1'
sys.path[:0]=[str(R/'work/continuous-single-scoop-20261014-v1'),str(R/'work/edge-first-strips-20261015-v1')]
from continuous_scoop import CpuRuntime,FixedRouteClearance,project,read,write,sha,ProgressAlignment
from layered_47_cycle_geometry import TransportClearance
from planning.cycle_geometry import tool_pose,MOUTH,TIP
from planning.rear_lower_prepoint import ground_completed_terrain,mechanism_and_truck_clearance,sampled_surface_clearance
from adjacent_selection import rank_adjacent
from transport_endpoint_candidate_v2 import locked_lift_pose
from contact_geometric_feedback import ContactGeometryFeedback,FeedbackConfig
from contact_bc_ground_guard import BucketGroundGuard
SPEED=np.array([.1,.018,.018,.018]);COMMAND_LIMITS=np.array([.12,.0225,.0225,.0225]);ACCEL=COMMAND_LIMITS

def smooth_path(r,start,goal,checker,*,duration_scale=1.,max_steps=5000):
 """Quintic rest-to-rest reference, with existing120Hz geometry checks."""
 start=np.asarray(start,float);goal=np.asarray(goal,float);delta=goal[r.chain.axes]-start[r.chain.axes];dt=1/60
 duration=max(dt,float(np.max(1.875*abs(delta)/SPEED)),float(np.sqrt(np.max((10/np.sqrt(3))*abs(delta)/ACCEL))))*duration_scale
 n=int(np.ceil(duration/dt))
 if n>max_steps:raise ValueError('cpu_path_frame_ceiling')
 q=start.copy();qs=[q.copy()];vel=[];diagnostics=[checker.check(q)];previous=np.zeros(4)
 for i in range(1,n+1):
  s=i/n;desired=start[r.chain.axes]+(10*s**3-15*s**4+6*s**5)*delta;v=(desired-q[r.chain.axes])/dt
  if np.max(abs(v)/SPEED)>1+1e-8 or np.max(abs((v-previous)/dt)/ACCEL)>1+1e-8:raise ValueError('original_speed_or_slew_bound')
  for _ in range(2):
   old=q.copy();r.chain.q=q.copy();q=r.chain.advance(v,dt/2)[0];diagnostics.append(checker.check(q,old))
  qs.append(q.copy());vel.append(v);previous=v
 if np.max(abs(q-goal))>1e-5:raise ValueError('original_passive_branch_mismatch')
 if np.max(abs(previous/dt)/ACCEL)>1+1e-8:raise ValueError('terminal_slew_bound')
 finite={k:min(d[k] for d in diagnostics) for k in diagnostics[0] if diagnostics[0][k] is not None}
 return dict(qpos=np.asarray(qs),velocity=np.asarray(vel),steps=n,seconds=n/60,minimum_clearance=finite,geometry_samples=len(diagnostics),endpoint_velocity=0.,endpoint_acceleration=0.,polynomial='10s^3-15s^4+6s^5',continuous_collision_proven=False,physical_admission=False)

class ReturnClearance(FixedRouteClearance):
 def __init__(self,r,terrain):
  # Existing transport helper explicitly models the known rigid ground plane
  # outside the stock ROI. This is not an expanded sensor observation.
  TransportClearance.__init__(self,r,terrain=ground_completed_terrain(terrain))
 material_clearance=TransportClearance.material_clearance
 def check(self,q,previous=None,*,physical=False):
  d=super().check(q,previous,physical=physical)
  if not self.runtime.projector.physical_bounds_ok(q):raise ValueError('original_physical_bounds')
  if d['bucket_terrain_m']<.32:raise ValueError('original_bucket_stock32cm:'+str(d['bucket_terrain_m']))
  _,b=self.runtime.assembly.assemble(q,include_meshes=False);other=mechanism_and_truck_clearance(self.geometries,b,self.terrain,self.truck_triangles.reshape(-1,3))
  if other['mechanism_terrain_clearance_m']<.02:raise ValueError('original_mechanism_stock2cm')
  return dict(d,mechanism_terrain_clearance_m=other['mechanism_terrain_clearance_m'])
 def full_roi_admission(self,q):
  _,b=self.runtime.assembly.assemble(q,include_meshes=False);pose=b['bucket'];points=self.samples@pose[:3,:3].T+pose[:3,3]
  result=sampled_surface_clearance(points,self.terrain)
  if result['clearance_m']<.325:raise ValueError('original_reference_positive_stock_margin')
  return result

def main():
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('D194_cpu240s_ceiling')));signal.alarm(240)
 start_time=time.monotonic();A=R/'runs/edge-first-two-scoop-20261015-continuation';B=R/'runs/edge-adjacent-59um-20261015-v1';O=W/'candidate-v1';O.mkdir(exist_ok=False)
 source=A/'dump.npz';source_sha=sha(source);r=CpuRuntime(source,O/'runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
 report=dict(status='cpu_candidate_running',source=str(source),source_sha256=source_sha,physics_steps=0,native_initializations=0,inference_calls=0,training_updates=0,physical_admission=False,attempts=[])
 write(O/'report.json',report)
 spec=importlib.util.spec_from_file_location('d194_observer',R/'work/contact-fixed-truck-continuation-20260925-v2/observation.py');observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
 terrain,observation=observer.stationary_terrain(r,r.arrays);first=read(A/'scoop-1-plan/actual-cut-trace.json')['trace'][0]['mouth_xy']
 candidates,rejected=rank_adjacent(terrain,first,inward_sign=-1);eligible=[x for x in candidates if abs(x['adjacency']['nominal_entry_boundary_gap_m'])<=float(terrain['resolution_m'])]
 if not eligible:raise ValueError('no_fresh_adjacent_candidate')
 selected=eligible[0];heading=float(np.arctan2(*np.asarray(selected['entry_xy_m'])[::-1]));prior=read(B/'scoop-2-plan/report.json')['selected'];prior_heading=float(np.arctan2(*np.asarray(prior['entry_xy_m'])[::-1]))
 with np.load(B/'scoop-2-plan/align.npz') as z:reference=z['reference_q'].copy()
 reference[:,7]+=heading-prior_heading;goal=reference[0].copy();goal[9]+=.001;goal=project(r,goal);initial=project(r,r.arrays['mechanism_qpos']);checker=ReturnClearance(r,terrain)
 poses=[r.assembly.assemble(q,include_meshes=False)[1]['bucket'] for q in reference];alignment=ProgressAlignment(reference,poses).observe(goal,r.assembly.assemble(goal,include_meshes=False)[1]['bucket'],0.)
 assert alignment['trusted'];admission=checker.full_roi_admission(goal)
 report.update(selected=selected,fresh_observation=observation,candidate_count=len(eligible),reference_alignment=alignment,full_roi_reference_admission=admission,original_after_unload_to_next_dig_s=(10504+4870-6549)/60,guard_semantics='Whole route static CAD4cm/2cm, bounds, closure, stock32cm and mechanism2cm. Outside stock ROI uses existing transport known-ground semantics. Terminal full ROI unchanged. No claim of full dynamic avoidance.')
 routes=[('DIRECT',[])]
 mouth,_=tool_pose(r.assembly,initial,MOUTH)
 for rise in [.2,.4]:
  try:
   exit_goal,detail=locked_lift_pose(r.assembly,r.chain,initial,mouth+np.array([0,0,rise]),MOUTH)
   routes.append(('OPEN_EXIT_'+str(rise),[exit_goal]));report.setdefault('exit_ik',[]).append(dict(rise_m=rise,detail=detail))
  except ValueError as e:report.setdefault('exit_ik_rejected',[]).append(dict(rise_m=rise,failure=str(e)))
 accepted=None
 for label,via in routes:
  path=[];point=initial.copy()
  try:
   for target in [*via,goal]:
    p=smooth_path(r,point,target,checker);path.append(p);point=target
   accepted=path;report['accepted_route']=label;break
  except ValueError as e:report['attempts'].append(dict(route=label,status='rejected',failure=str(e)));write(O/'report.json',report)
 if accepted is None:
  report['status']='cpu_return_rejected';report['fallback']='Existing verified route remains available; no admission of failed shortcut.'
 else:
  qs=np.concatenate([p['qpos'] if i==0 else p['qpos'][1:] for i,p in enumerate(accepted)]);vs=np.concatenate([p['velocity'] for p in accepted]);np.savez(O/'return-plan.npz',qpos=qs,velocity=vs,reference_q=reference)
  report.update(status='cpu_return_geometry_passed',reference_frames=len(vs),nominal_reference_seconds=len(vs)/60,intermediate_waypoints=len(accepted)-1,segments=[{k:v for k,v in p.items() if k not in ['qpos','velocity']} for p in accepted],endpoint_hold_frames=30,limitations='Nominal kinematics only. No settling, material response, online selection drift or frozen-policy success qualification. Timing is not a physical speedup measurement.')
 # Carry profile uses exactly the old held-cylinder endpoint and branch.
 with np.load(A/'transport-plan/plan.npz') as z:
  old=z['CLEAR_PILE_qpos'].copy();old_v=z['CLEAR_PILE_velocity'].copy()
 held=old[0,15];assert np.max(abs(old[:,15]-held))<1e-8
 class CarryChecker:
  def __init__(self):
   self.static=FixedRouteClearance(r);mouth,_=tool_pose(r.assembly,old[0],MOUTH)
   self.feedback=ContactGeometryFeedback(r.assembly,r.chain,BucketGroundGuard(r.assembly,old[0]),dict(entry_xy_m=mouth[:2],surface_z_m=mouth[2]),old[0],FeedbackConfig(late_curl_no_down=True))
  def check(self,q,previous=None):
   d,_=self.static.measure_clearance(q)
   if min(d['bucket_truck_m']-.04,d['mechanism_truck_m']-.02,d['mechanism_ground_m']-.02,.0001-d['closure_m'])<0:raise ValueError('original_carry_static_gate')
   if abs(q[15]-held)>1e-8:raise ValueError('carry_cylinder_changed')
   if not r.projector.physical_bounds_ok(q):raise ValueError('carry_physical_bounds')
   if previous is not None:
    if np.mean(self.feedback.body_heights(q)-self.feedback.body_heights(previous))<-1e-6:raise ValueError('original_carry_direction')
   return d
 try:
  carry=smooth_path(r,old[0],old[-1],CarryChecker(),max_steps=2400);np.savez(O/'carry-reference.npz',qpos=carry['qpos'],velocity=carry['velocity'])
  report['carry_reference']=dict(status='cpu_held_cylinder_geometry_passed',frames=carry['steps'],original_frames=len(old_v),cylinder_m=float(held),active_bucket_velocity_max=float(np.max(abs(carry['velocity'][:,3]))),profile=carry['polynomial'],minimum_clearance=carry['minimum_clearance'],damping_effect_verified=False)
 except ValueError as e:report['carry_reference']=dict(status='cpu_carry_rejected',failure=str(e))
 assert r.frames==0 and r.bridge is None and sha(source)==source_sha
 report['wall_seconds']=time.monotonic()-start_time;write(O/'report.json',report);print(json.dumps({k:report.get(k) for k in ['status','accepted_route','nominal_reference_seconds','attempts','carry_reference','wall_seconds']},indent=2))

if __name__=='__main__':main()
