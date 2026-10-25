"""Small CPU-only geometry diagnosis. Never initialize a native runtime."""
from pathlib import Path
import json,sys,time,signal,importlib.util
import numpy as np
R=Path('/home/qunwang/wajueji');O=R/'work/motion-review-20261015-v1';A=R/'runs/edge-first-two-scoop-20261015-continuation';B=R/'runs/edge-adjacent-59um-20261015-v1'
sys.path[:0]=[str(R/'work/continuous-single-scoop-20261014-v1')]
from continuous_scoop import CpuRuntime,FixedRouteClearance,project,segment,read,write,sha
from planning.cycle_geometry import tool_pose,MOUTH
from planning.rear_lower_prepoint import ground_completed_terrain,sampled_surface_clearance,mechanism_and_truck_clearance
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180s_cpu_probe_ceiling')));signal.alarm(180)
source=A/'dump.npz';digest=sha(source);t=time.monotonic();r=CpuRuntime(source,O/'cpu-runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
pitch={name:[] for name in ['CLEAR_PILE','ALIGN']}
for line in (A/'geometry-actions.ndjson').open():
 x=json.loads(line)
 if x['phase'] not in pitch:continue
 actual=float(x['detail']['pitch_diagnostic_rad']);_,desired=tool_pose(r.assembly,np.asarray(x['desired']),MOUTH)
 pitch[x['phase']].append([actual,desired])
metrics={}
for name,values in pitch.items():
 p=np.unwrap(np.array(values),axis=0);error=p[:,0]-p[:,1]
 metrics[name]=dict(max_abs_pitch_tracking_error_deg=float(np.rad2deg(max(abs(error)))),actual_pitch_total_travel_deg=float(np.rad2deg(np.sum(abs(np.diff(p[:,0]))))),reference_pitch_total_travel_deg=float(np.rad2deg(np.sum(abs(np.diff(p[:,1]))))),tracking_error_peak_to_peak_deg=float(np.rad2deg(np.ptp(error))))
report=dict(status='cpu_geometry_probe',physics_steps=0,native_initializations=0,inference_calls=0,source=str(source),source_sha256=digest,pitch=metrics,attempts=[],scope='Geometry only. Existing qualified next entry is used as a fixed diagnostic target; not a new selected-point or direct-path physical qualification.')
spec=importlib.util.spec_from_file_location('motion_review_observer',R/'work/contact-fixed-truck-continuation-20260925-v2/observation.py');observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
try:
 terrain,obs=observer.stationary_terrain(r,r.arrays);report['dump_end_observation']=obs
 static=FixedRouteClearance(r);collision=ground_completed_terrain(terrain)
 class Checker:
  def check(self,q,previous=None):
   d=static.check(q,previous);_,b=r.assembly.assemble(q,include_meshes=False);points=static.samples@b['bucket'][:3,:3].T+b['bucket'][:3,3]
   gap=sampled_surface_clearance(points,collision)['clearance_m'];other=mechanism_and_truck_clearance(static.geometries,b,collision,static.truck_triangles.reshape(-1,3))
   if gap<.32 or other['mechanism_terrain_clearance_m']<.02:raise ValueError('original_stock_clearance:'+str((gap,other['mechanism_terrain_clearance_m'])))
   return dict(d,bucket_observed_terrain_m=gap,mechanism_terrain_clearance_m=other['mechanism_terrain_clearance_m'])
 start=project(r,r.arrays['mechanism_qpos'])
 with np.load(B/'scoop-2-plan/align.npz') as z:goal=project(r,z['qpos'][-1])
 report['fixed_diagnostic_goal_axes']=goal[r.chain.axes].tolist();checker=Checker()
 try:
  p=segment(r,start,goal,checker);report['attempts'].append(dict(route='single_joint_space_segment_to_existing_entry',status='sampled_geometry_passed',frames=p['steps'],minimum_clearance=p['minimum_clearance']))
 except ValueError as e:report['attempts'].append(dict(route='single_joint_space_segment_to_existing_entry',status='rejected',failure=str(e)))
except ValueError as e:report['observation_or_geometry_failure']=str(e)
assert r.frames==0 and r.bridge is None and sha(source)==digest
report['wall_seconds']=time.monotonic()-t;write(O/'geometry-probe.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))
