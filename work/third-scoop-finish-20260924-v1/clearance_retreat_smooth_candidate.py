"""One bounded contact-exit lift; never relax the free-space transport gate."""
import argparse,json,time
from pathlib import Path
import numpy as np
from layered_47_cycle_runtime import CycleRuntime
from layered_47_cycle_geometry import TransportClearance
from planning.cycle_geometry import MOUTH,tool_pose,solve_pose
from planning.approach import sampled_approach
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');args=p.parse_args()
r=CycleRuntime(args.source,args.output);out=Path(args.output);checker=TransportClearance(r)
start=r.arrays['mechanism_qpos'].copy();r.chain.q=start.copy();start=r.chain.advance(np.zeros(4),1/60)[0]
origin,pitch=tool_pose(r.assembly,start,MOUTH);initial_pose=r.assembly.assemble(start,include_meshes=False)[1]['bucket'];initial_samples=checker.samples@initial_pose[:3,:3].T+initial_pose[:3,3]
initial,_=checker.measure_clearance(start)
# Preserve loaded target-to-actual offsets; do not issue a reverse correction
# toward the measured start before the upward ramp has begun.
target_offset=np.asarray(r.metadata['controller_state']['controller_target'])-start[r.chain.axes]
report=dict(schema='bounded_loaded_clearance_retreat_v1',status='planning',source=r.metadata['source_identity'],initial_clearance=initial,protocol=dict(boom_extension_m=.025,maximum_frames=600,maximum_wall_seconds=180,xy_drift_m=.20,minimum_sample_rise_m=-.002,minimum_endpoint_rise_m=.15,maximum_tracking_error_m=.015,minimum_endpoint_terrain_clearance_m=.15,preserve_initial_controller_target_offset=True,free_space_gate_unchanged=True,training_eligible=False))
def write(): (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
def check(q,previous=None,physical=False):
 values,mesh=checker.measure_clearance(q);pos,angle=tool_pose(r.assembly,q,MOUTH)
 pose=r.assembly.assemble(q,include_meshes=False)[1]['bucket'];points=checker.samples@pose[:3,:3].T+pose[:3,3]
 if values['bucket_truck_m']<.04 or values['mechanism_truck_m']<.02 or values['mechanism_ground_m']<.02 or values['closure_m']>(.001 if physical else .0001):raise ValueError('retreat_geometry:'+str(values))
 if values['bucket_terrain_m']<initial['bucket_terrain_m']-.002:raise ValueError('retreat_worsens_clearance:'+str(values))
 if np.max(np.linalg.norm(points[:,:2]-initial_samples[:,:2],axis=1))>.20 or np.min(points[:,2]-initial_samples[:,2])<-.002:raise ValueError('retreat_not_upward')
 if abs(angle-pitch)>.15:raise ValueError('retreat_pitch')
 if previous is not None:
  oldpos,_=tool_pose(r.assembly,previous,MOUTH)
  if pos[2]<oldpos[2]-1e-6:raise ValueError('retreat_descending')
 return values
write();started=time.monotonic()
try:
 goal=start.copy()
 for _ in range(150):
  r.chain.q=goal.copy();goal=r.chain.advance(np.array([0.,.01,0.,0.]),1/60)[0]
 def advance(q,v,dt):r.chain.q=q.copy();return r.chain.advance(v,dt)[0]
 path=sampled_approach(start,goal,axes=r.chain.axes,speed_limits=np.array([.05,.012,.012,.02]),advance=advance,check_pose=check,dt=1/60,max_steps=570,validation_substeps=2)
 report.update(status='geometry_passed',planned_frames=path['steps'],minimum_clearance={k:min(d[k] for d in path['diagnostics']) for k in initial});write()
 checker.check(goal)
 if checker.material_clearance(checker.samples@r.assembly.assemble(goal,include_meshes=False)[1]['bucket'][:3,:3].T+r.assembly.assemble(goal,include_meshes=False)[1]['bucket'][:3,3])<.15:raise ValueError('retreat_goal_clearance')
 if args.execute:
  with r:
   for i in range(len(path['velocity'])+30):
    if r.frames>=600 or time.monotonic()-started>180:raise RuntimeError('retreat_budget')
    if i<len(path['velocity']):
     desired=path['qpos'][i+1];target=np.asarray(r.bridge.state['controller_target']);command=np.clip(path['velocity'][i]+4*(desired[r.chain.axes]+target_offset-target),[-.12,-.0225,-.0225,-.0675],[.12,.0225,.0225,.0675])
    else:desired=goal;command=np.zeros(4)
    row=r.step(command);actual=np.asarray(r.bridge.state['mechanism_qpos']);check(actual,physical=True)
    if np.max(np.abs(actual[r.chain.axes]-desired[r.chain.axes]))>.015:raise ValueError('retreat_tracking')
    if r.frames%60==0:print('RETREAT',r.frames,flush=True)
   final=checker.check(actual,physical=True);position,_=tool_pose(r.assembly,actual,MOUTH)
   if position[2]-origin[2]<.15 or final['bucket_terrain_m']<.15:raise ValueError('retreat_endpoint')
   report.update(status='retreat_passed',final_clearance=final,actual_rise_m=float(position[2]-origin[2]),frames=r.frames,checkpoint=r.save(out/'cleared.npz',stage='transport'))
except BaseException as error:
 report.update(status='failed',failure=repr(error),frames=r.frames);raise
finally:
 report['wall_seconds']=time.monotonic()-started;write()
print(json.dumps(report,indent=2))
