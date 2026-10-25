"""Geometric last-waypoint repair: end descent at existing safe handoff."""
import numpy as np
from continuous_scoop import FixedRouteClearance,ManualResidualEnv,ProgressAlignment,project,segment
from planning.rear_lower_prepoint import ground_completed_terrain,sampled_surface_clearance,mechanism_and_truck_clearance

def direct_bridge(r,terrain,path,read_directory):
 assert path['segments'][-1]['name']=='REAR_LOWER_PREPOINT'
 old=path['segments'][-1];offset=len(path['velocity'])-old['frames'];start=path['qpos'][offset].copy()
 reference=ManualResidualEnv(read_directory,None,collect=False,waiting_action_mode='error_strength_v1').reference_q.copy()
 angle=float(path['qpos'][-1,7]-reference[0,7]);reference[:,7]+=angle;goal=reference[0].copy();goal[9]+=.001;goal=project(r,goal)
 poses=[r.assembly.assemble(q,include_meshes=False)[1]['bucket'] for q in reference]
 alignment=ProgressAlignment(reference,poses).observe(goal,r.assembly.assemble(goal,include_meshes=False)[1]['bucket'],0.)
 if not alignment['trusted']:raise ValueError('direct_reference_goal_untrusted')
 terrain=ground_completed_terrain(terrain)
 class Checker(FixedRouteClearance):
  def check(self,q,previous=None,*,physical=False):
   d=super().check(q,previous,physical=physical);_,b=self.runtime.assembly.assemble(q,include_meshes=False);pose=b['bucket'];points=self.samples@pose[:3,:3].T+pose[:3,3]
   gap=sampled_surface_clearance(points,terrain)['clearance_m'];other=mechanism_and_truck_clearance(self.geometries,b,terrain,self.truck_triangles.reshape(-1,3))
   if gap<.32 or other['mechanism_terrain_clearance_m']<.02:raise ValueError('unchanged_original_approach_gate')
   if not self.runtime.projector.physical_bounds_ok(np.asarray(q)):raise ValueError('unchanged_original_joint_bounds')
   return dict(d,bucket_observed_terrain_m=gap,mechanism_terrain_clearance_m=other['mechanism_terrain_clearance_m'])
 bridge=segment(r,start,goal,Checker(r))
 if bridge['minimum_clearance']['bucket_observed_terrain_m']<.325:raise ValueError('direct_bridge_inadequate_positive_margin')
 result=dict(path);result['qpos']=np.concatenate([path['qpos'][:offset+1],bridge['qpos'][1:]]);result['velocity']=np.concatenate([path['velocity'][:offset],bridge['velocity']])
 replacement=dict(old,frames=bridge['steps'],goal_qpos=goal.tolist(),minimum_bucket_terrain_clearance_m=bridge['minimum_clearance']['bucket_observed_terrain_m'],endpoint_bucket_terrain_clearance_m=bridge['minimum_clearance']['bucket_observed_terrain_m'],target_semantics='Existing reference handoff plus1mm boom, not old near-threshold prepoint')
 result['segments']=path['segments'][:-1]+[replacement];result['total_frames']=len(result['velocity']);result['direct_reference_bridge']=dict(alignment=alignment,minimum_clearance=bridge['minimum_clearance'],old_waypoint_replaced_not_protection_changed=True,reference_cylinders_commands_policy_unchanged=True)
 assert len(result['qpos'])==len(result['velocity'])+1
 return result
