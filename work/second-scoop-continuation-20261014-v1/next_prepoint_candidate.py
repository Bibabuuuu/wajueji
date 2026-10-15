"""CPU candidate: current stock clearance plus existing fixed-route CAD checks.

Reuse existing staged approach. Refine broad truck AABBs with the existing
2 cm expanded-box/CAD SAT test. Bucket/truck still requires the existing
fixed-route 4 cm guard. All terrain, joint and closure guards remain.
No source module is edited or globally monkey-patched.
"""
from types import FunctionType
import numpy as np
from continuous_scoop import FixedRouteClearance,segment,project
from layered_47_cycle_selection import rank_workspace_candidates
from planning.cycle_geometry import solve_pose,tool_pose,TIP
import planning.rear_lower_prepoint as prepoint
from layered_47_cycle_geometry import triangles_intersect_box,oriented_box_separation

def plan_next(runtime,terrain,observation):
 r=runtime;static=FixedRouteClearance(r)
 truck=static.truck_triangles
 refined=[]
 def clearance(geometries,bodies,map_,truck_vertices):
  result=prepoint.mechanism_and_truck_clearance(geometries,bodies,map_,truck_vertices)
  broad=result['static_truck_aabb_separation_m']
  if broad>=.02:return result
  gap=float('inf');worst=None
  for g in geometries:
   pose=bodies[g['body']];v=np.asarray(g['body_local_vertices']);lo=v.min(0);hi=v.max(0)
   corners=prepoint._local_aabb_corners(v);world=corners@pose[:3,:3].T+pose[:3,3]
   sep=oriented_box_separation(world,pose[:3,:3],static.truck_lo,static.truck_hi)
   if sep<.02:
    local=(truck-pose[:3,3])@pose[:3,:3]
    sep=0. if triangles_intersect_box(local,lo,hi,margin=.02) else .02
   if sep<gap:gap=sep;worst=g['mesh']
  # Keep original result key for the reused planner, explicitly identify
  # its refined meaning in the report. Source AABB value is retained below.
  refined.append(dict(original_world_aabb_gap_m=broad,refined_gap_lower_bound_m=gap,mesh=worst))
  return dict(result,static_truck_aabb_separation_m=gap)
 collision=prepoint.ground_completed_terrain(terrain)
 class Checker:
  def check(self,q,previous=None):
   result=static.check(q,previous)
   _,bodies=r.assembly.assemble(q,include_meshes=False);pose=bodies['bucket']
   p=static.samples@pose[:3,:3].T+pose[:3,3]
   soil=prepoint.sampled_surface_clearance(p,collision)['clearance_m']
   c=clearance(static.geometries,bodies,terrain,static.truck_triangles.reshape(-1,3))
   if soil<.32 or c['mechanism_terrain_clearance_m']<.02 or c['static_truck_aabb_separation_m']<.02:
    raise ValueError('candidate_original_clearance:'+str((soil,c)))
   return dict(result,bucket_observed_terrain_m=soil,**c)
 checker=Checker()
 inherited=project(r,r.arrays['mechanism_qpos'])
 tip,pitch=tool_pose(r.assembly,inherited,TIP)
 inherited_pose=r.assembly.assemble(inherited,include_meshes=False)[1]['bucket']
 inherited_points=static.samples@inherited_pose[:3,:3].T+inherited_pose[:3,3]
 inherited_gap=prepoint.sampled_surface_clearance(inherited_points,collision)['clearance_m']
 rise_parts=[];returned=inherited.copy()
 if inherited_gap<.32:
  # This is an extension of RETURN, with its original fixed-route guards;
  # it is not APPROACH with a waived 32 cm gate. The future physical run
  # still needs a fresh zero/contact audit before any command is admitted.
  class ReturnRiseChecker:
   def check(self,q,previous=None):
    d=static.check(q,previous)
    pose=r.assembly.assemble(q,include_meshes=False)[1]['bucket']
    points=static.samples@pose[:3,:3].T+pose[:3,3]
    gap=prepoint.sampled_surface_clearance(points,collision)['clearance_m']
    if gap<inherited_gap-1e-8:raise ValueError('return_rise_reduced_inherited_gap')
    if previous is not None:
     prev=r.assembly.assemble(previous,include_meshes=False)[1]['bucket']
     old=static.samples@prev[:3,:3].T+prev[:3,3]
     if np.min(points[:,2]-old[:,2])<-1e-6:raise ValueError('return_rise_bucket_vertex_descended')
    return dict(d,inherited_stock_clearance_m=gap)
  # Existing high-extend target is reachable at this boom endpoint, whereas
  # a pure vertical translation at fixed pitch was not (CPU-v3, retained).
  radial=tip[:2]/np.linalg.norm(tip[:2])
  goal=solve_pose(r.assembly,r.chain,returned,np.r_[5.2*radial,2.7],.2,TIP)
  part=segment(r,returned,goal,ReturnRiseChecker());rise_parts.append(part);returned=goal
  checker.check(returned)
 # Bind only this callable's helper; leave the imported module untouched.
 original=prepoint.plan_rear_lower_prepoint
 planner=FunctionType(original.__code__,dict(original.__globals__,mechanism_and_truck_clearance=clearance),
                      original.__name__,original.__defaults__,original.__closure__)
 planner.__kwdefaults__=original.__kwdefaults__.copy()
 candidates=rank_workspace_candidates(terrain,2);rejected=[]
 with np.load('/home/qunwang/wajueji/evidence/autonomous-multiscoop-2026-09-14-v1/receiver-mesh.npz') as f:vertices=f['world_vertices'].copy()
 for candidate in candidates:
  try:
   xy=np.array(candidate['entry_xy_m']);radial=xy/np.linalg.norm(xy)
   start=returned.copy();swing=start.copy();swing[7]=np.arctan2(xy[1],xy[0])
   goals=[('RETURN_ALIGN_HIGH',swing)];seed=swing
   # Existing high transfer is 2.7 m; keep that height while opening rather
   # than descend to the old fixed 2.5 m above the changed residual stock.
   for name,radius,pitch in [('RETURN_EXTEND_HIGH',5.2,.2),('RETURN_OPEN_HIGH',5.4,-.32)]:
    seed=solve_pose(r.assembly,r.chain,seed,np.r_[radius*radial,2.7],pitch,TIP)
    checker.check(seed);goals.append((name,seed.copy()))
   high=max(.55,float(np.nanmax(terrain['height_m']))+.55-candidate['contact']['crest_surface_z_m'])
   path=planner(terrain=terrain,contact_selection=candidate['contact'],operator_qpos=seed,
       open_bucket_proxy=r.proxy,truck_world_vertices=vertices,assembly=r.assembly,chain=r.chain,
       transfer_profiles=((high,-.65),(high,-.70),(high,-.75),(high+.05,-.75)),
       prepoint_offsets=((.30,.30),(.35,.40),(.40,.50)))
   qs=[];vs=[];segments=[];point=start.copy()
   for j,part in enumerate(rise_parts):
    qs.append(part['qpos'] if not qs else part['qpos'][1:]);vs.append(part['velocity'])
    segments.append(dict(name='RETURN_RISE_'+str(j),frames=part['steps'],minimum_clearance=part['minimum_clearance']))
   for name,goal in goals:
    part=segment(r,point,goal,checker)
    qs.append(part['qpos'] if not qs else part['qpos'][1:]);vs.append(part['velocity'])
    segments.append(dict(name=name,frames=part['steps'],minimum_clearance=part['minimum_clearance']));point=goal
   qs.append(path['qpos'][1:]);vs.append(path['velocity'])
   path['qpos']=np.concatenate(qs);path['velocity']=np.concatenate(vs)
   path['segments']=segments+path['segments'];path['total_frames']=len(path['velocity'])
   return dict(selected=candidate,path=path,observation=observation,terrain=terrain,rejected_pairs=rejected,
     candidate_count=len(candidates),cycle_index=2,truck_refinements=refined,
     return_rise=dict(inherited_bucket_stock_gap_m=inherited_gap,rise_tip_m=float(2.7-tip[2]) if rise_parts else 0.,
                     phase='RETURN extension, not APPROACH',physical_validation_required=True),
     selection_scope='Actual observed stock; original scoring and constraints. CPU candidate only; not admitted for physics.')
  except ValueError as e:
   rejected.append(dict(entry_xy_m=candidate['entry_xy_m'],failure=str(e)))
   print('CPU_CANDIDATE_REJECT',candidate['entry_xy_m'],str(e)[:180],flush=True)
 raise ValueError('no_safe_second_scoop_cpu_candidate:'+str(rejected))
