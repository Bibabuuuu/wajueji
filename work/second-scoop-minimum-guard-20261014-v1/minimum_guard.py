"""CPU candidate: lowest bucket height, rather than every corner's velocity.

No physical runner or admission is provided. Original static, terrain and
APPROACH gates remain in the inherited checker. Only RETURN's extra height
condition changes, under the user's separate CPU-only confirmation.
"""
import numpy as np
from run_short import LiveClearance,mechanism_and_truck_clearance
VERSION='return_minimum_height_guard_v1_cpu'

def minimum_rise(points,previous_points,gap,inherited_gap):
 points=np.asarray(points,float);previous_points=np.asarray(previous_points,float)
 if (points.ndim!=2 or points.shape[1]!=3 or len(points)==0 or points.shape!=previous_points.shape
     or not np.isfinite(points).all() or not np.isfinite(previous_points).all()
     or not np.isfinite([gap,inherited_gap]).all() or inherited_gap<0):
  raise ValueError('minimum_rise_input_contract')
 if gap<inherited_gap-1e-8:raise ValueError('return_rise_reduced_inherited_gap')
 before=float(previous_points[:,2].min());after=float(points[:,2].min())
 if after-before<-1e-6:raise ValueError('return_rise_minimum_bucket_height_descended')
 return dict(minimum_bucket_z_m=after,previous_minimum_bucket_z_m=before,minimum_height_step_m=after-before)

class MinimumHeightClearance(LiveClearance):
 def check(self,q,previous=None,*,physical=False):
  if not self.rise:return super().check(q,previous,physical=physical)
  d=self.static.check(q,previous,physical=physical);points,gap,b=self.measure(q)
  if not self.r.projector.physical_bounds_ok(np.asarray(q)):raise ValueError('original_joint_bounds')
  other=mechanism_and_truck_clearance(self.static.geometries,b,self.terrain,self.static.truck_triangles.reshape(-1,3))
  if other['mechanism_terrain_clearance_m']<.02:raise ValueError('original_mechanism_terrain_2cm')
  old=points if previous is None else self.measure(previous)[0]
  height=minimum_rise(points,old,gap,self.initial_gap)
  return dict(d,bucket_observed_terrain_m=gap,mechanism_terrain_clearance_m=other['mechanism_terrain_clearance_m'],
   diagnostic_old_truck_aabb_gap_m=other['static_truck_aabb_separation_m'],**height)
