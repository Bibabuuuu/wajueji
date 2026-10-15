"""CPU-only phase boundary: monotone rise, then qualified high-position hold.

No runtime entry or physical stepping is provided. The caller must complete
the nominal path before requesting the transition; failed admission is atomic.
"""
import numpy as np
from minimum_guard import MinimumHeightClearance
VERSION='return_rise_then_high_hold_59um_v2_cpu'

class RiseHoldGuard:
 def __init__(self,runtime,terrain,goal_qpos,nominal_steps):
  self.r=runtime;self.goal=np.asarray(goal_qpos,float).copy()
  if self.goal.shape!=(19,) or not np.isfinite(self.goal).all():raise ValueError('rise_hold_goal_contract')
  if isinstance(nominal_steps,bool) or not isinstance(nominal_steps,(int,np.integer)) or nominal_steps<=0:raise ValueError('rise_hold_steps_contract')
  self.nominal_steps=int(nominal_steps);self._mode='rise'
  self.rising=MinimumHeightClearance(runtime,terrain,rise=True)
  self.holding=MinimumHeightClearance(runtime,terrain,rise=False)
 @property
 def mode(self):return self._mode
 def endpoint_error(self,q):
  q=np.asarray(q,float)
  if q.shape!=(19,) or not np.isfinite(q).all():raise ValueError('rise_hold_state_contract')
  return float(np.max(abs(q[self.r.chain.axes]-self.goal[self.r.chain.axes])))
 def enter_hold(self,q,*,completed_nominal_steps):
  if self._mode!='rise':raise ValueError('rise_hold_repeated_transition')
  if isinstance(completed_nominal_steps,bool) or completed_nominal_steps!=self.nominal_steps:raise ValueError('rise_path_not_completed')
  if self.endpoint_error(q)>.015:raise ValueError('original_endpoint_guard_before_hold')
  measured=self.holding.check(q,physical=True)
  self._mode='hold'
  return dict(measured,mode='hold',endpoint_axis_error=self.endpoint_error(q))
 def check(self,q,previous=None,*,physical=False):
  self.endpoint_error(q)  # Reject malformed state before the geometry routines.
  if self._mode=='rise':return self.rising.check(q,previous,physical=physical)
  if self.endpoint_error(q)>.015:raise ValueError('original_endpoint_guard_during_hold')
  return dict(self.holding.check(q,previous,physical=physical),mode='hold',endpoint_axis_error=self.endpoint_error(q))
