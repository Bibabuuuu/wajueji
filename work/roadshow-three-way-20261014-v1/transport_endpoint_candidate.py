"""CPU candidate: preserve a source-bound closed cylinder at a planning endpoint."""
from contextlib import contextmanager
import numpy as np
from scipy.optimize import least_squares
from planning.cycle_geometry import tool_pose,MOUTH
import plan as original_plan

def locked_lift_pose(assembly,chain,seed,target,local=MOUTH):
    q=np.array(seed,float,copy=True);target=np.asarray(target,float)
    bucket=chain.axes[3];held=float(q[bucket]);low,high=chain.limits[bucket]
    if q.shape!=(19,) or target.shape!=(3,) or not np.isfinite(q).all() or not np.isfinite(target).all():
        raise ValueError('invalid_locked_pose_input')
    if not low<=held<=high:raise ValueError('held_cylinder_outside_physical_bounds')
    indices=np.array([i for i in range(7,19) if i!=bucket])
    bounds=np.array([chain.limits.get(int(i),(-3.,3.)) for i in indices],float)
    bounds[:,0]+=.008;bounds[:,1]-=.008
    scale=bounds[:,1]-bounds[:,0]
    if np.any(scale<=0):raise ValueError('invalid_locked_pose_bounds')
    def unpack(x):
        trial=q.copy();trial[indices]=bounds[:,0]+x*scale;trial[bucket]=held;return trial
    def residual(x):
        trial=unpack(x);point,_=tool_pose(assembly,trial,local)
        return np.r_[20*chain.residual(trial),point-target]
    initial=np.clip((q[indices]-bounds[:,0])/scale,0.,1.)
    solved=least_squares(residual,initial,bounds=(np.zeros(len(indices)),np.ones(len(indices))),
        xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=250)
    goal=unpack(solved.x);point,pitch=tool_pose(assembly,goal,local)
    closure=float(np.max(abs(chain.residual(goal))));error=float(np.linalg.norm(point-target))
    if closure>=1e-8 or error>=.005 or goal[bucket]!=held:
        raise ValueError('locked_pose_unresolved:'+str(dict(closure_m=closure,position_error_m=error)))
    return goal,dict(closure_m=closure,position_error_m=error,cylinder_m=held,pitch_diagnostic_rad=pitch,
        pitch_constraint='loaded cylinder is held; pitch follows linkage, not an extra independent target',
        physical_bounds=[low,high],original_other_joint_margin_m=.008)

@contextmanager
def bind_endpoint_solver(runtime,source):
    """Only the first loaded waypoint changes. Restore old module globals."""
    old_solver=original_plan.solve_pose;old_source=original_plan.SOURCE
    calls=[]
    def solve(assembly,chain,seed,target,pitch,local=MOUTH):
        if not calls:
            goal,detail=locked_lift_pose(assembly,chain,seed,target,local);calls.append(detail);return goal
        calls.append({'mode':'original_unload_pose_solver'})
        return old_solver(assembly,chain,seed,target,pitch,local)
    original_plan.solve_pose=solve;original_plan.SOURCE=source
    try:yield calls
    finally:original_plan.solve_pose=old_solver;original_plan.SOURCE=old_source

def build_plan(runtime,source,out):
    with bind_endpoint_solver(runtime,source) as calls:
        segments,checker,report=original_plan.build_plan(runtime,out)
    report.update(endpoint_candidate='source_bound_held_cylinder_v1',endpoint_solver_calls=calls,
        old_waypoint_positions_unchanged=True,mechanical_bounds_unchanged=True,
        clearance_and_closure_thresholds_unchanged=True,
        scope='CPU nominal plan only; loaded cylinder held, pitch is a diagnostic of coupled motion')
    return segments,checker,report
