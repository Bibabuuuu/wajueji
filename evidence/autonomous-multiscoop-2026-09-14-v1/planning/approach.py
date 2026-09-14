"""Sampled free-space approach checks; no simulator stepping or command publishing."""
import math
import numpy as np


def bucket_clearance(vertices, terrain, margin_m=.15, padding_m=.01):
    """Conservative whole-bucket AABB against observed cell maxima. Unknown rejects."""
    v=np.asarray(vertices,float);h=np.asarray(terrain['height_m'],float);counts=np.asarray(terrain['point_count'],float)
    origin=np.asarray(terrain['origin_xy_m'],float);res=float(terrain['resolution_m'])
    if v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all() or h.ndim!=2 or counts.shape!=h.shape or origin.shape!=(2,) or not np.isfinite(origin).all() or not math.isfinite(res) or res<=0 or not math.isfinite(margin_m) or margin_m<0 or not math.isfinite(padding_m) or padding_m<0:raise ValueError('invalid_clearance_input')
    if not terrain.get('valid'):raise ValueError('terrain_invalid')
    lo=v.min(axis=0)-padding_m;hi=v.max(axis=0)+padding_m
    upper=origin+res*np.array([h.shape[1],h.shape[0]])
    if np.any(lo[:2]<origin) or np.any(hi[:2]>=upper):raise ValueError('bucket_footprint_outside_observed_roi')
    start=np.floor((lo[:2]-origin)/res).astype(int);end=np.floor((hi[:2]-origin)/res).astype(int)
    patch=h[start[1]:end[1]+1,start[0]:end[0]+1];n=counts[start[1]:end[1]+1,start[0]:end[0]+1]
    if not np.isfinite(patch).all() or not np.isfinite(n).all() or (n<1).any():raise ValueError('unknown_bucket_corridor')
    clearance=float(lo[2]-patch.max())
    if clearance<margin_m:raise ValueError('bucket_corridor_clearance: '+format(clearance,'.6f'))
    return clearance


def boxes_overlap(a,b,margin_m=.01):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=3 or b.shape[1]!=3 or not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('invalid_collision_bounds')
    return bool(np.all(a.max(axis=0)+margin_m>=b.min(axis=0)) and np.all(b.max(axis=0)+margin_m>=a.min(axis=0)))


def sampled_approach(start_q,end_q,*,axes,speed_limits,advance,check_pose,dt=1/120,max_steps=14400):
    """advance(q,velocity,dt) must solve closure privately and return accepted q."""
    start=np.asarray(start_q,float);end=np.asarray(end_q,float);axes=np.asarray(axes,int);limit=np.asarray(speed_limits,float)
    if start.ndim!=1 or end.shape!=start.shape or not np.isfinite(start).all() or not np.isfinite(end).all() or limit.shape!=axes.shape or not np.isfinite(limit).all() or (limit<=0).any() or not 0<dt<=.05:raise ValueError('invalid_path_input')
    delta=end[axes]-start[axes];duration=max(dt,float(np.max(1.5*np.abs(delta)/limit)))
    steps=max(1,math.ceil(duration/dt))
    if steps>max_steps:raise ValueError('path_step_budget')
    q=start.copy();path=[q.copy()];velocities=[];diagnostics=[check_pose(q.copy(),None)]
    for i in range(1,steps+1):
        s=i/steps;desired=start[axes]+(3*s*s-2*s*s*s)*delta
        velocity=(desired-q[axes])/dt
        if np.any(np.abs(velocity)>limit+1e-10):raise ValueError('path_speed_budget')
        proposed=np.asarray(advance(q.copy(),velocity.copy(),dt),float)
        if proposed.shape!=q.shape or not np.isfinite(proposed).all() or np.max(np.abs(proposed[axes]-desired))>1e-8:raise ValueError('path_advance_mismatch')
        diagnostics.append(check_pose(proposed.copy(),q.copy()));q=proposed
        path.append(q.copy());velocities.append(velocity)
    # Same actuator endpoint can have another passive branch. Do not accept that silently.
    if np.max(np.abs(q-end))>1e-5:raise ValueError('endpoint_branch_mismatch')
    return {'qpos':np.array(path),'velocity':np.array(velocities),'diagnostics':diagnostics,'dt_s':dt,
            'duration_sim_s':steps*dt,'steps':steps,'validation_kind':'sampled_kinematic_path',
            'continuous_collision_proven':False,'physical_stop_validated':False,'execution_approved':False}
