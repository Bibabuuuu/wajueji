"""Closed-chain pose goals for the current fixed-base rig; no physics stepping."""
import numpy as np

TIP=np.array([.214,0.,-1.023])
MOUTH=np.array([.117,0.,-.590])


def tool_pose(assembly,q,local):
    _,bodies=assembly.assemble(q,include_meshes=False);pose=bodies['bucket']
    return pose[:3,:3]@local+pose[:3,3],float(np.arctan2(-pose[2,0],pose[2,2]))


def solve_pose(assembly,chain,seed,target,pitch,local=TIP):
    q=np.array(seed,float,copy=True);indices=np.arange(7,19)
    target=np.asarray(target,float);local=np.asarray(local,float)
    if q.shape!=(19,) or target.shape!=(3,) or local.shape!=(3,) or not np.isfinite(q).all() or not np.isfinite(target).all() or not np.isfinite(local).all() or not np.isfinite(pitch):
        raise ValueError('invalid_cycle_pose_input')
    bounds=np.array([chain.limits.get(int(i),(-3.0,3.0)) for i in indices],float)
    bounds[:,0]+=.008;bounds[:,1]-=.008
    if np.any(bounds[:,1]<=bounds[:,0]):raise ValueError('invalid_cycle_pose_bounds')
    q[indices]=np.clip(q[indices],bounds[:,0],bounds[:,1])
    scale=bounds[:,1]-bounds[:,0]
    def residual(q):
        point,angle=tool_pose(assembly,q,local)
        angle_error=np.arctan2(np.sin(angle-pitch),np.cos(angle-pitch))
        return np.r_[20*chain.residual(q),point-target,angle_error]
    damping=.003
    for iteration in range(160):
        r=residual(q)
        if np.max(abs(r[:12]))<2e-7 and np.linalg.norm(r[12:15])<.005 and abs(r[15])<.005:return q
        jac=[]
        for j,address in enumerate(indices):
            h=1e-5*scale[j]
            if q[address]+h>bounds[j,1]:h=-h
            other=q.copy();other[address]+=h
            jac.append((residual(other)-r)/(h/scale[j]))
        jac=np.array(jac).T
        delta=np.linalg.lstsq(np.vstack([jac,damping*np.eye(len(indices))]),np.r_[-r,np.zeros(len(indices))],rcond=None)[0]
        delta*=min(1.,.04/max(np.max(abs(delta)),1e-12));better=False
        for factor in (1.,.5,.25,.125,.0625):
            trial=q.copy();trial[indices]=np.clip(q[indices]+factor*delta*scale,bounds[:,0],bounds[:,1])
            if np.linalg.norm(residual(trial))<np.linalg.norm(r):q=trial;better=True;damping=max(.0001,damping*.7);break
        if not better:
            damping*=5
            if damping>5:break
    r=residual(q)
    active=indices[np.minimum(q[indices]-bounds[:,0],bounds[:,1]-q[indices])<1e-6].tolist()
    raise ValueError(f'cycle_pose_unresolved: closure_m={np.max(abs(r[:12]))/20:.9g}, '
                     f'position_error_m={np.linalg.norm(r[12:15]):.9g}, pitch_error_rad={r[15]:.9g}, active_bounds={active}')


def plan_cycle(assembly,chain,initial_q,selection,truck_contract):
    xy=np.array(selection['entry_xy_m']);radial=xy/np.linalg.norm(xy);h=selection['surface_z_m']
    yaw=np.arctan2(xy[1],xy[0]);goals={};seed=np.array(initial_q,float)
    # TIP is a reference point, not the lowest point of the CAD bucket.
    targets=[('APPROACH',np.r_[xy,h+.35],-.32,TIP),
             ('CUT',np.r_[xy-.32*radial,h-.50],.20,TIP),
             ('DEEPEN',np.r_[xy-.55*radial,h-.75],.50,TIP),
             ('CURL',np.r_[xy-1.2*radial,h+.05],1.50,TIP),
             # Original 1.30 rad target required q15=0.85376, beyond the
             # existing 0.852 planning bound. Keep the 8 mm margin.
             ('LIFT',np.r_[4.3*radial,3.30],1.00,MOUTH)]
    for stage,point,pitch,local in targets:
        seed=solve_pose(assembly,chain,seed,point,pitch,local);goals[stage]=seed.copy()
    transform=np.asarray(truck_contract['local_to_world'],float)
    center=transform@np.array([0.,np.mean(truck_contract['y_limits_m']),0.,1.])
    truck_yaw=np.arctan2(center[1],center[0]);seed=seed.copy();seed[7]+=truck_yaw-yaw;goals['TURN']=seed.copy()
    seed=solve_pose(assembly,chain,seed,np.r_[center[:2],3.50],1.00,MOUTH);goals['ALIGN']=seed.copy()
    # Extend into the current truck interior while opening, maintaining height.
    # These are geometry candidates; soil discharge still requires validation.
    dump=transform@np.array([.60,np.mean(truck_contract['y_limits_m']),0.,1.])
    seed=solve_pose(assembly,chain,seed,np.r_[dump[:2],3.60],-.45,MOUTH);goals['DUMP']=seed.copy()
    seed=solve_pose(assembly,chain,seed,np.r_[center[:2],3.50],1.00,MOUTH);goals['CLEAR_TRUCK']=seed.copy()
    # Keep the elevated linkage fixed until yaw clears the truck. A combined
    # return-to-initial joint interpolation dips toward the truck side wall.
    seed=seed.copy();seed[7]=initial_q[7];goals['RETURN_SWING']=seed.copy()
    goals['RETURN']=np.array(initial_q,float,copy=True)
    return goals
