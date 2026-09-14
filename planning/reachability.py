"""Bounded endpoint witness search. No path planning or actuator commands."""
import numpy as np


def search_endpoint(target_world, seeds, *, point_world, closure, variable_indices, limits,
                    check_pose, max_iterations=100, position_tolerance_m=.02, closure_tolerance_m=1e-7):
    target=np.asarray(target_world,float)
    indices=np.asarray(variable_indices,int)
    bounds=np.asarray(limits,float)
    seeds=[np.asarray(seed,float).copy() for seed in seeds]
    if target.shape!=(3,) or not np.isfinite(target).all() or not seeds:raise ValueError('invalid_target_or_seeds')
    if bounds.shape!=(len(indices),2) or not np.isfinite(bounds).all() or np.any(bounds[:,1]<=bounds[:,0]) or len(set(indices))!=len(indices):raise ValueError('invalid_bounds')
    if min(position_tolerance_m,closure_tolerance_m)<=0 or max_iterations<1:raise ValueError('invalid_search_budget')
    scale=bounds[:,1]-bounds[:,0];attempts=[];best=None
    def evaluate(q):
        c=np.asarray(closure(q),float);point=np.asarray(point_world(q),float)
        if c.ndim!=1 or point.shape!=(3,) or not np.isfinite(c).all() or not np.isfinite(point).all():raise ValueError('invalid_kinematics')
        return np.concatenate((20*c,point-target)),point,float(np.max(np.abs(c))) if len(c) else 0.
    for seed_index,seed in enumerate(seeds):
        if seed.ndim!=1 or not np.isfinite(seed).all() or np.any(indices<0) or np.any(indices>=len(seed)):raise ValueError('invalid_seed')
        q=seed.copy();q[indices]=np.clip(q[indices],bounds[:,0],bounds[:,1]);reason='search_budget';damping=.003
        for iteration in range(max_iterations):
            residual,point,closure_error=evaluate(q);position_error=float(np.linalg.norm(point-target));cost=float(residual@residual)
            if best is None or cost<best['cost']:best={'cost':cost,'position_error_m':position_error,'closure_error_m':closure_error}
            if position_error<=position_tolerance_m and closure_error<=closure_tolerance_m:
                try:check_pose(q.copy())
                except ValueError as error:reason='endpoint_guard_rejected: '+str(error);break
                return {'status':'endpoint_witness','qpos':q.tolist(),'point_world_m':point.tolist(),'target_world_m':target.tolist(),
                        'position_error_m':position_error,'closure_error_m':closure_error,'seed_index':seed_index,'iterations':iteration+1,
                        'joint_ranges_checked':True,'guard_checked':True,'orientation_constrained':False,
                        'path_checked':False,'terrain_collision_checked':False,'excavation_target_approved':False,'command':None}
            jac=[]
            for j,address in enumerate(indices):
                h=1e-5*scale[j]
                if q[address]+h>bounds[j,1]:h=-h
                shifted=q.copy();shifted[address]+=h
                jac.append((evaluate(shifted)[0]-residual)/(h/scale[j]))
            jac=np.asarray(jac).T
            delta=np.linalg.lstsq(np.vstack((jac,damping*np.eye(len(indices)))),np.concatenate((-residual,np.zeros(len(indices)))),rcond=None)[0]
            if not np.isfinite(delta).all():reason='nonfinite_step';break
            delta*=min(1.,.04/max(float(np.max(np.abs(delta))),1e-12))
            improved=False
            for factor in (1.,.5,.25,.125,.0625):
                proposed=q.copy();proposed[indices]=np.clip(q[indices]+factor*delta*scale,bounds[:,0],bounds[:,1])
                r,_,_=evaluate(proposed)
                if float(r@r)<cost-1e-16:q=proposed;improved=True;damping=max(.0001,damping*.7);break
            if not improved:
                damping*=5
                if damping>5:reason='search_stalled';break
        attempts.append({'seed_index':seed_index,'reason':reason,'iterations':iteration+1})
    return {'status':'no_witness_found','target_world_m':target.tolist(),'best_residual':best,'attempts':attempts,
            'unreachable_proven':False,'path_checked':False,'excavation_target_approved':False,'command':None}
