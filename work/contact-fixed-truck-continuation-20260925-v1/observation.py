"""Static terrain excludes unsupported airborne rocks; physical arrays unchanged."""
import sys
import numpy as np
from scipy.ndimage import label
from runtime import ROOT
sys.path.append(str(ROOT/'work/contact-first-transport-review-20260925-v1'))
from static_rocks import static_rock_ids
from layered_47_cycle_geometry import bucket_mask
from layered_excavation_approach_planner import rock_surface_points,terrain_from_points

def stationary_terrain(runtime,arrays=None):
    arrays=runtime.arrays if arrays is None else arrays
    q=arrays['particle_q'];pose=runtime.assembly.assemble(arrays['mechanism_qpos'],include_meshes=False)[1]['bucket']
    cell=.04;origin=q.min(0)-2*cell;ijk=np.floor((q-origin)/cell).astype(int);shape=ijk.max(0)+3
    if np.prod(shape)>12000000:raise ValueError('cycle_observation_grid_budget')
    grid=np.zeros(tuple(shape),bool);grid[tuple(ijk.T)]=True;components,_=label(grid)
    labels=components[tuple(ijk.T)];anchors=np.unique(labels[bucket_mask(q,pose)]);anchors=anchors[anchors>0]
    payload=np.isin(labels,anchors)
    if payload.any() and (q[payload,2].min()<.3 or np.max(np.ptp(q[payload],axis=0))>2.):
        raise ValueError('cycle_load_not_observably_detached')
    stationary=np.isin(labels,np.unique(labels[q[:,2]<.06]))
    if np.count_nonzero(stationary)<1000:raise ValueError('cycle_stationary_stock_unobserved')
    _,by_id=rock_surface_points(runtime.root,runtime.records,arrays['rigid_qpos'])
    # Overestimates possible support by more than one 10 cm MPM cell.
    static,distances=static_rock_ids(q[stationary],by_id,.15)
    kept=[]
    for key,points in by_id.items():
        indices=np.floor((points-origin)/cell).astype(int);inside=((indices>=0)&(indices<shape)).all(1)
        linked=np.zeros(len(points),bool);linked[inside]=np.isin(components[tuple(indices[inside].T)],anchors)
        if key in static and not (linked.any() or bucket_mask(points,pose).any()):kept.append(points)
    terrain=terrain_from_points(np.concatenate([q[stationary],*kept]))
    return terrain,dict(detached_payload_particles=int(payload.sum()),detached_dynamic_particles=int((~stationary).sum()),
        detached_rock_ids=sorted(set(by_id)-static),support_margin_m=.15,
        rock_to_stationary_soil_aabb_gap_m=distances,observation_only=True,cell_m=cell,
        scope='Ground-connected stock; all detached soil/rocks remain in physical collision solving')
