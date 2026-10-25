"""Locate the exact contributor to the stopped transport height-field check."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/contact-carry-transport-20260924-v2'))
from runtime import TransportRuntime,SOURCE,read,sha
from static_rocks import static_rock_ids
from layered_47_cycle_geometry import stationary_terrain,TransportClearance,bucket_mask
from layered_excavation_approach_planner import rock_surface_points,terrain_from_points

def main():
    source=ROOT/'runs/contact-carry-transport-20260924-v2/physical/recovery.npz'
    metadata=read(str(source)+'.json');assert sha(source)==metadata['checkpoint_sha256']
    r=TransportRuntime(SOURCE,ROOT/'runs/contact-first-transport-20260925-v1/geometry-support')
    with np.load(source) as f:arrays={k:f[k].copy() for k in f.files}
    q=arrays['particle_q'];pose=r.assembly.assemble(arrays['mechanism_qpos'],include_meshes=False)[1]['bucket']
    terrain,observation=stationary_terrain(r,arrays);checker=TransportClearance(r,terrain)
    samples=checker.samples@pose[:3,:3].T+pose[:3,3]
    cell=.04;origin=q.min(0)-2*cell;ijk=np.floor((q-origin)/cell).astype(int);shape=ijk.max(0)+3
    grid=np.zeros(tuple(shape),bool);grid[tuple(ijk.T)]=True;components,_=label(grid)
    labels=components[tuple(ijk.T)];anchors=np.unique(labels[bucket_mask(q,pose)]);anchors=anchors[anchors>0]
    payload=np.isin(labels,anchors);stationary=np.isin(labels,np.unique(labels[q[:,2]<.06]))
    tree=cKDTree(q[payload]);_,by_id=rock_surface_points(r.root,r.records,arrays['rigid_qpos'])
    details=[];kept=[]
    def gap(points):
        checker.terrain=terrain_from_points(points)
        return checker.material_clearance(samples)
    for identifier,points in by_id.items():
        indices=np.floor((points-origin)/cell).astype(int);inside=((indices>=0)&(indices<shape)).all(1)
        linked=np.zeros(len(points),bool);linked[inside]=np.isin(components[tuple(indices[inside].T)],anchors)
        cavity=bucket_mask(points,pose);exclude=bool(linked.any() or cavity.any())
        own=gap(points)
        if not exclude:kept.append((identifier,points))
        details.append(dict(rock_id=identifier,excluded=exclude,linked_vertices=int(linked.sum()),
            cavity_vertices=int(cavity.sum()),bucket_heightfield_gap_m=own,
            nearest_payload_center_m=float(tree.query(points)[0].min()),
            min_z_m=float(points[:,2].min()),max_z_m=float(points[:,2].max())))
    soil_gap=gap(q[stationary]);full_gap=gap(np.concatenate([q[stationary],*[p for _,p in kept]]))
    offenders=[d for d in details if not d['excluded'] and d['bucket_heightfield_gap_m']<.04]
    static,distances=static_rock_ids(q[stationary],by_id,.15)
    corrected_gap=gap(np.concatenate([q[stationary],*[points for key,points in kept if key in static]]))
    result=dict(static_support_margin_m=.15,corrected_gap_m=corrected_gap,detached_kept_rocks=[dict(rock_id=key,soil_aabb_gap_m=distances[key]) for key,_ in kept if key not in static],physics_steps_added=0,source=str(source),source_sha256=sha(source),
        observation=observation,stationary_soil_gap_m=soil_gap,full_gap_m=full_gap,
        kept_rocks=len(kept),offending_rocks=offenders,rocks=details,
        input_sha256={str(p):sha(p) for p in [source,Path(str(source)+'.json'),Path(__file__),Path(__file__).parent/'static_rocks.py',ROOT/'src/v4/layered_47_cycle_geometry.py',ROOT/'src/v4/layered_excavation_approach_planner.py']})
    (Path(__file__).parent/'terrain-support-review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rocks','input_sha256')},indent=2))

if __name__=='__main__':main()
