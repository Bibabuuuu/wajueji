from pathlib import Path
import json,numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
from layered_47_cycle_runtime import CycleRuntime
from layered_47_cycle_geometry import TransportClearance,bucket_mask
from layered_excavation_approach_planner import rock_surface_points
out=Path(__file__).resolve().parent
run=Path((out/'run-directory.txt').read_text().strip())
r=CycleRuntime(run/'source-audit/final.npz',out/'geometry-inspection')
c=TransportClearance(r);a=r.arrays;q=a['particle_q'];pose=r.assembly.assemble(a['mechanism_qpos'],include_meshes=False)[1]['bucket'];samples=c.samples@pose[:3,:3].T+pose[:3,3]
t=c.terrain;ij=np.floor((samples[:,:2]-t['origin_xy_m'])/t['resolution_m']).astype(int);h=np.asarray(t['height_m']);heights=h[ij[:,1],ij[:,0]];gaps=samples[:,2]-np.nan_to_num(heights)
worst=np.argsort(gaps)[:10]
cell=.04;origin=q.min(0)-2*cell;ijk=np.floor((q-origin)/cell).astype(int);shape=ijk.max(0)+3;grid=np.zeros(shape,bool);grid[tuple(ijk.T)]=True;comp,_=label(grid);labels=comp[tuple(ijk.T)];anchors=np.unique(labels[bucket_mask(q,pose)]);anchors=anchors[anchors>0];ground=np.isin(labels,np.unique(labels[q[:,2]<.06]));payload=np.isin(labels,anchors)
_,byid=rock_surface_points(r.root,r.records,a['rigid_qpos'])
rows=[]
for n in worst:
 point=samples[n];lo=np.asarray(t['origin_xy_m'])+ij[n]*t['resolution_m'];hi=lo+t['resolution_m'];mask=ground&((q[:,:2]>=lo)&(q[:,:2]<hi)).all(1)
 rocks=[]
 for rock_id,v in byid.items():
  ri=np.floor((v-origin)/cell).astype(int);inside=((ri>=0)&(ri<shape)).all(1);linked=np.zeros(len(v),bool);linked[inside]=np.isin(comp[tuple(ri[inside].T)],anchors)
  kept=not(linked.any() or bucket_mask(v,pose).any())
  same=((v[:,:2]>=lo)&(v[:,:2]<hi)).all(1)
  if same.any():rocks.append(dict(id=rock_id,kept=kept,height=float(v[same,2].max()),linked=int(linked.sum()),cavity=int(bucket_mask(v,pose).sum())))
 rows.append(dict(sample=point.tolist(),gap=float(gaps[n]),height=float(heights[n]),cell_lower_xy=lo.tolist(),ground_soil_count=int(mask.sum()),ground_soil_max=float(q[mask,2].max()) if mask.any() else None,rock_surface=rocks,nearest_ground_soil_distance=float(cKDTree(q[ground]).query(point)[0])))
result=dict(observation=c.observation,bucket_bounds=[samples.min(0).tolist(),samples.max(0).tolist()],worst=rows,payload_soil=int(payload.sum()),ground_soil=int(ground.sum()))
(out/'clearance-inspection.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
