"""Independent six-neighbor flood and per-solid nearest-point check, no physics."""
import json
from pathlib import Path
from collections import deque
import numpy as np
from contact_bc_inference import sha
from contact_phase_geometry import SolidBucketGap,ground_connected_mask,triangle_distances
from layered_excavation_approach_planner import _load_kinematics
W=Path(__file__).resolve().parent;ROOT=W.parents[1];run=ROOT/'runs/contact-geometric-probe-20260924-v7/scoop1500'
with np.load(run/'final.npz') as d:q=d['particle_q'].astype(float);joints=d['mechanism_qpos']
k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');assembly,_=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
pose=assembly.assemble(joints,include_meshes=False)[1]['bucket'];proxy=json.loads((ROOT/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json').read_text());observer=SolidBucketGap(proxy)
origin=q.min(0)-.08;ijk=np.floor((q-origin)/.04).astype(int);occupied=set(map(tuple,ijk));anchors=set(map(tuple,ijk[q[:,2]<.06]));visited=set(anchors);queue=deque(anchors)
while queue:
 cell=queue.popleft()
 for axis in range(3):
  for delta in (-1,1):
   n=list(cell);n[axis]+=delta;n=tuple(n)
   if n in occupied and n not in visited:visited.add(n);queue.append(n)
mask=np.array([tuple(v) in visited for v in ijk]);assert np.array_equal(mask,ground_connected_mask(q))
ids=np.flatnonzero(mask);local=(q[ids]-pose[:3,3])@pose[:3,:3]
bound=np.linalg.norm(np.maximum(np.maximum(observer.lower-local,local-observer.upper),0),axis=1);select=bound<=.005;ids=ids[select];local=local[select]
assert len(ids)>0
distance=np.full(len(ids),np.inf)
for center,rotation,h in observer.boxes:distance=np.minimum(distance,np.linalg.norm(np.maximum(np.abs((local-center)@rotation)-h,0),axis=1))
for lo,hi,triangles,planes in observer.sides:
 inside=np.ones(len(ids),bool)
 for normal,d in planes:inside &= local@normal-d<=0
 distance[inside]=0
 for tri in triangles:distance=np.minimum(distance,triangle_distances(local,tri))
order=np.argsort(distance)[:10];minimum=float(distance.min());original=observer.observe(q,pose)
assert abs(minimum-original['remaining_soil_gap_m'])<1e-12
result=dict(physics_steps_added=0,checkpoint=dict(path=str(run/'final.npz'),sha256=sha(run/'final.npz')),independent_flood_equals_production=True,ground_connected_particles=int(mask.sum()),occupied_voxels=len(occupied),ground_connected_voxels=len(visited),nearest_gap_m=minimum,production=original,nearest=[dict(particle_id=int(ids[i]),gap_m=float(distance[i]),world_m=q[ids[i]].tolist(),bucket_local_m=local[i].tolist()) for i in order],conclusion='v7 endpoint does not pass original 3mm gate; six-neighbor connectivity and solid distance independently reproduced',limits='Connectivity is still the frozen 4cm voxel heuristic. Not a calibrated material/contact truth; no threshold or particle exclusion changed.')
(W/'v7-gap-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
