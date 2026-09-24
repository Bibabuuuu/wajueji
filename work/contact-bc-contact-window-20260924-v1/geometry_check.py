from pathlib import Path
import json,numpy as np
from layered_excavation_approach_planner import _load_kinematics
K=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');assembly,_=_load_kinematics(K,K/'scene/input/model/A.xml',K/'scene/input/mesh')
run=Path('runs/contact-bc-contact-window-20260924-v1');rows=[json.loads(x) for x in (run/'physics/frames.ndjson').read_text().splitlines()]
results=[]
for frame in (300,360,420,480,540,600):
 geometries,bodies=assembly.assemble(np.array(rows[frame-1]['after_state']['mechanism_qpos']))
 pose=bodies['bucket'];points=np.concatenate([g['body_local_vertices'] for g in geometries if g['body']=='bucket'])@pose[:3,:3].T+pose[:3,3]
 results.append(dict(frame=frame,bucket_mesh_minimum_world_z_m=float(points[:,2].min()),bucket_mesh_maximum_world_z_m=float(points[:,2].max()),vertices_below_ground_z0=int(np.sum(points[:,2]<0))))
result=dict(ground_reference_z_m=0.,geometry='existing display assembly bucket vertices',samples=results)
Path(__file__).with_name('geometry.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
