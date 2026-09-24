"""Read-only upper mouth rim diagnostic; user identified this region in a screenshot.

Uses saved float32 display particles and existing 4cm connectivity heuristic.
Particle-center gap is not a physical surface clearance or calibrated trigger.
"""
import json
from pathlib import Path
import numpy as np
from contact_phase_geometry import ground_connected_mask
from layered_excavation_approach_planner import _load_kinematics
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
run=ROOT/'runs/contact-geometric-probe-20260924-v5/contact900'
proxy=json.loads((ROOT/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json').read_text())
states=json.loads((run/'display/states.json').read_text());particles=np.load(run/'display/particles.npy',mmap_mode='r')
k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');assembly,chain=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
x,z=proxy['mouth_endpoints_xz_m'][0];width=proxy['inner_clear_width_m'];rows=[]
for frame in (422,480,540,600,633,740,900):
 q=np.asarray(particles[frame],dtype=float);q=q[ground_connected_mask(q)]
 pose=assembly.assemble(np.asarray(states[frame]['mechanism_qpos']),include_meshes=False)[1]['bucket']
 local=(q-pose[:3,3])@pose[:3,:3]
 delta=local-np.array([x,0,z]);delta[:,1]=np.maximum(abs(delta[:,1])-width/2,0)
 d=np.linalg.norm(delta,axis=1);point=q[np.argmin(d)]
 rows.append(dict(frame_end=frame,upper_rim_particle_center_gap_m=float(d.min()),ground_connected_particles=len(q),nearest_world_point=point.tolist()))
report=dict(scope=__doc__,feature='representative upper mouth rim line across inner width; user identified upper root region, exact CAD edge remains an approximation',local_xz=[x,z],width_m=width,rows=rows,physics_steps=0,used_for_control=False)
(W/'root-gap-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
