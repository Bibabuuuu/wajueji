import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverImplicitMPM
from mixed_material_r1 import outward_side_plate_faces
from layered_excavation_approach_planner import quaternion_matrix_wxyz
from run_layered_47_excavation_preflight import _quat_xyzw_from_matrix
from moving_bucket_sweep_candidate import MovingBucketSweep
root=Path('/home/qunwang/wajueji');out=Path(__file__).resolve().parent;source=root/'work/loaded-transport-leak-path-20260923-v1';run=source/'replay-v2'
proxy=json.loads((root/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json').read_text())
wp.init();wp.set_device('cuda:0');newton.use_coord_layout_targets=True
b=newton.ModelBuilder();body=b.add_body(mass=1.0)
for box in proxy['shell_boxes']+proxy['tooth_boxes']:
 h=box['half_size_m'];w,x,y,z=box['quaternion_wxyz'];b.add_shape_box(body,xform=wp.transform(wp.vec3(*box['position_bucket_m']),wp.quat(x,y,z,w)),hx=h[0],hy=h[1],hz=h[2])
for side in proxy['side_plate_meshes']:
 faces,_=outward_side_plate_faces(side);b.add_shape_mesh(body,mesh=newton.Mesh(np.array(side['vertices_bucket_m'],np.float32),faces.reshape(-1),compute_inertia=False,is_solid=True))
model=b.finalize(device='cuda:0');sand=newton.ModelBuilder();SolverImplicitMPM.register_custom_attributes(sand);sand.add_particle(pos=(0,0,2),vel=(0,0,0),mass=1,radius=.01);solver=SolverImplicitMPM(sand.finalize(device='cuda:0'),config=SolverImplicitMPM.Config());solver.setup_collider(model=model);mesh=solver._mpm_model._collider_meshes[0]
def state(q,v):return SimpleNamespace(particle_q=wp.array(q,dtype=wp.vec3,device='cuda:0'),particle_qd=wp.array(v,dtype=wp.vec3,device='cuda:0'),mpm=SimpleNamespace(particle_qd_grad=wp.zeros(len(q),dtype=wp.mat33,device='cuda:0')))
def transform(p):return wp.transform(wp.vec3(*p[:3,3]),wp.quat(*_quat_xyzw_from_matrix(p[:3,:3])))
events=[e for e in json.loads((source/'first-exits.json').read_text()) if e['initial_cavity'] and e['face'].startswith('bucket_shell')]
pos=np.load(run/'post-projection-world.npy',mmap_mode='r');poses=np.load(run/'actual-bucket-poses.npy',mmap_mode='r');pre=np.load(run/'pre-projection-world.npy',mmap_mode='r');groups={};results=[]
for e in events:groups.setdefault((e['frame'],e['phase']),[]).append(e)
for (f,phase),es in groups.items():
 ix=np.array([e['tracked_index'] for e in es]);old=poses[f-1];new=poses[f] if phase=='rigid_pose_advance' else old
 previous=pos[f,ix] if phase=='rigid_pose_advance' else pos[f-1,ix]
 target=pos[f,ix];s=state(target,(target-previous)*60);guard=MovingBucketSweep(mesh,s,1.,.45,proxy=proxy)
 info=guard.apply(s,wp.array(previous,dtype=wp.vec3,device='cuda:0'),transform(old),transform(new),(0,0,0),(0,0,0))
 corrected=(s.particle_q.numpy()-new[:3,3])@new[:3,:3]
 for j,(e,q) in enumerate(zip(es,corrected)):
  box=next(b for b in proxy['shell_boxes'] if b['name']==e['face']);r=quaternion_matrix_wxyz(box['quaternion_wxyz']);local=(q-box['position_bucket_m'])@r
  results.append({'particle_id':e['particle_id'],'frame':f,'phase':phase,'face':e['face'],'corrected_local':q.tolist(),'previous_local':((previous[j]-old[:3,3])@old[:3,:3]).tolist(),'target_local':((target[j]-new[:3,3])@new[:3,:3]).tolist(),'hits':int(guard.flags.numpy()[j]),'outward_z_after_m':float(local[2]),'prevented_midplane_crossing':bool(local[2]<0)})
# Actual mouth must remain open in both directions; no hidden cap or adhesion.
center=np.array(proxy['mouth_center_bucket_m']);normal=np.array(proxy['mouth_inward_direction_bucket']);mouth_start=np.array([center+.1*normal,center-.1*normal]);mouth_end=mouth_start[::-1].copy();s=state(mouth_end,(mouth_end-mouth_start)*60);guard=MovingBucketSweep(mesh,s,1,.45,proxy=proxy);mouth_info=guard.apply(s,wp.array(mouth_start,dtype=wp.vec3,device='cuda:0'),wp.transform_identity(),wp.transform_identity(),(0,0,0),(0,0,0))
summary={'frames_advanced':0,'real_first_shell_crossing_events':len(results),'prevented_first_shell_midplane_crossings':sum(r['prevented_midplane_crossing'] for r in results),'failures':[r for r in results if not r['prevented_midplane_crossing']],'mouth_bidirectional_unchanged':bool(np.array_equal(s.particle_q.numpy(),mouth_end.astype(np.float32))),'mouth_hits':mouth_info['hits']}
(out/'geometry-convex.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='failures'},indent=2))
