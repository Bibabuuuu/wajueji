"""Read-only mesh sign audit; reversed faces exist only in an isolated candidate."""
import json
from pathlib import Path
import numpy as np
import warp as wp
from layered_excavation_approach_planner import _load_kinematics, quaternion_matrix_wxyz
from newton._src.solvers.implicit_mpm.rasterized_collisions import get_average_face_normal

@wp.kernel
def signed_query(mesh:wp.uint64,points:wp.array[wp.vec3],out:wp.array[float]):
    i=wp.tid();p=points[i]
    q=wp.mesh_query_point_no_sign(mesh,p,1.0)
    if q.result:
        cp=wp.mesh_eval_position(mesh,q.face,q.u,q.v)
        normal=get_average_face_normal(mesh,cp)
        out[i]=wp.length(p-cp)*wp.where(wp.dot(normal,p-cp)>0.0,1.0,-1.0)
    else:
        out[i]=999.0

root=Path('/home/qunwang/wajueji');out=root/'work/surface-render-diagnosis-20260923'
repo=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1/repo')
proxy=json.loads((repo/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json').read_text())
events=json.loads((out/'projection-summary.json').read_text())['large_projection_events']
trace=[json.loads(s) for s in (root/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-161451-732607/physics.ndjson').read_text().splitlines()]
kin=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
assembly,_=_load_kinematics(kin,kin/'scene/input/model/A.xml',kin/'scene/input/mesh')
local=[]
for e in events:
    pose=assembly.assemble(np.array(trace[e['frame']-1]['before_state']['mechanism_qpos']),include_meshes=False)[1]['bucket']
    local.append((np.asarray(e['before'])-pose[:3,3])@pose[:3,:3])
points=np.array(local)
def winding(v,f,p):
    a=v[f[:,0]]-p;b=v[f[:,1]]-p;c=v[f[:,2]]-p
    la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1);lc=np.linalg.norm(c,axis=1)
    num=np.einsum('ij,ij->i',a,np.cross(b,c))
    den=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
    return np.sum(2*np.arctan2(num,den))/(4*np.pi)
inside=np.zeros(len(points),dtype=bool)
for b in proxy['shell_boxes']+proxy['tooth_boxes']:
    r=quaternion_matrix_wxyz(b['quaternion_wxyz'])
    coords=(points-np.array(b['position_bucket_m']))@r
    inside |= np.all(np.abs(coords)<np.array(b['half_size_m']),axis=1)
side_results=[]
wp.init()
for side in proxy['side_plate_meshes']:
    v=np.array(side['vertices_bucket_m']);f=np.array(side['faces'])
    w=np.array([winding(v,f,p) for p in points]);inside|=(np.abs(w)>.5)
    volume=float(np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])).sum()/6)
    distances=[]
    for faces in [f,f[:,::-1]]:
        mesh=wp.Mesh(points=wp.array(v,dtype=wp.vec3,device='cpu'),indices=wp.array(faces.reshape(-1),dtype=int,device='cpu'))
        results=wp.zeros(len(points),dtype=float,device='cpu')
        wp.launch(signed_query,dim=len(points),inputs=[mesh.id,wp.array(points,dtype=wp.vec3,device='cpu'),results],device='cpu')
        distances.append(results.numpy().tolist())
    side_results.append({'name':side['name'],'signed_volume_m3':volume,'max_abs_winding_at_particles':float(np.abs(w).max()),'original_signed_distance':distances[0],'reversed_signed_distance':distances[1]})
result={'physics_steps':0,'particle_queries':len(points),'particles_inside_any_actual_proxy_solid':int(inside.sum()),'side_results':side_results,'candidate_only':True,'production_modified':False}
(out/'side-winding-audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({**{k:v for k,v in result.items() if k!='side_results'},'side_results':[{k:v for k,v in s.items() if not isinstance(v,list)}|{'original_negative_queries':sum(x<0 for x in s['original_signed_distance']),'reversed_negative_queries':sum(x<0 for x in s['reversed_signed_distance'])} for s in side_results]},indent=2))
