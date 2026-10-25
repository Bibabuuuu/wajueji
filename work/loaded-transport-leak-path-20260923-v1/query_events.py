"""Re-query recorded positions against production collision geometry; no physical steps."""
import json
from pathlib import Path
import numpy as np
import warp as wp
import newton
from newton._src.solvers.implicit_mpm.rasterized_collisions import Collider, collision_sdf, _query_collider_sdf
import run_layered_47_excavation_preflight as pf
import layered_47_truck_physics_candidate as truck
from layered_excavation_approach_planner import quaternion_matrix_wxyz

@wp.kernel
def query(points:wp.array[wp.vec3],collider:Collider,body_q:wp.array[wp.transform],body_qd:wp.array[wp.spatial_vector],empty:wp.array[wp.transform],result:wp.array2d[float]):
 i=wp.tid()
 sdf,grad,vel,cid,mat=collision_sdf(points[i],-2,collider,body_q,body_qd,empty,1.0/60.0)
 found,bsdf,bgrad,bvel,cp,bmat=_query_collider_sdf(points[i],collider,body_q,48)
 result[i,0]=sdf;result[i,1]=float(cid);result[i,2]=float(mat);result[i,3]=bsdf;result[i,4]=float(bmat)
 result[i,5]=grad[0];result[i,6]=grad[1];result[i,7]=grad[2]
 result[i,8]=vel[0];result[i,9]=vel[1];result[i,10]=vel[2]
 result[i,11]=bgrad[0];result[i,12]=bgrad[1];result[i,13]=bgrad[2]

root=Path('/home/qunwang/wajueji');work=Path(__file__).resolve().parent;run=work/'replay-v2'
source=root/'runs/layered-47-loaded-transport-v1/loaded-transport-20260923-215454-962382'
rows=[json.loads(l) for l in (source/'physics.ndjson').read_text().splitlines()]
proxy=json.loads((root/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json').read_text())
bundle=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1/repo')
base=json.loads((bundle/'configs/v4/layered-47-excavation-preflight-v1.json').read_text());manifest=json.loads((bundle/base['physical_geometry']['manifest_path']).read_text());records=[r for r in manifest['rocks'] if r['layer_id'] in pf.LAYERS]
pose=np.load(run/'actual-bucket-poses.npy',mmap_mode='r');pre=np.load(run/'pre-projection-world.npy',mmap_mode='r');post=np.load(run/'post-projection-world.npy',mmap_mode='r');trans=np.load(run/'projection-bucket-transforms.npy',mmap_mode='r')
events=[e for e in json.loads((work/'first-exits.json').read_text()) if e['initial_cavity'] and e['face'].startswith('bucket_shell') and e['outer_face_within_5_frames']]
state=rows[0]['before_state'];initial={'bucket_origin':pose[0,:3,3].tolist(),'bucket_rotation':pose[0,:3,:3].reshape(-1).tolist(),'bucket_linear_velocity_world_m_s':state['bucket_linear_velocity_world_m_s'],'bucket_angular_velocity_world_rad_s':state['bucket_angular_velocity_world_rad_s'],'rocks':[{**r,'linear_velocity_world_m_s':r['qvel'][:3],'angular_velocity_world_rad_s':[0,0,0]} for r in state['rocks']]}
wp.init();wp.set_device('cuda:0');newton.use_coord_layout_targets=True
_,fixtures=truck.load_candidate(root/'configs/v4/layered-47-truck-physics-v1.json',root)
profile=json.loads((root/'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089/report.json').read_text())['solver_profile']
fixture=pf.ExcavationFixture(base,bundle,records,np.asarray([[0,0,3]],dtype=np.float32),proxy,initial,'cuda:0',profile,fixtures)
empty=None
out=[]
# Query each event's crossing step and up to 5 following steps, until confirmed outer-face escape.
queries={}
for e in events:
 for f in range(e['frame'],e['outer_face_within_5_frames']['frame']+1):queries.setdefault(f,[]).append(e)
for f,es in sorted(queries.items()):
 before=rows[f-1]['before_state'];bodyq=[];bodyqd=[]
 for r in before['rocks']:
  bodyq.append(pf._make_transform(r['qpos'][:3],r['qpos'][3:7])); angular=quaternion_matrix_wxyz(r['qpos'][3:7])@np.asarray(r['qvel'][3:])
  lin=np.asarray(r['qvel'][:3])+np.asarray(base['mpm']['gravity_m_s2'])/60
  bodyqd.append(wp.spatial_vector(*lin,*angular))
 t=trans[f-1];bodyq.append(wp.transform(wp.vec3(*t[:3]),wp.quat(*t[3:])));bodyqd.append(wp.spatial_vector(*before['bucket_linear_velocity_world_m_s'],*before['bucket_angular_velocity_world_rad_s']))
 qarr=wp.array(bodyq,dtype=wp.transform,device='cuda:0');qdarr=wp.array(bodyqd,dtype=wp.spatial_vector,device='cuda:0')
 ix=np.array([e['tracked_index'] for e in es]);points=wp.array(pre[f-1,ix],dtype=wp.vec3,device='cuda:0');output=wp.empty((len(es),14),dtype=float,device='cuda:0')
 fixture.solver._mpm_model.collider.query_max_dist=rows[f-1]['postsolve_projection']['query_gap_m']
 wp.launch(query,dim=len(es),inputs=[points,fixture.solver._mpm_model.collider,qarr,qdarr,empty,output],device='cuda:0')
 vals=output.numpy()
 for e,v in zip(es,vals):
  i=e['tracked_index'];box=next(b for b in proxy['shell_boxes'] if b['name']==e['face']);rot=quaternion_matrix_wxyz(box['quaternion_wxyz']);h=box['half_size_m'][2]
  p0=(pre[f-1,i]-pose[f-1,:3,3])@pose[f-1,:3,:3];p1=(post[f,i]-pose[f-1,:3,3])@pose[f-1,:3,:3]
  z0=((p0-box['position_bucket_m'])@rot)[2];z1=((p1-box['position_bucket_m'])@rot)[2]
  out.append({'particle_id':e['particle_id'],'tracked_index':i,'frame':f,'first_crossing_frame':e['frame'],'face':e['face'],'first_crossing_phase':e['phase'],'pre_local':p0.tolist(),'post_local_same_collider_pose':p1.tolist(),'slab_outward_z_pre_m':float(z0),'slab_outward_z_post_m':float(z1),'slab_half_thickness_m':h,'query_sdf_m':float(v[0]),'query_owner':int(v[1]),'query_material':int(v[2]),'bucket_sdf_m':float(v[3]),'bucket_material':int(v[4]),'query_gradient_world':v[5:8].tolist(),'query_velocity_world':v[8:11].tolist(),'bucket_gradient_local':v[11:14].tolist(),'projection_displacement_world_m':(post[f,i]-pre[f-1,i]).tolist()})
(work/'event-collision-queries.json').write_text(json.dumps(out,indent=2)+'\n')
from collections import Counter
# Sign convention of these production boxes is +Z outward; explicit points provide auditability.
wrong=[r for r in out if r['query_owner']==48 and 0<r['slab_outward_z_pre_m']<r['slab_half_thickness_m'] and r['slab_outward_z_post_m']>r['slab_half_thickness_m']+1e-4]
summary={'query_only_no_physics':True,'query_rows':len(out),'owners':dict(Counter(r['query_owner'] for r in out)),'inside_outer_half_projected_beyond_wall_rows':len(wrong),'inside_outer_half_projected_beyond_wall_unique_particles':len(set(r['particle_id'] for r in wrong)),'examples':wrong[:12]}
(work/'query-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='examples'},indent=2))
