import json,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
sys.path.insert(0,'/home/qunwang/codex-work/fill-volume-2026-09-15-v1')
from fill_volume import bucket_volume,Calibration,cavity_mask
from layered_excavation_approach_planner import _load_kinematics,quaternion_matrix_wxyz
from run_layered_47_sparse_manual_contact_episode import _cache_rock_vertices
root=Path('/home/qunwang/wajueji');out=root/'work/manual-fill-20260923-163839-v1';run=root/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-163839-751464'
repo=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1/repo')
source=run/'final-001370.diagnostic.npz';report=json.loads((run/'report.json').read_text());data=np.load(source)
rows=[json.loads(x) for x in (run/'physics.ndjson').read_text().splitlines()];last=rows[-1]['after_state']
contract_path=Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json');front_path=contract_path.with_name('front-rim.npz');contract=json.loads(contract_path.read_text());front=np.load(front_path)['front']
calpath=Path('/home/qunwang/codex-work/fill-volume-2026-09-15-v1/calibration-known-geometry-v1/report.json');calreport=json.loads(calpath.read_text());cal=Calibration.from_dict(calreport['profiles']['reference-2p5cm']['selected']['calibration']);assert calreport['pass']
kin=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');assembly,_=_load_kinematics(kin,kin/'scene/input/model/A.xml',kin/'scene/input/mesh');pose=assembly.assemble(data['mechanism_qpos'],include_meshes=False)[1]['bucket'];assert np.max(np.abs(pose[:3,3]-last['bucket_origin_world_m']))<1e-6
records=json.loads((repo/'results/layered-mound-physical-preflight-v1/scene-manifest.json').read_text())['rocks'];records=[r for r in records if r['layer_id'] in ('lower','lower_middle','middle','upper')];assert len(records)==len(data['rigid_qpos'])==47
vertices=_cache_rock_vertices(repo,records)
soil=data['particle_q'];local=(soil-pose[:3,3])@pose[:3,:3];polygon=np.asarray(contract['bucket']['polygon_xz_m']);ylim=np.asarray(contract['bucket']['y_limits_m'])
inside=cavity_mask(local,polygon,ylim,front)
vbucket=np.array(last['bucket_linear_velocity_world_rad_s']) if 'bucket_linear_velocity_world_rad_s' in last else np.array(last['bucket_linear_velocity_world_m_s'])
w=np.array(last['bucket_angular_velocity_world_rad_s']);relative_speed=np.linalg.norm(data['particle_qd']-vbucket-np.cross(w,soil-pose[:3,3]),axis=1)
results={'soil_geometry_only':bucket_volume(soil,pose,contract,front,.025**3,cal)}
rock_info=[];clouds={}
for spacing in (.025,.0125):
 points=[];weights=[]
 for index,(record,v,q) in enumerate(zip(records,vertices,data['rigid_qpos'],strict=True)):
  rotation=quaternion_matrix_wxyz(q[3:]);worldv=v@rotation.T+q[:3];lv=(worldv-pose[:3,3])@pose[:3,:3]
  if np.linalg.norm(lv.mean(0)-np.r_[.4,0,-.5])>1.6:continue
  hull=ConvexHull(v)
  axes=[np.arange(v[:,j].min()+spacing/2,v[:,j].max(),spacing) for j in range(3)];pts=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3);pts=pts[np.all(pts@hull.equations[:,:3].T+hull.equations[:,3]<=1e-10,axis=1)]
  if not len(pts):continue
  world=pts@rotation.T+q[:3];points.append(world);weights.extend([record['volume_m3']/len(pts)]*len(pts))
  if spacing==.025:
   cp=cavity_mask((world-pose[:3,3])@pose[:3,:3],polygon,ylim,front)
   rock_info.append({'id':record['rock_id'],'index':index,'volume_liters':record['volume_m3']*1000,'fraction_in_cavity':float(cp.mean()),'center_bucket_m':((q[:3]-pose[:3,3])@pose[:3,:3]).tolist(),'minimum_world_z':float(worldv[:,2].min())})
 rp=np.concatenate(points) if points else np.empty((0,3));weights=np.asarray(weights);clouds[spacing]=(rp,weights)
 results[f'mixed_geometry_rock_samples_{spacing}']=bucket_volume(np.concatenate((soil,rp)),pose,contract,front,np.r_[np.full(len(soil),.025**3),weights],cal)
rp,weights=clouds[.0125]
for limit in (.25,.5,1.):
 keep=relative_speed<=limit
 results[f'mixed_velocity_screen_{limit}']=bucket_volume(np.concatenate((soil[keep],rp)),pose,contract,front,np.r_[np.full(keep.sum(),.025**3),weights],cal)
# Rock relative displacement in the final half-second, using recorded poses.
for info in rock_info:
 coords=[]
 for row in rows[-31:]:
  st=row['after_state'];bp=assembly.assemble(np.asarray(st['mechanism_qpos']),include_meshes=False)[1]['bucket'];p=np.asarray(st['rocks'][info['index']]['qpos'][:3]);coords.append((p-bp[:3,3])@bp[:3,:3])
 coords=np.array(coords);info['relative_center_drift_last_0p5s_m']=float(np.linalg.norm(coords-coords[0],axis=1).max())
result={'run':str(run),'snapshot':str(source),'snapshot_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'final_frame':1370,'final_simulation_seconds':1370/60,'all_hard_guards_passed':report['episode_summary']['all_frames_hard_safe'],'capacity_liters':255.164729950989,'measurement':results,'rock_candidates':rock_info,'cavity_soil_particles':int(inside.sum()),'cavity_soil_relative_speed_quantiles':np.quantile(relative_speed[inside],[0,.5,.9,1]).tolist(),'calibration':cal.__dict__,'scope':'Snapshot geometric diagnostic; pure-soil calibration extended with solid-rock interior samples is not qualified mixed-material fill. Velocity screens are sensitivity only, not validated retention rules. No stable truck-delivery qualification.','source_bindings':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,run/'report.json',contract_path,front_path,calpath,Path('/home/qunwang/codex-work/fill-volume-2026-09-15-v1/fill_volume.py'),Path(__file__))}}
(out/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps(result,ensure_ascii=False,indent=2))
