"""Read-only endpoint geometry and material-identity retention diagnostic."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
sys.path.insert(0,'/home/qunwang/codex-work/fill-volume-2026-09-15-v1')
from fill_volume import bucket_volume,Calibration,cavity_mask,density_field
from scipy.ndimage import label,binary_dilation,map_coordinates
from layered_excavation_approach_planner import quaternion_matrix_wxyz
from run_layered_47_sparse_manual_contact_episode import _cache_rock_vertices
root=Path('/home/qunwang/wajueji');out=Path(__file__).parent;mode=sys.argv[1];folder=out/mode
repo=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1/repo')
cp=Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json');fp=cp.with_name('front-rim.npz');contract=json.loads(cp.read_text());front=np.load(fp)['front']
calp=Path('/home/qunwang/codex-work/fill-volume-2026-09-15-v1/calibration-known-geometry-v1/report.json');calr=json.loads(calp.read_text());cal=Calibration.from_dict(calr['profiles']['reference-2p5cm']['selected']['calibration']);assert calr['pass']
records=json.loads((repo/'results/layered-mound-physical-preflight-v1/scene-manifest.json').read_text())['rocks'];records=[r for r in records if r['layer_id'] in ('lower','lower_middle','middle','upper')];assert len(records)==47
vertices=_cache_rock_vertices(repo,records);polygon=np.asarray(contract['bucket']['polygon_xz_m']);ylim=np.asarray(contract['bucket']['y_limits_m'])
rows=[json.loads(x) for x in (folder/'trace.ndjson').read_text().splitlines()];byframe={x['frame']:x for x in rows};cache={};masks={};rockmasks={};results={};bindings={}

def load_region_mask(points_world,pose,weights):
 local=(points_world-pose[:3,3])@pose[:3,:3]
 cavity=cavity_mask(local,polygon,ylim,front)
 low=np.array([polygon[:,0].min(),ylim[0],polygon[:,1].min()]);high=np.array([polygon[:,0].max(),ylim[1],polygon[:,1].max()])
 chosen=((local>=low-.8)&(local<=high+.8)).all(axis=1);cloud=local[chosen]
 field,origin,cell=density_field(cloud,weights[chosen],low-.8,high+.8,cell_m=cal.measurement_cell_m,sigma_cells=cal.smoothing_sigma_cells)
 labels,_=label(field>=cal.occupancy_level);idx=np.argwhere(field>=cal.occupancy_level);loc=origin+idx*cell
 components=labels[tuple(idx.T)] if len(idx) else np.empty(0,int);anchors=np.unique(components[cavity_mask(loc,polygon,ylim,front)])
 near=binary_dilation(np.isin(labels,anchors)&(labels!=0),iterations=2)
 point_near=map_coordinates(near.astype(np.uint8),((cloud-origin)/cell).T,order=0,mode='constant').astype(bool)
 vertical=pose[2,:3];rim=np.c_[front[:,0],np.zeros(len(front)),front[:,1]];min_height=(rim@vertical).min()
 included=cavity[chosen]|(point_near&(cloud@vertical>=min_height))
 mask=np.zeros(len(local),bool);mask[np.flatnonzero(chosen)]=included
 return mask

for frame in (1370,1490):
 source=folder/f'diagnostic-{frame:06}.npz';data=dict(np.load(source));row=byframe[frame];pose=np.array(row['bucket_pose']);state=json.loads((folder/f'bridge-state-{frame:06}.json').read_text());soil=data['particle_q'];local=(soil-pose[:3,3])@pose[:3,:3];inside=cavity_mask(local,polygon,ylim,front);masks[frame]=inside
 speed=np.linalg.norm(data['particle_qd']-np.array(state['bucket_linear_velocity_world_m_s'])-np.cross(state['bucket_angular_velocity_world_rad_s'],soil-pose[:3,3]),axis=1)
 points=[];weights=[];rockinfo=[];rockmasks[frame]={}
 for index,(r,v,q) in enumerate(zip(records,vertices,data['rigid_qpos'],strict=True)):
  rot=quaternion_matrix_wxyz(q[3:]);worldv=v@rot.T+q[:3];lv=(worldv-pose[:3,3])@pose[:3,:3]
  if np.linalg.norm(lv.mean(0)-np.r_[.4,0,-.5])>1.6:continue
  if index not in cache:
   spacing=.0125;hull=ConvexHull(v);axes=[np.arange(v[:,j].min()+spacing/2,v[:,j].max(),spacing) for j in range(3)];pts=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3);pts=pts[np.all(pts@hull.equations[:,:3].T+hull.equations[:,3]<=1e-10,axis=1)];assert len(pts);cache[index]=pts
  pts=cache[index];world=pts@rot.T+q[:3];points.append(world);weights.extend([r['volume_m3']/len(pts)]*len(pts));cm=cavity_mask((world-pose[:3,3])@pose[:3,:3],polygon,ylim,front);rockmasks[frame][index]=cm
  rockinfo.append({'id':r['rock_id'],'index':index,'volume_liters':r['volume_m3']*1000,'fraction_in_cavity':float(cm.mean())})
 rp=np.concatenate(points) if points else np.empty((0,3));weights=np.array(weights)
 mixed=bucket_volume(np.concatenate((soil,rp)),pose,contract,front,np.r_[np.full(len(soil),.025**3),weights],cal)
 counted=load_region_mask(np.concatenate((soil,rp)),pose,np.r_[np.full(len(soil),.025**3),weights]);assert counted.sum()==mixed['load_region_particles']
 masks[frame]=counted[:len(soil)];offset=len(soil)
 for info in rockinfo:
  index=info['index'];n=len(cache[index]);rockmasks[frame][index]=counted[offset:offset+n];offset+=n
 assert offset==len(counted)
 results[str(frame)]={'mixed_geometry':mixed,'cavity_soil_count':int(inside.sum()),'cavity_soil_relative_speed_quantiles_m_s':np.quantile(speed[inside],[.5,.9,1]).tolist() if inside.any() else [],'rocks':rockinfo,'particle_count':len(soil),'rock_count':len(data['rigid_qpos']),'snapshot_sha256':hashlib.sha256(source.read_bytes()).hexdigest()};bindings[str(source)]=results[str(frame)]['snapshot_sha256']
 print(mode,frame,mixed['fill_percent'],flush=True)
retention={}
for a,b in ((1370,1490),):
 initial=masks[a];final=masks[b];rocks=[]
 for index,mask in rockmasks[a].items():
  if not mask.any():continue
  last=rockmasks[b].get(index,np.zeros_like(mask));coords=[]
  for frame in range(a,b+1):
   row=byframe[frame];pose=np.array(row['bucket_pose']);q=np.array(row['rock_qpos'][index][:3]);coords.append((q-pose[:3,3])@pose[:3,:3])
  coords=np.array(coords);rocks.append({'id':records[index]['rock_id'],'initial_in_load_region_liters':records[index]['volume_m3']*1000*mask.mean(),'same_sample_retained_fraction':float(last[mask].mean()),'relative_center_max_drift_m':float(np.linalg.norm(coords-coords[0],axis=1).max())})
 retention[f'{a}-{b}']={'initial_load_region_soil_count':int(initial.sum()),'same_soil_ids_still_in_load_region':int((initial&final).sum()),'same_soil_retained_fraction':float(final[initial].mean()) if initial.any() else None,'new_soil_ids_in_load_region':int((~initial&final).sum()),'rocks':rocks,'scope':'cavity plus connected above-rim load-region identities using canonical fill_volume quality region; matches its particle count; this is geometric retention, not proven detached payload'}
result={'mode':mode,'endpoints':results,'retention':retention,'training_eligible':False,'off_soil_proven':False,'physical_force_qualified':False,'source_bindings':bindings,'analysis_bindings':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (cp,fp,calp,Path(__file__),Path('/home/qunwang/codex-work/fill-volume-2026-09-15-v1/fill_volume.py'))}}
(folder/'load-region-identity.json').write_text(json.dumps(result,indent=2));print('MEASUREMENT_SAVED',mode,flush=True)
