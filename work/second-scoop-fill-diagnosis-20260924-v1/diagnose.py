"""Read-only decomposition and sensitivity; never recalibrate from this scoop."""
from pathlib import Path
import sys, json
import numpy as np
from dataclasses import replace
from scipy.ndimage import label, map_coordinates, binary_fill_holes
# Reuse initialization only, without running or overwriting the prior review.
source=Path('/home/qunwang/wajueji/work/second-scoop-review-20260924-v1/review.py')
exec(compile(source.read_text().split('fill=e.measure_snapshot')[0],str(source),'exec'))
from mixed_material_r0 import quaternion_matrix_wxyz
m=sys.modules['_bound_fill_volume']
points=[a['particle_q']];weights=[np.full(len(a['particle_q']),.025**3)];rock_data=[]
for i,((verts,samples,weight),qpos,record) in enumerate(zip(e.rocks,a['rigid_qpos'],records,strict=True)):
 R=quaternion_matrix_wxyz(qpos[3:]);world=samples@R.T+qpos[:3]
 center=((verts@R.T+qpos[:3]).mean(0)-pose[:3,3])@pose[:3,:3]
 gate=np.linalg.norm(center-np.array([.4,0,-.5]))<=1.6
 rock_data.append(dict(index=i,rock_id=record['rock_id'],actual_volume_liters=record['volume_m3']*1000,center_bucket_m=center.tolist(),included_by_distance_gate=bool(gate),world=world))
 if gate:points.append(world);weights.append(np.full(len(world),weight))
cloud=np.concatenate(points);volume=np.concatenate(weights)
result=e.measure(cloud,pose,e.contract,e.front,volume,e.calibration)
local=(cloud-pose[:3,3])@pose[:3,:3]
poly=np.asarray(e.contract['bucket']['polygon_xz_m']);yl=np.asarray(e.contract['bucket']['y_limits_m'])
low=np.array([poly[:,0].min(),yl[0],poly[:,1].min()])-.8
high=np.array([poly[:,0].max(),yl[1],poly[:,1].max()])+.8
selected=((local>=low)&(local<=high)).all(1)
field,origin,cell=m.density_field(local[selected],volume[selected],low,high,cell_m=.02,sigma_cells=e.calibration.smoothing_sigma_cells)
bulk=field>=.4;labels,_=label(bulk);idx=np.argwhere(bulk);loc=origin+idx*cell
inside=m.cavity_mask(loc,poly,yl,e.front);ids=labels[tuple(idx.T)];anchors=np.unique(ids[inside]);connected=np.isin(ids,anchors)&(ids>0)
vertical=pose[2,:3];rim=np.c_[e.front[:,0],np.zeros(len(e.front)),e.front[:,1]];rimheight=(rim@vertical).min()
above=loc@vertical>=rimheight
connected_field=np.isin(labels,anchors)&(labels>0)
for rock in rock_data:
 world=rock.pop('world');locrock=(world-pose[:3,3])@pose[:3,:3]
 inside_rock=m.cavity_mask(locrock,poly,yl,e.front)
 density=map_coordinates(field,((locrock-origin)/cell).T,order=1,mode='constant')
 linked=map_coordinates(connected_field.astype(np.uint8),((locrock-origin)/cell).T,order=0,mode='constant').astype(bool)
 rock.update(cavity_fraction=float(inside_rock.mean()),fraction_in_anchored_bulk=float(linked.mean()),above_rim_fraction=float((locrock@vertical>=rimheight).mean()),density_supported_fraction=float((density>=.4).mean()))
# Diagnostics only: inspect enclosed voxel voids and parameter dependence.
holes=binary_fill_holes(bulk)&~bulk;hi=np.argwhere(holes);hl=origin+hi*cell
hole_inside=m.cavity_mask(hl,poly,yl,e.front)
sensitivity=[]
for level in (.30,.35,.40,.45,.50):
 v=e.measure(cloud,pose,e.contract,e.front,volume,replace(e.calibration,occupancy_level=level))
 sensitivity.append({k:v[k] for k in ('occupancy_level','fill_percent','inside_liters','heap_liters','valid_estimate')})
allidx=np.indices(field.shape).reshape(3,-1).T;allloc=origin+allidx*cell
cavity=m.cavity_mask(allloc,poly,yl,e.front)
empty=cavity&~bulk.ravel();emptyloc=allloc[empty]
sections=[]
for axis in range(3):
 bounds=np.linspace(allloc[cavity,axis].min(),allloc[cavity,axis].max()+1e-8,6)
 for j in range(5):
  region=cavity&(allloc[:,axis]>=bounds[j])&(allloc[:,axis]<bounds[j+1])
  sections.append(dict(axis=axis,lo=float(bounds[j]),hi=float(bounds[j+1]),capacity_liters=float(region.sum()*cell**3*1000*e.calibration.volume_scale),occupied_liters=float((region&bulk.ravel()).sum()*cell**3*1000*e.calibration.volume_scale)))
output=dict(schema='second_scoop_fill_decomposition_v1',same_snapshot=str(FINAL),official_unchanged=result,
 rock_details=rock_data,excluded_disconnected_above_rim_liters=float((~connected&above).sum()*cell**3*1000*e.calibration.volume_scale),
 connected_below_rim_outside_cavity_liters=float((connected&~inside&~above).sum()*cell**3*1000*e.calibration.volume_scale),
 enclosed_holes_inside_cavity_liters=float(hole_inside.sum()*cell**3*1000*e.calibration.volume_scale),
 enclosed_holes_total_liters=float(holes.sum()*cell**3*1000*e.calibration.volume_scale),
 cavity_voxel_capacity_liters=float(cavity.sum()*cell**3*1000*e.calibration.volume_scale),
 empty_cavity_voxel_liters=float(empty.sum()*cell**3*1000*e.calibration.volume_scale),
 sensitivity_diagnostic_only=sensitivity,cavity_sections=sections)
out=ROOT/'work/second-scoop-fill-diagnosis-20260924-v1';(out/'report.json').write_text(json.dumps(output,indent=2)+'\n')
np.savez_compressed(out/'diagnostic-grid.npz',field=field,origin=origin,cell=cell,pose=pose,polygon=poly,y_limits=yl,front=e.front)
print(json.dumps({k:v for k,v in output.items() if k not in ('cavity_sections','rock_details')},indent=2))
print('ROCKS',json.dumps([r for r in rock_data if r['included_by_distance_gate'] or r['cavity_fraction']>0],indent=2))
