"""CPU-only diagnosis of saved states; no simulator construction or scoring edits."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
from scipy.ndimage import label

ROOT=Path('/home/qunwang/wajueji')
sys.path[:0]=[str(ROOT/'src/v4'),str(ROOT)]
from layered_excavation_approach_planner import _load_kinematics
from live_bucket_fill import MixedFillEstimator,load_config
import run_layered_47_sparse_manual_contact_episode as manual

OUT=ROOT/'work/roadshow-three-way-20261014-v1/offline-boundary-v2'
OUT.mkdir(exist_ok=False)
started=time.monotonic()
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
kin=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
assembly,chain=_load_kinematics(str(kin),str(kin/'scene/input/model/A.xml'),str(kin/'scene/input/mesh'))
base=ROOT/'runs/roadshow-three-way-20261014-v1'
geo=base/'geometry-full'
scene=json.loads((geo/'display/scene.json').read_text())
config=load_config(ROOT/'configs/v4/live-bucket-fill-local-v1.json')
est=MixedFillEstimator(config,[r['volume_m3'] for r in scene['records']],manual._cache_rock_vertices(Path(scene['root']),scene['records']))
module=sys.modules[est.measure.__module__]
original_measure=est.measure
original_density=module.density_field
captures={}
def density(*args,**kwargs):
    result=original_density(*args,**kwargs)
    captures['field']=result
    return result
module.density_field=density
states=json.loads((geo/'display/states.json').read_text())
particles=np.load(geo/'display/particles.npy',mmap_mode='r')
paths=[Path(__file__),geo/'failure.npz',geo/'failure.npz.json',geo/'report.json',geo/'display/states.json',geo/'display/scene.json',ROOT/'configs/v4/live-bucket-fill-local-v1.json',Path(config['measurement_source'])]
before={str(p):digest(p) for p in paths}
rows=[]
def measure(snapshot,padding):
    def wrapped(*args,**kwargs):
        captures['inputs']=args
        raw=original_measure(*args,**kwargs,padding_m=padding)
        captures['raw']=raw
        return raw
    est.measure=wrapped
    result=est.measure_snapshot(snapshot)
    field,origin,cell=captures['field']
    labels,_=label(field>=est.calibration.occupancy_level)
    indices=np.argwhere(labels)
    locations=origin+indices*cell
    polygon=np.array(est.contract['bucket']['polygon_xz_m'])
    ylimits=np.array(est.contract['bucket']['y_limits_m'])
    inside=module.cavity_mask(locations,polygon,ylimits,est.front)
    ids=labels[tuple(indices.T)]
    anchors=np.unique(ids[inside]);connected=np.isin(ids,anchors)&(ids!=0)
    points=locations[connected]
    boundary=(indices[connected]<=1)|(indices[connected]>=np.array(field.shape)-2)
    local=(np.array(snapshot['soil'])-snapshot['pose'][:3,3])@snapshot['pose'][:3,:3]
    soil_inside=module.cavity_mask(local,polygon,ylimits,est.front)
    world=points@snapshot['pose'][:3,:3].T+snapshot['pose'][:3,3]
    return dict(**result,padding_m=padding,grid_shape=list(field.shape),grid_voxels=int(field.size),
        raw_estimator_diagnostic=captures['raw'],grid_origin=origin.tolist(),
        connected_voxels=int(len(points)),connected_cavity_voxels=int(inside.sum()),
        connected_local_bounds=[points.min(axis=0).tolist(),points.max(axis=0).tolist()] if len(points) else None,
        connected_world_z_bounds=[float(world[:,2].min()),float(world[:,2].max())] if len(world) else None,
        connected_boundary_axes=boundary.any(axis=0).tolist() if len(points) else [False]*3,
        soil_particle_centers_in_cavity=int(soil_inside.sum()),
        diagnostic_cavity_soil_equivalent_liters=float(soil_inside.sum()*.025**3*1000),
        geometry_pose=snapshot['pose'].tolist())

def checkpoint_snapshot(path,frame):
    with np.load(path) as data:
        pose=assembly.assemble(data['mechanism_qpos'],include_meshes=False)[1]['bucket']
        return dict(frame=frame,soil=data['particle_q'].copy(),pose=pose,rocks=data['rigid_qpos'].copy())

for frame in [2880,2895,2910]:
    state=states[frame]
    pose=assembly.assemble(np.array(state['mechanism_qpos']),include_meshes=False)[1]['bucket']
    assert np.allclose(pose[:3,3],state['bucket_origin_world_m'],atol=1e-8)
    snapshot=dict(frame=frame,soil=np.array(particles[frame]),pose=pose,rocks=np.array([x['qpos'] for x in state['rocks']]))
    for padding in ([.8,1.2,1.6] if frame==2910 else [.8]):
        assert time.monotonic()-started<180
        row=measure(snapshot,padding);row['case']='geometry'
        rows.append(row)
        print(json.dumps({k:row[k] for k in ['case','frame','padding_m','valid_estimate','fill_percent','connected_voxels','connected_world_z_bounds','soil_particle_centers_in_cavity']}),flush=True)
endpoint=checkpoint_snapshot(geo/'failure.npz',2910)
checks={'cpu_only':os.environ['CUDA_VISIBLE_DEVICES']=='','geometry_cache_final_particles_exact':np.array_equal(endpoint['soil'],particles[2910]),
        'geometry_cache_final_rocks_exact':np.array_equal(endpoint['rocks'],np.array([r['qpos'] for r in states[2910]['rocks']])),
        'geometry_cache_final_pose_exact':np.array_equal(endpoint['pose'],np.array(rows[-1]['geometry_pose']))}
for case,path,frame in [
    ('manual',base/'manual/endpoint.npz',1604),
    ('residual',ROOT/'runs/manual-strength-overnight-20261014/anchor-sweep-v1/seed44-weight100-independent-three/repetition-1/episode-0001/endpoint.npz',1206)]:
    paths.append(path);before[str(path)]=digest(path)
    snapshot=checkpoint_snapshot(path,frame)
    for padding in [.8,1.6]:
        assert time.monotonic()-started<180
        row=measure(snapshot,padding);row['case']=case;rows.append(row)
        print(json.dumps({k:row[k] for k in ['case','frame','padding_m','valid_estimate','fill_percent','connected_voxels']}),flush=True)
checks['original_sources_unchanged']=all(digest(p)==h for p,h in before.items())
checks['original_geometry_invalid_reproduced']=all(not r['valid_estimate'] for r in rows if r['case']=='geometry' and r['padding_m']==.8)
checks['manual_default_display_reproduced']=abs(next(r['fill_percent'] for r in rows if r['case']=='manual')-97.06992230854607)<1e-10
checks['residual_default_display_reproduced']=abs(next(r['fill_percent'] for r in rows if r['case']=='residual')-148.23212303789444)<1e-10
checks['geometry_has_no_stationary_hold']=not json.loads((geo/'report.json').read_text())['samples']
report=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,rows=rows,source_sha256=before,
    elapsed_seconds=time.monotonic()-started,new_physical_frames=0,neural_updates=0,
    score_scope='Diagnostic saved-state recomputation only. Expanded padding and cavity counts are not substitute short-hold scores.',
    geometry_terminal_qualified=False,official_geometry_fill_percent=None)
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':checks,'elapsed_seconds':report['elapsed_seconds']}))
assert all(checks.values())
