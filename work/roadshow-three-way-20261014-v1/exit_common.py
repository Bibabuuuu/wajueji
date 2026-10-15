"""Read-only receipt diagnostics and unchanged transport-controller handoff."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path('/home/qunwang/wajueji');WORK=ROOT/'work/roadshow-three-way-20261014-v1'
PREVIOUS=ROOT/'runs/d165-transport-full-pair-20261014-v1';CPU=WORK/'exit-settle-cpu-v1'
sys.path[:0]=[str(ROOT/'src/v4'),str(ROOT/'work/contact-second-fixed-route-20260925-v1'),str(ROOT)]
from layered_47_cycle_runtime import CycleRuntime
from controller import TransportController,LIMITS
from clearance import FixedRouteClearance
from layered_47_cycle_geometry import bucket_mask
from planning.rear_lower_prepoint import proxy_surface_samples
from terrain.truck_receiver import TruckReceiver,CAVITY,BOUNDARY,ABOVE_RIM,OUTSIDE
from scipy.spatial import cKDTree
from scipy.ndimage import label
import run_layered_47_sparse_manual_contact_episode as manual

def encode(x):return json.dumps(manual._clean(x),default=lambda v:v.item(),ensure_ascii=False,indent=2,allow_nan=False)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(encode(x)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

class CpuRuntime(CycleRuntime):
    def __enter__(self):raise RuntimeError('cpu_native_entry_forbidden')
    def step(self,command):raise RuntimeError('cpu_physical_step_forbidden')

def source_info(name,label='final'):
    source=PREVIOUS/name/(label+'.npz');side=Path(str(source)+'.json');control=Path(str(source)+'.control.json')
    info=dict(source=str(source),sha256=sha(source),metadata_sha256=sha(side),control_sha256=sha(control))
    meta=read(side);state=read(control)
    if meta['checkpoint_sha256']!=info['sha256'] or state['checkpoint_sha256']!=info['sha256']:
        raise ValueError('source_identity')
    if meta['stage'] not in ('transport','unloaded') or not meta['truck_enabled'] or meta.get('training_reset'):
        raise ValueError('source_scope')
    return info

def controller(r,state):
    c=TransportController(r,FixedRouteClearance(r),state['previous_command'])
    c.tracking.bias=np.array(state['load_bias']);c.first=False
    if c.previous.shape!=(4,) or c.tracking.bias.shape!=(4,) or not np.isfinite(np.r_[c.previous,c.tracking.bias]).all():
        raise ValueError('invalid_high_level_history')
    return c

def receiver():return TruckReceiver(read(ROOT/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-contract.json'))

def bucket_pose(r,q):return r.assembly.assemble(q,include_meshes=False)[1]['bucket']

def bucket_opening_gap(r,q):
    geometries,_=r.assembly.assemble(q)
    local=np.concatenate([g['body_local_vertices'] for g in geometries if g['body']=='bucket'])
    pose=bucket_pose(r,q);world=local@pose[:3,:3].T+pose[:3,3]
    truck=receiver();t=(world-truck.pose[:3,3])@truck.pose[:3,:3]
    low=np.array([truck.rim[0,0],truck.rim[:,1].min()]);high=np.array([truck.rim[-1,0],truck.rim[:,2].max()])
    gap=np.maximum(np.maximum(low-t[:,:2].max(0),t[:,:2].min(0)-high),0)
    return float(np.linalg.norm(gap))

def soil_observation(r,cloud,q,velocity=None,previous_cloud=None):
    truck=receiver();classes=truck.classify(cloud);pose=bucket_pose(r,q)
    local=(cloud-pose[:3,3])@pose[:3,:3]
    # Approximate shell proximity is a diagnostic, never an ownership/contact oracle.
    samples=proxy_surface_samples(r.proxy,maximum_spacing_m=.02)
    distance=cKDTree(samples).query(local,workers=1)[0]
    cavity=bucket_mask(cloud,pose);near=distance<=.05
    result=dict(partition={k:int(np.count_nonzero(classes==v)) for k,v in [('outside',OUTSIDE),('core',CAVITY),('boundary',BOUNDARY),('above_rim',ABOVE_RIM)]},
        bucket_cavity_count=int(cavity.sum()),bucket_near_50mm_count=int(near.sum()),
        above_rim_near_bucket_by_distance={str(d):int(np.count_nonzero((classes==ABOVE_RIM)&(distance<=d))) for d in [.02,.05,.1]},
        core_near_bucket_50mm=int(np.count_nonzero((classes==CAVITY)&near)),
        above_rim_bucket_cavity=int(np.count_nonzero((classes==ABOVE_RIM)&cavity)),
        bucket_outside_opening_aabb_gap_m=bucket_opening_gap(r,q),
        scope='Geometric stock, approximate shell proximity and motion only; no force-support or calibrated bulk-volume claim')
    upper=classes==ABOVE_RIM
    if upper.any():
        result['above_rim_shell_distance_p0_p50_p95_p100_m']=np.quantile(distance[upper],[0,.5,.95,1]).tolist()
        if velocity is not None:result['above_rim_speed_p50_p95_max_m_s']=np.quantile(np.linalg.norm(velocity[upper],axis=1),[.5,.95,1]).tolist()
        if previous_cloud is not None:result['above_rim_displacement_p50_p95_max_m']=np.quantile(np.linalg.norm(cloud[upper]-previous_cloud[upper],axis=1),[.5,.95,1]).tolist()
    return result,classes,distance

def graph_diagnostic(cloud,classes):
    ids=np.flatnonzero((classes==CAVITY)|(classes==ABOVE_RIM));p=cloud[ids];out={}
    for cell in [.025,.05]:
        ijk=np.floor((p-p.min(0))/cell).astype(int);shape=ijk.max(0)+1
        if np.prod(shape)>12000000:raise ValueError('diagnostic_grid_ceiling')
        grid=np.zeros(shape,bool);grid[tuple(ijk.T)]=True;components,count=label(grid,np.ones((3,3,3)))
        tags=components[tuple(ijk.T)];anchors=np.unique(tags[classes[ids]==CAVITY]);linked=np.isin(tags,anchors)
        out[str(cell)]=dict(components=int(count),above_rim_geometrically_connected_to_core=int(np.count_nonzero(linked&(classes[ids]==ABOVE_RIM))),
            above_rim_not_connected=int(np.count_nonzero(~linked&(classes[ids]==ABOVE_RIM))),
            scope='26-neighbor voxel adjacency; does not establish contact forces or supported cargo')
    return out
