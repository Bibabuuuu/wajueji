"""Geometry-only transport and workspace planning for the continuous pilot."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from planning.cycle_geometry import MOUTH, solve_pose
from planning.approach import sampled_approach
from planning.rear_lower_prepoint import proxy_surface_samples, _aabb_separation
from layered_excavation_approach_planner import terrain_from_points, rock_surface_points
from audit_layered_47_loaded_handoff import cavity_mask
from layered_47_truck_physics_candidate import clearance, load_candidate
from mixed_material_r0 import quaternion_matrix_wxyz
from terrain.truck_receiver import TruckReceiver, CAVITY


def bucket_mask(points,pose):
    contract=json.loads(Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json').read_text())
    with np.load('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/front-rim.npz') as data:front=data['front']
    local=(points-pose[:3,3])@pose[:3,:3]
    return cavity_mask(local,np.array(contract['bucket']['polygon_xz_m']),np.array(contract['bucket']['y_limits_m']),front)


def stationary_terrain(runtime):
    """Exclude a detached observed load component, never change physical arrays."""
    q=runtime.arrays['particle_q'];pose=runtime.assembly.assemble(runtime.arrays['mechanism_qpos'],include_meshes=False)[1]['bucket']
    cavity=bucket_mask(q,pose)
    cell=.04;origin=q.min(0)-2*cell;ijk=np.floor((q-origin)/cell).astype(int)
    shape=ijk.max(0)+3
    if np.prod(shape)>5000000:raise ValueError('cycle_observation_grid_budget')
    grid=np.zeros(tuple(shape),dtype=bool);grid[tuple(ijk.T)]=True
    components,_=label(grid)
    labels=components[tuple(ijk.T)];anchors=np.unique(labels[cavity]);anchors=anchors[anchors>0]
    payload=np.isin(labels,anchors)
    if payload.any() and (q[payload,2].min()<.3 or np.max(np.ptp(q[payload],axis=0))>2.):
        raise ValueError('cycle_load_not_observably_detached')
    rocks,by_id=rock_surface_points(runtime.root,runtime.records,runtime.arrays['rigid_qpos'])
    kept=[]
    for points in by_id.values():
        # Classify the rock geometry at the same observed boundary as the soil.
        indices=np.floor((points-origin)/cell).astype(int)
        inside=((indices>=0)&(indices<shape)).all(1)
        linked=np.zeros(len(points),bool)
        linked[inside]=np.isin(components[tuple(indices[inside].T)],anchors)
        if not (linked.any() or bucket_mask(points,pose).any()):kept.append(points)
    ground_labels=np.unique(labels[q[:,2]<.06])
    stationary=np.isin(labels,ground_labels)
    if np.count_nonzero(stationary)<1000:raise ValueError('cycle_stationary_stock_unobserved')
    terrain=terrain_from_points(np.concatenate([q[stationary],*kept]))
    return terrain,dict(detached_payload_particles=int(payload.sum()),
        detached_dynamic_particles=int(np.count_nonzero(~stationary)),
        observation_only=True,cell_m=cell,
        scope='ground-connected stock surface; detached material remains fully simulated')


def oriented_box_separation(corners, rotation, lower, upper):
    """SAT projected gap: conservative lower bound on convex box distance."""
    other=np.array([[x,y,z] for x in (lower[0],upper[0]) for y in (lower[1],upper[1]) for z in (lower[2],upper[2])])
    axes=[*np.eye(3),*rotation.T]
    axes += [np.cross(a,b) for a in np.eye(3) for b in rotation.T]
    separation=0.
    for axis in axes:
        norm=np.linalg.norm(axis)
        if norm<1e-10:continue
        axis=axis/norm;a=corners@axis;b=other@axis
        separation=max(separation,float(b.min()-a.max()),float(a.min()-b.max()))
    return separation


def triangles_intersect_box(triangles, lower, upper, margin=.02):
    """Triangle/expanded-AABB separating-axis test in the box's local frame."""
    half=(upper-lower)/2+margin;center=(upper+lower)/2
    tri=np.asarray(triangles)-center
    broad=np.all(tri.max(1)>=-half,axis=1)&np.all(tri.min(1)<=half,axis=1)
    tri=tri[broad]
    if not len(tri):return False
    edges=np.roll(tri,-1,axis=1)-tri
    axes=[np.cross(edges[:,0],edges[:,1])]
    for edge in range(3):
        for basis in np.eye(3):axes.append(np.cross(edges[:,edge],basis))
    separated=np.zeros(len(tri),dtype=bool)
    for axis in axes:
        projection=np.einsum('nij,nj->ni',tri,axis)
        radius=np.abs(axis)@half
        separated|=(projection.min(1)>radius+1e-10)|(projection.max(1)<-radius-1e-10)
    return bool(np.any(~separated))


class TransportClearance:
    def __init__(self,runtime):
        self.runtime=runtime
        self.terrain,self.observation=stationary_terrain(runtime)
        self.samples=proxy_surface_samples(runtime.proxy)
        self.geometries,_=runtime.assembly.assemble(runtime.arrays['mechanism_qpos'])
        _,self.fixtures=load_candidate(Path(__file__).resolve().parents[2]/'configs/v4/layered-47-truck-physics-v1.json',Path(__file__).resolve().parents[2])
        with np.load(Path(__file__).resolve().parents[2]/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-mesh.npz') as data:
            truck=data['world_vertices'];self.truck_lo=truck.min(0);self.truck_hi=truck.max(0)
            self.truck_triangles=truck[data['faces']]
        self.bounds={}
        for i,g in enumerate(self.geometries):
            v=np.asarray(g['body_local_vertices']);lo=v.min(0);hi=v.max(0)
            self.bounds[i]=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])

    def material_clearance(self,points):
        t=self.terrain;h=np.asarray(t['height_m']);idx=np.floor((points[:,:2]-t['origin_xy_m'])/t['resolution_m']).astype(int)
        inside=(idx[:,0]>=0)&(idx[:,0]<h.shape[1])&(idx[:,1]>=0)&(idx[:,1]<h.shape[0])
        height=np.zeros(len(points));surface=h[idx[inside,1],idx[inside,0]]
        height[inside]=np.where(np.isfinite(surface),np.maximum(surface,0),0)
        return float(np.min(points[:,2]-height))

    def check(self,q,previous=None,*,physical=False):
        _,bodies=self.runtime.assembly.assemble(q,include_meshes=False)
        pose=bodies['bucket'];points=self.samples@pose[:3,:3].T+pose[:3,3]
        bucket_truck=clearance(points,self.fixtures)['minimum_point_center_clearance_m']
        bucket_terrain=self.material_clearance(points)
        mechanism_truck=float('inf');mechanism_ground=float('inf');nearest_mesh=None
        for i,g in enumerate(self.geometries):
            if g['mesh'] in ('bucket','caterpillar','chassis'):continue
            p=bodies[g['body']];corners=self.bounds[i]@p[:3,:3].T+p[:3,3]
            separation=oriented_box_separation(corners,p[:3,:3],self.truck_lo,self.truck_hi)
            if separation<.02:
                # Full-truck box overlaps its empty bed and empty space below
                # a tilted boom. Refine conservatively against CAD triangles.
                local=(self.truck_triangles-p[:3,3])@p[:3,:3]
                hit=triangles_intersect_box(local,self.bounds[i].min(0),self.bounds[i].max(0))
                separation=0. if hit else .02
            if separation<mechanism_truck:mechanism_truck=separation;nearest_mesh=g['mesh']
            mechanism_ground=min(mechanism_ground,float(corners[:,2].min()))
        closure=float(np.max(np.abs(self.runtime.chain.residual(q))))
        result=dict(bucket_truck_m=bucket_truck,bucket_terrain_m=bucket_terrain,
                    mechanism_truck_m=mechanism_truck,mechanism_ground_m=mechanism_ground,closure_m=closure)
        if bucket_truck<.04 or bucket_terrain<.04 or mechanism_truck<.02 or mechanism_ground<.02 or closure>(.001 if physical else .0001):
            raise ValueError('cycle_transport_clearance:'+str(result)+':mesh='+str(nearest_mesh))
        return result


def transport_goals(runtime):
    contract=json.loads((Path(__file__).resolve().parents[2]/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-contract.json').read_text())
    floor=np.array(contract['triangles']['floor']).reshape(-1,3)
    ycenter=(floor[:,1].min()+floor[:,1].max())/2
    transform=np.array(contract['local_to_world'])
    center=transform@np.array([0,ycenter,0,1]);dump=transform@np.array([.6,ycenter,0,1])
    q=runtime.arrays['mechanism_qpos'].copy();pose=runtime.assembly.assemble(q,include_meshes=False)[1]['bucket']
    mouth=pose[:3,:3]@MOUTH+pose[:3,3];radial=mouth[:2]/np.linalg.norm(mouth[:2]);goals=[]
    for name,target,pitch in [('LIFT',np.r_[4.3*radial,3.3],1.),('ALIGN',np.r_[center[:2],3.5],1.),
            ('DUMP',np.r_[dump[:2],3.6],-.45),('CLEAR_TRUCK',np.r_[center[:2],3.5],1.),
            ('RETURN_HIGH',np.r_[4.3*radial,3.3],1.)]:
        q=solve_pose(runtime.assembly,runtime.chain,q,target,pitch,MOUTH)
        goals.append((name,q.copy()))
    return goals


def transport_plan(runtime):
    checker=TransportClearance(runtime);start=runtime.arrays['mechanism_qpos'].copy();segments=[]
    # Physical endpoint is slightly off the kinematic manifold. Start the
    # preview on the nearest closed chain, keeping the four actuated positions.
    runtime.chain.q=start.copy()
    start=runtime.chain.advance(np.zeros(4),1/60)[0]
    for name,goal in transport_goals(runtime):
        print('PLAN_SEGMENT',name,flush=True)
        def advance(q,v,dt):
            runtime.chain.q=q.copy();return runtime.chain.advance(v,dt)[0]
        path=sampled_approach(start,goal,axes=runtime.chain.axes,speed_limits=np.array([.1,.018,.018,.018]),
            advance=advance,check_pose=checker.check,dt=1/60,max_steps=2400,validation_substeps=2)
        diagnostics=path.pop('diagnostics')
        segments.append(dict(name=name,goal=goal,path=path,minimum_clearance={key:min(d[key] for d in diagnostics) for key in diagnostics[0]}))
        start=goal.copy()
    return segments,checker
