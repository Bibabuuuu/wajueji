"""Explicit contact-training branch: inventory transfer and direct empty prepoint.

Never a continuous-physics cycle. All particle/rock IDs remain present. Withdrawn
load is translated to an isolated ground inventory outside the observed workspace.
Unwithdrawn material position, velocity and constitutive fields are retained.
"""
from contextlib import nullcontext
from copy import deepcopy
import json
import numpy as np
import newton
from scipy.ndimage import label
from layered_47_cycle_runtime import CycleRuntime, REPO, APPROACH
from layered_47_cycle_bridge import controller_snapshot
from layered_47_cycle_geometry import bucket_mask, stationary_terrain, TransportClearance
from layered_47_cycle_selection import rank_workspace_candidates
from layered_excavation_approach_planner import rock_surface_points, terrain_from_points
from layered_frequency_bridge import HoldBridge
from planning.cycle_geometry import TIP, solve_pose
from planning.rear_lower_prepoint import (_joint_bounds_ok, mechanism_and_truck_clearance,
    ground_completed_terrain, proxy_surface_samples, sampled_surface_clearance)
import run_layered_47_sparse_motion as motion


MODE = 'fast_contact_collection_v1'


def detached_inventory(runtime):
    """Require the existing detachment gate before assigning any withdrawn IDs."""
    terrain, observation = stationary_terrain(runtime)
    a = runtime.arrays
    q = a['particle_q']
    pose = runtime.assembly.assemble(a['mechanism_qpos'],include_meshes=False)[1]['bucket']
    cell = .04
    origin = q.min(0)-2*cell
    ijk = np.floor((q-origin)/cell).astype(int)
    shape = ijk.max(0)+3
    if np.prod(shape)>12000000:
        raise ValueError('training_inventory_grid_budget')
    grid = np.zeros(tuple(shape),bool)
    grid[tuple(ijk.T)] = True
    components,_ = label(grid)
    ids = components[tuple(ijk.T)]
    anchors = np.unique(ids[bucket_mask(q,pose)])
    anchors = anchors[anchors>0]
    soil = np.isin(ids,anchors)
    if soil.sum()<100:
        raise ValueError('training_reset_requires_identified_load')
    _, rocks = rock_surface_points(runtime.root,runtime.records,a['rigid_qpos'])
    chosen = []
    for i,record in enumerate(runtime.records):
        points = rocks[record['rock_id']]
        indices = np.floor((points-origin)/cell).astype(int)
        inside = ((indices>=0)&(indices<shape)).all(1)
        linked = np.zeros(len(points),bool)
        linked[inside] = np.isin(components[tuple(indices[inside].T)],anchors)
        if linked.any() or bucket_mask(points,pose).any():
            if points[:,2].min()<.3:
                raise ValueError('training_withdrawn_rock_not_detached')
            chosen.append(i)
    return terrain,observation,np.flatnonzero(soil),np.asarray(chosen,dtype=int),rocks


def select_direct_prepoint(runtime, terrain, clearance_terrain):
    """Check target geometry only: no fictitious free-space approach path."""
    candidates = rank_workspace_candidates(terrain,3-runtime.metadata['cycle_index'])
    with np.load(APPROACH/'approach-handoff.npz') as data:
        open_seed = data['mechanism_qpos'].copy()
    checker = TransportClearance(runtime,terrain=clearance_terrain)
    collision = ground_completed_terrain(clearance_terrain)
    samples = proxy_surface_samples(runtime.proxy)
    with np.load(REPO/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-mesh.npz') as data:
        truck = data['world_vertices'].copy()
    geometries,_ = runtime.assembly.assemble(open_seed)
    rejected=[]
    for candidate in candidates[:16]:
        face=candidate['contact'];xy=np.asarray(face['contact_xy_m']);radial=xy/np.linalg.norm(xy)
        for outward,upward in ((.5,.6),(.5,.7),(.6,.6)):
            target=np.r_[xy+outward*radial,face['contact_surface_z_m']+upward]
            seed=open_seed.copy();seed[7]=np.arctan2(xy[1],xy[0])
            try:
                q=solve_pose(runtime.assembly,runtime.chain,seed,target,-.32,TIP)
                if not _joint_bounds_ok(q,runtime.chain):
                    raise ValueError('training_prepoint_joint_bounds')
                clearance=checker.check(q)
                _,bodies=runtime.assembly.assemble(q,include_meshes=False);pose=bodies['bucket']
                surface=sampled_surface_clearance(samples@pose[:3,:3].T+pose[:3,3],collision)['clearance_m']
                other=mechanism_and_truck_clearance(geometries,bodies,ground_completed_terrain(terrain),truck)
                if surface<.4 or any(v<.02 for v in other.values()):
                    raise ValueError('training_prepoint_clearance:'+str((surface,other)))
                return dict(selected=candidate,qpos=q,tip_world_m=target.tolist(),
                    bucket_surface_clearance_m=surface,clearance=clearance,rejected=rejected,
                    planning_scope='direct training initial pose; no approach motion executed')
            except ValueError as error:
                rejected.append(dict(entry_xy_m=candidate['entry_xy_m'],offset=[outward,upward],failure=str(error)))
    raise ValueError('training_no_safe_direct_prepoint:'+str(rejected))


class TrainingResetRuntime(CycleRuntime):
    def __init__(self,source,output):
        super().__init__(source,output,enable_truck=True,freeze_passive_jp=True,allow_training_reset=True)
        if self.metadata['cycle_index']>=3:
            raise ValueError('training_chain_three_scoops_complete')
        terrain,observation,soil,rocks,rock_points=detached_inventory(self)
        remaining = np.ones(len(self.arrays['particle_q']), dtype=bool)
        remaining[soil] = False
        remaining_rocks = [rock_points[record['rock_id']] for i,record in enumerate(self.records) if i not in rocks]
        # Rank the actual ground-connected stock, but clear ALL remaining material,
        # including airborne spill that has not yet reached the stock surface.
        clearance_terrain = terrain_from_points(np.concatenate([self.arrays['particle_q'][remaining],*remaining_rocks]))
        plan=select_direct_prepoint(self,terrain,clearance_terrain)
        previous=deepcopy(self.metadata.get('training_reset',{}))
        entries=previous.get('entries',[])
        old_soil={i for e in entries for i in e['soil_indices']}
        old_rocks={i for e in entries for i in e['rock_indices']}
        if old_soil.intersection(soil.tolist()) or old_rocks.intersection(rocks.tolist()):
            raise ValueError('training_withdrawal_double_count')
        cloud=np.concatenate([self.arrays['particle_q'][soil],*[rock_points[self.records[i]['rock_id']] for i in rocks]])
        lo=cloud.min(0);hi=cloud.max(0)
        destination=np.array([1.0,-3.5-1.75*len(entries),.03])
        shift=destination-np.r_[(lo[:2]+hi[:2])/2,lo[2]]
        if np.max(hi[:2]-lo[:2])>1.6:
            raise ValueError('training_inventory_packet_extent')
        entry=dict(source_checkpoint=self.metadata['source_identity'],source_cycle_index=self.metadata['cycle_index'],
            soil_indices=soil.tolist(),rock_indices=rocks.tolist(),soil_count=len(soil),rock_count=len(rocks),
            translation_m=shift.tolist(),inventory_center_xy_m=destination[:2].tolist(),
            ownership='withdrawn_for_training_only_not_truck_delivery')
        self.arrays['particle_q'][soil]+=shift
        for name in ('particle_qd','particle_f','particle_qd_grad'):
            self.arrays[name][soil]=0
        self.arrays['rigid_qpos'][rocks,:3]+=shift
        self.arrays['rigid_qvel'][rocks]=0
        self.arrays['rigid_reported_linear_velocity'][rocks]=0
        self.arrays['rigid_reported_angular_velocity'][rocks]=0
        self.arrays['mechanism_qpos']=plan['qpos'].copy()
        self.arrays['mechanism_qvel']=np.zeros_like(self.arrays['mechanism_qvel'])
        self.arrays['grid_reference_particle_q']=self.arrays['particle_q'].copy()
        self.metadata['training_reset']=dict(mode=MODE,continuous_physics=False,entries=entries+[entry],
            resets=['withdrawn material rigid translation and velocity','mechanism pose and velocity',
                    'controller targets/integrals/native acceleration history','MPM grid/warmstart/contact history',
                    'pending bucket reaction ledger'],unwithdrawn_material_state_preserved=True)
        self.metadata['moving_bucket_sweep_state']=dict(pending_velocity_reaction=None,
            total_generated_reaction_ns=[0.,0.,0.],total_routed_reaction_ns=[0.,0.,0.],resume_qualified=False)
        self.metadata['cycle_index']+=1
        self.metadata['selected_cut']=plan['selected']
        self.plan=plan;self.observation=observation;self.withdrawal=entry
        name='layered_47_training_reset.py'
        self.bindings[name]=motion.probe.bench.sha256(REPO/'src/v4'/name)
        for path in (REPO/'planning/contact_footprint.py', REPO/'src/v4/layered_47_cycle_selection.py'):
            self.bindings[path.name]=motion.probe.bench.sha256(path)

    def _grid_restore_context(self):
        return nullcontext()  # Fresh topology for changed geometry; never restore an old grid under a new scene.

    def _initialize_fixture(self,work,fixtures):
        probe=motion.audit.probe
        model,binary,mm,bm,command=probe.bench.build_model_and_bridge(self.args,self.root,
            self.root/self.base['capacity_bridge']['config_path'],
            probe.bench.records_at_qpos(self.records,self.arrays['rigid_qpos']),work,'training-reset')
        held=HoldBridge(probe.bench.RestoredRuntimeBridge(binary,model,len(self.records),
            self.arrays['mechanism_qpos'].tolist(),self.arrays['mechanism_qvel'].tolist(),
            self.arrays['rigid_qpos'],self.arrays['rigid_qvel'],None))
        try:
            fixture=probe.p47.ExcavationFixture(self.base,self.root,self.records,
                self.arrays['grid_reference_particle_q'],self.proxy,held.bridge.state,self.args.device,
                solver_profile=self.solver_profile,static_fixtures=fixtures)
            fields=probe.bench.restore_fixture_prefix(fixture,self.arrays)
            fixture.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
            fixture.assign(held.bridge.state)
            self.metadata['controller_state']=controller_snapshot(held.bridge.state)
            for i,rock in enumerate(held.bridge.state['rocks']):
                self.arrays['rigid_reported_linear_velocity'][i]=rock['linear_velocity_world_m_s']
                self.arrays['rigid_reported_angular_velocity'][i]=rock['angular_velocity_world_rad_s']
            return fixture,held,dict(initialization_kind=MODE,exact=False,
                reset_state_particle_readback_exact=all(fields['field_readback_exact'].values()),
                particles=fields,solver_history='fresh; not original-continuation history',
                runtime_build=dict(model=mm,bridge=bm,command=command))
        except BaseException:
            held.close(kill=True)
            raise
