"""Single-owner runtime with phase-local views; existing policy/control stay frozen."""
from pathlib import Path
from copy import deepcopy
import sys,json,hashlib,time
import numpy as np
ROOT=Path('/home/qunwang/wajueji');W=ROOT/'work/continuous-single-scoop-20261014-v1'
OLD=ROOT/'work/roadshow-three-way-20261014-v1'
sys.path[:0]=[str(ROOT/'local-data/runtime/sb3-2.9.0-py312'),str(ROOT/'src/v4'),str(OLD),str(ROOT)]
from exit_common import CycleRuntime,TransportController,FixedRouteClearance,read,write,sha,encode,bucket_opening_gap
from layered_47_cycle_bridge import controller_snapshot
from manual_residual_sac_env import ManualResidualEnv,SOURCE,SOURCE_SHA,INPUT,INPUT_SHA,environment_contract,require_environment_contract
from manual_trajectory_alignment import ProgressAlignment
from contact_bc_ground_guard import BucketGroundGuard
from contact_phase_geometry import SolidBucketGap
from planning.approach import sampled_approach
from planning.cycle_geometry import solve_pose,tool_pose,MOUTH
from planning.rear_lower_prepoint import ground_completed_terrain,proxy_surface_samples,sampled_surface_clearance,mechanism_and_truck_clearance
from layered_excavation_approach_planner import build_fused_terrain
import run_layered_47_sparse_manual_contact_episode as manual
MODEL=ROOT/'work/manual-strength-overnight-20261013-v1/anchor-grid-v1/seed44-weight100.zip'
MODEL_SHA='84aade9a3d8819368afb72223b8452ebfbc5e9ac9ad190818207d1142692aab1'
RECEIPT=MODEL.parent/'review.json'
APP=ROOT/'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089/report.json'

class PhaseView:
    """Read-only entry snapshot + local counter; never owns or reinitializes physics."""
    def __init__(self,core,*,arrays=None,native=None,fresh_policy_history=False):
        self.core=core;self.start_frame=core.frames
        self.arrays={k:v.copy() for k,v in (core.capture() if arrays is None else arrays).items()}
        self.metadata=deepcopy(core.metadata)
        self.metadata['controller_state']=deepcopy(controller_snapshot(core.bridge.state) if native is None else native)
        self._captured={} if fresh_policy_history else core.captured
    @property
    def frames(self):return self.core.frames-self.start_frame
    @property
    def captured(self):return self._captured
    def __getattr__(self,name):return getattr(self.core,name)
    def step(self,command):
        row=self.core.step(command);self._captured=self.core.captured;return row
    def capture(self):return self.core.capture()
    def save(self,*args,**kwargs):return self.core.save(*args,**kwargs)
    def close(self):pass
    def __enter__(self):raise RuntimeError('phase_view_cannot_initialize_physics')

class AttachedResidualEnv(ManualResidualEnv):
    """Reuse original reset-free step/observation/guards and frozen221/5 contract."""
    def reset(self,*args,**kwargs):raise RuntimeError('attached_policy_reset_forbidden')
    def attach(self,core,directory,previous_command):
        if self.runtime is not None:raise ValueError('policy_already_attached')
        self.directory=Path(directory);self.directory.mkdir(exist_ok=False);self.counter+=1;self.started=time.monotonic()
        self.runtime=PhaseView(core,fresh_policy_history=True)
        self.report=dict(status='initializing',episode=self.counter,input_sha256=INPUT_SHA,source_sha256=SOURCE_SHA,
            actual_entry_global_frame=core.frames,actual_entry_simulation_time_s=float(self.runtime.arrays['simulation_time_s'][0]),
            source_scope='Reference/model provenance; actual entry is live approach endpoint, not restored reference state',
            cap_fraction=self.cap_fraction,alignment_checks=0,alignment_wait_frames=0,maximum_reference_lead_frames=0.,maximum_alignment_error_m=0.,
            waiting_action_mode=self.waiting_action_mode,action_handoff_version='invalidate_on_mode_change_v1',terminal_score_version='manual_endpoint_hold_no_gap_v1',
            scores=[],gap_checks=[],success=False,runtime_bindings=core.bindings,control_hz=20,mpm_hz=60,rigid_hz=4800,
            fresh_policy_history='zero previous policy action and no within-phase force observation until first phase step; physical force/history unchanged',training_updates=0)
        self.ground=BucketGroundGuard(core.assembly,core.bridge.state['mechanism_qpos']);self.geometry=SolidBucketGap(core.proxy)
        self.vertices=manual._cache_rock_vertices(core.root,core.records);self.estimator=None;self.cursor=0.;self.hold=None
        self.previous_action=np.zeros(5);self.previous_applied=np.array(previous_command,copy=True)
        poses=[core.assembly.assemble(q,include_meshes=False)[1]['bucket'] for q in self.reference_q]
        self.alignment=ProgressAlignment(self.reference_q,poses);entry=self.alignment_state('attach')
        if not entry['trusted']:raise ValueError('live_entry_outside_original_alignment_corridor')
        self.report['status']='running';write(self.directory/'report.json',self.report)
        return self.observation()

class CpuRuntime(CycleRuntime):
    def __enter__(self):raise RuntimeError('cpu_native_entry_forbidden')
    def step(self,*args):raise RuntimeError('cpu_physical_step_forbidden')

class ApproachClearance(FixedRouteClearance):
    def __init__(self,r,arrays=None):
        super().__init__(r);a=r.arrays if arrays is None else arrays
        terrain,self.fresh_observation=build_fused_terrain(r.root,r.records,a['particle_q'],a['rigid_qpos'])
        self.collision_terrain=ground_completed_terrain(terrain);self.dense=proxy_surface_samples(r.proxy)
        with np.load(ROOT/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-mesh.npz') as f:self.truck_vertices=f['world_vertices'].copy()
    def check(self,q,previous=None,*,physical=False):
        result=super().check(q,previous,physical=physical);_,bodies=self.runtime.assembly.assemble(q,include_meshes=False)
        pose=bodies['bucket'];points=self.dense@pose[:3,:3].T+pose[:3,3]
        terrain=sampled_surface_clearance(points,self.collision_terrain)['clearance_m']
        other=mechanism_and_truck_clearance(self.geometries,bodies,self.collision_terrain,self.truck_vertices)
        if terrain<.32 or other['mechanism_terrain_clearance_m']<.02 or other['static_truck_aabb_separation_m']<.02:
            raise ValueError('original_approach_clearance:'+str((terrain,other)))
        return dict(result,bucket_observed_terrain_m=terrain,**other)

def project(r,q):
    # Reuse the existing transport projection: solve passive closure while
    # preserving the requested actuators; a zero velocity step keeps the old
    # source's small native closure residual and is insufficient here.
    from plan import project_passive
    return project_passive(r.chain,np.array(q,copy=True))

def segment(r,start,goal,checker):
    def advance(q,v,dt):r.chain.q=q.copy();return r.chain.advance(v,dt)[0]
    p=sampled_approach(start,goal,axes=r.chain.axes,speed_limits=np.array([.1,.018,.018,.018]),advance=advance,
        check_pose=checker.check,dt=1/60,max_steps=3000,validation_substeps=2)
    diagnostics=p.pop('diagnostics');p['minimum_clearance']={k:min(d[k] for d in diagnostics) for k in diagnostics[0] if diagnostics[0][k] is not None}
    return p

def initial_plan(r,out):
    out=Path(out);out.mkdir(exist_ok=False);app=read(APP)['dispatch_plan'];q=r.arrays['mechanism_qpos'].copy();original=q.copy()
    goals=[('PREPARE_OUTWARD',app['segments'][1]['goal_qpos']),('PREPARE_HIGH',app['segments'][0]['goal_qpos']),('SAFE_START',app['start_qpos']),
           ('APPROACH_HIGH',app['segments'][0]['goal_qpos']),('APPROACH_OUTWARD',app['segments'][1]['goal_qpos']),('APPROACH_PREPOINT',original)]
    checker=ApproachClearance(r);start=project(r,q);segments=[]
    for name,goal in goals:
        goal=project(r,np.array(goal));p=segment(r,start,goal,checker);segments.append(dict(name=name,path=p));start=goal
        print('CPU',name,p['steps'],flush=True)
    np.savez(out/'plan.npz',**{s['name']+'_qpos':s['path']['qpos'] for s in segments},**{s['name']+'_velocity':s['path']['velocity'] for s in segments})
    report=dict(status='geometry_passed',source_sha256=SOURCE_SHA,stages=[dict(name=s['name'],frames=s['path']['steps'],minimum_clearance=s['path']['minimum_clearance']) for s in segments],
        geometry_frames=sum(s['path']['steps']+30 for s in segments),fresh_observation=checker.fresh_observation,safe_return_goal=goals[2][1],
        scope='Start from qualified prepoint, physically retreat to high safe start then approach; no earlier schema relabel or physical reset')
    write(out/'report.json',report);return segments,checker,report

def exit_and_return_plan(view,safe_goal,out):
    out=Path(out);out.mkdir(exist_ok=False);checker=FixedRouteClearance(view);start=project(view,view.arrays['mechanism_qpos'])
    contract=read(ROOT/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-contract.json');rim=np.array(contract['rim_stations']);tf=np.array(contract['local_to_world'])
    center=tf@np.array([0.,(rim[:,1].min()+rim[:,2].max())/2,0.,1.])
    clear=solve_pose(view.assembly,view.chain,start,np.r_[center[:2],3.5],1.,MOUTH);swing=clear.copy();swing[7]=safe_goal[7]
    goals=[('CLEAR_TRUCK',clear),('WITHDRAW_SWING',swing),('RETURN_SAFE',project(view,np.array(safe_goal)))];segments=[]
    for name,goal in goals:
        p=segment(view,start,goal,checker);segments.append(dict(name=name,path=p));start=goal
        print('PLAN',name,p['steps'],flush=True)
    np.savez(out/'plan.npz',**{s['name']+'_qpos':s['path']['qpos'] for s in segments},**{s['name']+'_velocity':s['path']['velocity'] for s in segments})
    write(out/'report.json',dict(stages=[dict(name=s['name'],frames=s['path']['steps'],minimum_clearance=s['path']['minimum_clearance']) for s in segments],physics_frames=0))
    return segments,checker

def self_check():
    from types import SimpleNamespace
    core=SimpleNamespace(frames=2500,metadata={},captured={'old':True},closed=False)
    v=PhaseView(core,arrays={'a':np.array([1])},native={'controller_target':[0]*4},fresh_policy_history=True)
    assert v.frames==0 and v.captured=={};core.frames+=3;assert v.frames==3
    v.close();assert not core.closed and core.frames==2503 and v.arrays['a'][0]==1
    try:v.__enter__()
    except RuntimeError:pass
    else:raise AssertionError('phase initialized a second environment')
    e=AttachedResidualEnv.__new__(AttachedResidualEnv)
    try:e.reset()
    except RuntimeError:pass
    else:raise AssertionError('attached reset accepted')
    print('Phase ownership, nonzero global clock and reset rejection checks passed; no native initialization')
