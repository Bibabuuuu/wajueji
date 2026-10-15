"""Authorized45-minute native continuation, approach and <=120 residual frames.

Reuses the original runtime/controller/policy step. Does not train or finish a
second scoop. The CPU-only adapter remains disabled outside this bound runner.
"""
from pathlib import Path
import sys,traceback,hashlib,json,time,shutil
ROOT=Path('/home/qunwang/wajueji');A=ROOT/'work/second-scoop-continuation-20261014-v1'
C=ROOT/'work/second-scoop-minimum-guard-20261014-v1';OLD_SHORT=ROOT/'work/second-scoop-short-20261014-v1'
G=ROOT/'work/second-scoop-rise-hold-20261014-v1'
sys.path[:0]=[str(G),str(C),str(OLD_SHORT),str(A),str(ROOT/'work/continuous-single-scoop-20261014-v1')]
from continuous_scoop import *
from yaw_transfer import YawTransferEnv
from next_prepoint_candidate import plan_next
from view_contact_bc_probe import ProbeDisplayCache
from stable_baselines3 import SAC
from planning.rear_lower_prepoint import sampled_surface_clearance,ground_completed_terrain,mechanism_and_truck_clearance
import torch,importlib.util
W=ROOT/'work/second-scoop-short-phased-20261014-v1';OUT=ROOT/'runs/second-scoop-short-phased-20261014-v1'
SOURCE2=ROOT/'runs/continuous-single-scoop-20261014-v1/final.npz'

class Budget:
 def __init__(self):
  self.path=W/'budget.json';self.data=read(self.path);d=self.data
  assert not (W/'closure.json').exists(),'budget_closed'
  assert (d['purpose'],d['seconds'],d['max_frames'],d['max_residual_frames'],d['training_updates_allowed'])==('second_scoop_phased_restore_approach_short',2700,5000,120,0)
  assert abs(d['epoch_deadline']-d['epoch_start']-2700)<1e-6
 def start(self):self.check()
 def remaining(self):return self.data['epoch_deadline']-time.time()
 def check(self):
  if self.remaining()<=self.data['reserve_seconds']:raise TimeoutError('45min_budget_save_audit_reserve')

class BoundRuntime(CycleRuntime):
 def __init__(self,output,budget):
  self.budget=budget;self.step_authorized=False;self.native_initializations=0;budget.check()
  assert sha(SOURCE2)==budget.data['source_sha256']
  super().__init__(SOURCE2,output,enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
  assert self.metadata['stage']=='returned' and self.metadata['cycle_index']==1 and not self.metadata.get('training_reset')
 def __enter__(self):
  if self.native_initializations:raise RuntimeError('second_native_initialization_forbidden')
  self.native_initializations+=1;return super().__enter__()
 def step(self,command):
  self.budget.check()
  if not self.step_authorized:raise ValueError('native_zero_restore_gate_required')
  if self.frames>=self.budget.data['max_frames']:raise ValueError('5000_frame_ceiling')
  return super().step(command)

class PhysicalYawTransferEnv(YawTransferEnv):
 def attach(self,core,*args,**kwargs):
  assert isinstance(core,BoundRuntime) and core.step_authorized and core.native_initializations==1
  obs=super().attach(core,*args,**kwargs)
  self.report.update(physical_admission=True,physical_scope='authorized120-frame-short-only',budget_sha256=sha(core.budget.path))
  write(self.directory/'report.json',self.report);return obs
 def step(self,action):
  if self.runtime is None or not isinstance(self.runtime.core,BoundRuntime):raise ValueError('bound_short_runtime_required')
  limit=self.budget.data['max_residual_frames']
  if self.runtime.frames+3>limit:raise ValueError('120_residual_frame_ceiling')
  return AttachedResidualEnv.step(self,action)

def digest(r):
 arrays=r.capture()
 return dict(arrays={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()},
  native=hashlib.sha256(encode(controller_snapshot(r.bridge.state)).encode()).hexdigest(),
  sweep=hashlib.sha256(encode(r.sweep.diagnostic_state()).encode()).hexdigest(),
  frame=r.frames,fixture_id=id(r.fixture),bridge_id=id(r.bridge))

from minimum_guard import MinimumHeightClearance as LiveClearance
from rise_hold_guard import RiseHoldGuard,VERSION as HEIGHT_GUARD_VERSION

def main():
 budget=Budget();budget.check();protocol=read(W/'protocol.json')
 for n,h in protocol['code_sha256'].items():assert sha(n)==h,('frozen_code_changed',n)
 assert sha(MODEL)==MODEL_SHA and read(A/'adapter-independent-audit.json')['status']=='passed'
 assert read(C/'cpu-v1/report.json')['status']=='cpu_candidate_passed'
 assert read(G/'cpu-v1/report.json')['status']=='cpu_phase_candidate_passed'
 require_environment_contract(read(RECEIPT),'error_strength_v1');torch.set_num_threads(1)
 model=SAC.load(MODEL,device='cpu');updates=model._n_updates
 OUT.mkdir(exist_ok=False);started=time.monotonic()
 report=dict(status='initializing',height_guard_version=HEIGHT_GUARD_VERSION,physical_variant='rise-then-high-hold-new45min',phase_switches=[],frames=0,phases=[],handoffs=[],native_initializations=0,neural_updates=0,
  source=str(SOURCE2),source_sha256=sha(SOURCE2),source_metadata_sha256=sha(str(SOURCE2)+'.json'),model_sha256=MODEL_SHA,
  budget_sha256=sha(budget.path),protocol_sha256=sha(W/'protocol.json'),scope='native restore, second approach and <=120 residual frames only; no full scoop/training/three-scoop qualification')
 cache=None;r=None;env=None;phase='INITIALIZE';previous=np.asarray(read(SOURCE2.parent/'final.control.json')['previous_command'])
 frame_phases=[]
 def persist():write(OUT/'report.json',report)
 def save(name,stage,cycle=1):
  s=r.save(OUT/(name+'.npz'),stage=stage,cycle_index=cycle,selected_cut=report.get('selected_cut'))
  write(OUT/(name+'.control.json'),dict(frame=r.frames,phase=phase,previous_command=previous,checkpoint_sha256=s['sha256'],continuation_qualified=False))
  return s
 def handoff(label,fn):
  before=digest(r);v=fn();after=digest(r);ok=before==after
  report['handoffs'].append(dict(label=label,before=before,after=after,physical_and_native_unchanged=ok));persist()
  if not ok:raise ValueError('handoff_mutated_physics:'+label)
  return v
 def geometry(name,path,checker):
  nonlocal phase
  phase=name;begin=r.frames;c=handoff(name,lambda:TransportController(PhaseView(r),checker,previous))
  for index,(desired,velocity) in enumerate(list(zip(path['qpos'][1:],path['velocity'],strict=True))+[(path['qpos'][-1],np.zeros(4))]*30):
   if index==len(path['velocity']) and isinstance(checker,RiseHoldGuard):
    admission=handoff(name+'_TO_HIGH_HOLD',lambda:checker.enter_hold(np.asarray(r.bridge.state['mechanism_qpos']),completed_nominal_steps=index))
    report['phase_switches'].append(dict(phase=name,global_frame=r.frames,completed_nominal_steps=index,admission=admission));persist()
   q=np.array(r.bridge.state['mechanism_qpos']);c.check(q,name,physical=True)
   command,detail=c.command(name,q,r.bridge.state['controller_target'],desired,velocity)
   manual.append_json(OUT/'geometry-actions.ndjson',dict(global_frame=r.frames+1,phase=name,before_state=manual.compact_state(r.bridge.state),desired=desired,velocity=velocity,command=command,detail=detail))
   row=r.step(command);actual=np.array(row['after_state']['mechanism_qpos']);d=checker.check(actual,q,physical=True)
   error=float(np.max(abs(actual[r.chain.axes]-desired[r.chain.axes])))
   manual.append_json(OUT/'geometry-clearance.ndjson',dict(global_frame=r.frames,phase=name,guard_mode=(checker.mode if isinstance(checker,RiseHoldGuard) else 'approach'),axis_error=error,**d))
   if error>.025:raise ValueError('original_tracking_guard:'+str(error))
  for _ in range(30):
   q=np.array(r.bridge.state['mechanism_qpos']);row=r.step(np.zeros(4));checker.check(np.array(row['after_state']['mechanism_qpos']),q,physical=True)
  error=float(np.max(abs(np.asarray(r.bridge.state['mechanism_qpos'])[r.chain.axes]-path['qpos'][-1][r.chain.axes])))
  if error>.015:raise ValueError('original_endpoint_guard:'+name)
  report['phases'].append(dict(name=name,start_frame=begin,end_frame=r.frames,endpoint_axis_error=error));persist()
  report['latest_checkpoint']=save(name.lower(),'returned' if name.startswith('RETURN') else 'prepoint',2);persist()
 persist()
 try:
  with BoundRuntime(OUT/'physics',budget) as r:
   try:
    actual=r.capture();exact={k:np.array_equal(v,r.arrays[k]) for k,v in actual.items()};dt=abs(float(actual['simulation_time_s'][0])-float(r.arrays['simulation_time_s'][0]))
    assert all(v for k,v in exact.items() if k!='simulation_time_s') and dt<1e-12 and max(r.restored['cycle_errors'].values())<=1e-12,'native_zero_restore_inexact'
    report.update(zero_arrays_exact=exact,zero_time_error_s=dt,native_restore_errors=r.restored['cycle_errors'],native_initializations=r.native_initializations,
      initial_simulation_time_s=float(actual['simulation_time_s'][0]),zero_checkpoint=save('zero','returned'))
    cache=ProbeDisplayCache(OUT/'display',5000,r.arrays['particle_q'].shape,dict(records=r.records,root=str(r.root),selected_cut=r.metadata['selected_cut'],particle_spacing_m=r.base['mpm']['particle_spacing_m']))
    cache.capture(0,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy());frame_phases.append('INITIAL')
    raw_step=r.step
    def tracked(command):
     nonlocal previous
     manual.append_json(OUT/'dispatch.ndjson',dict(global_frame=r.frames+1,phase=phase,command=np.asarray(command)))
     try:row=raw_step(command)
     finally:report['frames']=r.frames
     previous=np.array(command,copy=True);cache.capture(r.frames,row['after_state'],r.fixture.state.particle_q.numpy());frame_phases.append(phase)
     if not row['hencky_material_guard']['safe'] or not row['moving_bucket_sweep']['safe']:raise ValueError('original_material_or_sweep_guard')
     if r.frames%120==0:report.update(status='running',current_phase=phase);print('GLOBAL',r.frames,phase,'remaining',round(budget.remaining()),flush=True);persist()
     return row
    r.step=tracked;r.step_authorized=True;phase='RESTORE_ZERO_COMMAND'
    fixed=FixedRouteClearance(r)
    for _ in range(3):row=r.step(np.zeros(4));fixed.check(np.asarray(row['after_state']['mechanism_qpos']),physical=True)
    report['phases'].append(dict(name=phase,start_frame=0,end_frame=r.frames));persist()
    folder=ROOT/'work/contact-fixed-truck-continuation-20260925-v2'
    spec=importlib.util.spec_from_file_location('short_live_observer',folder/'observation.py');observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
    view=PhaseView(r);terrain,observation=handoff('FRESH_STOCK',lambda:observer.stationary_terrain(view,view.arrays))
    plan=handoff('LIVE_PLAN',lambda:plan_next(view,terrain,observation));path=plan.pop('path');plan.pop('terrain')
    report['selected_cut']=plan['selected']['contact']
    write(OUT/'live-plan.json',dict(**plan,segments=path['segments'],frames=path['total_frames']))
    np.savez(OUT/'live-plan.npz',qpos=path['qpos'],velocity=path['velocity']);np.savez(OUT/'live-terrain.npz',**{k:np.asarray(v) for k,v in terrain.items() if isinstance(v,(int,float,list,np.ndarray))})
    assert path['total_frames']+600+120+r.frames<=5000,'live_plan_exceeds_scope'
    offset=0
    for s in path['segments']:
     n=s['frames'];p=dict(qpos=path['qpos'][offset:offset+n+1],velocity=path['velocity'][offset:offset+n]);offset+=n
     checker=RiseHoldGuard(r,terrain,p['qpos'][-1],n) if s['name'].startswith('RETURN_RISE') else LiveClearance(r,terrain)
     geometry(s['name'],p,checker)
    # Reference/canonical observation only; never rotate the physical scene.
    temp=ManualResidualEnv(OUT/'reference-read',budget,collect=False,waiting_action_mode='error_strength_v1')
    yaw=float(np.asarray(r.bridge.state['mechanism_qpos'])[7]-temp.reference_q[0,7])
    env=PhysicalYawTransferEnv(OUT/'policy',budget,collect=False,waiting_action_mode='error_strength_v1',yaw_offset_rad=yaw)
    view=PhaseView(r);goal=project(view,env.reference_q[0]);align=segment(view,project(view,np.asarray(r.bridge.state['mechanism_qpos'])),goal,LiveClearance(r,terrain))
    np.savez(OUT/'reference-align.npz',qpos=align['qpos'],velocity=align['velocity']);report['yaw_offset_rad']=yaw
    geometry('REFERENCE_ALIGN',align,LiveClearance(r,terrain))
    phase='RESIDUAL_SHORT';begin=r.frames;obs=handoff('RESIDUAL_SHORT',lambda:env.attach(r,OUT/'policy/episode-0001',previous))
    while env.runtime.frames<120:
     d=dict(global_frame_before=r.frames,local_frame_before=env.runtime.frames,observation=obs,action=model.predict(obs,deterministic=True)[0],model_sha256=MODEL_SHA,world_bucket_wrench_before=env.runtime.captured.get('wrenches',{}).get('bucket'))
     manual.append_json(OUT/'policy-decisions.ndjson',d);obs,_,done,truncated,info=env.step(d['action'])
     if done or truncated:raise ValueError('short_policy_terminal:'+env.report['status'])
    report.update(status='second_scoop_short_completed',residual_frames=env.runtime.frames,reference_cursor=env.cursor,residual_report=dict(env.report))
    report['phases'].append(dict(name=phase,start_frame=begin,end_frame=r.frames));handoff('RESIDUAL_RELEASE',lambda:env.close())
    # "hold" is the existing native checkpoint schema tag, not completed
    # excavation or a measured retention endpoint. This partial save has no
    # continuation/endpoint qualification; its exact scope lives in the report.
    report['final_checkpoint']=save('short-final','hold',2);report['final_simulation_time_s']=float(r.capture()['simulation_time_s'][0]);persist()
   except BaseException as e:
    report.update(status='failed',failure=repr(e),traceback=traceback.format_exc(),phase=phase,frames=r.frames)
    try:report['recovery_checkpoint']=save('recovery','failure',2)
    except BaseException as s:report['save_error']=repr(s)
    persist();raise
 except BaseException as e:
  report.update(status='failed',failure=repr(e));persist()
 finally:
  if cache is not None:
   try:
    display=cache.finish(report['status']);display.update(capture_origin='second_scoop_native_short',native_initializations=report['native_initializations']);report['display']=display
    write(OUT/'display/reconstruction.json',display);write(OUT/'display/phases.json',frame_phases)
   except BaseException as e:report['display_error']=repr(e)
  report.update(frames=r.frames if r else 0,wall_seconds=time.monotonic()-started,neural_updates=model._n_updates-updates,remaining_budget_seconds=budget.remaining(),physical_run_finished_epoch=time.time())
  assert sha(MODEL)==MODEL_SHA and report['neural_updates']==0
  persist();print(encode({k:report.get(k) for k in ['status','frames','phase','failure','residual_frames','wall_seconds']}),flush=True)

if __name__=='__main__':main()
