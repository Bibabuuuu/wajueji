"""D190: new2h, one native edge-first/adjacent pair; no training or resets."""
from pathlib import Path
import sys,time,json,hashlib,traceback,importlib.util
from types import FunctionType
R=Path('/home/qunwang/wajueji');C=R/'work/edge-first-strips-20261015-v1';W=R/'work/edge-first-two-scoop-20261015-v1';OUT=R/'runs/edge-first-two-scoop-20261015-attempt2'
sys.path[:0]=[str(C),str(R/'work/second-scoop-wait-qerror-20261014-v1'),str(R/'work/second-scoop-continuation-20261014-v1'),str(R/'work/continuous-single-scoop-20261014-v1')]
from continuous_scoop import *
from actual_joint_wait import ActualJointWaitEnv,VERSION as WAIT_VERSION
from adjacent_selection import rank_adjacent
from next_prepoint_candidate import plan_next
from layered_47_cycle_selection import rank_workspace_candidates
from view_contact_bc_probe import ProbeDisplayCache
from stable_baselines3 import SAC
import torch
sys.path[:0]=[str(R/'work/second-scoop-rise-hold-20261014-v1'),str(R/'work/second-scoop-minimum-guard-20261014-v1'),str(R/'work/second-scoop-short-20261014-v1')]
from minimum_guard import MinimumHeightClearance
from rise_hold_guard import RiseHoldGuard
from inplace_cache import InPlaceDisplayCache as ProbeDisplayCache
from direct_reference_bridge import direct_bridge
W=R/'work/edge-first-two-scoop-20261015-v1'

class Budget:
 def __init__(self):
  self.path=W/'budget.json';self.data=read(self.path);d=self.data
  assert not (W/'closure.json').exists(),'budget_closed'
  assert (d['purpose'],d['seconds'],d['max_frames'],d['max_dig_frames'],d['training_updates_allowed'])==('edge_first_adjacent_two_scoop',7200,30000,7200,0)
  assert abs(d['epoch_deadline']-d['epoch_start']-7200)<1e-6
  assert d['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 def remaining(self):return min(self.data['epoch_deadline']-time.time(),self.data['monotonic_deadline']-time.monotonic())
 def check(self):
  if self.remaining()<=self.data['reserve_seconds']:raise TimeoutError('new2h_save_audit_reserve')
 def start(self):self.check()

class BoundRuntime(CycleRuntime):
 def __init__(self,budget):
  self.budget=budget;self.step_authorized=False;self.prefix_verified=False;self.native_initializations=0;self.active_cycle=1;budget.check();d=budget.data
  source=Path(d['source'])
  for p,h in [(source,d['source_sha256']),(Path(str(source)+'.json'),d['source_metadata_sha256']),(source.parent/'safe_start.control.json',d['source_control_sha256'])]:assert sha(p)==h,('source_identity',p)
  super().__init__(source,OUT/'physics',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
  assert self.metadata['cycle_index']==1 and self.metadata['stage']=='transport' and not self.metadata.get('training_reset')
 def __enter__(self):
  if self.native_initializations:raise RuntimeError('second_native_initialization_forbidden')
  self.native_initializations+=1;return super().__enter__()
 def step(self,command):
  self.budget.check()
  if not self.step_authorized:raise ValueError('native_zero_gate_required')
  if self.frames>=3 and not self.prefix_verified:raise ValueError('prefix3_gate_required')
  if self.frames+4654>=30000:raise ValueError('whole_pair_frame_ceiling')
  return super().step(command)

 # Metadata routing only; physical arrays and native controller remain unchanged.
 def save(self,path,stage='hold',cycle_index=None,selected_cut=None):
  return super().save(path,stage=stage,cycle_index=(cycle_index or self.active_cycle),selected_cut=selected_cut)

class EdgeEnv(ActualJointWaitEnv):
 def attach(self,core,*args,**kwargs):
  assert isinstance(core,BoundRuntime) and core.step_authorized and core.prefix_verified and core.native_initializations==1
  obs=super().attach(core,*args,**kwargs)
  self.report.update(physical_admission=True,physical_scope='D190_new2h_edge_pair',budget_sha256=sha(core.budget.path))
  write(self.directory/'report.json',self.report);return obs
 def step(self,action):
  if self.runtime is None or not isinstance(self.runtime.core,BoundRuntime):raise ValueError('bound_edge_pair_required')
  return AttachedResidualEnv.step(self,action)

def digest(r):
 return dict(arrays={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in r.capture().items()},native=hashlib.sha256(encode(controller_snapshot(r.bridge.state)).encode()).hexdigest(),sweep=hashlib.sha256(encode(r.sweep.diagnostic_state()).encode()).hexdigest(),frame=r.frames,fixture_id=id(r.fixture),bridge_id=id(r.bridge))

def main():
 budget=Budget();budget.check();protocol=read(W/'live-protocol-attempt2.json')
 for p,h in protocol['bindings_sha256'].items():assert sha(p)==h,('binding_changed',p)
 assert sha(MODEL)==MODEL_SHA;require_environment_contract(read(RECEIPT),'error_strength_v1');torch.set_num_threads(1)
 model=SAC.load(MODEL,device='cpu');updates=model._n_updates
 OUT.mkdir(exist_ok=False);started=time.monotonic();r=None;cache=None;env=None;phase='INITIALIZE';cycle=1;previous=np.zeros(4);labels=[]
 report=dict(status='initializing',frames=0,phases=[],handoffs=[],scoops=[],prefix_checks=[],native_initializations=0,neural_updates=0,source=budget.data['source'],source_sha256=budget.data['source_sha256'],model_sha256=MODEL_SHA,budget_sha256=sha(budget.path),protocol_sha256=sha(W/'live-protocol-attempt2.json'),action_mapping_version=WAIT_VERSION,observation_time_denominator=1560,scope='same-instance edge-first dig, real geometric unload/return, fresh-stock inward adjacent second dig; no matched-central rerun or training')
 def persist():write(OUT/'report.json',report)
 def save(name,stage):
  s=r.save(OUT/(name+'.npz'),stage=stage,cycle_index=cycle,selected_cut=report.get('selected_cut'))
  write(OUT/(name+'.control.json'),dict(frame=r.frames,phase=phase,previous_command=previous,checkpoint_sha256=s['sha256'],continuation_qualified=False));return s
 def handoff(name,fn):
  before=digest(r);v=fn();after=digest(r);ok=before==after
  report['handoffs'].append(dict(label=name,before=before,after=after,physical_and_native_unchanged=ok));persist()
  if not ok:raise ValueError('handoff_mutated_physics:'+name)
  return v
 def geometry(name,path,checker,hold=30):
  nonlocal phase
  phase=name;begin=r.frames;c=handoff(name,lambda:TransportController(PhaseView(r),checker,previous));n=len(path['velocity'])
  desired_frames=list(zip(path['qpos'][1:],path['velocity'],strict=True))+[(path['qpos'][-1],np.zeros(4))]*hold
  for index,(desired,velocity) in enumerate(desired_frames):
   if index==n and isinstance(checker,RiseHoldGuard):
    transition=handoff(name+'_HIGH_HOLD',lambda:checker.enter_hold(np.asarray(r.bridge.state['mechanism_qpos']),completed_nominal_steps=n))
    report.setdefault('phase_switches',[]).append(dict(phase=name,frame=r.frames,admission=transition));persist()
   q=np.array(r.bridge.state['mechanism_qpos']);c.check(q,name,physical=True)
   command,detail=c.command(name,q,r.bridge.state['controller_target'],desired,velocity)
   manual.append_json(OUT/'geometry-actions.ndjson',dict(global_frame=r.frames+1,phase=name,before_state=manual.compact_state(r.bridge.state),desired=desired,velocity=velocity,command=command,detail=detail))
   row=r.step(command);actual=np.asarray(row['after_state']['mechanism_qpos']);d=checker.check(actual,q,physical=True)
   error=float(np.max(abs(actual[r.chain.axes]-desired[r.chain.axes])))
   manual.append_json(OUT/'geometry-clearance.ndjson',dict(global_frame=r.frames,phase=name,axis_error=error,**d))
   if error>.025:raise ValueError('original_tracking_guard:'+str(error))
  for _ in range(30):
   old=np.asarray(r.bridge.state['mechanism_qpos']);row=r.step(np.zeros(4));checker.check(np.asarray(row['after_state']['mechanism_qpos']),old,physical=True)
  error=float(np.max(abs(np.asarray(r.bridge.state['mechanism_qpos'])[r.chain.axes]-path['qpos'][-1][r.chain.axes])))
  if error>.015:raise ValueError('original_endpoint_guard:'+name)
  report['phases'].append(dict(name=name,start_frame=begin,end_frame=r.frames,endpoint_axis_error=error));report['latest_checkpoint']=save(name.lower(),'transport');persist()
 def observer(view):
  folder=R/'work/contact-fixed-truck-continuation-20260925-v2';spec=importlib.util.spec_from_file_location('edge_live_observer',folder/'observation.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module.stationary_terrain(view,view.arrays)
 def approach_and_dig(first_entry=None):
  nonlocal phase,env
  r.active_cycle=cycle
  view=PhaseView(r);terrain,observation=handoff('SCOOP%d_FRESH_STOCK'%cycle,lambda:observer(view));candidates=rank_workspace_candidates(terrain,3 if cycle==1 else 1)
  if cycle==1:
   eligible=[c for c in candidates if c['future_strip_count']>=1];outermost=max(abs(c['entry_xy_m'][1]) for c in eligible)
   assert outermost>.3,'not_an_outer_edge'
   ordered=sorted([c for c in eligible if abs(abs(c['entry_xy_m'][1])-outermost)<1e-9],key=lambda c:(c['workspace_score'],c['score']),reverse=True)
  else:
   ordered,rejected=rank_adjacent(terrain,first_entry,inward_sign=-1)
   # A 0.1m observation-cell gap is an experiment placement boundary, not a
   # relaxed machine/soil protection. Refuse a non-adjacent fallback.
   ordered=[c for c in ordered if abs(c['adjacency']['nominal_entry_boundary_gap_m'])<=float(terrain['resolution_m'])]
   assert ordered,'no_fresh_full_width_adjacent_candidate'
   report['second_adjacency_rejected']=rejected
  fn=plan_next;planner=FunctionType(fn.__code__,dict(fn.__globals__,rank_workspace_candidates=lambda *_:ordered),fn.__name__,fn.__defaults__,fn.__closure__)
  result=handoff('SCOOP%d_PLAN'%cycle,lambda:planner(view,terrain,observation));path=result.pop('path');result.pop('terrain');result['cycle_index']=cycle
  path=handoff('SCOOP%d_DIRECT_REFERENCE_BRIDGE'%cycle,lambda:direct_bridge(view,terrain,path,OUT/('reference-bridge-read-%d'%cycle)))
  result['direct_reference_bridge']=path['direct_reference_bridge']
  report['selected_cut']=result['selected']['contact'];folder=OUT/('scoop-%d-plan'%cycle);folder.mkdir()
  np.savez(folder/'terrain.npz',**{k:np.asarray(v) for k,v in terrain.items() if isinstance(v,(list,np.ndarray,int,float))})
  np.savez(folder/'approach.npz',qpos=path['qpos'],velocity=path['velocity']);write(folder/'report.json',dict(result,segments=path['segments'],total_frames=path['total_frames']))
  offset=0
  for item in path['segments']:
   n=item['frames'];part=dict(qpos=path['qpos'][offset:offset+n+1],velocity=path['velocity'][offset:offset+n]);offset+=n
   checker=RiseHoldGuard(r,terrain,part['qpos'][-1],n) if item['name'].startswith('RETURN_RISE') else MinimumHeightClearance(r,terrain)
   geometry('SCOOP%d_'%cycle+item['name'],part,checker)
  readref=ManualResidualEnv(OUT/('reference-read-%d'%cycle),budget,collect=False,waiting_action_mode='error_strength_v1')
  yaw=float(np.asarray(r.bridge.state['mechanism_qpos'])[7]-readref.reference_q[0,7])
  env=EdgeEnv(OUT/('policy-%d'%cycle),budget,collect=False,waiting_action_mode='error_strength_v1',max_frames=7200,yaw_offset_rad=yaw)
  live=PhaseView(r);goal=env.reference_q[0].copy();goal[9]+=.001;goal=project(live,goal)
  poses=[live.assembly.assemble(q,include_meshes=False)[1]['bucket'] for q in env.reference_q]
  admission=ProgressAlignment(env.reference_q,poses).observe(goal,live.assembly.assemble(goal,include_meshes=False)[1]['bucket'],0.)
  assert admission['trusted'],'reference_goal_outside_original_corridor'
  checker=MinimumHeightClearance(r,terrain);align=handoff('SCOOP%d_ALIGN_PLAN'%cycle,lambda:segment(live,project(live,np.asarray(r.bridge.state['mechanism_qpos'])),goal,checker))
  np.savez(folder/'align.npz',qpos=align['qpos'],velocity=align['velocity'],reference_q=env.reference_q,reference_targets=env.reference_targets,yaw_offset_rad=yaw)
  write(folder/'alignment-admission.json',dict(yaw_offset_rad=yaw,admission=admission,minimum_clearance=align['minimum_clearance']))
  geometry('SCOOP%d_REFERENCE_ALIGN'%cycle,align,checker)
  phase='SCOOP%d_RESIDUAL'%cycle;begin=r.frames;obs=handoff(phase,lambda:env.attach(r,OUT/('policy-%d'%cycle)/'episode-0001',previous));trace=[]
  while True:
   action=model.predict(obs,deterministic=True)[0]
   manual.append_json(OUT/'policy-decisions.ndjson',dict(scoop=cycle,global_frame_before=r.frames,local_frame_before=env.runtime.frames,observation=obs,action=action,world_bucket_wrench_before=env.runtime.captured.get('wrenches',{}).get('bucket'),model_sha256=MODEL_SHA))
   obs,_,done,truncated,info=env.step(action)
   q=np.asarray(r.bridge.state['mechanism_qpos']);pose=r.assembly.assemble(q,include_meshes=False)[1]['bucket']
   if 120<=env.cursor<=840:
    trace.append(dict(frame=r.frames,cursor=env.cursor,mouth_xy=(pose[:3,:3]@MOUTH+pose[:3,3])[:2].tolist(),q=q.tolist()))
   if done or truncated:break
  item=dict(scoop=cycle,entry_global_frame=begin,end_global_frame=r.frames,yaw_offset_rad=yaw,selected=result['selected'],residual_result=info['episode_summary'],actual_entry_mouth_xy_m=(trace[0]['mouth_xy'] if trace else None))
  write(folder/'actual-cut-trace.json',dict(window=[120,840],trace=trace,scope='Actual FK mouth centres at policy decisions, nominal0.6m width, not excavated-volume boundaries'))
  report['scoops'].append(item);report['phases'].append(dict(name=phase,start_frame=begin,end_frame=r.frames));persist()
  if not env.report['success'] or env.report['status']!='completed':raise ValueError('scoop%d_dig_failed:'%cycle+env.report['status'])
  assert env.cursor==1170 and env.hold==30,'completion_contract'
  checkpoint=env.report['checkpoint'];handoff('SCOOP%d_RELEASE'%cycle,lambda:env.close());env=None
  return checkpoint,item
 persist()
 try:
  with BoundRuntime(budget) as r:
   try:
    actual=r.capture();exact={k:np.array_equal(v,r.arrays[k]) for k,v in actual.items()};dt=abs(float(actual['simulation_time_s'][0])-float(r.arrays['simulation_time_s'][0]))
    assert all(v for k,v in exact.items() if k!='simulation_time_s') and dt<=1e-12 and max(r.restored['cycle_errors'].values())<=1e-12,'native_zero_restore_inexact'
    report.update(zero_arrays_exact=exact,zero_time_error_s=dt,native_restore_errors=r.restored['cycle_errors'],native_initializations=1,initial_simulation_time_s=float(actual['simulation_time_s'][0]),zero_checkpoint=save('zero','transport'))
    cache=ProbeDisplayCache(OUT/'display',30000,r.arrays['particle_q'].shape,dict(records=r.records,root=str(r.root),selected_cut=r.metadata['selected_cut'],particle_spacing_m=r.base['mpm']['particle_spacing_m']))
    cache.capture(0,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy());labels.append('INITIAL');raw=r.step
    prior=R/'runs/continuous-single-scoop-20261014-v1';prefix_frame=2813
    old_states=read(prior/'display/states.json');old_cloud=np.load(prior/'display/particles.npy',mmap_mode='r')
    prefix_commands=[json.loads(s)['command'] for s in (prior/'dispatch.ndjson').read_text().splitlines()[prefix_frame:prefix_frame+3]]
    def tracked(command):
     nonlocal previous
     label=phase+'_HOLD' if env is not None and env.hold is not None else phase
     manual.append_json(OUT/'dispatch.ndjson',dict(global_frame=r.frames+1,phase=label,cycle=cycle,command=np.asarray(command)))
     try:row=raw(command)
     finally:report['frames']=r.frames
     previous=np.array(command,copy=True);cloud=r.fixture.state.particle_q.numpy();cache.capture(r.frames,row['after_state'],cloud);labels.append(label)
     if not row['hencky_material_guard']['safe'] or not row['moving_bucket_sweep']['safe']:raise ValueError('original_material_or_sweep_guard')
     if r.frames<=3:
      check=dict(frame=r.frames,particles_exact=np.array_equal(cloud,old_cloud[prefix_frame+r.frames]),compact_state_exact=row['after_state']==old_states[prefix_frame+r.frames],command_exact=np.array_equal(command,prefix_commands[r.frames-1]))
      report['prefix_checks'].append(check);persist()
      if not all(check[k] for k in ('particles_exact','compact_state_exact','command_exact')):raise ValueError('saved_prefix_mismatch:'+str(check))
      if r.frames==3:r.prefix_verified=True;print('PREFIX3_EXACT',flush=True)
     if r.frames%600==0:report['periodic_checkpoint']=save('periodic-%05d'%r.frames,'transport')
     if r.frames%120==0:report.update(status='running',phase=label,reference_cursor=(env.cursor if env else None));print('GLOBAL',r.frames,label,'cursor',round(env.cursor,1) if env else None,'remaining',round(budget.remaining()),flush=True);persist()
     return row
    r.step=tracked;r.step_authorized=True;phase='RESTORE_PREFIX3';fixed=FixedRouteClearance(r)
    for command in prefix_commands:
     row=r.step(command);fixed.check(np.asarray(row['after_state']['mechanism_qpos']),physical=True)
    assert r.prefix_verified;report['prefix_checkpoint']=save('prefix3','transport');persist()
    safe_goal=np.array(r.arrays['mechanism_qpos']);endpoint,first=approach_and_dig()
    view=PhaseView(r)
    from transport_endpoint_candidate_v2 import build_plan
    stages,checker,planned=handoff('TRANSPORT_PLAN',lambda:build_plan(view,Path(endpoint['path']),OUT/'transport-plan'))
    report['transport_plan']=planned;persist()
    for stage in stages:geometry(stage['name'],stage['path'],checker,hold=30+(180 if stage['name']=='DUMP' else 0))
    view=PhaseView(r);stages,checker=handoff('EXIT_RETURN_PLAN',lambda:exit_and_return_plan(view,safe_goal,OUT/'exit-return-plan'))
    for stage in stages:geometry(stage['name'],stage['path'],checker)
    report['first_return_checkpoint']=save('first-returned','returned');cycle=2
    assert first['actual_entry_mouth_xy_m'] is not None,'missing_actual_first_entry'
    endpoint,second=approach_and_dig(first['actual_entry_mouth_xy_m'])
    phase='FINAL_HOLD';report.update(status='edge_adjacent_pair_completed',final_checkpoint=save('final','hold'),final_simulation_time_s=float(r.capture()['simulation_time_s'][0]));persist()
   except BaseException as e:
    report.update(status='failed',failure=repr(e),traceback=traceback.format_exc(),frames=r.frames,phase=phase,reference_cursor=(env.cursor if env else None))
    if env is not None:report['active_residual_report']=dict(env.report)
    try:report['recovery_checkpoint']=save('recovery','failure')
    except BaseException as error:report['save_error']=repr(error)
    persist();raise
 except BaseException as e:report.update(status='failed',failure=repr(e));persist()
 finally:
  if cache is not None:
   try:
    d=cache.finish(report['status']);d.update(capture_origin='D190_same_native_edge_adjacent_pair',native_initializations=report['native_initializations']);write(OUT/'display/reconstruction.json',d);write(OUT/'display/phases.json',labels);report['display']=d
   except BaseException as e:report['display_error']=repr(e)
  report.update(frames=r.frames if r else 0,neural_updates=model._n_updates-updates,wall_seconds=time.monotonic()-started,physical_run_finished_epoch=time.time(),remaining_budget_seconds=budget.remaining())
  assert report['neural_updates']==0 and sha(MODEL)==MODEL_SHA;persist();print(encode({k:report.get(k) for k in ['status','frames','failure','wall_seconds']}),flush=True)

if __name__=='__main__':main()

