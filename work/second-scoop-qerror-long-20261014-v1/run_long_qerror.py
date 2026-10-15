"""New30-minute actual-joint waiting scope: restore actual pre-dig source, verify prefix, full dig.

No training, transport or physical reset between prefix and later residual.
The unchanged original residual step owns progress, guards and hold scoring.
"""
from pathlib import Path
import sys,time,json,hashlib,traceback
R=Path('/home/qunwang/wajueji');OLD=R/'work/second-scoop-short-phased-20261014-v1';A=R/'work/second-scoop-continuation-20261014-v1'
sys.path[:0]=[str(A),str(R/'work/continuous-single-scoop-20261014-v1')]
from continuous_scoop import *
from yaw_transfer import YawTransferEnv
sys.path.insert(0,str(R/'work/second-scoop-wait-qerror-20261014-v1'))
from actual_joint_wait import ActualJointWaitEnv,VERSION as WAIT_MAPPING_VERSION
from view_contact_bc_probe import ProbeDisplayCache
from stable_baselines3 import SAC
import torch
PRIOR=R/'work/second-scoop-short-phased-20261014-v1'
W=R/'work/second-scoop-qerror-long-20261014-v1';OUT=R/'runs/second-scoop-qerror-long-20261014-v1'
CANDIDATE=read(PRIOR/'aligned-source-candidate.json');SOURCE2=Path(CANDIDATE['source']);SOURCE_FRAME=CANDIDATE['actual_frame']

class Budget:
 def __init__(self):
  self.path=W/'budget.json';self.data=read(self.path);d=self.data
  assert not (W/'closure.json').exists(),'budget_closed'
  assert (d['purpose'],d['seconds'],d['max_frames'],d['prefix_frames'],d['training_updates_allowed'])==('second_scoop_actual_joint_wait_long_validation',1800,7200,3,0)
  assert abs(d['epoch_deadline']-d['epoch_start']-1800)<1e-6
 def start(self):self.check()
 def remaining(self):return self.data['epoch_deadline']-time.time()
 def check(self):
  if self.remaining()<=self.data['reserve_seconds']:raise TimeoutError('30min_save_audit_reserve')

class BoundRuntime(CycleRuntime):
 def __init__(self,out,budget):
  self.budget=budget;self.step_authorized=False;self.prefix_verified=False;self.native_initializations=0;budget.check()
  if sha(SOURCE2)!=budget.data['source_sha256'] or sha(str(SOURCE2)+'.json')!=budget.data['source_metadata_sha256'] or sha(SOURCE2.parent/'reference_align.control.json')!=budget.data['source_control_sha256']:raise ValueError('source_identity')
  super().__init__(SOURCE2,out,enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
  assert self.metadata['stage']=='prepoint' and self.metadata['cycle_index']==2 and not self.metadata.get('training_reset')
 def __enter__(self):
  if self.native_initializations:raise RuntimeError('second_native_initialization_forbidden')
  self.native_initializations+=1;return super().__enter__()
 def step(self,command):
  self.budget.check()
  if not self.step_authorized:raise ValueError('native_zero_gate_required')
  if self.frames>=3 and not self.prefix_verified:raise ValueError('saved_first3_prefix_gate_required')
  if self.frames>=7200:raise ValueError('7200_frame_ceiling')
  return super().step(command)

class FullYawEnv(ActualJointWaitEnv):
 def attach(self,core,*args,**kwargs):
  assert isinstance(core,BoundRuntime) and core.step_authorized and core.native_initializations==1
  obs=super().attach(core,*args,**kwargs)
  self.report.update(action_mapping_version=WAIT_MAPPING_VERSION,replay_compatible_with_old_mapping=False,physical_admission=True,physical_scope='new30min7200_actual_joint_wait_dig_only',budget_sha256=sha(core.budget.path))
  write(self.directory/'report.json',self.report);return obs
 def step(self,action):
  if self.runtime is None or not isinstance(self.runtime.core,BoundRuntime):raise ValueError('bound_full_dig_runtime_required')
  return AttachedResidualEnv.step(self,action)

def digest(r):
 return dict(arrays={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in r.capture().items()},native=hashlib.sha256(encode(controller_snapshot(r.bridge.state)).encode()).hexdigest(),
  sweep=hashlib.sha256(encode(r.sweep.diagnostic_state()).encode()).hexdigest(),frame=r.frames,fixture_id=id(r.fixture),bridge_id=id(r.bridge))

def main():
 b=Budget();b.check();protocol=read(W/'protocol.json')
 assert protocol['action_mapping_version']==WAIT_MAPPING_VERSION and protocol['old_replay_direct_mixing_allowed'] is False
 for n,h in protocol['code_sha256'].items():assert sha(n)==h,('frozen_code_changed',n)
 for n,h in protocol['input_sha256'].items():assert sha(n)==h,('frozen_input_changed',n)
 assert sha(MODEL)==MODEL_SHA and read(SOURCE2.parent/'read-only-audit.json')['status']=='passed'
 require_environment_contract(read(RECEIPT),'error_strength_v1');torch.set_num_threads(1);model=SAC.load(MODEL,device='cpu');updates=model._n_updates
 ref_states=read(SOURCE2.parent/'display/states.json');ref_cloud=np.load(SOURCE2.parent/'display/particles.npy',mmap_mode='r')
 ref_actions=[json.loads(s) for s in (SOURCE2.parent/'policy/episode-0001/actions.ndjson').read_text().splitlines()]
 first_decision=json.loads((SOURCE2.parent/'policy-decisions.ndjson').read_text().splitlines()[0])
 OUT.mkdir(exist_ok=False);started=time.monotonic();r=None;env=None;cache=None;previous=np.asarray(read(SOURCE2.parent/'reference_align.control.json')['previous_command']);labels=[]
 report=dict(status='initializing',frames=0,native_initializations=0,source=str(SOURCE2),source_sha256=sha(SOURCE2),source_metadata_sha256=sha(str(SOURCE2)+'.json'),
  model_sha256=MODEL_SHA,budget_sha256=sha(b.path),protocol_sha256=sha(W/'protocol.json'),source_global_frame=SOURCE_FRAME,prefix_checks=[],handoffs=[],neural_updates=0,
  action_mapping_version=WAIT_MAPPING_VERSION,old_replay_direct_mixing_allowed=False,observation_time_denominator=1560,maximum_physical_frames=7200,scope='actual-joint waiting full second-target dig and30-frame retention from actual pre-dig restore; no transport/training or continuous-three-scoop qualification')
 def persist():write(OUT/'report.json',report)
 def save(name,stage):
  s=r.save(OUT/(name+'.npz'),stage=stage,cycle_index=2)
  write(OUT/(name+'.control.json'),dict(frame=r.frames,previous_command=previous,checkpoint_sha256=s['sha256'],continuation_qualified=False));return s
 persist()
 try:
  with BoundRuntime(OUT/'physics',b) as r:
   try:
    actual=r.capture();exact={k:np.array_equal(v,r.arrays[k]) for k,v in actual.items()};dt=abs(float(actual['simulation_time_s'][0])-float(r.arrays['simulation_time_s'][0]))
    assert all(v for k,v in exact.items() if k!='simulation_time_s') and dt<=1e-12 and max(r.restored['cycle_errors'].values())<=1e-12,'native_zero_restore_inexact'
    report.update(zero_arrays_exact=exact,zero_time_error_s=dt,native_restore_errors=r.restored['cycle_errors'],native_initializations=1,initial_simulation_time_s=float(actual['simulation_time_s'][0]),zero_checkpoint=save('zero','prepoint'))
    cache=ProbeDisplayCache(OUT/'display',7200,r.arrays['particle_q'].shape,dict(records=r.records,root=str(r.root),selected_cut=r.metadata['selected_cut'],particle_spacing_m=r.base['mpm']['particle_spacing_m']))
    cache.capture(0,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy());labels.append('INITIAL');raw=r.step
    def tracked(command):
     nonlocal previous
     phase='PREFIX_CHECK' if r.frames<3 else 'RESIDUAL_HOLD' if env is not None and env.hold is not None else 'RESIDUAL'
     manual.append_json(OUT/'dispatch.ndjson',dict(global_frame=r.frames+1,phase=phase,command=np.asarray(command)))
     try:row=raw(command)
     finally:report['frames']=r.frames
     previous=np.array(command,copy=True);cloud=r.fixture.state.particle_q.numpy();cache.capture(r.frames,row['after_state'],cloud);labels.append(phase)
     if not row['hencky_material_guard']['safe'] or not row['moving_bucket_sweep']['safe']:raise ValueError('original_material_or_sweep_guard')
     if r.frames<=3:
      expected=ref_states[SOURCE_FRAME+r.frames]
      item=dict(frame=r.frames,particles_exact=np.array_equal(cloud,ref_cloud[SOURCE_FRAME+r.frames]),compact_state_exact=row['after_state']==expected,
       applied_command_exact=np.array_equal(command,ref_actions[r.frames-1]['applied_si']))
      report['prefix_checks'].append(item);persist()
      if not all(item[k] for k in ['particles_exact','compact_state_exact','applied_command_exact']):raise ValueError('saved_SAC_prefix_mismatch:'+str(item))
      if r.frames==3:r.prefix_verified=True;report['prefix_checkpoint']=save('prefix3','hold');report['prefix_verified']=True;print('PREFIX3_EXACT',flush=True);persist()
     if r.frames%60==0:report.update(status='running',reference_cursor=env.cursor if env else None);print('FRAME',r.frames,'cursor',round(env.cursor,2),'remaining',round(b.remaining()),flush=True);persist()
     return row
    r.step=tracked;r.step_authorized=True
    env=FullYawEnv(OUT/'policy',b,collect=False,waiting_action_mode='error_strength_v1',max_frames=7200,yaw_offset_rad=CANDIDATE['yaw_offset_rad'])
    before=digest(r);obs=env.attach(r,OUT/'policy/episode-0001',previous);after=digest(r);report['handoffs'].append(dict(label='FULL_RESIDUAL_ATTACH',before=before,after=after,physical_and_native_unchanged=before==after))
    assert before==after,'attach_mutated_physics'
    assert np.array_equal(obs,np.asarray(first_decision['observation'],np.float32)),'restored_initial_policy_observation_mismatch'
    report['initial_observation_exact']=True;persist()
    while True:
     action=model.predict(obs,deterministic=True)[0]
     if r.frames==0:assert np.array_equal(action,first_decision['action']),'initial_frozen_action_mismatch'
     manual.append_json(OUT/'policy-decisions.ndjson',dict(global_frame_before=r.frames,local_frame_before=env.runtime.frames,observation=obs,action=action,world_bucket_wrench_before=env.runtime.captured.get('wrenches',{}).get('bucket'),model_sha256=MODEL_SHA))
     obs,_,done,truncated,info=env.step(action)
     if done or truncated:break
    report.update(residual_result=info['episode_summary'],reference_cursor=env.cursor,residual_frames=r.frames)
    if not env.report['success'] or env.report['status']!='completed':raise ValueError('full_second_dig_stopped:'+env.report['status'])
    assert r.prefix_verified and env.cursor==1170 and env.hold==30,'full_completion_contract'
    before=digest(r);env.close();after=digest(r);report['handoffs'].append(dict(label='FULL_RESIDUAL_RELEASE',before=before,after=after,physical_and_native_unchanged=before==after));assert before==after
    report.update(status='full_second_dig_completed',final_checkpoint=save('final','hold'),final_simulation_time_s=float(r.capture()['simulation_time_s'][0]));persist()
   except BaseException as e:
    report.update(status='failed',failure=repr(e),traceback=traceback.format_exc(),frames=r.frames,reference_cursor=env.cursor if env else None)
    if env is not None:report['residual_report']=dict(env.report)
    try:report['recovery_checkpoint']=save('recovery','failure')
    except BaseException as save_error:report['save_error']=repr(save_error)
    persist();raise
 except BaseException as e:report.update(status='failed',failure=repr(e));persist()
 finally:
  if cache is not None:
   try:
    d=cache.finish(report['status']);d.update(capture_origin='full_second_dig_native_restore',native_initializations=report['native_initializations']);report['display']=d;write(OUT/'display/reconstruction.json',d);write(OUT/'display/phases.json',labels)
   except BaseException as e:report['display_error']=repr(e)
  report.update(frames=r.frames if r else 0,neural_updates=model._n_updates-updates,wall_seconds=time.monotonic()-started,physical_run_finished_epoch=time.time(),remaining_budget_seconds=b.remaining())
  assert report['neural_updates']==0 and sha(MODEL)==MODEL_SHA;persist();print(encode({k:report.get(k) for k in ['status','frames','reference_cursor','failure','wall_seconds']}),flush=True)
if __name__=='__main__':main()
