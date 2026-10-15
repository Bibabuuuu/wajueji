"""Independent CPU-only audit of the saved short run; never advances physics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,hashlib,subprocess
ROOT=Path('/home/qunwang/wajueji');W=ROOT/'work/second-scoop-short-20261014-v1';sys.path.insert(0,str(W))
import run_short as m
import numpy as np,torch
from types import SimpleNamespace
from stable_baselines3 import SAC
rows=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines()]

def main():
 start=time.monotonic();D=m.OUT;p=m.read(D/'report.json');protocol=m.read(W/'protocol.json');b=m.read(W/'budget.json');checks={}
 def check(n,v):checks[n]=bool(v)
 f=rows(D/'physics/frames.ndjson');dispatch=rows(D/'dispatch.ndjson');states=m.read(D/'display/states.json');phases=m.read(D/'display/phases.json');cloud=np.load(D/'display/particles.npy',mmap_mode='r');n=p['frames']
 check('one_native_continuation_instance',p['native_initializations']==1)
 check('source_unchanged',m.sha(m.SOURCE2)==p['source_sha256']==b['source_sha256'] and m.sha(str(m.SOURCE2)+'.json')==p['source_metadata_sha256'])
 check('frozen_model_unchanged',m.sha(m.MODEL)==m.MODEL_SHA==p['model_sha256'])
 check('all_bound_code_unchanged',all(m.sha(k)==h for k,h in protocol['code_sha256'].items()))
 check('protocol_and_budget_identity',m.sha(W/'protocol.json')==p['protocol_sha256'] and m.sha(W/'budget.json')==p['budget_sha256']==protocol['budget_sha256'])
 check('zero_training',p['neural_updates']==0)
 check('within_fixed_time_and_frame_scope',p['physical_run_finished_epoch']<b['epoch_deadline'] and 0<n<=5000)
 check('actual_frame_and_dispatch_count',n==len(f)==len(dispatch))
 check('physical_and_dispatch_frames_contiguous',[x['frame'] for x in f]==list(range(1,n+1)) and [x['global_frame'] for x in dispatch]==list(range(1,n+1)))
 check('dispatched_commands_equal_physical_commands',all(np.array_equal(x['command'],y['applied_command_si']) for x,y in zip(dispatch,f)))
 check('native_hard_guards_recomputed',all(m.manual.hard_guard_checks(x,allow_truck=True)==x['hard_guard_checks'] for x in f))
 check('all_handoffs_preserve_physics',all(x['before']==x['after'] and x['physical_and_native_unchanged'] for x in p['handoffs']))
 ids={(x['before']['fixture_id'],x['before']['bridge_id']) for x in p['handoffs']};check('same_fixture_and_bridge_through_handoffs',len(ids)==1)
 check('cache_prefix_counts',len(states)==len(cloud)==len(phases) and len(states)<=n+1 and len(states)>=n)
 check('cache_prefix_states_are_actual',all(s==x['after_state'] for s,x in zip(states[1:],f)))
 check('cache_file_hashes',all(m.sha(D/'display'/k)==h for k,h in p['display']['cache_sha256'].items()))
 with np.load(D/'zero.npz') as z:zero={k:z[k].copy() for k in z.files}
 with np.load(m.SOURCE2) as z:check('24_source_arrays_restored',len(z.files)==24 and all(np.array_equal(z[k],zero[k]) for k in z.files if k!='simulation_time_s'))
 check('native_zero_controller_restore',max(p['native_restore_errors'].values())<=1e-12 and p['zero_time_error_s']<=1e-12)
 checkpoint=p.get('final_checkpoint',p.get('recovery_checkpoint'))
 check('terminal_checkpoint_hash',m.sha(checkpoint['path'])==checkpoint['sha256'])
 with np.load(checkpoint['path']) as z:final={k:z[k].copy() for k in z.files}
 check('continuous_physical_clock',abs(float(final['simulation_time_s'][0])-float(zero['simulation_time_s'][0])-n/60)<1e-10)
 check('particle_and_rock_inventory_preserved',np.array_equal(zero['particle_flags'],final['particle_flags']) and np.array_equal(zero['rigid_ids'],final['rigid_ids']) and final['particle_q'].shape==zero['particle_q'].shape)
 check('final_state_cache_or_failure_checkpoint',np.array_equal(final['particle_q'],cloud[-1]) if len(states)==n+1 else p['status']=='failed')
 r=m.CpuRuntime(D/'zero.npz',W/('audit-runtime-'+str(time.time_ns())),enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
 ga=rows(D/'geometry-actions.ndjson') if (D/'geometry-actions.ndjson').exists() else [];gc=rows(D/'geometry-clearance.ndjson') if (D/'geometry-clearance.ndjson').exists() else []
 max_command_error=0.;command_count=0;phase_names=list(dict.fromkeys(x['phase'] for x in ga))
 for name in phase_names:
  part=[x for x in ga if x['phase']==name];first=part[0];q=np.asarray(first['before_state']['mechanism_qpos']);arrays=dict(r.arrays,mechanism_qpos=q);native=dict(r.metadata['controller_state'],controller_target=first['before_state']['controller_target'])
  view=m.PhaseView(r,arrays=arrays,native=native,fresh_policy_history=True);prev=np.asarray(dispatch[first['global_frame']-2]['command']) if first['global_frame']>1 else np.zeros(4)
  c=m.TransportController(view,m.FixedRouteClearance(view),prev)
  for x in part:
   q=np.asarray(x['before_state']['mechanism_qpos']);cmd,detail=c.command(name,q,x['before_state']['controller_target'],x['desired'],x['velocity'])
   max_command_error=max(max_command_error,float(np.max(abs(cmd-np.asarray(x['command'])))));command_count+=1
  print('AUDIT',name,len(part),flush=True)
 check('geometry_commands_independently_recomputed',max_command_error<=1e-15)
 check('logged_geometry_has_original_guards',all(x['axis_error']<=.025 and x['bucket_truck_m']>=.04 and x['mechanism_truck_m']>=.02 and x['mechanism_ground_m']>=.02 and x['closure_m']<=.001 and x['mechanism_terrain_clearance_m']>=.02 and (x['phase'].startswith('RETURN_RISE') or x['bucket_observed_terrain_m']>=.32) for x in gc))
 policy=D/'policy/episode-0001';decisions=rows(D/'policy-decisions.ndjson') if (D/'policy-decisions.ndjson').exists() else [];actions=rows(policy/'actions.ndjson') if (policy/'actions.ndjson').exists() else []
 prediction_errors=[];observed_errors=[];grid_errors=[]
 if decisions:
  torch.set_num_threads(1);model=SAC.load(m.MODEL,device='cpu');env=m.YawTransferEnv(W/'audit-reference',SimpleNamespace(),collect=False,waiting_action_mode='error_strength_v1',yaw_offset_rad=p['yaw_offset_rad'])
  R=env.canonical_rotation;rot=p['yaw_offset_rad'];vertices=m.manual._cache_rock_vertices(r.root,r.records)
  from mixed_material_r0 import quaternion_matrix_wxyz
  from yaw_transfer import height_grid,ANCHOR
  for d in decisions:
   obs=np.asarray(d['observation'],np.float32);prediction_errors.append(float(np.max(abs(model.predict(obs,deterministic=True)[0]-d['action']))))
   state=states[d['global_frame_before']];q=np.asarray(state['mechanism_qpos']).copy();q[7]-=rot
   vectors=[q,np.asarray(state['mechanism_qvel']),R.T@np.asarray(state['bucket_origin_world_m'])-ANCHOR,R.T@np.asarray(state['bucket_linear_velocity_world_m_s']),R.T@np.asarray(state['bucket_angular_velocity_world_rad_s'])]
   expected=np.clip(np.r_[*vectors],-10,10).astype(np.float32);actual=np.r_[obs[:37],obs[56:65]]
   observed_errors.append(float(np.max(abs(expected-actual))))
   points=[cloud[d['global_frame_before']]]
   for v,rock in zip(vertices,state['rocks']):
    pose=np.asarray(rock['qpos']);points.append(v@quaternion_matrix_wxyz(pose[3:]).T+pose[:3])
   h,k=height_grid(np.concatenate(points),np.asarray(state['bucket_origin_world_m']),R,rot)
   grid_errors.append(float(np.max(abs(np.r_[h,k]-obs[87:215]))))
  first=decisions[0]['global_frame_before'];check('residual_local_clock_and120_cap',len(actions)<=120 and all(d['global_frame_before']-first==d['local_frame_before'] and np.float32(d['local_frame_before']/1560)==np.float32(d['observation'][85]) for d in decisions))
  check('residual_dispatch_matches',all(np.array_equal(x['applied_si'],dispatch[first+x['frame']-1]['command']) for x in actions))
  check('frozen_predictions_recomputed',max(prediction_errors)<=1e-7)
  check('policy_observes_actual_canonical_geometry',max(observed_errors)<=1e-6 and max(grid_errors)<=1e-6)
  check('policy_first_force_history_is_fresh',not np.any(decisions[0]['observation'][65:71]) and decisions[0]['observation'][86]==0)
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip();check('GPU_idle_after_run',not gpu)
 actual_pass=p['status']=='second_scoop_short_completed' and p.get('residual_frames')==120 and all(x['hard_guard'] and x['hencky_material_guard']['safe'] and x['moving_bucket_sweep']['safe'] for x in f)
 result=dict(status=('passed' if actual_pass else 'diagnostic_audit_passed') if all(checks.values()) else 'failed',checks=checks,check_count=len(checks),physical_short_passed=actual_pass,run_status=p['status'],failure=p.get('failure'),frames=n,
  geometry_commands_recomputed=command_count,geometry_max_command_error_si=max_command_error,policy_decisions=len(decisions),residual_frames=len(actions),maximum_policy_prediction_error=max(prediction_errors,default=None),maximum_observed_geometry_error=max(observed_errors,default=None),maximum_terrain_grid_error=max(grid_errors,default=None),
  new_physics_steps_by_audit=0,training_updates=0,not_full_second_or_three_scoops=True,wall_seconds=time.monotonic()-start,
  limitation='World force/torque source values are not retained separately in the old frame schema; audit recomputes policy from logged inputs and checks geometry/grid against actual cached state. No claim of independent raw wrench reconstruction.')
 m.write(D/'read-only-audit.json',result);print(m.encode({k:v for k,v in result.items() if k!='checks'}),flush=True)
 assert all(checks.values()),[k for k,v in checks.items() if not v]
if __name__=='__main__':main()
