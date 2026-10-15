"""CPU-only audit of source restoration, exact prefix and frozen full-dig attempt."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,subprocess
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-full-dig-20261014-v1';sys.path.insert(0,str(W))
import run_full_second as m
import numpy as np,torch
from stable_baselines3 import SAC
from manual_residual_sac_env import LIMITS_SI
from mixed_material_r0 import quaternion_matrix_wxyz
from yaw_transfer import height_grid,ANCHOR
rows=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines()]

def main():
 start=time.monotonic();D=m.OUT;p=m.read(D/'report.json');protocol=m.read(W/'protocol.json');b=m.read(W/'budget.json');checks={}
 def check(n,v):checks[n]=bool(v)
 f=rows(D/'physics/frames.ndjson');dispatch=rows(D/'dispatch.ndjson');states=m.read(D/'display/states.json');cloud=np.load(D/'display/particles.npy',mmap_mode='r');n=p['frames']
 check('one_native_instance_no_reset',p['native_initializations']==1)
 check('actual_source_identity',m.sha(m.SOURCE2)==p['source_sha256']==b['source_sha256'] and m.sha(str(m.SOURCE2)+'.json')==p['source_metadata_sha256'])
 check('all_bound_core_code_unchanged',all(m.sha(k)==h for k,h in protocol['code_sha256'].items()))
 check('all_bound_inputs_unchanged',all(m.sha(k)==h for k,h in protocol['input_sha256'].items()))
 check('frozen_model_unchanged',m.sha(m.MODEL)==m.MODEL_SHA==p['model_sha256'] and p['neural_updates']==0)
 check('fixed30minute_budget',m.sha(W/'budget.json')==p['budget_sha256']==protocol['budget_sha256'] and p['physical_run_finished_epoch']<b['epoch_deadline'] and 0<n<=1560)
 check('protocol_identity',m.sha(W/'protocol.json')==p['protocol_sha256'])
 check('frame_dispatch_cache_counts',n==len(f)==len(dispatch)==len(states)-1==len(cloud)-1)
 check('frames_contiguous',[x['frame'] for x in f]==list(range(1,n+1)) and [x['global_frame'] for x in dispatch]==list(range(1,n+1)))
 check('all_dispatch_matches_native',all(np.array_equal(d['command'],x['applied_command_si']) for d,x in zip(dispatch,f)))
 check('cached_states_are_actual',all(s==x['after_state'] for s,x in zip(states[1:],f)))
 check('all_hard_guards_recomputed_and_safe',all(m.manual.hard_guard_checks(x,allow_truck=True)==x['hard_guard_checks'] and x['hard_guard'] for x in f))
 check('all_material_and_sweep_guards_safe',all(x['hencky_material_guard']['safe'] and x['moving_bucket_sweep']['safe'] for x in f))
 check('cache_file_hashes',all(m.sha(D/'display'/k)==h for k,h in p['display']['cache_sha256'].items()))
 check('all_handoffs_preserve_native_and_physics',all(x['before']==x['after'] and x['physical_and_native_unchanged'] for x in p['handoffs']))
 with np.load(D/'zero.npz') as z:zero={k:z[k].copy() for k in z.files}
 with np.load(m.SOURCE2) as z:check('zero24_arrays_exact',len(z.files)==24 and all(np.array_equal(z[k],zero[k]) for k in z.files if k!='simulation_time_s'))
 check('zero_native_restore_exact',max(p['native_restore_errors'].values())<=1e-12 and p['zero_time_error_s']<=1e-12)
 old_states=m.read(m.SOURCE2.parent/'display/states.json');old_cloud=np.load(m.SOURCE2.parent/'display/particles.npy',mmap_mode='r');old_actions=rows(m.SOURCE2.parent/'policy/episode-0001/actions.ndjson')
 prefix_ok=all(np.array_equal(cloud[i],old_cloud[m.SOURCE_FRAME+i]) and states[i]==old_states[m.SOURCE_FRAME+i] and np.array_equal(dispatch[i-1]['command'],old_actions[i-1]['applied_si']) for i in range(1,min(n,3)+1))
 check('first3_saved_prefix_exact',n>=3 and prefix_ok and p.get('prefix_verified') and all(x['particles_exact'] and x['compact_state_exact'] and x['applied_command_exact'] for x in p['prefix_checks']))
 checkpoint=p.get('final_checkpoint',p.get('recovery_checkpoint'));check('saved_terminal_hash',m.sha(checkpoint['path'])==checkpoint['sha256'])
 with np.load(checkpoint['path']) as z:final={k:z[k].copy() for k in z.files}
 check('physical_clock_continuous',abs(float(final['simulation_time_s'][0])-float(zero['simulation_time_s'][0])-n/60)<1e-10)
 check('final_particles_rocks_and_ids_match_cache',np.array_equal(final['particle_q'],cloud[-1]) and np.array_equal(final['rigid_qpos'],[x['qpos'] for x in states[-1]['rocks']]) and np.array_equal(zero['rigid_ids'],final['rigid_ids']) and np.array_equal(zero['particle_flags'],final['particle_flags']))
 r=m.CpuRuntime(D/'zero.npz',W/('audit-runtime-'+str(time.time_ns())),enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
 env=m.YawTransferEnv(W/'audit-reference',None,collect=False,waiting_action_mode='error_strength_v1',max_frames=1560,yaw_offset_rad=m.CANDIDATE['yaw_offset_rad'])
 vertices=m.manual._cache_rock_vertices(r.root,r.records);rot=env.canonical_rotation;angle=m.CANDIDATE['yaw_offset_rad'];model=SAC.load(m.MODEL,device='cpu');torch.set_num_threads(1)
 decisions=rows(D/'policy-decisions.ndjson');actions=rows(D/'policy/episode-0001/actions.ndjson');alignment=rows(D/'policy/episode-0001/alignment.ndjson')
 check('all_policy_actions_are_actual_dispatch',len(actions)==n and [x['frame'] for x in actions]==list(range(1,n+1)) and all(np.array_equal(x['applied_si'],dispatch[x['frame']-1]['command']) for x in actions))
 predictions=[];input_errors=[]
 for index,d in enumerate(decisions):
  local=d['local_frame_before'];obs=np.asarray(d['observation'],np.float32);state=states[local];q=np.asarray(state['mechanism_qpos']).copy();q[7]-=angle
  predictions.append(float(np.max(abs(model.predict(obs,deterministic=True)[0]-d['action']))))
  record=next(x for x in reversed(alignment) if x['frame']==local and x['stage'] in ('attach','after'));cursor=record['reference_progress'];phase=min(int(cursor),1170);last=min(phase+1,1170);fraction=cursor-phase
  target=env.reference_targets[phase]*(1-fraction)+env.reference_targets[last]*fraction
  points=[cloud[local]]
  for v,rock in zip(vertices,state['rocks']):
   pose=np.asarray(rock['qpos']);points.append(v@quaternion_matrix_wxyz(pose[3:]).T+pose[:3])
  height,known=height_grid(np.concatenate(points),np.asarray(state['bucket_origin_world_m']),rot,angle)
  world=d['world_bucket_wrench_before'];wrench=np.zeros(6) if world is None else np.r_[rot.T@np.asarray(world['force']),rot.T@np.asarray(world['torque'])]/1e5
  prev=np.asarray(dispatch[local-1]['command']) if local else np.asarray(m.read(m.SOURCE2.parent/'reference_align.control.json')['previous_command'])
  prev_action=np.zeros(5) if index==0 else np.asarray(decisions[index-1]['action'])
  residual=p.get('residual_result',p.get('residual_report',{}));hold_started=any(s['frame']<=local for s in residual.get('scores',[]))
  base=np.zeros(4) if hold_started else env.commands[min(phase,1169)]/LIMITS_SI
  expected=np.r_[q,state['mechanism_qvel'],np.asarray(state['mechanism_qpos'])-env.reference_q[phase],rot.T@np.asarray(state['bucket_origin_world_m'])-ANCHOR,
   rot.T@np.asarray(state['bucket_linear_velocity_world_m_s']),rot.T@np.asarray(state['bucket_angular_velocity_world_rad_s']),wrench,base,prev/LIMITS_SI,prev_action,
   cursor/1170,local/1560,float(world is not None),height,known,(target-state['controller_target'])/LIMITS_SI,record['actual_progress']/1170,float(record['mode'].startswith('wait'))]
  input_errors.append(float(np.max(abs(np.clip(expected,-10,10).astype(np.float32)-obs))))
  if index%120==0:print('AUDIT_POLICY',index,'of',len(decisions),flush=True)
 check('all_frozen_predictions_recomputed',max(predictions)<=1e-7)
 check('all221_policy_inputs_reconstructed',max(input_errors)<=1e-6)
 check('local_policy_clock_not_restored_world_clock',all(d['global_frame_before']==d['local_frame_before'] and np.float32(d['observation'][85])==np.float32(d['local_frame_before']/1560) for d in decisions))
 check('initial_observation_and_action_match_saved_prefix',p['initial_observation_exact'] and np.array_equal(decisions[0]['observation'],rows(m.SOURCE2.parent/'policy-decisions.ndjson')[0]['observation']))
 check('GPU_idle',not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip())
 residual=p.get('residual_result',p.get('residual_report',{}));complete=p['status']=='full_second_dig_completed' and residual.get('success') and p['reference_cursor']==1170 and sum(x['hold_frame'] is not None for x in actions)==30
 if complete:check('completed1170_plus30_hold',residual['status']=='completed' and len(residual['scores'])==3)
 result=dict(status=('passed' if complete else 'diagnostic_audit_passed') if all(checks.values()) else 'failed',checks=checks,check_count=len(checks),full_second_dig_passed=bool(complete),run_status=p['status'],failure=p.get('failure'),
  frames=n,reference_cursor=p.get('reference_cursor'),policy_decisions=len(decisions),maximum_prediction_error=max(predictions),maximum221_input_error=max(input_errors),prefix3_exact=prefix_ok,
  retained_fill_fraction=residual.get('retained_fill_fraction'),new_physics_steps_by_audit=0,training_updates=0,wall_seconds=time.monotonic()-start,
  limitation='A second-target restored-source excavation test; no transport, third scoop, original-instance continuous-three-scoop or cross-terrain loading advantage qualification.')
 m.write(D/'read-only-audit.json',result);print(m.encode({k:v for k,v in result.items() if k!='checks'}),flush=True);assert all(checks.values()),[k for k,v in checks.items() if not v]
if __name__=='__main__':main()
