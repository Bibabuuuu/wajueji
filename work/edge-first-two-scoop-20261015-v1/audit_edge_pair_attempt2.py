"""Read-only evidence audit; a trustworthy failure is not a successful pair."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,subprocess
R=Path('/home/qunwang/wajueji');W=R/'work/edge-first-two-scoop-20261015-v1';sys.path.insert(0,str(W))
import run_edge_pair_attempt2 as m
import numpy as np,torch
from stable_baselines3 import SAC
from manual_residual_sac_env import command_at,LIMITS_SI
from mixed_material_r0 import quaternion_matrix_wxyz
from yaw_transfer import height_grid,ANCHOR
rows=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines()] if Path(p).exists() else []

def main():
 start=time.monotonic();D=m.OUT;p=m.read(D/'report.json');protocol=m.read(W/'live-protocol-attempt2.json');b=m.read(W/'budget.json');checks={}
 def check(name,value):checks[name]=bool(value)
 f=rows(D/'physics/frames.ndjson');dispatch=rows(D/'dispatch.ndjson');states=m.read(D/'display/states.json');cloud=np.load(D/'display/particles.npy',mmap_mode='r');n=p['frames'];executed=dispatch[:n]
 check('one_native_instance',p['native_initializations']==1)
 check('bound_code_model_inputs_unchanged',all(m.sha(k)==h for k,h in protocol['bindings_sha256'].items()) and m.sha(m.MODEL)==m.MODEL_SHA)
 check('no_training',p['neural_updates']==0)
 check('new2h_budget_identity',m.sha(W/'budget.json')==p['budget_sha256'] and b['seconds']==7200 and b['reserve_seconds']==900)
 check('physical_finished_within_budget',p['physical_run_finished_epoch']<=b['epoch_deadline'] and 0<n and n+4654<=30000)
 check('protocol_identity',m.sha(W/'live-protocol-attempt2.json')==p['protocol_sha256'])
 check('native_and_cache_counts',n==len(f)==len(states)-1==len(cloud)-1 and n<=len(dispatch)<=n+1)
 check('global_frames_contiguous',[x['frame'] for x in f]==list(range(1,n+1)) and [x['global_frame'] for x in executed]==list(range(1,n+1)))
 check('dispatch_equals_native',all(np.array_equal(a['command'],x['applied_command_si']) for a,x in zip(executed,f,strict=True)))
 check('cached_states_are_actual',all(s==x['after_state'] for s,x in zip(states[1:],f,strict=True)))
 check('native_hard_guards_recorded_correctly',all(m.manual.hard_guard_checks(x,allow_truck=True)==x['hard_guard_checks'] for x in f))
 guards_safe=all(x['hard_guard'] and x['hencky_material_guard']['safe'] and x['moving_bucket_sweep']['safe'] for x in f)
 check('cache_hashes',all(m.sha(D/'display'/k)==h for k,h in p['display']['cache_sha256'].items()))
 check('all_handoffs_preserve_state',all(x['before']==x['after'] and x['physical_and_native_unchanged'] for x in p['handoffs']))
 check('same_fixture_all_handoffs',len({(x['before']['fixture_id'],x['before']['bridge_id']) for x in p['handoffs']})==1)
 phases=p['phases'];check('reported_stage_ranges_contiguous',not phases or (phases[0]['start_frame']==3 and all(x['end_frame']==y['start_frame'] for x,y in zip(phases,phases[1:])) and phases[-1]['end_frame']<=n))
 with np.load(D/'zero.npz') as z:zero={k:z[k].copy() for k in z.files}
 with np.load(b['source']) as z:check('zero24_source_arrays',len(z.files)==24 and all(np.array_equal(z[k],zero[k]) for k in z.files if k!='simulation_time_s'))
 check('native_zero_gate',max(p['native_restore_errors'].values())<=1e-12 and p['zero_time_error_s']<=1e-12)
 old=R/'runs/continuous-single-scoop-20261014-v1';oldstates=m.read(old/'display/states.json');oldcloud=np.load(old/'display/particles.npy',mmap_mode='r');oldrows=rows(old/'dispatch.ndjson')
 check('first3_prefix_independently_exact',n>=3 and all(np.array_equal(cloud[i],oldcloud[2813+i]) and states[i]==oldstates[2813+i] and np.array_equal(executed[i-1]['command'],oldrows[2813+i-1]['command']) for i in range(1,4)))
 terminal=p.get('final_checkpoint',p.get('recovery_checkpoint'));check('terminal_checkpoint_hash',m.sha(terminal['path'])==terminal['sha256'])
 with np.load(terminal['path']) as z:final={k:z[k].copy() for k in z.files}
 check('clock_advanced_exactly_once',abs(float(final['simulation_time_s'][0])-float(zero['simulation_time_s'][0])-n/60)<1e-10)
 check('terminal_state_and_inventory',np.array_equal(final['particle_q'],cloud[-1]) and np.array_equal(final['rigid_qpos'],[x['qpos'] for x in states[-1]['rocks']]) and np.array_equal(final['rigid_ids'],zero['rigid_ids']) and np.array_equal(final['particle_flags'],zero['particle_flags']))
 r=m.CpuRuntime(D/'zero.npz',W/('audit-runtime-'+str(time.time_ns())),enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
 measurement_bindings=m.read(W/'measurement-bindings.json');check('measurement_inputs_frozen',all(m.sha(k)==h for k,h in measurement_bindings.items()))
 geometry=rows(D/'geometry-actions.ndjson');clearance=rows(D/'geometry-clearance.ndjson');maxcmd=0.;maxbias=0.
 for name in dict.fromkeys(x['phase'] for x in geometry):
  selected=[x for x in geometry if x['phase']==name and x['global_frame']<=n]
  if not selected:continue
  first=selected[0];state=first['before_state'];q=np.asarray(state['mechanism_qpos']);arrays=dict(r.arrays,mechanism_qpos=q);native=dict(r.metadata['controller_state'],controller_target=state['controller_target'])
  view=m.PhaseView(r,arrays=arrays,native=native,fresh_policy_history=True);offset=first['global_frame']-1;previous=np.asarray(executed[offset-1]['command']) if offset else np.zeros(4);controller=m.TransportController(view,m.FixedRouteClearance(view),previous)
  for item in selected:
   s=item['before_state'];command,detail=controller.command(name,np.asarray(s['mechanism_qpos']),s['controller_target'],np.asarray(item['desired']),np.asarray(item['velocity']))
   maxcmd=max(maxcmd,float(np.max(abs(command-item['command']))));maxbias=max(maxbias,float(np.max(abs(np.asarray(detail['load_bias'])-item['detail']['load_bias']))))
   assert np.array_equal(item['command'],executed[item['global_frame']-1]['command'])
  print('AUDIT_GEOMETRY',name,len(selected),flush=True)
 check('all_geometry_commands_and_biases_recomputed',maxcmd<=1e-15 and maxbias<=1e-15)
 check('logged_geometry_tracking_and_static_guards',all(x['axis_error']<=.025 and x['bucket_truck_m']>=.04 and x['mechanism_truck_m']>=.02 and x['mechanism_ground_m']>=.02 and x['closure_m']<=.001 for x in clearance))
 check('logged_approach_32cm_guard',all(x.get('bucket_observed_terrain_m',1.)>=.32 for x in clearance if x['phase'].startswith('SCOOP') and 'RETURN_RISE' not in x['phase']))
 decisions=rows(D/'policy-decisions.ndjson');vertices=m.manual._cache_rock_vertices(r.root,r.records);model=SAC.load(m.MODEL,device='cpu');torch.set_num_threads(1)
 maxprediction=0.;maxinput=0.;maxlaw=0.;maxfillerror=0.;scoops=[];allpolicy=[]
 from live_bucket_fill import MixedFillEstimator,load_config
 estimator=MixedFillEstimator(load_config(R/'configs/v4/live-bucket-fill-local-v1.json'),[x['volume_m3'] for x in r.records],vertices)
 for folder in sorted(D.glob('policy-*/episode-0001')):
  residual=m.read(folder/'report.json');cycle=int(folder.parent.name.split('-')[-1]);offset=residual['actual_entry_global_frame'];angle=residual['yaw_offset_rad'];env=m.ActualJointWaitEnv(W/('audit-reference-%d'%cycle),None,collect=False,waiting_action_mode='error_strength_v1',max_frames=7200,yaw_offset_rad=angle)
  rot=env.canonical_rotation;alignment=rows(folder/'alignment.ndjson');actions=rows(folder/'actions.ndjson');ds=[d for d in decisions if d['scoop']==cycle];allpolicy.extend(actions)
  latest={(x['frame']):x for x in alignment if x['stage'] in ('attach','after')}
  check('scoop%d_actions_match_actual_dispatch'%cycle,all(x['frame']+offset<=n and np.array_equal(x['applied_si'],executed[offset+x['frame']-1]['command']) for x in actions))
  for index,d in enumerate(ds):
   local=d['local_frame_before'];globalframe=d['global_frame_before'];assert offset+local==globalframe;state=states[globalframe];obs=np.asarray(d['observation'],np.float32);q=np.asarray(state['mechanism_qpos']).copy();q[7]-=angle
   maxprediction=max(maxprediction,float(np.max(abs(model.predict(obs,deterministic=True)[0]-d['action']))));record=latest[local];cursor=record['reference_progress'];i=min(int(cursor),1170);j=min(i+1,1170);fraction=cursor-i;target=env.reference_targets[i]*(1-fraction)+env.reference_targets[j]*fraction
   points=[cloud[globalframe]]
   for v,rock in zip(vertices,state['rocks']):pose=np.asarray(rock['qpos']);points.append(v@quaternion_matrix_wxyz(pose[3:]).T+pose[:3])
   height,known=height_grid(np.concatenate(points),np.asarray(state['bucket_origin_world_m']),rot,angle);world=d['world_bucket_wrench_before'];wrench=np.zeros(6) if world is None else np.r_[rot.T@np.asarray(world['force']),rot.T@np.asarray(world['torque'])]/1e5
   prev=np.asarray(executed[globalframe-1]['command']) if globalframe else np.zeros(4);prevaction=np.zeros(5) if index==0 else np.asarray(ds[index-1]['action']);hold=any(s['frame']<=local for s in residual.get('scores',[]));base=np.zeros(4) if hold else env.commands[min(i,1169)]/LIMITS_SI
   expected=np.r_[q,state['mechanism_qvel'],np.asarray(state['mechanism_qpos'])-env.reference_q[i],rot.T@np.asarray(state['bucket_origin_world_m'])-ANCHOR,rot.T@np.asarray(state['bucket_linear_velocity_world_m_s']),rot.T@np.asarray(state['bucket_angular_velocity_world_rad_s']),wrench,base,prev/LIMITS_SI,prevaction,cursor/1170,local/1560,float(world is not None),height,known,(target-state['controller_target'])/LIMITS_SI,record['actual_progress']/1170,float(record['mode'].startswith('wait'))]
   maxinput=max(maxinput,float(np.max(abs(np.clip(expected,-10,10).astype(np.float32)-obs))))
  for item in actions:
   local=item['frame']-1;state=states[offset+local];cursor=item['cursor_before'];i=min(int(cursor),1170);j=min(i+1,1170);fraction=cursor-i;error=(env.reference_q[i]*(1-fraction)+env.reference_q[j]*fraction-np.asarray(state['mechanism_qpos']))[[7,9,12,15]]
   if item['hold_frame'] is not None:expected=np.zeros(4)
   else:
    _,_,expected,pace=command_at(env.commands,cursor,np.asarray(item['effective_action']),.1,reference_limit=max(cursor,item['alignment']['reference_limit']),wait_target_error=error,waiting_action_mode='error_strength_v1');maxlaw=max(maxlaw,abs(pace-item['pace']))
   maxlaw=max(maxlaw,float(np.max(abs(expected-item['request_si']))),float(np.max(abs(error-item['alignment']['actual_joint_error_si']))))
  for saved in residual.get('scores',[]):
   g=offset+saved['frame'];state=states[g];pose=r.assembly.assemble(np.asarray(state['mechanism_qpos']),include_meshes=False)[1]['bucket']
   recalculated=estimator.measure_snapshot(dict(frame=saved['frame'],soil=cloud[g],pose=pose,rocks=np.asarray([x['qpos'] for x in state['rocks']])))
   key='scoop%d_fill_validity_and_reasons'%cycle
   check(key,checks.get(key,True) and recalculated['valid_estimate']==saved['valid_estimate'] and recalculated['quality_reasons']==saved['quality_reasons'])
   if saved['valid_estimate'] and recalculated['valid_estimate']:maxfillerror=max(maxfillerror,abs(recalculated['fill_percent']-saved['fill_percent']))
  waits=sum(x['execution_mode']=='wait' for x in actions);holds=sum(x['hold_frame'] is not None for x in actions)
  scoops.append(dict(scoop=cycle,status=residual['status'],complete=bool(residual.get('success') and residual.get('cursor')==1170 and holds==30),frames=len(actions),wait_frames=waits,wait_fraction=waits/max(1,len(actions)),hold_frames=holds,retained_fill_fraction=residual.get('retained_fill_fraction'),policy_decisions=len(ds)))
 check('frozen_predictions_recomputed',maxprediction<=1e-7)
 check('actual221_inputs_reconstructed',maxinput<=1e-6)
 check('actual_joint_wait_law_recomputed',maxlaw<=1e-12)
 check('hold_fill_values_independently_recomputed',maxfillerror<=1e-9)
 check('model_updates_unchanged',model._n_updates==1000 and p['neural_updates']==0)
 check('wait_direction_and10percent_cap',all(np.all(np.asarray(x['request_si'])*np.asarray(x['alignment']['actual_joint_error_si'])>=-1e-16) and np.all(abs(np.asarray(x['request_si']))<=LIMITS_SI*.1+1e-15) for x in allpolicy if x['execution_mode']=='wait'))
 check('cross_mode_invalidated_wait_stops',all(np.array_equal(x['request_si'],np.zeros(4)) for x in allpolicy if x['execution_mode']=='wait' and x['action_invalidated']))
 check('gpu_physics_stopped',not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip())
 complete=p['status']=='edge_adjacent_pair_completed' and len(scoops)==2 and all(s['complete'] for s in scoops)
 if complete:
  names=[x['name'] for x in phases];first=names.index('SCOOP1_RESIDUAL');second=names.index('SCOOP2_RESIDUAL');between=names[first+1:second]
  required=['CLEAR_PILE','ALIGN','DUMP','CLEAR_TRUCK','WITHDRAW_SWING','RETURN_SAFE']
  check('real_unload_and_return_between_digs',all(name in between for name in required) and [between.index(name) for name in required]==sorted(between.index(name) for name in required))
  check('final_stage_covers_final_frame',phases[-1]['end_frame']==n and names[-1]=='SCOOP2_RESIDUAL')
  check('terminal_snapshot_cycle2',m.read(str(terminal['path'])+'.json')['cycle_index']==2)
 # Rebuild observed maps from the exact saved native phase snapshots.
 spec=m.importlib.util.spec_from_file_location('audit_fresh_stock',R/'work/contact-fixed-truck-continuation-20260925-v2/observation.py');observer=m.importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
 for cycle in (1,2):
  folder=D/('scoop-%d-plan'%cycle)
  if not folder.exists():continue
  snapshot=D/('prefix3.npz' if cycle==1 else 'first-returned.npz')
  if not snapshot.exists():continue
  cr=m.CpuRuntime(snapshot,W/('audit-stock-%d-'%cycle+str(time.time_ns())),enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
  terrain,observation=observer.stationary_terrain(cr,cr.arrays);plan=m.read(folder/'report.json')
  with np.load(folder/'terrain.npz') as z:check('scoop%d_fresh_observed_map_exact'%cycle,all(np.array_equal(np.asarray(terrain[k]),z[k],equal_nan=True) for k in z.files))
  check('scoop%d_observation_metadata_exact'%cycle,observation==plan['observation'])
  if cycle==1:
   candidates=m.rank_workspace_candidates(terrain,3);outermost=max(abs(c['entry_xy_m'][1]) for c in candidates if c['future_strip_count']>=1)
   check('first_scoop_actually_outermost_eligible',abs(abs(plan['selected']['entry_xy_m'][1])-outermost)<1e-9 and outermost>.3)
  else:
   first=m.read(D/'scoop-1-plan/actual-cut-trace.json')['trace'][0]['mouth_xy'];ranked,_=m.rank_adjacent(terrain,first,inward_sign=-1)
   selected=next(x for x in ranked if x['entry_xy_m']==plan['selected']['entry_xy_m'])
   check('second_adjacency_uses_actual_first_entry_and_fresh_map',selected['adjacency']==plan['selected']['adjacency'] and abs(selected['adjacency']['nominal_entry_boundary_gap_m'])<=float(terrain['resolution_m']))
 result=dict(status=('passed' if complete else 'diagnostic_audit_passed') if all(checks.values()) else 'audit_failed',record_integrity_passed=all(checks.values()),checks=checks,check_count=len(checks),two_scoop_goal_complete=complete,original_native_material_sweep_guards_safe=guards_safe,run_status=p['status'],failure=p.get('failure'),frames=n,cumulative_budget_physical_frames=n+4654,scoops=scoops,maximum_geometry_command_error=maxcmd,maximum_geometry_bias_error=maxbias,maximum_prediction_error=maxprediction,maximum221_input_error=maxinput,maximum_wait_law_error=maxlaw,maximum_fill_display_error_percentage_points=maxfillerror,new_physics_steps=0,training_updates=0,wall_seconds=time.monotonic()-start,comparison_scope='Old central-first pair is historical trend reference only; no matched central branch was authorized.')
 m.write(D/'read-only-audit.json',result);print(m.encode({k:v for k,v in result.items() if k!='checks'}),flush=True)
 assert all(checks.values()),[k for k,v in checks.items() if not v]
if __name__=='__main__':main()

