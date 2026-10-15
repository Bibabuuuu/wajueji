"""CPU-only continuity, command and actual frozen-policy participation audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from continuous_scoop import *
from stable_baselines3 import SAC
import argparse,torch
rows=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines()]

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run;started=time.monotonic()
    report=read(out/'report.json');protocol=read(W/'protocol.json');budget=read(protocol['budget_path'])
    assert report['status']=='continuous_single_scoop_completed'
    frames=rows(out/'physics/frames.ndjson');dispatch=rows(out/'dispatch.ndjson');states=read(out/'display/states.json');phases=read(out/'display/phases.json')
    cloud=np.load(out/'display/particles.npy',mmap_mode='r');n=report['frames']
    checks=dict(completed=report['status']=='continuous_single_scoop_completed',one_native_instance=report['native_initializations']==1,
        original_source_unchanged=sha(SOURCE)==SOURCE_SHA and sha(str(SOURCE)+'.json')==report['source_metadata_sha256'],frozen_model_unchanged=sha(MODEL)==MODEL_SHA==report['model_sha256'],
        no_training=report['neural_updates']==0,within_frame_ceiling=0<n<=15000,within_fixed_budget=(out/'report.json').stat().st_mtime<budget['epoch_deadline'],
        input_code_unchanged=all(sha(k)==v for k,v in protocol['code_sha256'].items()),protocol_identity=sha(W/'protocol.json')==report['protocol_sha256'],
        budget_identity=sha(protocol['budget_path'])==report['budget_sha256'],
        counts=n==len(frames)==len(dispatch)==len(states)-1==len(cloud)-1==len(phases)-1,
        global_frames_contiguous=[x['frame'] for x in frames]==list(range(1,n+1)) and [x['global_frame'] for x in dispatch]==list(range(1,n+1)),
        dispatch_equals_native=all(np.array_equal(d['command'],f['applied_command_si']) for d,f in zip(dispatch,frames,strict=True)),
        actual_cache_states_match=all(s==f['after_state'] for s,f in zip(states[1:],frames,strict=True)),
        hard_guards_recomputed=all(manual.hard_guard_checks(f,allow_truck=True)==f['hard_guard_checks'] and f['hard_guard'] for f in frames),
        all_material_and_sweep_guards=all(f['hencky_material_guard']['safe'] and f['moving_bucket_sweep']['safe'] for f in frames),
        all_handoffs_preserve_state=all(h['physical_and_native_unchanged'] and h['before']==h['after'] for h in report['handoffs']))
    ids={(h['before']['fixture_id'],h['before']['bridge_id']) for h in report['handoffs']};checks['one_fixture_and_bridge_across_all_phases']=len(ids)==1
    expected=['PREPARE_OUTWARD','PREPARE_HIGH','SAFE_START','APPROACH_HIGH','APPROACH_OUTWARD','APPROACH_PREPOINT','RESIDUAL','CLEAR_PILE','ALIGN','DUMP','CLEAR_TRUCK','WITHDRAW_SWING','RETURN_SAFE','FINAL_HOLD']
    checks['all_stages_in_order']=[x['name'] for x in report['phases']]==expected
    checks['stage_ranges_are_continuous']=report['phases'][0]['start_frame']==0 and report['phases'][-1]['end_frame']==n and all(x['end_frame']==y['start_frame'] for x,y in zip(report['phases'],report['phases'][1:]))
    checks['all_cache_hashes']=all(sha(out/'display'/k)==v for k,v in report['display']['cache_sha256'].items())
    with np.load(out/'zero.npz') as data:zero={k:data[k].copy() for k in data.files}
    with np.load(SOURCE) as data:checks['zero_source_23_arrays_exact']=all(np.array_equal(zero[k],data[k]) for k in data.files if k!='simulation_time_s') and len(data.files)==24
    checks['zero_source_clock']=abs(float(zero['simulation_time_s'][0])-report['initial_simulation_time_s'])<1e-12
    with np.load(out/'final.npz') as data:final={k:data[k].copy() for k in data.files}
    checks['final_saved_hash']=sha(out/'final.npz')==report['final_checkpoint']['sha256']
    checks['final_cache_particles_and_rocks_exact']=np.array_equal(final['particle_q'],cloud[-1]) and np.array_equal(final['rigid_qpos'],[r['qpos'] for r in states[-1]['rocks']])
    checks['particle_inventory_and_ids_unchanged']=np.array_equal(zero['particle_flags'],final['particle_flags']) and np.array_equal(zero['rigid_ids'],final['rigid_ids']) and final['particle_q'].shape==(256441,3)
    checks['physical_clock_advanced_once']=abs(float(final['simulation_time_s'][0])-float(zero['simulation_time_s'][0])-n/60)<1e-10
    r=CpuRuntime(out/'zero.npz',out/'audit-cpu-v2',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
    ga=rows(out/'geometry-actions.ndjson');gc=rows(out/'geometry-clearance.ndjson');all_exact=[];all_bias=[];numeric_errors=[];dispatch_exact=[]
    for phase in report['phases']:
        name=phase['name']
        if name in ('RESIDUAL','FINAL_HOLD'):continue
        selected=[x for x in ga if x['phase']==name];first=selected[0];q=np.array(first['before_state']['mechanism_qpos']);arrays=dict(r.arrays,mechanism_qpos=q)
        native=dict(r.metadata['controller_state'],controller_target=first['before_state']['controller_target'])
        view=PhaseView(r,arrays=arrays,native=native);previous=np.zeros(4) if phase['start_frame']==0 else np.array(dispatch[phase['start_frame']-1]['command'])
        c=TransportController(view,FixedRouteClearance(view),previous)
        for action in selected:
            state=action['before_state'];command,detail=c.command(name,np.array(state['mechanism_qpos']),state['controller_target'],np.array(action['desired']),np.array(action['velocity']))
            all_exact.append(np.array_equal(command,action['command']));numeric_errors.append(float(np.max(np.abs(command-np.asarray(action['command'])))));dispatch_exact.append(np.array_equal(action['command'],dispatch[action['global_frame']-1]['command']));all_bias.append(np.array_equal(detail['load_bias'],action['detail']['load_bias']))
        last=selected[-1]['global_frame'];checks[name+'_zero_stop_tail']=phase['end_frame']-last==30 and all(not np.any(dispatch[f-1]['command']) for f in range(last+1,phase['end_frame']+1))
        print('AUDIT_GEOMETRY',name,len(selected),flush=True)
    checks['all_geometry_commands_recomputed_within_1e15']=max(numeric_errors)<=1e-15;checks['all_geometry_dispatch_exact']=all(dispatch_exact);checks['all_geometry_biases_recomputed_exact']=all(all_bias)
    checks['all_geometry_tracking_guards']=max(c['axis_error'] for c in gc)<=.025
    checks['all_original_static_clearances']=all(c['bucket_truck_m']>=.04 and c['mechanism_truck_m']>=.02 and c['mechanism_ground_m']>=.02 and c['closure_m']<=.001 for c in gc)
    app=[c for c in gc if c['phase'].startswith(('PREPARE','SAFE_START','APPROACH'))]
    checks['all_actual_approach_clearances']=all(c['bucket_observed_terrain_m']>=.32 and c['mechanism_terrain_clearance_m']>=.02 and c['static_truck_aabb_separation_m']>=.02 for c in app)
    residual=next(x for x in report['phases'] if x['name']=='RESIDUAL');offset=residual['start_frame'];actions=rows(out/'policy/episode-0001/actions.ndjson');decisions=rows(out/'policy-decisions.ndjson')
    checks['residual_local_frames_contiguous']=[x['frame'] for x in actions]==list(range(1,residual['end_frame']-offset+1))
    checks['residual_actual_dispatch_matches']=all(np.array_equal(x['applied_si'],dispatch[offset+x['frame']-1]['command']) for x in actions)
    checks['residual_1170_reference_and_hold']=report['residual_result']['success'] and report['residual_result']['cursor']==1170 and sum(x['hold_frame'] is not None for x in actions)==30
    torch.set_num_threads(1);model=SAC.load(MODEL,device='cpu');updates=model._n_updates;prediction=[];local_clock=[];state_match=[]
    for d in decisions:
        obs=np.asarray(d['observation'],np.float32);prediction.append(np.array_equal(model.predict(obs,deterministic=True)[0],d['action']))
        local_clock.append(d['global_frame_before']-offset==d['local_frame_before'] and obs[85]==np.float32(d['local_frame_before']/1560))
        state_match.append(np.array_equal(obs[:19],np.clip(np.array(states[d['global_frame_before']]['mechanism_qpos']),-10,10).astype(np.float32)))
    checks['all_frozen_predictions_recomputed_exact']=all(prediction) and len(prediction)>0
    checks['policy_uses_local_phase_clock']=all(local_clock) and decisions[0]['local_frame_before']==0 and offset>0
    checks['policy_observes_actual_state']=all(state_match)
    checks['model_updates_unchanged']=model._n_updates==updates==1000
    checks['cached_playback_no_new_physics']=report['display']['live_inference'] is False and report['display']['playback_physics_steps']==0 and report['display']['capture_origin']=='single_native_continuous_scoop'
    checks['final_phase_zero_hold']=all(not np.any(d['command']) for d in dispatch[report['phases'][-1]['start_frame']:])
    result=dict(status='passed' if all(checks.values()) else 'failed',check_count=len(checks),checks=checks,global_frames=n,simulation_seconds=n/60,
        neural_updates=0,native_initializations=1,policy_decisions=len(decisions),residual_frames=residual['end_frame']-offset,residual_start_global_frame=offset,
        retained_fill_fraction=report['residual_result'].get('retained_fill_fraction'),geometry_commands_recomputed=len(all_exact),geometry_exact_commands=sum(all_exact),geometry_nonexact_commands=sum(not x for x in all_exact),geometry_max_abs_difference_si=max(numeric_errors),geometry_numeric_tolerance_si=1e-15,phases=report['phases'],audit_new_physical_frames=0,
        full_three_scoop_qualified=False,scope='One fixed-scene current-target continuous scoop, including preparation from known prepoint, true approach, policy inference, unload and safe return. No formal throughput/volume/generalization claim.',wall_seconds=time.monotonic()-started)
    write(out/'read-only-audit.json',result);print(encode({k:result[k] for k in ['status','check_count','global_frames','policy_decisions','residual_frames','retained_fill_fraction','wall_seconds']}),flush=True)
    assert all(checks.values()),[k for k,v in checks.items() if not v]

if __name__=='__main__':main()
