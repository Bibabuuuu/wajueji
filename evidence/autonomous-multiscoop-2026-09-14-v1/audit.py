"""Independent checks for a continuous multi-cycle dynamics run."""
from common import *
import argparse

ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args()
r=read(a.run/'report.json');rows=[json.loads(s) for s in (a.run/'tracking.jsonl').read_text().splitlines()]
assert rows,'no_dynamic_evidence'
states=[x['mechanism'] for x in rows];q=np.array([s['qpos'] for s in states])
target=np.array([s['target'] for s in states]);bounds=np.array(r['target_bounds'])
times=np.array([x['time_s'] for x in rows]);cycle_ids=np.array([x['cycle'] for x in rows])
applied=np.array([x['applied_force_n'] for x in rows]);measured=np.array([x['measured_force_n'] for x in rows])
at=np.array([x['applied_torque_nm'] for x in rows]);mt=np.array([x['measured_torque_nm'] for x in rows])
acc=np.array([s['maximum_reference_acceleration'] for s in states]);ref=np.array([x['reference'] for x in rows])
equations=[abs((np.array(s['last_hold_control'])-s['last_control_target'])*s['hold_gain']-
               s['unloaded_feedforward']-s['external_load_compensation']).max() for s in states]
cycles=r['cycles'];entries=np.array([c['selection']['entry_xy_m'] for c in cycles])
separations=[] if len(entries)<2 else [float(np.linalg.norm(entries[i]-entries[j]))
    for i in range(len(entries)) for j in range(i)]
checks={
    'requested_cycles_completed':r['status']=='passed' and r['cycles_completed']==r['cycles_requested']==len(cycles),
    'each_cycle_path_and_holds':all(c['status']=='passed' and c.get('hold_duration_s',0)>=10-1e-8
        and c.get('final_hold_duration_s',0)>=10-1e-8 for c in cycles),
    'all_actual_stage_completions':all(all(c['stages'].get(s,{}).get('status')=='passed' for s in STAGES) for c in cycles),
    'terrain_reobserved_and_dispatch_checked':all(c['dispatch_validation']['valid'] for c in cycles),
    'planning_retries_bounded':all(1<=len(c.get('planning_attempts',[]))<=r.get('planning_replan_limit',0)
        and c['planning_attempts'][-1]['status']=='passed'
        and all(x['status']=='rejected' and 'cut_selection_invalid:' in x.get('error','')
                for x in c['planning_attempts'][:-1]) for c in cycles),
    'planning_retries_do_not_advance_physics':all(len({(x['start_sim_s'],x['start_particle_step'])
        for x in c.get('planning_attempts',[])})==1 for c in cycles),
    'selection_history_separation':not separations or min(separations)>=.58-1e-8,
    'finite_q_and_target':bool(np.isfinite(q).all() and np.isfinite(target).all()),
    'dt_120hz':bool(np.max(abs(np.diff(times)-1/120))<1e-8),
    'cycle_ids_ordered':np.array_equal(np.unique(cycle_ids),np.arange(1,r['cycles_requested']+1)),
    'internal_target_8mm':bool(np.all(target>=bounds[:,0]-1e-10) and np.all(target<=bounds[:,1]+1e-10)),
    'reference_8mm':bool(np.all(ref[:,[7,9,12,15]]>=bounds[:,0]-1e-10) and np.all(ref[:,[7,9,12,15]]<=bounds[:,1]+1e-10)),
    'one_step_wrench_delay':bool(np.array_equal(applied[1:],measured[:-1]) and np.array_equal(at[1:],mt[:-1])),
    'control_port_equation':max(equations)<1e-7,
    'load_compensation_sign':all(np.array_equal(np.array(s['external_load_compensation']),-np.array(s['equivalent_drive_load'])) for s in states),
    'reference_acceleration':bool(np.all(acc<=np.array([1.5,1,1,1])+1e-10)),
    'zero_emergency_clamps':all(s['emergency_reference_substeps']==0 for s in states),
    'closure_under_1mm':max(s['closure_m'] for s in states)<.001,
    'perception_fresh_at_dispatch':all(0<=x['perception_age_wall_s']<=.3 and 0<=x['perception_age_sim_s']<=.3 for x in rows),
    'live_initial_selection_matches_plan':r['live_initial_selection_match'],
    'sources_unchanged':r['sources_unchanged']}
checks['stage_confirmation_and_end_state']=all(e['actual_reached'] and e['confirmation_s']>=.25-1e-9 for e in r['events']) and all(
    np.max(abs(np.array(v.get('endpoint_error',[1]))))<=.001+1e-10 and
    np.all(abs(np.array(v.get('endpoint_velocity',[1,1,1,1])))<=np.array([.005,.002,.002,.002])+1e-10)
    for c in cycles for v in c['stages'].values())
checks['physical_clock_and_log_count']=(r['mechanism_steps']==r['logged_steps']==len(rows)
    and abs(r['sim_s']-len(rows)/120)<1e-7 and (r['mode']=='empty' or r['mpm_steps']==len(rows)))
checks['cycle_state_continuity']=all(cycles[i]['start_particle_step']==cycles[i-1]['end_particle_step'] and
    abs(cycles[i]['start_sim_s']-cycles[i-1]['end_sim_s'])<1e-8 for i in range(1,len(cycles)))
if r['mode']=='contact':
    checks['per_cycle_new_acquisition']=all(c.get('hold_end_newly_acquired_kg',-1)>=20 for c in cycles)
    checks['per_cycle_stable_delivery']=all(c.get('stable_delivery_increment_kg',-1)>=10 for c in cycles)
    checks['recorded_residual']=np.isfinite(r['retention_last']['cavity_kg']) and r['retention_last']['cavity_kg']>=0
result={'passed':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},
    'status':r['status'],'steps':len(rows),'cycles':len(cycles),'sim_s':times[-1],
    'selections':entries.tolist(),'minimum_entry_separation_m':None if not separations else min(separations),
    'max_axis_error':abs(q[:,[7,9,12,15]]-ref[:,[7,9,12,15]]).max(0).tolist(),
    'maximum_internal_bound_excess':np.maximum(bounds[:,0]-target,target-bounds[:,1]).max(0).tolist(),
    'maximum_reference_acceleration':acc.max(0).tolist(),'max_closure_m':max(s['closure_m'] for s in states),
    'max_force_n':float(np.linalg.norm(measured,axis=1).max()),'nonzero_force_steps':int(np.any(applied!=0,axis=1).sum()),
    'inputs_sha256':{n:sha(a.run/n) for n in ('report.json','tracking.jsonl')}}
if (a.run/'soil.jsonl').exists():
    soil=[json.loads(s) for s in (a.run/'soil.jsonl').read_text().splitlines()]
    result['maximum_outlier_count']=max(x['outlier_count'] for x in soil)
    domain=[json.loads(s) for s in (a.run/'scene-envelope.jsonl').read_text().splitlines()]
    result['maximum_scene_outlier_count']=max(x['scene_outlier_count'] for x in domain)
    result['checks']['no_sampled_scene_outliers']=result['maximum_scene_outlier_count']==0
    result['checks']['domain_sampling_complete']=len(domain)==len(rows)//12
    result['passed']=bool(all(result['checks'].values()))
(a.run/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
raise SystemExit(0 if result['passed'] else 1)
