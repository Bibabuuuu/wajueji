"""Read-only native-rate staging references, not a model/feature contract."""
from bisect import bisect_left
from collections import Counter
from pathlib import Path
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
DATA=ROOT/'local-data/contact-bc-native-staging-20260924-v2'
CATALOG=ROOT/'work/contact-collection-progress-20260924-v5/catalog.json'
LIMITS=[.12,.0225,.0225,.0675]

def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(v) for v in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(condition,reason):
    if not condition:raise ValueError(reason)
def anchor_before(frame,anchors):
    i=bisect_left(anchors,frame)-1
    if i<0:raise ValueError('no_prior_terrain')
    return anchors[i]
def expert_control(a):
    return a['input_device'] in ('gamepad','keyboard') and a['armed'] and a['connected'] and a['window_focused'] and not a.get('emergency_stop_latched',False)
def onset(flags,start=0,width=3):
    return next((i+1 for i in range(start,len(flags)-width+1) if all(flags[i:i+width])),None)
def phase_audit(actions,physics):
    u=[[x/y for x,y in zip(a['applied_command_si'],LIMITS)] for a in actions]
    first=onset([max(map(abs,a))>=.05 for a in u])
    contact=onset([p['bucket_mpm_contact_samples']>=10 for p in physics],width=5)
    curl=onset([a[3]>=.5 for a in u],width=30)
    z=[p['after_state']['bucket_origin_world_m'][2] for p in physics]
    lift=onset([a[1]>=.5 and i+14<len(z) and z[i+14]-z[i]>=.02 for i,a in enumerate(u)],start=max(0,(curl or 1)-1),width=15)
    return dict(first_motion=first,first_five_contact_samples_ge10=contact,dominant_curl=curl,dominant_lift=lift,
                posthoc_only=True,contains_future_state=True,allowed_as_policy_input=False,physical_detachment_proven=False)

def main():
    catalog=read(CATALOG);DATA.mkdir(parents=True,exist_ok=False)
    result=dict(schema='contact_bc_native_staging_v1',catalog_sha256=sha(CATALOG),native_action_hz=60,
                limits_si=LIMITS,training_eligible=False,model_action_rate_decision='not_decided_by_this_reference_index',episodes=[])
    for e in catalog['episodes']:
        run=Path(e['snapshot']['path']).parent
        paths={n:run/n for n in ('report.json','input.ndjson','physics.ndjson','observation.ndjson')}
        bindings={n:sha(p) for n,p in paths.items()}
        require(all(bindings[n]==e['source_bindings'][str(p)] for n,p in paths.items()),'catalog_source_changed')
        r=read(paths['report.json']);actions=rows(paths['input.ndjson']);physics=rows(paths['physics.ndjson']);obs=rows(paths['observation.ndjson']);n=len(actions)
        require(r['protocol']['hz']==60 and r['protocol']['maximum_command_si']==LIMITS,'rate_or_action_contract')
        require(r['protocol']['decision_observation_stride_frames']==3,'terrain_stride')
        require(len(physics)==n and [a['frame'] for a in actions]==list(range(1,n+1)),'frame_sequence')
        require([o['frame'] for o in obs]==list(range(3,n+1,3)),'observation_sequence')
        require(all(o['state']==physics[o['frame']-1]['after_state'] for o in obs),'post_step_observation_mismatch')
        require(all(a['applied_command_si']==p['applied_command_si'] and a['requested_command_si']==p['requested_command_si'] for a,p in zip(actions,physics)),'action_mismatch')
        require(all(p['hard_guard'] and p['hencky_material_guard']['safe'] and p['moving_bucket_sweep']['safe'] for p in physics),'physics_guard')
        require(all(a['input_device']=='system_safety_hold' and a['applied_command_si']==[0.]*4 for a in actions[-30:]),'system_tail')
        anchors=[0]+[o['frame'] for o in obs];eligible=[];excluded=Counter();last=None;sequence=0;sequence_start=None
        commands=[a['applied_command_si'] for a in actions]
        for a,p in zip(actions,physics):
            f=a['frame'];require(p['frame']==f,'physics_frame_identity')
            require(abs(a['sim_time_before_s']-(f-1)/60)<1e-8,'pre_action_time')
            require(all(math.isfinite(x) and abs(x)<=limit+1e-10 for name in ('applied_command_si','requested_command_si') for x,limit in zip(a[name],LIMITS)),'nonfinite_or_out_of_bounds_action')
            if not expert_control(a):
                excluded[a['input_device'] if a['input_device'] not in ('gamepad','keyboard') else 'inactive_manual_control']+=1;continue
            reset=last is None or f!=last+1
            if reset:sequence+=1;sequence_start=f
            anchor=anchor_before(f,anchors);age=f-1-anchor;require(0<=age<=2,'stale_terrain')
            eligible.append(dict(frame=f,sequence_id=sequence,reset_learning_history=reset,lineage=e['lineage_root'],scoop_index=e['scoop_index'],
                terrain_anchor_frame=anchor,terrain_age_frames=age,terrain_grid_available=anchor!=0,tensor_candidate=anchor!=0,terrain_reference=None if anchor==0 else f'observation.ndjson:frame={anchor}:terrain',
                proprioception_reference=f'physics.ndjson:frame={f}:before_state',
                previous_applied_action_si=commands[f-2] if f>1 else [0.]*4,
                load_history_frames=None if reset else [sequence_start,f-1],
                requested_action_si=a['requested_command_si'],applied_action_si=a['applied_command_si'],
                requested_action_normalized=[x/y for x,y in zip(a['requested_command_si'],LIMITS)],
                applied_action_normalized=[x/y for x,y in zip(a['applied_command_si'],LIMITS)],
                command_projection_reason=a['command_projection_reason'],
                recorded_goal=r['selected_cut'],goal_is_posthoc=False,training_eligible=False))
            last=f
        dest=DATA/(e['run']+'.ndjson');dest.write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in eligible))
        phases=phase_audit(actions,physics)
        source_events=r['phase_events'];comparison=[]
        candidates={'penetrate_deepen':'first_five_contact_samples_ge10','curl_load':'dominant_curl','lift_clear':'dominant_lift'}
        for event in source_events:
            k=candidates.get(event['label']);candidate=phases.get(k) if k else None
            comparison.append(dict(operator_phase=event['label'],operator_frame=event['next_frame'],candidate_kind=k,candidate_frame=candidate,
                                   operator_minus_candidate_frames=None if candidate is None else event['next_frame']-candidate))
        phase_doc=dict(original_events=source_events,posthoc_candidates=phases,comparison=comparison,manual_review_required=True,not_online_features=True)
        phase_path=DATA/(e['run']+'.phases.json');phase_path.write_text(json.dumps(phase_doc,indent=2)+'\n')
        projected=sum(a['command_projection_reason'] is not None for a in actions if expert_control(a))
        swings=sum(abs(a['applied_command_si'][0])>1e-6 for a in actions if expert_control(a))
        require(bindings=={name:sha(p) for name,p in paths.items()},'source_changed_during_read')
        require(len(eligible)==e['candidate_manual_frames'],'candidate_count_differs_from_catalog')
        result['episodes'].append(dict(run=e['run'],lineage=e['lineage_root'],scoop_index=e['scoop_index'],candidate_frames=len(eligible),tensor_candidate_frames=sum(x['tensor_candidate'] for x in eligible),sequence_count=sequence,
            excluded=dict(excluded),nonzero_swing_frames=swings,projected_manual_frames=projected,phase_comparison=comparison,
            source_bindings=bindings,index_path=str(dest),index_sha256=sha(dest),phase_path=str(phase_path),phase_sha256=sha(phase_path),
            recorded_goal=r['selected_cut'],operator_retargeting_review=e['operator_retargeting_review'],reset_mode=e['reset_mode'],material_history_policy=e['material_history_policy']))
    result['candidate_frames']=sum(e['candidate_frames'] for e in result['episodes'])
    require(result['candidate_frames']==catalog['candidate_manual_frames'],'total_count')
    result['tensor_candidate_frames']=sum(e['tensor_candidate_frames'] for e in result['episodes'])
    result['action_rate_decision']='user_confirmed_native60Hz_four_axes'
    result['checks']=dict(source_hashes_match_catalog=True,action_limits_from_runtime=True,states_actions_aligned=True,terrain_strictly_before_action=True,
                         maximum_terrain_age_frames=2,system_tail_excluded=True,sequence_gaps_reset_history=True,raw_and_applied_preserved=True,posthoc_labels_separate=True)
    result['split_proposal']=dict(status='draft_not_used_for_training',development_lineages=[k for k in catalog['initial_prepoint_by_lineage'] if not k.startswith('659f1360')],
        validation_lineages=[k for k in catalog['initial_prepoint_by_lineage'] if k.startswith('659f1360')],
        sealed_test_lineages=[],scope='All outcomes already inspected; validation is development validation, not a sealed or unseen-scene test.')
    (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n');(DATA/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(episodes=len(result['episodes']),frames=result['candidate_frames'],swing_frames=sum(e['nonzero_swing_frames'] for e in result['episodes']),projected_frames=sum(e['projected_manual_frames'] for e in result['episodes']))))

if __name__=='__main__':main()
