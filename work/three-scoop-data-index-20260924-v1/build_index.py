"""Native 60 Hz causal candidate references; no training or source rewriting."""
from pathlib import Path
from bisect import bisect_left
import hashlib,json
import numpy as np
ROOT=Path('/home/qunwang/wajueji');OUT=Path(__file__).resolve().parent
DATA=ROOT/'local-data/three-scoop-contact-candidates-20260924-v1'
RUNS=['manual-contact-20260924-010900-1088617','manual-contact-20260924-083547-1327233','manual-contact-20260924-093202-1362929']
GROUP='48b1f3785beec8b0062f8b18f36ff756852708d9df305cc24eb375e4837011b3'
LIMITS=np.array([.12,.0225,.0225,.0675])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def anchor_before(frame,anchors):
 i=bisect_left(anchors,frame)-1
 if i<0:raise ValueError('no_causal_terrain_anchor')
 return anchors[i]
def onset(condition,start=0,width=3):
 for i in range(start,len(condition)-width+1):
  if np.all(condition[i:i+width]):return i+1
 return None
assert [anchor_before(f,[0,3,6]) for f in (1,3,4,6,7)]==[0,0,3,3,6]
try:anchor_before(0,[0,3])
except ValueError:pass
else:raise AssertionError('future anchor accepted')
DATA.mkdir(parents=True,exist_ok=True)
summary=dict(schema='three_scoop_native_contact_candidate_index_v1',split_group=GROUP,split_assignment='unassigned_whole_chain_only',training_eligible=False,episodes=[],scope='Native 60Hz references, not a legacy 20Hz actor tensor dataset; no automatic transport expert labels.')
for scoop,runid in enumerate(RUNS,1):
 run=ROOT/'runs/layered-47-sparse-manual-contact-v1'/runid
 paths={n:run/n for n in ('report.json','input.ndjson','physics.ndjson','observation.ndjson')}
 bindings={n:sha(p) for n,p in paths.items()};report=read(paths['report.json']);actions=lines(paths['input.ndjson']);physics=lines(paths['physics.ndjson']);observations=lines(paths['observation.ndjson']);n=len(actions)
 assert report['status']=='manual_contact_pilot_recorded' and report['stop_reason']=='user_finished' and all(report['checks'].values())
 assert len(physics)==n and [r['frame'] for r in actions]==list(range(1,n+1))
 assert [r['frame'] for r in observations]==list(range(3,n+1,3))
 assert all(a['applied_command_si']==p['applied_command_si'] for a,p in zip(actions,physics))
 assert all(o['state']==physics[o['frame']-1]['after_state'] for o in observations)
 assert all(p['hard_guard'] and p['hencky_material_guard']['safe'] and p['moving_bucket_sweep']['safe'] for p in physics)
 if scoop==1:assert report['diagnostic_snapshots'][-1]['sha256']==GROUP
 else:assert report['cycle_lineage']['cycle_index']==scoop and report['cycle_lineage']['lineage_root']==GROUP
 anchors=[0]+[o['frame'] for o in observations]
 commands=np.array([a['applied_command_si'] for a in actions]);normalized=commands/LIMITS
 assert np.max(np.abs(normalized))<=1+1e-9
 first=onset(np.max(np.abs(normalized),axis=1)>=.05)
 curl=onset(normalized[:,3]>=.5,start=first-1,width=30)
 z=np.array([p['after_state']['bucket_origin_world_m'][2] for p in physics]);rise=np.r_[z[14:]-z[:-14],np.zeros(14)]
 lift=onset((normalized[:,1]>=.5)&(rise>=.02),start=curl-1,width=15)
 assert first and curl and lift and first<=curl<=lift<=n-30
 terminal=n-29
 assert all(a['input_device']=='system_safety_hold' and a['applied_command_si']==[0.,0.,0.,0.] for a in actions[-30:])
 phase=lambda f: 'system_terminal_hold' if f>=terminal else 'dominant_lift_candidate' if f>=lift else 'dominant_curl_candidate' if f>=curl else 'positioning_and_penetration_candidate' if f>=first else 'initial_manual_hold'
 intervals=[dict(label=phase(a),start_frame=a,end_frame=b) for a,b in [(1,first-1),(first,curl-1),(curl,lift-1),(lift,terminal-1),(terminal,n)] if b>=a]
 count=0;manualzero=0;excluded={};max_age=0;last_included_frame=None;sequence_id=0
 dest=DATA/(runid+'.ndjson')
 with dest.open('w') as stream:
  for a,p in zip(actions,physics):
   f=a['frame'];device=a['input_device']
   if device not in ('gamepad','keyboard') or not a['armed'] or not a['connected'] or not a['window_focused']:
    reason=device if device not in ('gamepad','keyboard') else 'unarmed_disconnected_or_unfocused';excluded[reason]=excluded.get(reason,0)+1;continue
   anchor=anchor_before(f,anchors);age=f-1-anchor;max_age=max(max_age,age);assert 0<=age<=2
   assert (f-1)/60==a['sim_time_before_s'] or abs((f-1)/60-a['sim_time_before_s'])<1e-8
   tags=[name for j,(neg,pos) in enumerate([('swing_negative','swing_positive'),('boom_lower','boom_raise'),('arm_extend','arm_retract'),('bucket_open','bucket_curl')]) for name in ([pos] if normalized[f-1,j]>.05 else [neg] if normalized[f-1,j]<-.05 else [])]
   new_sequence=last_included_frame is None or f!=last_included_frame+1
   if new_sequence:sequence_id+=1
   row=dict(sequence_id=sequence_id,reset_learning_history=new_sequence,source_run=str(run),frame=f,split_group=GROUP,scoop_index=scoop,sim_time_before_s=(f-1)/60,native_action_hz=60,
    terrain_anchor_frame=anchor,terrain_reference='report.json:initial_observation.terrain' if anchor==0 else f'observation.ndjson:frame={anchor}:terrain',terrain_age_frames=age,
    proprioception_reference=f'physics.ndjson:frame={f}:before_state',previous_action_si=commands[f-2].tolist() if f>1 else [0.,0.,0.,0.],
    load_history_reference=None if f==1 else f'physics.ndjson:frames=1..{f-1}',target_applied_action_si=a['applied_command_si'],target_normalized_action=normalized[f-1].tolist(),
    original_operator_phase=a['phase'],posthoc_action_phase=phase(f),action_tags=tags,phase_feature_allowed=False,training_eligible=False)
   stream.write(json.dumps(row,separators=(',',':'))+'\n');count+=1;manualzero+=int(not tags);last_included_frame=f
 assert bindings=={name:sha(p) for name,p in paths.items()}
 labels=dict(source_run=str(run),source_bindings=bindings,original_operator_events=report['phase_events'],intervals=intervals,
  method=dict(first_motion='3 consecutive frames any normalized action magnitude >=0.05',dominant_curl='30 consecutive frames normalized bucket curl >=0.5',dominant_lift='15 consecutive frames normalized boom raise >=0.5 and bucket origin rises >=2cm over next14frames',terminal='runner-owned last30 zero-command frames'),
  posthoc_only=True,contains_future_state_for_labeling=True,not_a_runtime_router=True,not_proof_of_detachment=True,manual_review_required=True)
 (DATA/(runid+'.phases.json')).write_text(json.dumps(labels,indent=2)+'\n')
 summary['episodes'].append(dict(run=runid,scoop_index=scoop,frames=n,sequence_count=sequence_id,recorded_selected_cut=report.get('selected_cut'),entry_conditioning_review_required=True,operator_changed_lateral_entry=(scoop==1),candidate_frames=count,manual_low_action_frames=manualzero,excluded=excluded,maximum_terrain_age_frames=max_age,phase_intervals=intervals,operator_phase_events=report['phase_events'],source_bindings=bindings,index_path=str(dest),index_sha256=sha(dest),phase_path=str(DATA/(runid+'.phases.json')),phase_sha256=sha(DATA/(runid+'.phases.json')),historical_native_history_missing=True))
summary['candidate_frames']=sum(e['candidate_frames'] for e in summary['episodes'])
summary['checks']=dict(source_unchanged=True,states_actions_aligned=True,all_runtime_guards=True,terrain_strictly_pre_action=True,terrain_age_at_most_two_frames=True,all_three_same_group=True,system_actions_excluded=True,posthoc_labels_not_features=True)
summary['remaining_before_training']=['review coarse phase boundaries and entry conditioning','choose explicit native60Hz or20Hz action adapter; never silently average labels','additional independent lineage groups for validation/test','local diagnostic-domain admission remains separate from formal A30 qualification']
(OUT/'report.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
(DATA/'manifest.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2,ensure_ascii=False))
