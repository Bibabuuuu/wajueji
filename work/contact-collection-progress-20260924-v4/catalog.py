"""Read-only completed-recording catalog; the currently active run is excluded."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUNS=['manual-contact-20260924-005742-1081585','manual-contact-20260924-010900-1088617',
      'manual-contact-20260924-083547-1327233','manual-contact-20260924-093202-1362929',
      'manual-contact-20260924-100857-1387193','manual-contact-20260924-110051-1421079',
      'manual-contact-20260924-110932-1426831','manual-contact-20260924-113555-1443279','manual-contact-20260924-114908-1452582']


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def rows(path): return [json.loads(line) for line in path.read_text().splitlines()]


episodes=[]
for runid in RUNS:
    run=ROOT/'runs/layered-47-sparse-manual-contact-v1'/runid
    paths=[run/name for name in ('report.json','input.ndjson','physics.ndjson','observation.ndjson')]
    before={str(p):sha(p) for p in paths}
    r=read(paths[0]);actions=rows(paths[1]);physics=rows(paths[2]);observations=rows(paths[3])
    assert r['status']=='manual_contact_pilot_recorded' and r['stop_reason']=='user_finished'
    assert all(r['checks'].values()) and len(actions)==len(physics)
    assert [a['frame'] for a in actions]==list(range(1,len(actions)+1))
    assert [o['frame'] for o in observations]==list(range(3,len(actions)+1,3))
    assert all(a['applied_command_si']==p['applied_command_si'] for a,p in zip(actions,physics))
    assert all(p['hard_guard'] and p['hencky_material_guard']['safe'] and p['moving_bucket_sweep']['safe'] for p in physics)
    assert all(o['state']==physics[o['frame']-1]['after_state'] for o in observations)
    assert all(a['input_device']=='system_safety_hold' and a['applied_command_si']==[0.]*4 for a in actions[-30:])
    snapshots=r.get('cycle_checkpoints') or r['diagnostic_snapshots']
    snapshot=snapshots[-1];point=Path(snapshot['path']);assert sha(point)==snapshot['sha256']
    metadata=read(Path(str(point)+'.json'))
    lineage=metadata.get('lineage_root',snapshot['sha256'])
    scoop=metadata.get('cycle_index',1)
    candidate=[a for a in actions if a['input_device'] in ('gamepad','keyboard') and a['armed'] and a['connected'] and a['window_focused']]
    assert before=={str(p):sha(p) for p in paths}
    source_reset=metadata.get('training_reset')
    parent=r['handoff_checkpoint']
    episodes.append(dict(run=runid,scoop_index=scoop,lineage_root=lineage,frames=len(actions),
        candidate_manual_frames=len(candidate),system_hold_frames=30,excluded_nonexpert_frames=len(actions)-len(candidate),
        snapshot=dict(path=str(point),sha256=snapshot['sha256']),
        input_prepoint=parent,source_bindings=before,
        material_history_policy=metadata.get('material_history_policy','legacy_accumulated_Jp'),
        reset_mode=source_reset['mode'] if source_reset else 'continuous_or_original_prepoint',
        completed_inventory_entries=len(source_reset['entries']) if source_reset else 0,
        operator_retargeting_review=('confirmed_separate_annotation' if runid.endswith('110932-1426831')
                                    else 'reported_first_entry_shift' if runid.endswith('010900-1088617') else 'pending'),
        recording_review_pass=True,training_eligible=False))
groups={}
for e in episodes: groups.setdefault(e['lineage_root'],[]).append(e)
initial_conditions={group:next((e['input_prepoint']['sha256'] for e in entries if e['scoop_index']==1),None)
                    for group,entries in groups.items()}
report=dict(schema='contact_collection_catalog_v2',episodes=episodes,total_episodes=len(episodes),
    candidate_manual_frames=sum(e['candidate_manual_frames'] for e in episodes),
    scoop_counts=dict(Counter(str(e['scoop_index']) for e in episodes)),
    lineage_groups=len(groups),completed_three_scoop_recording_groups=sum(sorted(e['scoop_index'] for e in es)==[1,2,3] for es in groups.values()),
    initial_prepoint_by_lineage=initial_conditions,distinct_initial_prepoints=len(set(initial_conditions.values())),
    split_rule='Keep entire lineage together; roots share the same original scene, so held-out trajectories do not prove unseen-scene generalization.',
    training_eligible=False,remaining_before_training=[
        'Review phase boundaries and retargeting; never treat a posthoc target as an online observation.',
        'Agree the native 60 Hz action/training adapter; do not silently average into legacy 20 Hz labels.',
        'Retain reset/material policy domains and choose whole-lineage holdout before fitting normalization.',
        'Export and verify causal tensors, then run bounded initial training and independent closed-loop evaluation.'],
    current_collection_target=dict(completed_episodes=9,planned_additional_scoops=1,expected_total=10,
        expected_scoop_counts={'1':4,'2':3,'3':3},reason='An additional first scoop provides the real excavated prefix for the missing second/third scoops.'))
(HERE/'catalog.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2))
