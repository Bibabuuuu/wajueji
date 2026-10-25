"""Apply a fixed human-calibrated angle to other trajectories, without fitting them."""
from pathlib import Path
import json,hashlib
W=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
c=read(W/'curl-angle-candidate-v1.json');annotation=Path(c['calibration_annotation'])
assert sha(annotation)==c['calibration_annotation_sha256']
threshold=c['angle_threshold_deg'];result=[]
for suffix in ('010900-1088617','113555-1443279','114908-1452582','115812-1458627'):
 source=W/(suffix+'-features.json');d=read(source);tr=d['trace'];hits=[]
 for prev,row in zip(tr,tr[1:]):
  if prev['mouth_chord_from_down_deg']>threshold and row['mouth_chord_from_down_deg']<=threshold:hits.append(row)
 first=hits[0] if hits else None
 keys=[e['next_frame'] for e in d['annotations'] if e.get('kind')=='phase' and e.get('label')=='curl_load']
 result.append(dict(run=d['run'],role='calibration' if d['run']==c['calibration_run'] else 'transfer_review',
  candidate_physical_frame_end=None if first is None else first['frame'],candidate_next_decision_frame=None if first is None else first['frame']+1,
  candidate_time_s=None if first is None else first['time_s'],angle_deg=None if first is None else first['mouth_chord_from_down_deg'],
  already_past_threshold_at_start=tr[0]['mouth_chord_from_down_deg']<=threshold,
  downward_crossings=[h['frame'] for h in hits],original_key2_next_frames=keys,
  at_candidate=None if first is None else {k:first[k] for k in ('soil_force_n','soil_samples','rock_substeps','q_bucket','tip_z_m')},
  human_reference_frame_end=610 if d['run']==c['calibration_run'] else None,
  human_reference_available=d['run']==c['calibration_run'],
  error_vs_human_frames=(first['frame']-610) if first is not None and d['run']==c['calibration_run'] else None,
  boundary_accuracy_verified=False,
  scope='Calibration comparison is not independent validation; other keypresses are comparison only, not truth. Contact gate and other phases not validated.',
  source_inputs=d['inputs'],features_sha256=sha(source)))
assert len(result)==4 and all(e['candidate_next_decision_frame']==e['candidate_physical_frame_end']+1 for e in result if e['candidate_physical_frame_end'] is not None)
report=dict(schema='contact_curl_angle_transfer_review_v1',threshold_deg=threshold,rule_sha256=sha(W/'curl-angle-candidate-v1.json'),
 status='candidate_frames_generated_human_transfer_review_pending',threshold_refitted_on_transfer=False,automatic_phase_router_qualified=False,episodes=result)
(W/'transfer-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
