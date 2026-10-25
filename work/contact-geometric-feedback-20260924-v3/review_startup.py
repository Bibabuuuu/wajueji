"""Read-only candidate startup qualification; no physics or model training."""
from pathlib import Path
import json,numpy as np
from contact_bc_inference import sha
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
RUN=ROOT/'runs/contact-geometric-probe-20260924-v3/startup120'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def ident(p):return dict(path=str(p),sha256=sha(p))
r=read(RUN/'report.json');zero=read(RUN.parent/'zero/report.json')
source=Path(r['source']['path']);policy=Path(r['policy']['path'])
a=rows(RUN/'actions.ndjson');p=rows(RUN/'physics/frames.ndjson');t=rows(RUN/'terrain.ndjson');g=rows(RUN/'ground-guard.ndjson')
with np.load(source) as f:initial={k:f[k] for k in f.files}
with np.load(RUN/'final.npz') as f:final={k:f[k] for k in f.files}
meta=read(RUN/'final.npz.json');sm=read(Path(str(source)+'.json'))
checks=dict(completed=r['status']=='bounded_probe_completed' and r['frames_completed']==120,
 zero_step=zero['status']=='zero_step_passed' and all(zero['zero_step_arrays_exact'].values()),
 exact_restore=all(r['zero_step_arrays_exact'].values()),
 frames=[x['frame'] for x in a]==[x['frame'] for x in p]==list(range(1,121)),
 actions_match=all(x['applied_command_si']==y['applied_command_si'] for x,y in zip(a,p,strict=True)),
 state_chain=all(a[i]['before_state']==p[i-1]['after_state'] for i in range(1,120)),
 startup_identity=a[0]['before_state']==zero['zero_step_prediction']['before_state'] and a[0]['action_normalized']==zero['zero_step_prediction']['action_normalized'],
 causal_terrain=all(x['terrain_anchor_frame']<x['frame'] and x['frame']-1-x['terrain_anchor_frame']<=2 for x in a),
 terrain_anchors=[x['frame'] for x in t]==list(range(0,120,3)),
 previous_action=all(a[i]['previous_applied_action_si']==a[i-1]['applied_command_si'] for i in range(1,120)),
 hard_guards=all(x['hard_guard'] and all(x['hard_guard_checks'].values()) for x in p),
 material=all(x['hencky_material_guard']['safe'] for x in p),
 sweep=all(x['moving_bucket_sweep']['safe'] for x in p),
 ground=len(g)==241 and all(x['safe'] for x in g),
 finite=all(np.isfinite(v).all() for v in final.values() if v.dtype.kind in "fciu"),
 particles=initial['particle_q'].shape==final['particle_q'].shape,
 rocks=np.array_equal(initial['rigid_ids'],final['rigid_ids']),
 final_state=np.array_equal(final['mechanism_qpos'],p[-1]['after_state']['mechanism_qpos']),
 elapsed=bool(np.max(np.abs(final['simulation_time_s']-initial['simulation_time_s']-2))<1e-12),
 source=sha(source)==r['source']['sha256']==zero['source']['sha256'],
 policy=sha(policy)==r['policy']['sha256']==zero['policy']['sha256'],
 lineage=meta['lineage_root']==sm['lineage_root'],inventory=meta['training_reset']==sm['training_reset'])
phases=rows(RUN/'phase-trace.ndjson')
checks.update(phase_frames=[x['frame'] for x in phases]==list(range(1,121)),
 phase_before_matches_action=all(x['phase']==y['phase']['phase_before_observation'] for x,y in zip(a,phases,strict=True)),
 phase_next_decision=all(a[i]['phase']==phases[i-1]['phase']['phase_for_next_decision'] for i in range(1,120)),
 evidence_previous=all(x['latest_allowed_phase_evidence_frame']==x['frame']-1 for x in a),
 no_unresolved=not any(x['phase']['curl_boundary_unresolved'] for x in phases),
 phase_protocol=sha(r['phase_contract']['path'])==r['phase_contract']['sha256'])
checks.update(feedback_present=all(x['feedback']['mode'] in ('geometric_approach','geometric_penetrate','bc_with_lateral_depth_feedback') for x in a),
 approach_pitch_tracking=all(abs(x['feedback']['coordinates'][3]-x['feedback']['target'][3])<.03 for x in a if x['phase']=='approach_contact'),
 maximum_lateral_error=all(abs(x['feedback']['coordinates'][1])<=.10 for x in a),
 preview_ground=all(x['feedback']['preview_minimum_clearance_m']>=.03-1e-6 for x in a))
for name,digest in r['code_sha256'].items():checks['code_'+name]=sha(ROOT/'src/v4'/name)==digest
for name,digest in r['runtime_bindings'].items():checks['runtime_'+name]=sha(ROOT/'src/v4'/name)==digest
result=dict(run=str(RUN),frames=120,passed=all(checks.values()),checks=checks,
 inputs=[ident(source),ident(policy)],outputs=[ident(RUN/x) for x in ('actions.ndjson','phase-trace.ndjson','terrain.ndjson','physics/frames.ndjson','ground-guard.ndjson','final.npz','final.npz.json','report.json')],
 qualification='geometric feedback startup only, not excavation success',source_unchanged=r['source_unchanged'])
(OUT/'startup-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['passed']
