"""Audit actual candidate failure and compare commands; never advances physics."""
from pathlib import Path
import json,numpy as np
from contact_bc_inference import sha
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
RUN=ROOT/'runs/contact-bc-repair-probe-20260924-v1/contact600'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def ident(p):return dict(path=str(p),sha256=sha(p))
r=read(RUN/'report.json');a=rows(RUN/'actions.ndjson');p=rows(RUN/'physics/frames.ndjson');g=rows(RUN/'ground-guard.ndjson');n=len(p)
failure=RUN/'physics/failure.npz'
with np.load(failure) as f:final={k:f[k] for k in f.files}
source=Path(r['source']['path']);policy=Path(r['policy']['path'])
checks=dict(frame_count=n==len(a)==r['frames_completed']==330,
 frames=[x['frame'] for x in p]==list(range(1,n+1)),
 prefix=r['prefix_frames_verified']==120 and all(r['prefix_final_arrays_exact'].values()),
 exact_restore=all(r['zero_step_arrays_exact'].values()),
 actions_match=all(x['applied_command_si']==y['applied_command_si'] for x,y in zip(a,p,strict=True)),
 state_chain=all(a[i]['before_state']==p[i-1]['after_state'] for i in range(1,n)),
 legacy_hard_guards=all(x['hard_guard'] and all(x['hard_guard_checks'].values()) for x in p),
 material=all(x['hencky_material_guard']['safe'] for x in p),sweep=all(x['moving_bucket_sweep']['safe'] for x in p),
 ground_stopped_at_first_violation=all(x['safe'] for x in g[:-1]) and not g[-1]['safe'] and g[-1]['frame']==n,
 sampled_no_ground_penetration=all(x['minimum_bucket_z_m']>0 for x in g),
 failure_saved=np.array_equal(final['mechanism_qpos'],p[-1]['after_state']['mechanism_qpos']),
 finite_numeric_snapshot=all(np.isfinite(v).all() for v in final.values() if v.dtype.kind in 'fciu'),
 source=sha(source)==r['source']['sha256'],policy=sha(policy)==r['policy']['sha256'],
 display=r['display_cache']['frames']==n and r['display_cache']['probe_status']=='failed')
old=rows(ROOT/'runs/contact-bc-contact-window-20260924-v1/actions.ndjson')
human=rows(ROOT/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260924-115812-1458627/physics.ndjson')
limits=np.array([.12,.0225,.0225,.0675])
windows=[]
for start,end in ((1,60),(61,120),(121,180),(181,240),(241,300),(311,330)):
 selection=slice(start-1,end)
 windows.append(dict(frames=[start,end],candidate=np.mean([x['action_normalized'] for x in a[selection]],axis=0).tolist(),
 old=np.mean([x['action_normalized'] for x in old[selection]],axis=0).tolist(),
 human=np.mean([x['applied_command_si'] for x in human[selection]],axis=0).__truediv__(limits).tolist()))
comparison=read(OUT/'offline-comparison.json')
def val_mae(row):
 # Original three validation episodes, frame-weighted to match original report.
 counts={e['run']:e['rows'] for e in read(ROOT/'local-data/contact-bc-native-tensors-20260924-v1/manifest.json')['episodes']}
 val=row['episodes'][-3:]
 return float(sum(np.mean(e['mae'])*counts[e['run']] for e in val)/sum(counts[e['run']] for e in val))
result=dict(status='behavior_failed',evidence_checks=checks,evidence_checks_passed=all(checks.values()),frames=n,
 minimum_bucket_z_m=min(x['minimum_bucket_z_m'] for x in g),last_command=a[-1]['action_normalized'],
 command_windows=windows,wall_seconds=r['wall_seconds'],simulation_seconds=n/60,
 old_validation_mae=val_mae(comparison[0]),candidate_validation_mae=val_mae(comparison[-1]),
 old_teacher_forced_release_mean= comparison[0]['episodes'][-1]['boom_release_320_600_mean'],
 candidate_teacher_forced_release_mean=comparison[-1]['episodes'][-1]['boom_release_320_600_mean'],
 failure=r['failure'],scope='Actual policy behavior failed; checks qualify evidence and safety stop only. No autonomous scoop, fill, lift, or G7 qualification.',
 model_selection='Adaptive development comparison of five valid fixed-budget candidates; selected using already-viewed validation errors, not a sealed evaluation. One additional full-command-mask prototype rejected before evaluation for train/inference mismatch.',
 inputs=[ident(source),ident(policy)],outputs=[ident(RUN/x) for x in ('actions.ndjson','terrain.ndjson','physics/frames.ndjson','ground-guard.ndjson','physics/failure.npz','physics/failure.npz.json','report.json','display/reconstruction.json')])
(OUT/'contact-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['evidence_checks_passed']
