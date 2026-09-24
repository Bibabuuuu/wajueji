"""Read-only audit of saved bounded rollout; does not recreate physics."""
import json
from pathlib import Path
import numpy as np
from contact_bc_inference import sha
from run_contact_bc_probe import execution_summary
from layered_excavation_approach_planner import _load_kinematics
from planning.cycle_geometry import TIP
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
RUN=ROOT/'runs/contact-bc-closed-loop-20260924-v1/probe120-v2'
SOURCE=ROOT/'runs/fast-contact-collection/20260924-115633-1457317/prepare/prepoint.npz'
POLICY=ROOT/'runs/contact-bc-native60-pilot-20260924-v1/pilot.pt'
MANUAL=ROOT/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260924-115812-1458627'
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def ident(p):return dict(path=str(p),sha256=sha(p))
actions=rows(RUN/'actions.ndjson');physics=rows(RUN/'physics/frames.ndjson');terrain=rows(RUN/'terrain.ndjson')
zero=json.loads((RUN.parent/'zero-step-v2/report.json').read_text())
with np.load(SOURCE) as f:initial={k:f[k] for k in f.files}
with np.load(RUN/'final.npz') as f:final={k:f[k] for k in f.files}
meta=json.loads((RUN/'final.npz.json').read_text());sm=json.loads(Path(str(SOURCE)+'.json').read_text())
checks=dict(frames_complete=[r['frame'] for r in physics]==list(range(1,121)),
 actions_complete=[r['frame'] for r in actions]==list(range(1,121)),
 actions_match=all(a['applied_command_si']==p['applied_command_si'] for a,p in zip(actions,physics,strict=True)),
 state_chain=all(actions[i]['before_state']==physics[i-1]['after_state'] for i in range(1,120)),
 initial_state_matches_zero_step=actions[0]['before_state']==zero['zero_step_prediction']['before_state'],
 first_action_matches_zero_step=actions[0]['action_normalized']==zero['zero_step_prediction']['action_normalized'],
 terrain_causal=all(a['terrain_anchor_frame']<a['frame'] and 0<=a['frame']-1-a['terrain_anchor_frame']<=2 for a in actions),
 terrain_anchors=[r['frame'] for r in terrain]==list(range(0,120,3)),
 previous_action_causal=all(actions[i]['previous_applied_action_si']==actions[i-1]['applied_command_si'] for i in range(1,120)),
 safe=all(p['hard_guard'] and all(p['hard_guard_checks'].values()) for p in physics),
 materials=all(p['hencky_material_guard']['safe'] for p in physics),sweep=all(p['moving_bucket_sweep']['safe'] for p in physics),
 finite_final=all(np.isfinite(v).all() for v in final.values() if v.dtype.kind in 'fciu'),
 particle_count_preserved=initial['particle_q'].shape==final['particle_q'].shape,
 rock_ids_preserved=np.array_equal(initial['rigid_ids'],final['rigid_ids']),
 final_mechanism_matches_log=np.array_equal(final['mechanism_qpos'],physics[-1]['after_state']['mechanism_qpos']),
 simulation_advance_2_seconds=bool(np.max(np.abs(final['simulation_time_s']-initial['simulation_time_s']-2))<1e-12),
 source_unchanged=sha(SOURCE)=='a6fed857a335c95b6fb6222530b33fe73267841d77e527a712731d5edd0d896b',
 policy_unchanged=sha(POLICY)=='4bd50c9f2bd9d5d31b0adee2335f2320b19056b518a097a23a11d2373bcf371c',
 lineage_preserved=meta['lineage_root']==sm['lineage_root'],inventory_preserved=meta['training_reset']==sm['training_reset'])
K=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
assembly,_=_load_kinematics(K,K/'scene/input/model/A.xml',K/'scene/input/mesh')
def tip(q):
 pose=assembly.assemble(np.asarray(q),include_meshes=False)[1]['bucket'];return pose[:3,:3]@TIP+pose[:3,3]
m=rows(MANUAL/'physics.ndjson')[:120]
t0=tip(initial['mechanism_qpos']);t1=tip(final['mechanism_qpos']);tm=tip(m[-1]['after_state']['mechanism_qpos'])
commands=np.array([a['action_normalized'] for a in actions]);before=actions[0]['before_state'];after=physics[-1]['after_state']
loads=[dict(frame=p['frame'],soil_samples=p['bucket_mpm_contact_samples'],rock_substeps=p['rock_bucket_contact_substeps'],hard_guard=p['hard_guard']) for p in physics]
summary=execution_summary(actions,[a['timing']['roundtrip_seconds'] for a in actions],loads)
summary.pop('contact_summary')
result=dict(schema='contact_bc_bounded_probe_readonly_review_v1',run=str(RUN),checks=checks,passed=all(checks.values()),frames=120,
 original_process_status='physical_120_frames_and_final_snapshot_completed_then_report_serialization_failed',
 original_report_failure="TypeError: Object of type int64 is not JSON serializable; command_nonzero_frames was a NumPy scalar",
 repair='Boolean conversion before sum; regression test passes. This independent report is reconstructed from original saved logs, not a rerun.',
 **summary,tip_initial_m=t0.tolist(),tip_final_m=t1.tolist(),tip_displacement_m=(t1-t0).tolist(),
 human_same_source_first120_tip_m=tm.tolist(),human_same_source_first120_tip_displacement_m=(tm-t0).tolist(),
 bucket_origin_displacement_m=(np.array(after['bucket_origin_world_m'])-before['bucket_origin_world_m']).tolist(),
 normalized_command_mean=commands.mean(axis=0).tolist(),normalized_command_first=commands[0].tolist(),normalized_command_last=commands[-1].tolist(),
 normalized_command_min=commands.min(axis=0).tolist(),normalized_command_max=commands.max(axis=0).tolist(),
 total_soil_contact_samples=sum(x['soil_samples'] for x in loads),total_rock_contact_substeps=sum(x['rock_substeps'] for x in loads),
 physics_wall_seconds=sum(p['wall_seconds'] for p in physics),formal_training_admission=False,full_excavation_qualified=False,policy_resume_qualified=False,
 inputs=[ident(SOURCE),ident(POLICY)],outputs=[ident(RUN/x) for x in ('actions.ndjson','terrain.ndjson','physics/frames.ndjson','final.npz','final.npz.json')])
(OUT/'probe120-review.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,indent=2,ensure_ascii=False));assert result['passed']
