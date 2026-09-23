"""Summarize the bounded real manual-entry restore probe without new physics."""
from pathlib import Path
import json
from legacy_push_shelf_cleanup import sha256
w=Path(__file__).resolve().parent;root=w.parents[1]
base=root/'runs/layered-47-sparse-manual-contact-v1'
run=base/'manual-contact-20260924-005017-1076447'
old=base/'manual-contact-20260924-004002-1069238'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
r=read(run/'report.json');ready=read(run/'viewer-ready.json');physics=rows(run/'physics.ndjson');inputs=rows(run/'input.ndjson');before=rows(old/'physics.ndjson')
pre=root/'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089/approach-handoff.npz'
checks={'completed_30_frame_probe':r['status']=='manual_contact_zero_probe_completed' and len(physics)==30,'runtime_10_checks_pass':len(r['checks'])==10 and all(r['checks'].values()),'all_frames_hard_safe':all(x['hard_guard'] for x in physics),'all_commands_zero':len(inputs)==30 and all(x['applied_command_si']==[0.,0.,0.,0.] for x in inputs),'ready_paused_without_physics':ready['paused'] and ready['physics_frames']==0,'first_three_after_states_exact_to_previous_shelf_model':len(before)==3 and all(a['after_state']==b['after_state'] for a,b in zip(before,physics[:3])),'exact_manual_model_collision_check_pass':read(w/'manual-model-check.json')['passed'],'original_prepoint_sha_unchanged':sha256(pre)=='6344e74f809c8b5323942b211f5d62cccb6356da87bf8c85698910e1f6dfb3f4'}
result={'passed':all(checks.values()),'checks':checks,'runtime_checks':r['checks'],'run':str(run),'previous_three_frame_probe':str(old),'timing':r['timing'],'cleanup':r['legacy_push_shelf_cleanup'],'bindings':{str(run/n):sha256(run/n) for n in ['report.json','viewer-ready.json','physics.ndjson','input.ndjson']},'scope':'30 zero-command frames only; not a full scoop or training qualification'}
(w/'restore-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['passed']
