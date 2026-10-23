"""Audit the new first-to-second continuation after all stages complete."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/home/qunwang/wajueji');OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=read(OUT/'manifest.json');run=Path(m['run_directory']);source=Path(m['source'])
transport=read(run/'transport/report.json');prepoint=read(run/'prepoint/report.json')
qualified=read(run/'qualified-prepoint/report.json');audit=read(run/'restore-audit/audit.json')
retreat=read(run/'contact-exit-physical/report.json')
point=run/'qualified-prepoint/prepoint.npz';metadata=read(Path(str(point)+'.json'))
initial=np.load(source);final=np.load(point)
stages={}
for name in ('contact-exit-physical','transport','prepoint','qualified-prepoint','restore-audit'):
    rows=[json.loads(s) for s in (run/name/'frames.ndjson').read_text().splitlines()]
    stages[name]=dict(frames=len(rows),all_hard_guards=all(r['hard_guard'] for r in rows),
        all_material_guards=all(r['hencky_material_guard']['safe'] for r in rows),
        all_sweep_guards=all(r['moving_bucket_sweep']['safe'] for r in rows),
        contact_frames=sum(r['bucket_mpm_contact_samples']>0 or r['rock_bucket_contact_substeps']>0 for r in rows))
checks=dict(original_first_unchanged=sha(source)==m['source_sha256'],
    code_unchanged=all(sha(ROOT/p)==h for p,h in m['source_bindings'].items()),
    bounded_exit_passed=retreat['status']=='retreat_passed',
    transport_passed=transport['status']=='transport_unload_return_passed',
    prepoint_saved=prepoint['status']=='prepoint_saved_requires_independent_audit',
    qualified_trailing_zero=qualified['trailing_zero_contact_frames']>=30,
    independent_restore_passed=audit['status']=='hold_passed' and audit['all_hold_frames_zero_bucket_contact'],
    independent_exact=audit['zero_step']['exact'],
    restored_point_sha=audit['source']['sha256']==sha(point),
    restored_metadata_sha=audit['source_metadata']['sha256']==sha(Path(str(point)+'.json')),
    second_scoop=metadata['cycle_index']==2,
    same_new_first_lineage=metadata['lineage_root']==m['source_sha256'],
    passive_history_policy=metadata['material_history_policy']==m['policy'],
    truck_preserved=metadata['truck_enabled'],
    soil_count_preserved=len(initial['particle_q'])==len(final['particle_q'])==256441,
    rock_identity_preserved=np.array_equal(initial['rigid_ids'],final['rigid_ids']) and len(final['rigid_qpos'])==47,
    finite_final=all(np.isfinite(final[k]).all() for k in final.files if np.issubdtype(final[k].dtype,np.number)),
    passive_Jp_preserved=np.array_equal(initial['particle_Jp'],final['particle_Jp']),
    all_runtime_guards=all(s[k] for s in stages.values() for k in ('all_hard_guards','all_material_guards','all_sweep_guards')))
result=dict(schema='new_chain_second_preparation_review_v1',passed=all(checks.values()),checks=checks,
    source=str(source),source_sha256=sha(source),prepoint=str(point),prepoint_sha256=sha(point),
    metadata_sha256=sha(Path(str(point)+'.json')),lineage_root=metadata['lineage_root'],cycle_index=metadata['cycle_index'],
    material_history_policy=metadata['material_history_policy'],stages=stages,
    selected=prepoint['selected'],unload_point_inventory=transport['unload_inventory'],
    tip_world_m=qualified['tip_world_m'] if 'tip_world_m' in qualified else None,
    scope='Local guarded second-scoop manual prepoint after the actual new first scoop; not autonomous G5, stable delivered mixed volume or training admission.')
(OUT/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps(result,ensure_ascii=False));assert result['passed']
