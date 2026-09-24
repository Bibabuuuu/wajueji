"""Read-only evidence review; full states remain in ignored local runs/."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'runs/fast-contact-collection-20260924-v1'


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


prepare=read(RUN/'prepare-clear/report.json')
qualified=read(RUN/'qualified/report.json')
audit=read(RUN/'restore-audit/audit.json')
output_line=next(line for line in (HERE/'manual-probe-final.log').read_text().splitlines() if line.startswith('Output: '))
manual_dir=Path(output_line.removeprefix('Output: '))
manual=read(manual_dir/'report.json')
source=Path(prepare['source']['path'])
point=RUN/'prepare-clear/prepoint.npz'
meta=read(Path(str(point)+'.json'))
final=manual_dir/'final-000003.cycle.npz'
final_meta=read(Path(str(final)+'.json'))
with np.load(final) as m, np.load(RUN/'restore-audit/final.npz') as a:
    equal={name:bool(np.array_equal(m[name],a[name])) for name in a.files}
checks=dict(source_unchanged=sha(source)==prepare['source']['sha256'],
    unwithdrawn_state_exact=all(qualified['unwithdrawn_state_exact'].values()),
    initial_material_native_readback_exact=all(qualified['initial_state_readback_exact'].values()),
    all_ids_retained=len(meta['training_reset']['entries'])==1,
    prepare_hold_passed=prepare['status']=='hold_passed' and prepare['frames']==30,
    qualified_same_arrays=sha(point)==sha(RUN/'qualified/prepoint.npz'),
    independent_restore_exact=audit['zero_step']['exact'] and max(audit['zero_step']['cycle_errors'].values())==0,
    independent_hold_no_contact=audit['all_hold_frames_zero_bucket_contact'] and audit['frames_completed']==3,
    manual_checks_passed=all(manual['checks'].values()),
    manual_matches_audit=all(equal.values()),
    reset_metadata_inherited=final_meta['training_reset']==meta['training_reset'],
    same_lineage=final_meta['lineage_root']==meta['lineage_root'],
    explicit_noncontinuous=meta['training_reset']['continuous_physics'] is False)
with np.load(point) as a:
    checks['all_ids_retained']=checks['all_ids_retained'] and len(a['particle_q'])==256441 and np.array_equal(a['rigid_ids'],np.load(source)['rigid_ids'])
for folder in (RUN/'prepare-clear',RUN/'restore-audit',manual_dir):
    rows=[json.loads(line) for line in (folder/('physics.ndjson' if folder==manual_dir else 'frames.ndjson')).read_text().splitlines()]
    checks[str(folder.relative_to(ROOT))+'_zero_contact']=all(r['bucket_mpm_contact_samples']==0 and r['rock_bucket_contact_substeps']==0 for r in rows)
assert all(checks.values()),checks
withdrawal=prepare['withdrawal']
report=dict(status='fast_second_prepoint_qualified',checks=checks,tests=46,
    source=prepare['source'],prepoint=dict(path=str(point),sha256=sha(point)),
    lineage_root=meta['lineage_root'],cycle_index=meta['cycle_index'],
    inventory={k:withdrawal[k] for k in ('soil_count','rock_count','rock_indices','translation_m','ownership')},
    inventory_ids_sha256=hashlib.sha256(json.dumps(withdrawal['soil_indices']).encode()).hexdigest(),
    selected_contact=prepare['plan']['selected']['contact']['contact_xy_m'],
    selected_contact_z_m=prepare['plan']['selected']['contact']['contact_surface_z_m'],
    initial_tip_world_m=prepare['plan']['tip_world_m'],
    future_entry_xy_m=prepare['plan']['selected']['future_entry_xy_m'],
    initial_bucket_clearance_m=prepare['plan']['bucket_surface_clearance_m'],
    prepare_wall_seconds=prepare['wall_seconds'],independent_audit=str(RUN/'restore-audit/audit.json'),
    manual_probe=str(manual_dir),manual_checks=manual['checks'],manual_audit_equal_arrays=equal,
    boundaries=['explicit training reset, not continuous physical cycle','not truck delivery or G4/G5',
                'candidate demonstration entry, no training admission'],
    rejected_first_candidate=read(RUN/'prepare-contact-diagnosis/report.json')['unexpected_contact'],
    implementation_sha256={name:sha(ROOT/name) for name in (
        'src/v4/layered_47_training_reset.py','src/v4/prepare_47_fast_collection.py',
        'src/v4/layered_47_cycle_checkpoint.py','src/v4/layered_47_cycle_runtime.py',
        'src/v4/run_layered_47_sparse_manual_contact_episode.py')})
(HERE/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('implementation_sha256','manual_checks','manual_audit_equal_arrays')},indent=2))
