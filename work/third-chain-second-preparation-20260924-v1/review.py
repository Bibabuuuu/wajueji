"""Read-only qualification of the second fast-collection prepoint."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'runs/fast-contact-collection/20260924-114750-1451324'
def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

prepare=read(RUN/'prepare/report.json')
audit=read(RUN/'restore-audit/audit.json')
point=RUN/'prepare/prepoint.npz'
metadata=read(Path(str(point)+'.json'))
source=Path(prepare['source']['path'])
source_metadata=read(Path(str(source)+'.json'))
manual_dir=Path(next(line[8:] for line in (HERE/'manual-probe.log').read_text().splitlines() if line.startswith('Output: ')))
manual=read(manual_dir/'report.json')
final=manual_dir/'final-000003.cycle.npz'
final_metadata=read(Path(str(final)+'.json'))
with np.load(final) as left, np.load(RUN/'restore-audit/final.npz') as right:
    equal={key:bool(np.array_equal(left[key],right[key])) for key in right.files}
entries=metadata['training_reset']['entries']
checks=dict(source_unchanged=sha(source)==prepare['source']['sha256'],
    normal_first_endpoint=source_metadata['stage']=='loaded_manual_endpoint' and source_metadata['cycle_index']==1,
    unwithdrawn_arrays_exact=all(prepare['unwithdrawn_state_exact'].values()),
    initialized_arrays_readback_exact=all(prepare['initial_state_readback_exact'].values()),
    prepare_passed=prepare['status']=='hold_passed' and prepare['frames']==30,
    independent_restore_exact=audit['zero_step']['exact'] and max(audit['zero_step']['cycle_errors'].values())==0,
    audit_passed=audit['status']=='hold_passed' and audit['all_hold_frames_zero_bucket_contact'],
    manual_passed=manual['status']=='manual_contact_zero_probe_completed' and all(manual['checks'].values()),
    manual_audit_arrays_equal=all(equal.values()),
    second_scoop=metadata['cycle_index']==2,
    source_lineage_preserved=metadata['lineage_root']==source_metadata['lineage_root']==final_metadata['lineage_root'],
    prior_inventory_preserved=entries[:-1]==source_metadata.get('training_reset',{}).get('entries',[]),
    no_duplicate_withdrawal=all(len([i for e in entries for i in e[k]])==len({i for e in entries for i in e[k]}) for k in ('soil_indices','rock_indices')),
    reset_persists=final_metadata['training_reset']==metadata['training_reset'],
    explicitly_not_continuous=metadata['training_reset']['continuous_physics'] is False)
for directory,name in ((RUN/'prepare','frames.ndjson'),(RUN/'restore-audit','frames.ndjson'),(manual_dir,'physics.ndjson')):
    rows=[json.loads(line) for line in (directory/name).read_text().splitlines()]
    checks[str(directory.relative_to(ROOT))+'_safe_zero_contact']=bool(rows) and all(r['hard_guard'] and r['bucket_mpm_contact_samples']==0 and r['rock_bucket_contact_substeps']==0 for r in rows)
assert all(checks.values()),checks
report=dict(status='fast_second_prepoint_qualified',checks=checks,source=prepare['source'],
    prepoint=dict(path=str(point),sha256=sha(point)),lineage_root=metadata['lineage_root'],
    cycle_index=2,withdrawal_counts=[dict(soil=e['soil_count'],rocks=e['rock_count']) for e in entries],
    prepare_wall_seconds=prepare['wall_seconds'],tip_world_m=prepare['plan']['tip_world_m'],
    contact_xy_m=prepare['plan']['selected']['contact']['contact_xy_m'],
    contact_z_m=prepare['plan']['selected']['contact']['contact_surface_z_m'],
    manual_probe=str(manual_dir),manual_audit_array_equal=equal,
    scope='Fast training reset preserves excavated stock; not truck delivery or continuous three-scoop completion.',
    evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in (
        RUN/'prepare/report.json',RUN/'restore-audit/audit.json',manual_dir/'report.json',Path(__file__))})
(HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('qualified:',sha(point),'checks:',len(checks))
