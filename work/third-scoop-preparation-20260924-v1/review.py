"""Read-only audit of the actual second-to-third scoop continuation."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('/home/qunwang/wajueji')
OUT = Path(__file__).resolve().parent
manifest = json.loads((OUT/'manifest.json').read_text())
RUN = Path(manifest['run_directory'])
SOURCE = ROOT/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260924-083547-1327233/final-001723.cycle.npz'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

source_audit = read(RUN/'source-audit/audit.json')
transport = read(RUN/'transport/report.json')
prepoint = read(RUN/'prepoint/report.json')
qualified = read(RUN/'qualified-prepoint/report.json')
restore = read(RUN/'restore-audit/audit.json')
point = RUN/'qualified-prepoint/prepoint.npz'
metadata = read(Path(str(point)+'.json'))
source_metadata = read(Path(str(SOURCE)+'.json'))
source = np.load(SOURCE)
final = np.load(point)
stages = {}
for name in ('source-audit', 'transport', 'prepoint', 'qualified-prepoint', 'restore-audit'):
    rows = [json.loads(s) for s in (RUN/name/'frames.ndjson').read_text().splitlines()]
    stages[name] = dict(frames=len(rows), all_hard_guards=all(r['hard_guard'] for r in rows),
        all_material_guards=all(r['hencky_material_guard']['safe'] for r in rows),
        all_sweep_guards=all(r['moving_bucket_sweep']['safe'] for r in rows),
        bucket_contact_frames=sum(r['bucket_mpm_contact_samples']>0 or r['rock_bucket_contact_substeps']>0 for r in rows))
checks = dict(
    source_unchanged=sha(SOURCE)==manifest['source_sha256'],
    runtime_unchanged=all(sha(ROOT/p)==digest for p,digest in manifest['source_bindings'].items()),
    source_hold_passed=source_audit['status']=='hold_passed',
    transport_passed=transport['status']=='transport_unload_return_passed',
    prepoint_saved=prepoint['status']=='prepoint_saved_requires_independent_audit',
    qualified_zero_contact=qualified['trailing_zero_contact_frames']>=30,
    independent_restore_passed=restore['status']=='hold_passed' and restore['all_hold_frames_zero_bucket_contact'],
    restore_binds_point=restore['source']['sha256']==sha(point),
    restore_binds_metadata=restore['source_metadata']['sha256']==sha(Path(str(point)+'.json')),
    third_scoop=metadata['cycle_index']==3 and source_metadata['cycle_index']==2,
    same_lineage=metadata['lineage_root']==source_metadata['lineage_root'],
    truck_preserved=metadata['truck_enabled'],
    soil_count_preserved=len(source['particle_q'])==len(final['particle_q'])==256441,
    rocks_preserved=len(source['rigid_qpos'])==len(final['rigid_qpos'])==47,
    time_advanced=float(final['simulation_time_s'][0])>float(source['simulation_time_s'][0]),
    finite_final=all(np.isfinite(final[k]).all() for k in final.files if np.issubdtype(final[k].dtype,np.number)),
    all_runtime_guards=all(s[k] for s in stages.values() for k in ('all_hard_guards','all_material_guards','all_sweep_guards')))
report = dict(schema='third_scoop_preparation_review_v1', checks=checks, passed=all(checks.values()),
    source=str(SOURCE), source_sha256=sha(SOURCE), prepoint=str(point), prepoint_sha256=sha(point),
    metadata_sha256=sha(Path(str(point)+'.json')), cycle_index=metadata['cycle_index'],
    lineage_root=metadata['lineage_root'], stages=stages,
    source_simulation_time_s=float(source['simulation_time_s'][0]),
    prepoint_simulation_time_s=float(final['simulation_time_s'][0]),
    selected=prepoint['selected'], prepoint_tip_world_m=qualified['prepoint_tip_world_m'],
    fresh_clearance_m=qualified['fresh_clearance_m'], refinement_attempts=qualified['attempts'],
    unload_inventory=transport['unload_inventory'],
    scope='Actual third manual prepoint only; point inventory is not stable mixed volume or formal G4/G5 qualification.')
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
assert all(checks.values()), checks
