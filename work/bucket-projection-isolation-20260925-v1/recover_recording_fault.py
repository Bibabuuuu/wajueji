"""Admit only this audited display-format exception, never a physics guard fault."""
from pathlib import Path
import json
import numpy as np
import layered_47_cycle_checkpoint as cp
from layered_mound_checkpoint import checkpoint_identity
root=Path(__file__).resolve().parent
run=root/'manual-prefix-1800'; path=run/'failure.npz'
report=json.loads((run/'report.json').read_text()); failure=json.loads((run/'failure.json').read_text())
assert report['frames']==180 and report['error']=='TypeError("type NoneType doesn\'t define __round__ method")'
assert report['measurements'][-1]['valid_estimate'] is False
assert report['measurements'][-1]['fill_percent'] is None
assert 'round(fill[\'fill_percent\'],2)' in report['traceback']
assert checkpoint_identity(path)==failure['checkpoint']
meta=json.loads(Path(str(path)+'.json').read_text())
assert meta['stage']=='failure' and meta['checkpoint_sha256']==failure['checkpoint']['sha256']
rows=[json.loads(x) for x in (run/'frames.ndjson').read_text().splitlines()]
assert len(rows)==180 and all(x['hard_guard'] and all(x['hard_guard_checks'].values()) for x in rows)
assert all(x['hencky_material_guard']['safe'] and x['moving_bucket_sweep']['safe'] for x in rows)
with np.load(path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
cp.sparse._validate(arrays,256441,47)
for key in ('mechanism_qpos','mechanism_qvel'):
    np.testing.assert_array_equal(arrays[key],rows[-1]['after_state'][key])
for key in ('controller_target','controller_target_velocity','bucket_integral_effort','bucket_active_curl_effort','bucket_linear_velocity_world_m_s','bucket_angular_velocity_world_rad_s'):
    np.testing.assert_array_equal(meta['controller_state'][key],rows[-1]['after_state'][key])
for name,sha in meta['runtime_bindings'].items():
    assert checkpoint_identity(Path(cp.__file__).with_name(Path(name).name))['sha256']==sha
cp.validate_controller(meta['controller_state'],47);cp.validate_sweep(meta['moving_bucket_sweep_state'])
cp.validate_bucket_motion_policy(meta,meta['runtime_bindings'],allowed=True)
parent=dict(meta,source_identity=checkpoint_identity(path))
new=cp.save(root/'recording-fault-recovery-180.npz',arrays,meta['controller_state'],meta['moving_bucket_sweep_state'],parent=parent,stage='hold',cycle_index=1,truck_enabled=True,solver_profile=meta['solver_profile'],runtime_bindings=meta['runtime_bindings'],selected_cut=meta['selected_cut'])
summary={'status':'reporting_fault_only_recovery_audited','frames_audited':180,'all_physical_guards_passed':True,'source_failure_preserved':True,'source':checkpoint_identity(path),'report':checkpoint_identity(run/'report.json'),'recovery':new,'qualification':'diagnostic continuation only; not a manual prepoint or training result'}
(root/'recording-fault-recovery-audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
