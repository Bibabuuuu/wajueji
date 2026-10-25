"""Restore original saved first-scoop handoff; initialize as original manual.
No travel or source/schema rewrite. Reproduce the original first neutral step; save only a contact-development initial state.
"""
import sys,json
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
sys.path.insert(0,str(ROOT/'work/contact-derived-batch-20260924-v1'))
from approach_runtime import ApproachRuntime
from contact_bc_inference import sha
import run_layered_47_sparse_manual_contact_episode as manual
run='manual-contact-20260924-113555-1443279'
folder=ROOT/'runs/layered-47-sparse-manual-contact-v1'/run
teacher=json.loads((folder/'report.json').read_text());source=Path(teacher['handoff_checkpoint']['path'])
original_row=json.loads(next((folder/'physics.ndjson').open()));original=original_row['before_state']
assert original_row['applied_command_si']==[0.,0.,0.,0.]
out=ROOT/'runs/contact-multisource-validation-20260924-v1/first-initialization-v3'
report=dict(source=dict(path=str(source),sha256=sha(source)),teacher=dict(path=str(folder/'report.json'),sha256=sha(folder/'report.json')),physics_steps=0,status='initializing')
try:
 with ApproachRuntime(source,out,teacher) as r:
  captured=r.capture();report['source_arrays_equal']={k:bool(np.array_equal(v,captured[k])) for k,v in r.arrays.items()}
  required=set(r.arrays)-{'rigid_reported_linear_velocity','rigid_reported_angular_velocity'}
  assert all(report['source_arrays_equal'][k] for k in required)
  actual=manual.compact_state(r.bridge.state)
  fields=['mechanism_qpos','mechanism_qvel','controller_target','controller_target_velocity','bucket_integral_effort','bucket_active_curl_effort']
  report['teacher_errors']={k:float(np.max(abs(np.asarray(actual[k])-original[k]))) for k in fields}
  assert max(report['teacher_errors'].values())<=1e-12
  report['initialization']=r.initialization_audit
  report['hold_checks']=[]
  for frame in range(1):
   row=r.step([0.,0.,0.,0.]);report['physics_steps']=frame+1
   check=dict(frame=frame+1,hard_guard=row['hard_guard'],soil_samples=row['bucket_mpm_contact_samples'],rock_substeps=row['rock_bucket_contact_substeps'])
   report['hold_checks'].append(check)
   assert check['hard_guard'] and check['rock_substeps']==0
   # Require reproduction of an already-recorded neutral initialization step,
   # not a newly invented threshold on the legacy raw grid samples.
   assert check['soil_samples']==original_row['bucket_mpm_contact_samples']
   errors={k:float(np.max(abs(np.asarray(row['after_state'][k])-original_row['after_state'][k]))) for k in fields}
   for i,rock in enumerate(row['after_state']['rocks']):
    for key in ('qpos','qvel'):
     errors[f'rock_{i}_{key}']=float(np.max(abs(np.asarray(rock[key])-original_row['after_state']['rocks'][i][key])))
   report['original_neutral_step_errors']=errors
   assert max(errors.values())<=1e-12
  report['checkpoint']=r.save(out/'prepoint.npz',stage='prepoint')
  report.update(status='legacy_contact_initialization_reproduced_and_saved',strict_zero_sample_qualified=False,full_G2_qualified=False,original_neutral_frame=1)
except BaseException as error:
 report.update(status='failed',error=repr(error));raise
finally:
 report['source_unchanged']=sha(source)==report['source']['sha256']
 (W/'first-initialization-v3.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
