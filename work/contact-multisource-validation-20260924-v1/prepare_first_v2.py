"""Restore original saved first-scoop handoff; initialize as original manual.
No travel or source/schema rewrite. Thirty zero-command frames qualify a NEW native state.
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
original=json.loads(next((folder/'physics.ndjson').open()))['before_state']
out=ROOT/'runs/contact-multisource-validation-20260924-v1/first-initialization-v2'
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
  for frame in range(30):
   row=r.step([0.,0.,0.,0.]);report['physics_steps']=frame+1
   check=dict(frame=frame+1,hard_guard=row['hard_guard'],soil_samples=row['bucket_mpm_contact_samples'],rock_substeps=row['rock_bucket_contact_substeps'])
   report['hold_checks'].append(check)
   assert check['hard_guard'] and check['soil_samples']==0 and check['rock_substeps']==0
  report['checkpoint']=r.save(out/'prepoint.npz',stage='hold')
  report['status']='new_full_state_saved_from_original_manual_initialization'
except BaseException as error:
 report.update(status='failed',error=repr(error));raise
finally:
 report['source_unchanged']=sha(source)==report['source']['sha256']
 (W/'first-initialization-v2.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
