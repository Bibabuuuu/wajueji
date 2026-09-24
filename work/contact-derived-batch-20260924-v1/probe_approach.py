"""Zero-step and original first-three-command check of approach initialization."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from approach_runtime import ApproachRuntime
import run_layered_47_sparse_manual_contact_episode as manual
from contact_bc_ground_guard import BucketGroundGuard
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 run='manual-contact-20260924-010900-1088617';p=json.loads((W/(run+'.protocol.json')).read_text())
 folder=ROOT/'runs/layered-47-sparse-manual-contact-v1'/run;teacher=json.loads((folder/'report.json').read_text())
 rows=[json.loads(s) for s in (folder/'physics.ndjson').read_text().splitlines()];out=ROOT/'runs/contact-derived-batch-v1/approach-initialization-probe'
 started=time.monotonic();report=dict(schema='approach_replay_initialization_probe_v1',teacher_run=run,source=p['source'],status='initializing',frames=[],training_eligible=False)
 try:
  with ApproachRuntime(p['source']['path'],out,teacher) as r:
   a=r.capture();report['zero_step_array_equal']={k:bool(np.array_equal(v,a[k])) for k,v in r.arrays.items()}
   # Reported velocities at native restore need not equal the pre-restore source.
   required=set(r.arrays)-{'rigid_reported_linear_velocity','rigid_reported_angular_velocity'}
   assert all(report['zero_step_array_equal'][k] for k in required),'approach_nonvelocity_restore_difference'
   actual=manual.compact_state(r.bridge.state);target=rows[0]['before_state']
   fields=['mechanism_qpos','mechanism_qvel','controller_target','controller_target_velocity','bucket_integral_effort','bucket_active_curl_effort']
   report['initial_teacher_error']={k:float(np.max(np.abs(np.asarray(actual[k])-target[k]))) for k in fields}
   assert max(report['initial_teacher_error'].values())<=1e-12
   guard=BucketGroundGuard(r.assembly,actual['mechanism_qpos']);assert guard.measure(actual['mechanism_qpos'])['safe']
   report['initialization']=r.initialization_audit
   for n,original in enumerate(rows[:3],start=1):
    assert time.monotonic()-started<120
    f=r.step(original['applied_command_si']);state=f['after_state'];ref=original['after_state']
    differences={k:float(np.max(np.abs(np.asarray(state[k])-ref[k]))) for k in fields}
    report['frames'].append(dict(frame=n,hard_guard=f['hard_guard'],ground=guard.measure(state['mechanism_qpos']),state_max_errors=differences))
    assert max(differences.values())<=1e-9 and report['frames'][-1]['ground']['safe']
   report['checkpoint']=r.save(out/'final.npz',stage='hold');report['status']='zero_and_three_frame_initialization_passed'
 except BaseException as error:
  report['status']='failed';report['error']=repr(error);raise
 finally:
  report['source_unchanged']=sha(p['source']['path'])==p['source']['sha256'];report['wall_s']=time.monotonic()-started
  report['script_sha256']=sha(__file__);(W/'approach-initialization-probe.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
