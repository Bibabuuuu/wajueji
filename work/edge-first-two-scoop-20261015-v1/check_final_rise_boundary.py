"""CPU cached-state check of the existing holding admission, not a repair run."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from types import SimpleNamespace
import sys,numpy as np
R=Path('/home/qunwang/wajueji');W=R/'work/edge-first-two-scoop-20261015-v1';sys.path.insert(0,str(W));import run_prepoint_continuation as m
D=m.OUT;r=m.CpuRuntime(D/'first-returned.npz',W/'boundary-cpu-runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True);r.bridge=SimpleNamespace(state={'mechanism_qpos':r.arrays['mechanism_qpos'].copy()})
with np.load(D/'audit-derived/scoop-2-terrain.npz') as z:terrain={k:z[k] for k in z.files}
plan=m.read(D/'scoop-2-plan/report.json');n=plan['segments'][0]['frames']
with np.load(D/'scoop-2-plan/approach.npz') as z:goal=z['qpos'][n].copy()
q=np.asarray(m.read(D/'display/states.json')[-1]['mechanism_qpos']);guard=m.RiseHoldGuard(r,terrain,goal,n);early_rejected=False
try:guard.enter_hold(q,completed_nominal_steps=n-1)
except ValueError as e:early_rejected=str(e)=='rise_path_not_completed'
admission=guard.enter_hold(q,completed_nominal_steps=n)
assert early_rejected and guard.mode=='hold'
m.write(D/'phase-boundary-cpu.json',dict(status='cached_holding_admission_passed',early_transition_rejected=early_rejected,completed_nominal_frames=n,admission=admission,
 old_failure_remains_failed=True,new_physics_steps=0,training_updates=0,
 candidate='The final nominal step reaches count N, but the runner checks its post-state with rise protection before switching at the next loop. Cached state meets original holding admission. Last-step protection ownership/order still requires a decision; no live semantics changed.'))
print('CACHED_HOLD_ADMISSION_PASSED; old physical failure preserved',flush=True)
