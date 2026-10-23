"""Replay one recorded handoff through existing controllers; no physical advance."""
from pathlib import Path
import json,sys,time
import numpy as np
R=Path('/home/qunwang/wajueji');W=R/'work/motion-review-20261015-v1';A=R/'runs/edge-first-two-scoop-20261015-continuation'
sys.path.insert(0,str(R/'work/continuous-single-scoop-20261014-v1'))
from continuous_scoop import CpuRuntime,TransportController,FixedRouteClearance,write
from exit_common import controller as inherited_controller
limits=np.array([.12,.0225,.0225,.0225]);last=None;first=None
for line in (A/'geometry-actions.ndjson').open():
 x=json.loads(line)
 if x['phase']=='CLEAR_PILE':last=x
 elif x['phase']=='ALIGN':first=x;break
assert last and first
previous=None
for line in (A/'dispatch.ndjson').open():
 x=json.loads(line)
 if x['global_frame']==first['global_frame']-1:previous=np.array(x['command']);break
assert previous is not None
O=W/'control-handoff-cpu-v2';O.mkdir(exist_ok=False);r=CpuRuntime(A/'clear_pile.npz',O/'runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
q=np.asarray(first['before_state']['mechanism_qpos']);target=np.asarray(first['before_state']['controller_target']);desired=np.asarray(first['desired']);velocity=np.asarray(first['velocity'])
assert np.array_equal(r.arrays['mechanism_qpos'],q)
old=TransportController(r,FixedRouteClearance(r),previous);command0,detail0=old.command('ALIGN',q,target,desired,velocity)
state=dict(previous_command=previous.tolist(),load_bias=last['detail']['load_bias']);continued=inherited_controller(r,state);bias0=continued.tracking.bias.copy();command1,detail1=continued.command('ALIGN',q,target,desired,velocity)
checks=dict(original_snapshot_command_reproduced=np.max(abs(command0-first['command']))<1e-12,
 original_load_bias_reproduced=np.max(abs(np.array(detail0['load_bias'])-first['detail']['load_bias']))<1e-12,
 recorded_bias_inherited=np.array_equal(bias0,state['load_bias']),
 original_loaded_bucket_command_zero=command0[3]==0 and command1[3]==0,
 original_speed_bound=bool(np.all(abs(command1)<=limits+1e-12)),
 original_slew_bound=bool(np.all(abs(command1-previous)<=2/60*limits+1e-12)),
 previous_command_not_replaced_by_zero=np.array_equal(np.asarray(state['previous_command']),previous),
 input_physical_arrays_unchanged=np.array_equal(r.arrays['mechanism_qpos'],q),
 no_physical_entry=r.bridge is None and r.frames==0)
checks={k:bool(v) for k,v in checks.items()}
report=dict(status='cpu_control_handoff_passed' if all(checks.values()) else 'cpu_control_handoff_failed',checks=checks,old_command=command0.tolist(),continued_command=command1.tolist(),before_bias=bias0.tolist(),after_bias=continued.tracking.bias.tolist(),original_snapshot_stage=first['phase'],original_snapshot_frame=first['global_frame'],maximum_normalized_command_change=float(np.max(abs(command1-previous)/limits)),physics_steps=0,native_initializations=0,inference_calls=0,implementation='Reuse exit_common.controller with saved previous_command and load_bias; no new controller class or native-state reset.',limitation='One same-state CPU handoff check; does not reproduce later dynamics or prove damping/retention improvement.')
write(O/'report.json',report);print(json.dumps(report,indent=2));assert all(checks.values()),checks
