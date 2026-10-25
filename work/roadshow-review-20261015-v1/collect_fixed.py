"""Fresh, bounded physical capture of the verified loaded-lift controller."""
from pathlib import Path
import hashlib, json, shutil, sys, time, traceback
import numpy as np
R=Path('/home/qunwang/wajueji')
W=R/'work/roadshow-review-20261015-v1'; O=R/'runs/roadshow-review-20261015-v1'
V=R/'runs/loaded-closedloop-20261015-v5'; B=R/'runs/motion-physical-short-20261015-v2/loaded'
sys.path[:0]=[str(R/'src/v4'),str(R/'work/continuous-single-scoop-20261014-v1')]
from continuous_scoop import FixedRouteClearance,PhaseView,TransportController,manual
from layered_47_loaded_transport_support import LoadedTransportRuntime
from planning.cycle_geometry import tool_pose,MOUTH
def read(p): return json.loads(Path(p).read_text())
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not O.exists(); O.mkdir(); (O/'display').mkdir()
start=time.monotonic(); deadline=start+600; states=[]; sampled=[]; errors=[]; r=None; cloud=None
result=dict(status='starting',physics_steps=0,training_updates=0,inference_calls=0,new_budget_seconds=600,save_reserve_seconds=120,old_budget_reused=False,frames_expected=767,particle_stride=2)
source=R/'runs/edge-first-two-scoop-20261015-continuation/policy-1/episode-0001/endpoint.npz'
p=read(R/'work/motion-review-20261015-v1/loaded-closedloop-20261015-v5/protocol.json')
assert sha(source)=='46d3f04fa2dfc18082c17f7496e4c146f2491fdb8e08ccd0339cc6169203947a'
assert shutil.disk_usage(R).free>8*1024**3
for path,h in p['bindings'].items():assert sha(path)==h,path
write(W/'protocol.json',dict(source=str(source),source_sha256=sha(source),support_start_s=p['support_start_s'],support_end_s=p['support_end_s'],scope='same-source first 764 lift frames; recompute original closed-loop controller; compare exact v5 state at every frame',capture_hz=30,max_frames=767,budget_seconds=600,reserve_seconds=120))
write(W/'budget.json',dict(start_epoch=time.time(),start_monotonic=start,deadline_monotonic=deadline,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
oldstates=read(B/'display/states.json'); expected=read(V/'states.json')
oldcloud=np.load(B/'display/particles.npy',mmap_mode='r')
prefix=[json.loads(x)['command'] for x in (B/'dispatch.ndjson').read_text().splitlines()[:3]]
refs=np.load(R/'work/motion-review-20261015-v1/loaded-closedloop-20261015-v5/reference.npz')
try:
 with LoadedTransportRuntime(source,O/'physics',support_start_s=p['support_start_s'],support_end_s=p['support_end_s'],enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True) as r:
  assert len(r.capture())==24 and all(np.array_equal(v,r.arrays[k]) for k,v in r.capture().items())
  assert max(r.restored['cycle_errors'].values())<=1e-12
  assert manual.compact_state(r.bridge.state)==oldstates[0] and np.array_equal(r.fixture.state.particle_q.numpy(),oldcloud[0])
  cloud=np.lib.format.open_memmap(O/'display/particles.npy',mode='w+',dtype=np.float32,shape=(384,*oldcloud.shape[1:]))
  def sample():
   state=manual.compact_state(r.bridge.state);assert state==expected[r.frames],f'v5 state drift at {r.frames}'
   states.append(state)
   if r.frames%2==0:
    cloud[len(sampled)]=r.fixture.state.particle_q.numpy();sampled.append(r.frames)
  sample()
  def step(command):
   if time.monotonic()>deadline-120:raise TimeoutError('save_reserve')
   row=r.step(command)
   assert row['hard_guard'] and row['hencky_material_guard']['safe'] and row['moving_bucket_sweep']['safe']
   if r.frames<=3:
    assert row['after_state']==oldstates[r.frames] and np.array_equal(r.fixture.state.particle_q.numpy(),oldcloud[r.frames])
   sample()
   if r.frames%120==0:print('CAPTURE',r.frames,'elapsed',round(time.monotonic()-start),flush=True)
   return row
  for cmd in prefix:step(cmd)
  controller=TransportController(PhaseView(r),FixedRouteClearance(PhaseView(r)),prefix[-1])
  q,v=refs['CLEAR_PILE_qpos'],refs['CLEAR_PILE_velocity']
  for desired,velocity in list(zip(q[1:],v,strict=True))+[(q[-1],np.zeros(4))]*30:
   before=np.asarray(r.bridge.state['mechanism_qpos']);controller.check(before,'CLEAR_PILE',physical=True)
   command,detail=controller.command('CLEAR_PILE',before,r.bridge.state['controller_target'],desired,velocity)
   assert command[3]==0
   e=tool_pose(r.assembly,before,MOUTH)[1]-tool_pose(r.assembly,desired,MOUTH)[1]
   errors.append(dict(frame=r.frames+1,phase='CLEAR_PILE',simulation_time_s=(r.frames-3)/60,pitch_error_deg=float(np.rad2deg(np.arctan2(np.sin(e),np.cos(e))))))
   row=step(command);actual=np.asarray(row['after_state']['mechanism_qpos']);controller.check(actual,'CLEAR_PILE',physical=True)
   assert np.max(abs(actual[r.chain.axes]-desired[r.chain.axes]))<=.025
  assert r.frames==767 and len(sampled)==384
  assert np.max(abs(actual[r.chain.axes]-q[-1][r.chain.axes]))<=.015
  e=np.array([x['pitch_error_deg'] for x in errors]); prior=read(V/'pitch-errors.json')[:764]
  assert errors and all(a['frame']==b['frame'] and a['pitch_error_deg']==b['pitch_error_deg'] for a,b in zip(errors,prior,strict=True))
  result.update(status='completed',all_768_states_exact_v5=True,all_764_errors_exact_v5=True,all_original_guards_pass=True,peak_error_deg=float(abs(e).max()),rms_error_deg=float(np.sqrt(np.mean(e*e))))
except BaseException as exc:
 result.update(status='failed',exception=repr(exc),traceback=traceback.format_exc());raise
finally:
 result.update(physics_steps=r.frames if r else 0,wall_seconds=time.monotonic()-start,sampled_frames=sampled)
 if cloud is not None:cloud.flush()
 write(O/'display/states.json',states);write(O/'pitch-errors.json',errors);write(O/'report.json',result)
 shutil.copy2(B/'display/scene.json',O/'display/scene.json')
 write(W/'closure.json',dict(status=result['status'],physics_steps=result['physics_steps'],finished_epoch=time.time(),old_budget_reused=False))
 print(result['status'],result['physics_steps'],flush=True)
