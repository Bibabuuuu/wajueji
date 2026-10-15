"""Qualify a completed pre-dig phase snapshot, never a failure snapshot."""
from pathlib import Path
import sys,time,numpy as np
R=Path('/home/qunwang/wajueji');W=R/'work/edge-first-two-scoop-20261015-v1';sys.path.insert(0,str(W));import run_edge_pair_attempt2 as m
b=m.Budget();b.check();prior=m.OUT;source=prior/'scoop1_rear_lower_prepoint.npz';control=prior/'scoop1_rear_lower_prepoint.control.json';meta=m.read(str(source)+'.json');state=m.read(control)
assert meta['stage']=='transport' and meta['cycle_index']==1 and state['phase']=='SCOOP1_REAR_LOWER_PREPOINT' and state['frame']==4790
assert source.name!='recovery.npz' and source.name!='failure.npz' and not np.any(state['previous_command'])
assert m.sha(source)==meta['checkpoint_sha256']==state['checkpoint_sha256']
with np.load(source) as z:source_arrays={k:z[k].copy() for k in z.files}
assert np.array_equal(source_arrays['particle_q'],np.load(prior/'display/particles.npy',mmap_mode='r')[-1])
assert m.read(prior/'display/states.json')[-1]['mechanism_qpos']==source_arrays['mechanism_qpos'].tolist()
out=W/'prepoint-qualification';out.mkdir(exist_ok=False);start=time.monotonic();results=[]
for label in ('replica-a','replica-b'):
 b.check();folder=out/label;folder.mkdir();states=[];clouds=[]
 with m.CycleRuntime(source,folder/'physics',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True) as r:
  actual=r.capture();exact={k:np.array_equal(actual[k],source_arrays[k]) for k in source_arrays};dt=abs(float(actual['simulation_time_s'][0])-float(source_arrays['simulation_time_s'][0]));assert all(v for k,v in exact.items() if k!='simulation_time_s') and dt<=1e-12 and max(r.restored['cycle_errors'].values())<=1e-12
  with np.load(prior/'scoop-1-plan/terrain.npz') as z:terrain={k:z[k] for k in z.files}
  checker=m.MinimumHeightClearance(r,terrain);checker.check(np.asarray(r.bridge.state['mechanism_qpos']),physical=True)
  states.append(m.manual.compact_state(r.bridge.state));clouds.append(r.fixture.state.particle_q.numpy().copy())
  for _ in range(3):
   b.check();row=r.step(np.zeros(4));assert row['hard_guard'] and row['hencky_material_guard']['safe'] and row['moving_bucket_sweep']['safe'];checker.check(np.asarray(r.bridge.state['mechanism_qpos']),physical=True)
   states.append(row['after_state']);clouds.append(r.fixture.state.particle_q.numpy().copy())
  saved=r.save(folder/'prefix3.npz',stage='transport',cycle_index=1);results.append(dict(label=label,zero_exact=exact,zero_clock_error=dt,native_restore_errors=r.restored['cycle_errors'],saved=saved))
 np.save(folder/'particles.npy',np.stack(clouds));m.write(folder/'states.json',states)
with np.load(out/'replica-a/prefix3.npz') as a,np.load(out/'replica-b/prefix3.npz') as c:all24={k:np.array_equal(a[k],c[k]) for k in a.files}
assert all(all24.values()) and m.read(out/'replica-a/states.json')==m.read(out/'replica-b/states.json')
assert np.array_equal(np.load(out/'replica-a/particles.npy'),np.load(out/'replica-b/particles.npy'))
report=dict(status='completed_phase_prepoint_restore_qualified',source=str(source),source_sha256=m.sha(source),source_metadata_sha256=m.sha(str(source)+'.json'),source_control=str(control),source_control_sha256=m.sha(control),source_phase='completed prepoint before first dig; software exception occurred before any later physical step',source_prior_run=str(prior),source_prior_frame=4790,
 qualification='Two independent exact native zero restores and3 zero-command steps; all24 end arrays and all4 cached states/particle clouds match. No old unsaved live suffix is claimed.',replicas=results,replica_end24_exact=all24,physical_frames=6,prior_physical_frames=4654+4790,cumulative_before_production=9450,training_updates=0,remaining_seconds=b.remaining(),wall_seconds=time.monotonic()-start)
m.write(out/'report.json',report);m.write(W/'qualified-prepoint-source.json',report);print('PREPOINT_QUALIFIED 6frames remaining',round(b.remaining()),flush=True)
