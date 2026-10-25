"""One bounded exact D196 replay with native force observation; no training."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np

ROOT = Path('/home/qunwang/wajueji')
WORK = ROOT/'work/motion-review-20261015-v1/force-probe-20261015-v1'
OUT = ROOT/'runs/loaded-force-probe-20261015-v1'
OLD = ROOT/'runs/motion-physical-short-20261015-v2/loaded'
sys.path[:0] = [str(WORK), str(ROOT/'work/continuous-single-scoop-20261014-v1')]
from observe_bridge import instrument
from continuous_scoop import CycleRuntime, FixedRouteClearance, PhaseView, TransportController, manual
import layered_mound_checkpoint_runtime as bridge_generator


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')


class ObservedRuntime(CycleRuntime):
    def _initialize_fixture(self,*args):
        original=bridge_generator.render_capacity_bridge
        def render(*a):
            source, manifest=original(*a)
            observed,recovered=instrument(source,OUT/'substeps.csv')
            assert recovered==source
            (OUT/'bridge-original.cc').write_text(source)
            (OUT/'bridge-observed.cc').write_text(observed)
            return observed,manifest
        bridge_generator.render_capacity_bridge=render
        try:return super()._initialize_fixture(*args)
        finally:bridge_generator.render_capacity_bridge=original
    def step(self,command):
        if time.time()>=budget['epoch_deadline']-120 or time.monotonic()>=budget['monotonic_deadline']-120:
            raise TimeoutError('save_reserve')
        if self.frames>=120:raise ValueError('frame_ceiling')
        return super().step(command)


budget=json.loads((WORK/'budget.json').read_text())
assert budget['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert not (WORK/'closure.json').exists()
OUT.mkdir(exist_ok=False)
result=dict(status='starting',physics_steps=0,native_initializations=0,training_updates=0,inference_calls=0,checks=[])
write(OUT/'report.json',result)
source=ROOT/'runs/edge-first-two-scoop-20261015-continuation/policy-1/episode-0001/endpoint.npz'
frozen=json.loads((ROOT/'work/motion-physical-short-20261015-v2/preflight.json').read_text())['bindings_sha256']
assert all(sha(p)==h for p,h in frozen.items()),'original_binding_changed'
bindings={p:sha(p) for p in (source,Path(str(source)+'.json'),WORK/'run_probe.py',WORK/'observe_bridge.py',OLD/'dispatch.ndjson',OLD/'display/states.json')}
assert bindings[source]=='46d3f04fa2dfc18082c17f7496e4c146f2491fdb8e08ccd0339cc6169203947a'
states=json.loads((OLD/'display/states.json').read_text());cloud=np.load(OLD/'display/particles.npy',mmap_mode='r')
commands=[json.loads(x)['command'] for x in (OLD/'dispatch.ndjson').read_text().splitlines()][:120]
actions={x['frame']:x for x in map(json.loads,(OLD/'actions.ndjson').read_text().splitlines())}
r=None
try:
    with ObservedRuntime(source,OUT/'physics',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True) as r:
        result['native_initializations']=1
        captured=r.capture()
        assert len(captured)==24 and all(np.array_equal(a,r.arrays[k]) for k,a in captured.items())
        assert max(r.restored['cycle_errors'].values())<=1e-12
        assert manual.compact_state(r.bridge.state)==states[0]
        assert np.array_equal(r.fixture.state.particle_q.numpy(),cloud[0])
        checker=FixedRouteClearance(PhaseView(r))
        controller=TransportController(PhaseView(r),checker,commands[2])
        for i,command in enumerate(commands,1):
            if i>3:controller.check(np.asarray(r.bridge.state['mechanism_qpos']),'CLEAR_PILE',physical=True)
            row=r.step(command);result['physics_steps']=r.frames
            if i>3:controller.check(np.asarray(row['after_state']['mechanism_qpos']),'CLEAR_PILE',physical=True)
            assert row['after_state']==states[i],('state_mismatch',i)
            assert np.array_equal(r.fixture.state.particle_q.numpy(),cloud[i]),('particle_mismatch',i)
            assert row['hard_guard'] and row['hencky_material_guard']['safe'] and row['moving_bucket_sweep']['safe']
            result['checks'].append(dict(frame=i,state_exact=True,particles_exact=True,hard_guards=True))
            if i%30==0:print('EXACT',i,'remaining',round(budget['epoch_deadline']-time.time()),flush=True)
        result['checkpoint']=r.save(OUT/'final.npz',stage='transport')
        result['status']='exact_120_frame_force_probe_completed'
except BaseException as exc:
    result.update(status='failed',exception=repr(exc),traceback=traceback.format_exc())
    raise
finally:
    result['physics_steps']=r.frames if r is not None else 0
    result['input_hashes']={str(p):h for p,h in bindings.items()}
    result['inputs_unchanged']=all(sha(p)==h for p,h in bindings.items())
    result['finished_epoch']=time.time()
    write(OUT/'report.json',result)
    write(WORK/'closure.json',dict(status=result['status'],physics_steps=result['physics_steps'],finished_epoch=time.time(),remaining_seconds=max(0,budget['epoch_deadline']-time.time()),old_budget_reused=False))
    print(result['status'],result['physics_steps'],flush=True)
