"""Separate native instances verify active/inactive policy continuation."""
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT=Path('/home/qunwang/wajueji')
WORK=ROOT/'work/motion-review-20261015-v1/loaded-restore-20261015-v1'
SOURCE=ROOT/'runs/loaded-closedloop-20261015-v5'
OUT=ROOT/'runs/loaded-restore-20261015-v1'
sys.path[:0]=[str(ROOT/'work/continuous-single-scoop-20261014-v1')]
from continuous_scoop import CycleRuntime,manual
from layered_47_loaded_transport_support import LoadedTransportRuntime


def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
budget=json.loads((WORK/'budget.json').read_text());assert budget['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and not (WORK/'closure.json').exists()
report=json.loads((SOURCE/'report.json').read_text());assert report['status']=='full_closedloop_lift_swing_completed'
cache=json.loads((SOURCE/'restore-cache.json').read_text());OUT.mkdir(exist_ok=False)
result=dict(status='starting',native_initializations=0,physics_steps=0,inference_calls=0,training_updates=0,cases=[],default_runtime_rejects_active=False)
try:
    try:
        CycleRuntime(SOURCE/'active120.npz',OUT/'wrong-runtime-cpu',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
    except ValueError as exc:
        assert str(exc)=='active_loaded_support_requires_LoadedTransportRuntime';result['default_runtime_rejects_active']=True
    else:raise AssertionError('active policy silently lost')
    for label,runtime,path in [('active',LoadedTransportRuntime,SOURCE/'active120.npz'),('inactive',CycleRuntime,SOURCE/'inactive.npz')]:
        if min(budget['epoch_deadline']-time.time(),budget['monotonic_deadline']-time.monotonic())<=120:raise TimeoutError('save_reserve')
        case=dict(name=label,checks=[]);result['cases'].append(case)
        with runtime(path,OUT/label,enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True) as r:
            result['native_initializations']+=1
            assert all(np.array_equal(v,r.arrays[k]) for k,v in r.capture().items()) and len(r.capture())==24
            assert max(r.restored['cycle_errors'].values())<=1e-12
            assert manual.compact_state(r.bridge.state)==cache[label][0]['before_state']
            for i,item in enumerate(cache[label]):
                assert result['physics_steps']<6
                row=r.step(item['command']);result['physics_steps']+=1
                check=dict(frame=i+1,state_exact=row['after_state']==item['after_state'],particles_exact=np.array_equal(r.fixture.state.particle_q.numpy(),np.load(SOURCE/(label+'-restore-'+str(i)+'.npy'))),guards=bool(row['hard_guard'] and row['hencky_material_guard']['safe'] and row['moving_bucket_sweep']['safe']))
                case['checks'].append(check);assert check['state_exact'] and check['particles_exact'] and check['guards']
            case['status']='exact_three_frame_restore_passed'
    assert len(cache['active'])==len(cache['inactive'])==3 and result['native_initializations']==2 and result['physics_steps']==6
    result['status']='active_and_inactive_independent_restore_passed'
except BaseException as exc:
    result.update(status='failed',exception=str(exc)[:1200],traceback=traceback.format_exc()[-4000:]);raise
finally:
    write(OUT/'report.json',result);write(WORK/'closure.json',dict(status=result['status'],physics_steps=result['physics_steps'],finished_epoch=time.time(),remaining_seconds=max(0,budget['epoch_deadline']-time.time()),old_budget_reused=False));print(json.dumps(result),flush=True)
