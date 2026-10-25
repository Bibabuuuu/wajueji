import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from continuous_scoop import *
import signal

def main():
    self_check();assert sha(MODEL)==MODEL_SHA and sha(SOURCE)==SOURCE_SHA
    require_environment_contract(read(RECEIPT),'error_strength_v1')
    out=W/'cpu-v2';out.mkdir(exist_ok=False);started=time.monotonic()
    def timeout(*_):raise TimeoutError('CPU 600s ceiling')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    report=dict(status='checking',new_physical_frames=0,neural_updates=0)
    try:
        r=CpuRuntime(SOURCE,out/'initial',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True);original={k:v.copy() for k,v in r.arrays.items()}
        segments,checker,plan=initial_plan(r,out/'approach')
        from stable_baselines3 import SAC
        model=SAC.load(MODEL,device='cpu');assert model.observation_space.shape==(221,) and model.action_space.shape==(5,)
        policy=model.predict(np.zeros(221,dtype=np.float32),deterministic=True)[0];assert np.isfinite(policy).all() and np.max(abs(policy))<=1
        endpoint=ROOT/'runs/manual-strength-overnight-20261014/anchor-sweep-v1/seed44-weight100-independent-three/repetition-1/episode-0001/endpoint.npz'
        t=CpuRuntime(endpoint,out/'transport',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
        from transport_endpoint_candidate_v2 import build_plan
        ts,tc,tr=build_plan(t,endpoint,out/'transport-plan')
        unloaded=ROOT/'runs/d165-transport-full-pair-20261014-v1/frozen_residual/final.npz'
        u=CpuRuntime(unloaded,out/'unloaded',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
        es,ec=exit_and_return_plan(u,np.array(plan['safe_return_goal']),out/'exit-return')
        expected=plan['geometry_frames']+1560+sum(s['path']['steps']+30+(180 if s['name']=='DUMP' else 0) for s in ts)+sum(s['path']['steps']+30 for s in es)+300
        report.update(status='cpu_passed',initial_plan=plan,max_nominal_physical_frames=expected,
            original_arrays_unchanged=all(np.array_equal(v,r.arrays[k]) for k,v in original.items()),model_sha256=sha(MODEL),source_sha256=sha(SOURCE),
            frozen_updates=model._n_updates,original_contract=environment_contract('error_strength_v1'),
            scope='Geometry/model/ownership checks only. Live dynamic-stage plans must be checked again from actual stage entry.')
        assert report['original_arrays_unchanged'] and expected<=15000
    except BaseException as e:report.update(status='failed',failure=repr(e));raise
    finally:
        signal.alarm(0);report['wall_seconds']=time.monotonic()-started;write(out/'report.json',report);print(encode(report),flush=True)

if __name__=='__main__':main()
