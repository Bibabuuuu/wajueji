"""One original-reset frozen rollout, adding only existing display capture."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def attach_capture(runtime, cache, compact):
    cache.capture(0, compact(runtime.bridge.state), runtime.fixture.state.particle_q.numpy())
    original = runtime.step
    def step(command):
        row = original(command)
        cache.capture(runtime.frames, row['after_state'], runtime.fixture.state.particle_q.numpy())
        return row
    runtime.step = step


def self_check():
    from types import SimpleNamespace
    calls=[]; frames=[]
    class R:
        frames=0
        bridge=SimpleNamespace(state={'value': 0})
        fixture=SimpleNamespace(state=SimpleNamespace(particle_q=SimpleNamespace(numpy=lambda:[[1.,2.,3.]])))
        def step(self, command):
            calls.append(command);self.frames+=1
            return {'after_state': {'value':self.frames}, 'hard_guard': True}
    r=R();cache=SimpleNamespace(capture=lambda f,s,p:frames.append((f,s,p)))
    attach_capture(r,cache,lambda s:s.copy())
    assert r.step([.1])=={'after_state':{'value':1},'hard_guard':True}
    assert r.step([.2])=={'after_state':{'value':2},'hard_guard':True}
    assert calls==[[.1],[.2]] and [x[0] for x in frames]==[0,1,2]
    print('capture_wrapper_self_check_passed')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path('/home/qunwang/wajueji'))
    p.add_argument('--output',type=Path)
    p.add_argument('--self-check',action='store_true')
    a=p.parse_args()
    if a.self_check: self_check(); return
    root=a.repo.resolve();out=a.output.resolve()
    sys.path[:0]=[str(root/'local-data/runtime/sb3-2.9.0-py312'),str(root/'src/v4'),str(root)]
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from manual_residual_sac_env import ManualResidualEnv,Budget,require_environment_contract,environment_contract,sha,write
    from run_manual_residual_sac import episode
    from view_contact_bc_probe import ProbeDisplayCache
    import run_layered_47_sparse_manual_contact_episode as manual
    model=root/'work/manual-strength-overnight-20261013-v1/anchor-grid-v1/seed44-weight100.zip'
    receipt_path=model.parent/'review.json'
    require_environment_contract(json.loads(receipt_path.read_text()),'error_strength_v1')
    assert sha(model)=='84aade9a3d8819368afb72223b8452ebfbc5e9ac9ad190818207d1142692aab1'
    assert not out.exists()
    out.mkdir(parents=True)
    torch.set_num_threads(1);frozen=SAC.load(model,device='cpu');updates=frozen._n_updates
    budget=Budget(out/'budget.json',seconds=600,purpose='frozen_media_only')
    review=dict(status='running',training_updates=0,maximum_wall_seconds=600,
        model=dict(path=str(model),sha256=sha(model),updates=updates),
        environment_contract=environment_contract('error_strength_v1'),
        script_sha256=sha(__file__),old_overnight_budget_reopened=False)
    write(out/'review.json',review)
    env=ManualResidualEnv(out/'rollout',budget,waiting_action_mode='error_strength_v1',collect=True)
    cache=None;original_reset=env.reset
    def reset(**kwargs):
        nonlocal cache
        obs,info=original_reset(**kwargs)
        r=env.runtime
        cache=ProbeDisplayCache(out/'display',1560,r.arrays['particle_q'].shape,
            dict(records=r.records,root=str(r.root),selected_cut=r.metadata['selected_cut'],
                 particle_spacing_m=r.base['mpm']['particle_spacing_m']))
        attach_capture(r,cache,manual.compact_state)
        return obs,info
    env.reset=reset
    try:
        result=episode(env,lambda obs:frozen.predict(obs,deterministic=True)[0])
        review['result']=result
        assert frozen._n_updates==updates and sha(model)==review['model']['sha256']
        assert result['status']=='completed' and result['physical_frames']==1206
        expected=root/'runs/manual-strength-overnight-20261014/anchor-sweep-v1/seed44-weight100-independent-three/repetition-1/episode-0001'
        old=[json.loads(x) for x in (expected/'actions.ndjson').read_text().splitlines()]
        new=[json.loads(x) for x in (env.directory/'actions.ndjson').read_text().splitlines()]
        review['actions_equal_verified_episode']=old==new
        with np.load(expected/'endpoint.npz') as x,np.load(env.directory/'endpoint.npz') as y:
            exact={k:bool(np.array_equal(x[k],y[k])) for k in x.files}
        review['endpoint_arrays_exact']=exact
        assert review['actions_equal_verified_episode'] and all(exact.values())
        review['status']='captured_frozen_verified_repeat'
    except BaseException as e:
        review.update(status='failed',error=type(e).__name__+':'+str(e));raise
    finally:
        if cache is not None:
            captured=cache.finish(review['status'])
            captured['label']='FROZEN ANCHORED SAC / SAME-SOURCE DIG AND SHORT HOLD'
            captured['controller']='anchored_SAC_seed44_weight100'
            captured['physical_frame_hz']=60
            write(out/'display/reconstruction.json',captured)
            review['display']=captured
        env.close()
        review['remaining_seconds']=budget.remaining()
        review['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        write(out/'review.json',review)
        print(json.dumps(dict(status=review['status'],output=str(out),remaining_seconds=review['remaining_seconds'])))


if __name__=='__main__': main()
