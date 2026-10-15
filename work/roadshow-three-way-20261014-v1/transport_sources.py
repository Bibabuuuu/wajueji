"""Check existing D165 envelopes on CPU; never restore a native runtime."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json, hashlib, sys, time
import numpy as np
ROOT=Path('/home/qunwang/wajueji');sys.path[:0]=[str(ROOT/'src/v4'),str(ROOT)]
import layered_47_cycle_checkpoint as checkpoint
OUT=ROOT/'work/roadshow-three-way-20261014-v1/offline-boundary-v2'
started=time.monotonic();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources={
 'better_manual':ROOT/'runs/manual-strength-recovery-collection-20261013-v1/manual_baseline/episode-0001/endpoint.npz',
 'selected_lower_manual':ROOT/'runs/roadshow-three-way-20261014-v1/manual/endpoint.npz',
 'frozen_residual':ROOT/'runs/manual-strength-overnight-20261014/anchor-sweep-v1/seed44-weight100-independent-three/repetition-1/episode-0001/endpoint.npz',
}
rows={};checks={}
for name,path in sources.items():
    before=sha(path);meta_path=Path(str(path)+'.json');meta_before=sha(meta_path)
    arrays,metadata=checkpoint.load(path,allow_bucket_motion_consistency=True)
    with np.load(path,allow_pickle=False) as raw:
        exact={k:bool(np.array_equal(v,raw[k])) for k,v in arrays.items()}
    state=metadata['controller_state']
    checks[name+'_envelope_exact']=len(exact)==24 and all(exact.values())
    checks[name+'_full_hold']=metadata['stage']=='hold' and state['native_state_kind']=='full' and metadata['cycle_index']==1
    checks[name+'_same_material_and_motion']=metadata['material_history_policy']=='zero_hardening_frozen_jp_v1' and metadata['bucket_motion_policy']=='bucket_origin_predicted_endpoint_v1'
    checks[name+'_unchanged']=before==sha(path) and meta_before==sha(meta_path)
    try:checkpoint.load(path)
    except ValueError as e:legacy_error=str(e)
    else:raise AssertionError('Existing transport entry unexpectedly admits D165 without explicit flag')
    checks[name+'_old_entry_rejects']=legacy_error=='cycle_requires_explicit_bucket_motion_consistency'
    rows[name]={'source':str(path),'sha256':before,'metadata_sha256':meta_before,'array_exact':exact,
        'schema':metadata['schema'],'stage':metadata['stage'],'cycle_index':metadata['cycle_index'],
        'native_time_s':state['native_time_s'],'controller_target':state['controller_target'],
        'controller_target_velocity':state['controller_target_velocity'],
        'old_transport_constructor_error':legacy_error,'native_runtime_restored':False}
path=ROOT/'runs/roadshow-three-way-20261014-v1/geometry-full/failure.npz'
try:checkpoint.load(path,allow_bucket_motion_consistency=True)
except ValueError as e:failure_error=str(e)
else:raise AssertionError('Failure source was admitted')
checks['geometry_failure_source_not_resumed']=failure_error=='cycle_failure_checkpoint_not_resumable'
checks['cpu_only']=os.environ['CUDA_VISIBLE_DEVICES']==''
result={'status':'passed' if all(checks.values()) else 'failed','checks':checks,'sources':rows,
        'geometry_failure_error':failure_error,'new_physical_frames':0,'training_updates':0,
        'native_zero_step_restore_tested':False,'elapsed_seconds':time.monotonic()-started,
        'next_interface':'Explicit D165 flags and source SHA binding; preserve existing runtime and control internal references; reuse fixed-route retention controller after scoped audit',
        'next_measurement':'Stable delivered mixed volume still needs defined/calibrated estimator. Existing receiver counts cannot be called formal delivered volume.'}
(OUT/'transport-source-preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':result['status'],'checks':checks,'elapsed_seconds':result['elapsed_seconds']}))
assert all(checks.values())
