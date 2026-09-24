"""Exact diagnostic restore; physical continuation requires a source exception."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
PARENT=ROOT/'work/contact-carry-transport-20260924-v2'
spec=importlib.util.spec_from_file_location('_frozen_d157_runtime',PARENT/'runtime.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
sha=parent.sha;read=parent.read;checkpoint=parent.checkpoint
SOURCE=ROOT/'runs/contact-carry-transport-20260924-v2/physical/recovery.npz'
SOURCE_SHA='09e4bcb46872fd10d47d0cbb92d0d2b4cbdb7eb5b83191374266675a1397946a'

def load_endpoint(path,*,allow_training_reset=False):
    path=Path(path).resolve()
    if path!=SOURCE or sha(path)!=SOURCE_SHA or allow_training_reset:
        raise ValueError('unbound_observation_stop_source')
    meta=read(str(path)+'.json');audit=read(PARENT/'audit-review.json');report=read(path.parent/'report.json')
    if (meta['schema']!='layered_47_full_response_carry_transport_candidate_v2'
        or meta['stage']!='failure' or meta['cycle_index']!=1 or meta.get('training_reset') is not None
        or meta['checkpoint_sha256']!=SOURCE_SHA or not audit['audit_passed']
        or audit['endpoint_sha256']!=SOURCE_SHA or audit['frames']!=1054
        or report['stage']!='CLEAR_PILE' or report['stage_step']!=1054
        or not report['failure'].startswith('ValueError("cycle_transport_clearance:')
        or meta['controller_state']['native_state_kind']!='full'
        or (meta['coupling_hz'],meta['control_hz'],meta['rigid_hz'])!=(60,60,4800)):
        raise ValueError('observation_stop_not_independently_audited')
    for field in ('candidate_bindings','transport_bindings'):
        for p,h in meta[field].items():
            if sha(p)!=h:raise ValueError('changed_source_code:'+p)
    for p,h in meta['runtime_bindings'].items():
        if sha(ROOT/'src/v4'/p)!=h:raise ValueError('changed_source_runtime:'+p)
    with np.load(path,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    checkpoint.sparse._validate(arrays,256441,47)
    checkpoint.validate_controller(meta['controller_state'],47)
    checkpoint.validate_material_history_policy(meta,meta['runtime_bindings'])
    checkpoint.validate_sweep(meta['moving_bucket_sweep_state'])
    meta['source_identity']=checkpoint.checkpoint_identity(path)
    meta['source_metadata_identity']=checkpoint.checkpoint_identity(str(path)+'.json')
    return arrays,meta

class TransportRuntime(parent.CandidateRuntime):
    def __init__(self,source,output,*,authorize_steps=False,**kw):
        self.authorize_steps=authorize_steps
        if authorize_steps and read(W/'protocol.json')['status']!='source_exception_authorized':
            raise ValueError('source_exception_not_authorized')
        original=checkpoint.load
        try:
            checkpoint.load=load_endpoint
            super().__init__(source,output,enable_truck=True,**kw)
        finally:checkpoint.load=original

    def step(self,command):
        if not self.authorize_steps:raise ValueError('diagnostic_restore_cannot_step')
        return super().step(command)

    def save(self,path,**kw):
        info=super().save(path,**kw)
        sidecar=Path(str(path)+'.json');meta=read(sidecar)
        control=getattr(self,'transport_controller',None)
        if control is not None:
            meta['transport_controller_state']=dict(previous_command=control.previous.tolist(),
                load_bias=control.tracking.bias.tolist(),first=control.first,
                minimum_cylinder=control.minimum_cylinder,source_pitch=control.pitch,source_cylinder=control.cylinder)
        meta.update(schema='layered_47_full_response_fixed_truck_continuation_v1',
            transport_bindings={str(p):sha(p) for p in PARENT.glob('*.py')},
            continuation_bindings={str(p):sha(p) for p in [*W.glob('*.py'),ROOT/'work/contact-first-transport-review-20260925-v1/static_rocks.py']},
            source_exception_authorized=self.authorize_steps,continuation_qualified=False)
        sidecar.write_text(json.dumps(meta,indent=2)+'\n');return info
