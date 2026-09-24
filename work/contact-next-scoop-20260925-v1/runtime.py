"""Bound full-response runtime for the real second prepoint; no training reset."""
import json,sys
from pathlib import Path
import numpy as np
from check_source import ROOT,W,SOURCE,read,sha
sys.path.insert(0,str(ROOT/'work/contact-full-response-20260924-v1'))
from candidate_runtime import CandidateRuntime
import layered_47_cycle_checkpoint as checkpoint

def load_endpoint(path,*,allow_training_reset=False):
    contract=read(W/'protocol.json');review=read(W/'source-review.json');path=Path(path).resolve()
    if path!=SOURCE or allow_training_reset or not review['passed'] or sha(path)!=contract['source_sha256'] or sha(path)!=review['source_sha256']:
        raise ValueError('next_contact_requires_bound_real_prepoint')
    if sha(str(path)+'.json')!=review['source_metadata_sha256']:raise ValueError('prepoint_metadata_changed')
    meta=read(str(path)+'.json')
    if (meta['schema']!='layered_47_full_response_next_prepoint_v1' or meta['stage']!='prepoint'
        or meta['cycle_index']!=2 or meta.get('training_reset') is not None or not meta['truck_enabled']
        or meta['controller_state']['native_state_kind']!='full'
        or meta['checkpoint_sha256']!=review['source_sha256']):raise ValueError('next_contact_source_identity')
    for field in ('candidate_bindings','prepoint_bindings','prepoint_dependency_bindings'):
        for p,h in meta[field].items():
            if sha(p)!=h:raise ValueError('source_code_changed:'+p)
    for p,h in meta['runtime_bindings'].items():
        if sha(ROOT/'src/v4'/p)!=h:raise ValueError('source_runtime_changed:'+p)
    with np.load(path,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    checkpoint.sparse._validate(arrays,256441,47)
    checkpoint.validate_controller(meta['controller_state'],47)
    checkpoint.validate_material_history_policy(meta,meta['runtime_bindings'])
    checkpoint.validate_sweep(meta['moving_bucket_sweep_state'])
    meta['source_identity']=checkpoint.checkpoint_identity(path);meta['source_metadata_identity']=checkpoint.checkpoint_identity(str(path)+'.json')
    return arrays,meta

class ContactRuntime(CandidateRuntime):
    def __init__(self,source,output,**kw):
        original=checkpoint.load
        try:
            checkpoint.load=load_endpoint
            super().__init__(source,output,freeze_passive_jp=True,**kw)
        finally:checkpoint.load=original
    def step(self,command):
        if read(W/'protocol.json')['status']!='authorized_after_preflight':raise ValueError('next_contact_not_preflighted')
        return super().step(command)
    def save(self,path,**kw):
        result=super().save(path,**kw);side=Path(str(path)+'.json');meta=read(side)
        meta.update(schema='layered_47_full_response_real_second_contact_v1',
            contact_bindings={str(p):sha(p) for p in W.glob('*.py')},
            control_bindings={str(p):sha(p) for p in (ROOT/'work/contact-preview-lift-20260924-v1').glob('*.py')},
            multisource_bindings={str(p):sha(p) for p in (ROOT/'work/contact-multisource-lift-20260924-v1').glob('*.py')},
            preview_binding={str(ROOT/'work/contact-fixed-truck-continuation-20260925-v2/yaw_preview.py'):sha(ROOT/'work/contact-fixed-truck-continuation-20260925-v2/yaw_preview.py')},
            continuation_qualified=False)
        side.write_text(json.dumps(meta,indent=2)+'\n');return result
