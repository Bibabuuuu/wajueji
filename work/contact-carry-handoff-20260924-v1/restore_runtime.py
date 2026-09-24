"""Exact D154 first-endpoint import, retaining its candidate identity."""
import json
import sys
from pathlib import Path
import numpy as np
from manual_key4_review import ROOT, W, sha

sys.path.insert(0,str(ROOT/'work/contact-full-response-20260924-v1'))
from candidate_runtime import CandidateRuntime
import layered_47_cycle_checkpoint as checkpoint

SOURCE=ROOT/'runs/contact-multisource-lift-20260924-v1/first-candidate/final.npz'
SOURCE_SHA='3e95b08e05d6fe4eddac7fdbc7f1d459954e83becf328678997313b18b7f5eb9'

def load_first_endpoint(path, *, allow_training_reset=False):
    path=Path(path).resolve()
    if path != SOURCE or sha(path) != SOURCE_SHA:
        raise ValueError('D155_unbound_source')
    meta=json.loads(Path(str(path)+'.json').read_text())
    if (meta['schema']!='layered_47_full_response_multisource_lift_candidate_v1'
            or meta['base_schema']!=checkpoint.SCHEMA or meta['stage']!='hold'
            or meta['checkpoint_sha256']!=SOURCE_SHA
            or meta['controller_state']['native_state_kind']!='full'
            or (meta['coupling_hz'],meta['control_hz'],meta['rigid_hz'])!=(60,60,4800)
            or meta['cycle_index']!=1 or meta.get('training_reset') is not None
            or type(meta['truck_enabled']) is not bool):
        raise ValueError('D155_source_contract')
    for field in ('candidate_bindings','control_bindings','multisource_bindings'):
        if not meta[field]:raise ValueError('D155_missing_binding:'+field)
        for name,digest in meta[field].items():
            if sha(name)!=digest:raise ValueError('D155_parent_code_changed:'+name)
    for name,digest in meta['runtime_bindings'].items():
        if sha(ROOT/'src/v4'/name)!=digest:raise ValueError('D155_runtime_changed:'+name)
    protocol=json.loads((ROOT/'work/contact-multisource-lift-20260924-v1/first-protocol.json').read_text())
    for item in protocol['inputs']:
        if sha(item['path'])!=item['sha256']:raise ValueError('D155_source_dependency_changed')
    with np.load(path,allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
    checkpoint.sparse._validate(arrays,256441,47)
    checkpoint.validate_material_history_policy(meta,meta['runtime_bindings'])
    checkpoint.validate_controller(meta['controller_state'],47)
    checkpoint.validate_sweep(meta['moving_bucket_sweep_state'])
    meta['source_identity']=checkpoint.checkpoint_identity(path)
    meta['source_metadata_identity']=checkpoint.checkpoint_identity(str(path)+'.json')
    return arrays,meta

class HandoffRuntime(CandidateRuntime):
    def __init__(self,source,output,**kw):
        original=checkpoint.load
        try:
            checkpoint.load=load_first_endpoint
            super().__init__(source,output,**kw)
        finally:checkpoint.load=original

    def save(self,path,**kw):
        info=super().save(path,**kw)
        sidecar=Path(str(path)+'.json');meta=json.loads(sidecar.read_text())
        meta.update(schema='layered_47_full_response_carry_handoff_candidate_v1',
                    handoff_bindings={str(p):sha(p) for p in W.glob('*.py')},
                    handoff_gate_binding={str(ROOT/'src/v4/contact_transport_handoff.py'):sha(ROOT/'src/v4/contact_transport_handoff.py')},
                    continuation_qualified=False)
        sidecar.write_text(json.dumps(meta,indent=2)+'\n')
        return info
