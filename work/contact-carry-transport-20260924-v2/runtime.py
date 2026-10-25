"""Explicit D155 endpoint import for the authorized first-scoop transport."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
W=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'work/contact-full-response-20260924-v1'))
from candidate_runtime import CandidateRuntime
import layered_47_cycle_checkpoint as checkpoint

SOURCE=ROOT/'runs/contact-carry-handoff-20260924-v1/first-resume/final.npz'
SOURCE_SHA='dc7378f44d2b8204186354d0c58fd72993d3abb8ebfe29e8e84a31b9b868ae06'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load_endpoint(path,*,allow_training_reset=False):
    path=Path(path).resolve()
    if path!=SOURCE or sha(path)!=SOURCE_SHA:raise ValueError('transport_unbound_source')
    meta=read(str(path)+'.json')
    if (meta['schema']!='layered_47_full_response_carry_handoff_candidate_v1'
        or meta['base_schema']!=checkpoint.SCHEMA or meta['stage']!='hold'
        or meta['checkpoint_sha256']!=SOURCE_SHA or meta['controller_state']['native_state_kind']!='full'
        or meta['cycle_index']!=1 or meta.get('training_reset') is not None
        or (meta['coupling_hz'],meta['control_hz'],meta['rigid_hz'])!=(60,60,4800)):
        raise ValueError('transport_source_contract')
    for field in ('candidate_bindings','handoff_bindings','handoff_gate_binding'):
        if not meta[field]:raise ValueError('missing_source_bindings')
        for p,h in meta[field].items():
            if sha(p)!=h:raise ValueError('source_code_changed:'+p)
    for p,h in meta['runtime_bindings'].items():
        if sha(ROOT/'src/v4'/p)!=h:raise ValueError('source_runtime_changed:'+p)
    review=read(ROOT/'work/contact-carry-handoff-20260924-v1/resume-audit.json')
    if not review['audit_passed'] or review['endpoint_sha256']!=SOURCE_SHA:raise ValueError('source_not_audited')
    gate=read(ROOT/'work/contact-carry-handoff-tolerant-20260924-v1/review.json')
    if not gate['final']['ready'] or gate['final']['frame_end']!=2927:raise ValueError('handoff_gate_not_ready')
    for p,h in gate['inputs'].items():
        if sha(p)!=h:raise ValueError('gate_input_changed:'+p)
    with np.load(path,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    checkpoint.sparse._validate(arrays,256441,47)
    checkpoint.validate_controller(meta['controller_state'],47)
    checkpoint.validate_material_history_policy(meta,meta['runtime_bindings'])
    checkpoint.validate_sweep(meta['moving_bucket_sweep_state'])
    meta['source_identity']=checkpoint.checkpoint_identity(path)
    meta['source_metadata_identity']=checkpoint.checkpoint_identity(str(path)+'.json')
    return arrays,meta

class TransportRuntime(CandidateRuntime):
    def __init__(self,source,output,**kw):
        original=checkpoint.load
        try:
            checkpoint.load=load_endpoint
            super().__init__(source,output,enable_truck=True,**kw)
        finally:checkpoint.load=original
    def save(self,path,**kw):
        info=super().save(path,**kw);sidecar=Path(str(path)+'.json');meta=read(sidecar)
        control=getattr(self,'transport_controller',None)
        if control is not None:
            meta['transport_controller_state']=dict(previous_command=control.previous.tolist(),load_bias=control.tracking.bias.tolist(),first=control.first,minimum_cylinder=control.minimum_cylinder,source_pitch=control.pitch,source_cylinder=control.cylinder)
        meta.update(schema='layered_47_full_response_carry_transport_candidate_v2',
            transport_bindings={str(p):sha(p) for p in W.glob('*.py')},continuation_qualified=False)
        sidecar.write_text(json.dumps(meta,indent=2)+'\n');return info
