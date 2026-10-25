"""Explicit full-response continuation from the audited first return only."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
PARENT=ROOT/'work/contact-carry-transport-20260924-v2'
sys.path.insert(0,str(ROOT/'work/contact-full-response-20260924-v1'))
from candidate_runtime import CandidateRuntime
import layered_47_cycle_checkpoint as checkpoint
SOURCE=ROOT/'runs/contact-fixed-truck-continuation-20260925-v2/physical/final.npz'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load_endpoint(path,*,allow_training_reset=False):
    protocol=read(W/'protocol.json');path=Path(path).resolve()
    if path!=SOURCE or allow_training_reset or sha(path)!=protocol['source_sha256']:
        raise ValueError('next_prepoint_unbound_source')
    if sha(str(path)+'.json')!=protocol['source_metadata_sha256']:
        raise ValueError('next_prepoint_source_metadata_changed')
    meta=read(str(path)+'.json')
    audit=read(ROOT/'work/contact-fixed-truck-continuation-20260925-v2/audit-review.json')
    result=read(path.parent/'report.json')
    if (not audit['audit_passed'] or audit['endpoint_sha256']!=sha(path)
        or result['status']!='transport_unload_return_completed'
        or meta['schema']!='layered_47_full_response_fixed_truck_continuation_v2'
        or meta['stage']!='returned' or meta['cycle_index']!=1
        or meta['checkpoint_sha256']!=protocol['source_sha256']
        or meta.get('training_reset') is not None or not meta['truck_enabled']
        or meta['controller_state']['native_state_kind']!='full'
        or (meta['coupling_hz'],meta['control_hz'],meta['rigid_hz'])!=(60,60,4800)):
        raise ValueError('next_prepoint_requires_audited_return')
    for field in ('candidate_bindings','transport_bindings','continuation_bindings'):
        for p,h in meta[field].items():
            if sha(p)!=h:raise ValueError('source_code_changed:'+p)
    for p,h in meta['runtime_bindings'].items():
        if sha(ROOT/'src/v4'/p)!=h:raise ValueError('source_runtime_changed:'+p)
    with np.load(path,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    checkpoint.sparse._validate(arrays,256441,47)
    checkpoint.validate_controller(meta['controller_state'],47)
    checkpoint.validate_material_history_policy(meta,meta['runtime_bindings'])
    checkpoint.validate_sweep(meta['moving_bucket_sweep_state'])
    meta['source_identity']=checkpoint.checkpoint_identity(path)
    meta['source_metadata_identity']=checkpoint.checkpoint_identity(str(path)+'.json')
    return arrays,meta

class PrepointRuntime(CandidateRuntime):
    def __init__(self,source,output,*,authorize_steps=False,**kw):
        self.authorize_steps=authorize_steps
        if authorize_steps and read(W/'protocol.json')['status']!='authorized_after_preflight':
            raise ValueError('prepoint_not_authorized_after_preflight')
        original=checkpoint.load
        try:
            checkpoint.load=load_endpoint
            super().__init__(source,output,enable_truck=True,**kw)
        finally:checkpoint.load=original
    def step(self,command):
        if not self.authorize_steps:raise ValueError('preparation_cannot_step')
        return super().step(command)
    def save(self,path,**kw):
        result=super().save(path,**kw);side=Path(str(path)+'.json');meta=read(side)
        c=getattr(self,'transport_controller',None)
        if c is not None:
            meta['transport_controller_state']=dict(previous_command=c.previous.tolist(),load_bias=c.tracking.bias.tolist(),first=c.first)
        meta.update(schema='layered_47_full_response_next_prepoint_v1',
            prepoint_bindings={str(p):sha(p) for p in W.glob('*.py')},continuation_qualified=False)
        meta['prepoint_dependency_bindings']={str(p):sha(p) for p in [
            ROOT/'work/contact-fixed-truck-continuation-20260925-v2/controller.py',
            ROOT/'work/contact-fixed-truck-continuation-20260925-v2/yaw_preview.py',
            ROOT/'work/contact-fixed-truck-continuation-20260925-v2/observation.py',
            ROOT/'work/contact-fixed-truck-continuation-20260925-v1/controller.py',
            PARENT/'controller.py',PARENT/'load_tracking.py',
            ROOT/'work/contact-preview-lift-20260924-v1/continuous_preview.py',
            ROOT/'work/contact-first-transport-review-20260925-v1/static_rocks.py',
            ROOT/'src/v4/layered_47_cycle_selection.py']}
        side.write_text(json.dumps(meta,indent=2)+'\n');return result
