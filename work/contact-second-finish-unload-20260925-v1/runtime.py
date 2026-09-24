"""Continuous second-scoop tail; review follows completion or protection exit."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
SOURCE=ROOT/'runs/contact-next-scoop-20260925-v1/candidate/final.npz'
sys.path.insert(0,str(ROOT/'work/contact-full-response-20260924-v1'))
from candidate_runtime import CandidateRuntime
import layered_47_cycle_checkpoint as checkpoint
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_endpoint(path,*,allow_training_reset=False):
    path=Path(path).resolve();protocol=read(W/'protocol.json')
    if path!=SOURCE or allow_training_reset or sha(path)!=protocol['source_sha256']:raise ValueError('second_tail_source')
    meta=read(str(path)+'.json');report=read(path.parent/'report.json');control=read(path.parent/'control-state.json')
    if (meta['schema']!='layered_47_full_response_real_second_contact_v1' or meta['stage']!='hold'
        or meta['cycle_index']!=2 or meta.get('training_reset') is not None
        or meta['checkpoint_sha256']!=sha(path) or not meta['truck_enabled']
        or meta['controller_state']['native_state_kind']!='full'
        or (meta['coupling_hz'],meta['control_hz'],meta['rigid_hz'])!=(60,60,4800)
        or report['status']!='bounded_probe_completed' or not all(report['zero_step_arrays_exact'].values())
        or control['frame']!=report['frames_completed'] or control['feature_last_frame']!=control['frame']
        or control['feedback']['last_frame']!=control['frame']):raise ValueError('second_tail_checkpoint_contract')
    for field in ('candidate_bindings','contact_bindings','control_bindings','multisource_bindings','preview_binding'):
        for p,h in meta[field].items():
            if sha(p)!=h:raise ValueError('second_source_code_changed:'+p)
    for p,h in meta['runtime_bindings'].items():
        if sha(ROOT/'src/v4'/p)!=h:raise ValueError('second_source_runtime_changed:'+p)
    with np.load(path,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    checkpoint.sparse._validate(arrays,256441,47);checkpoint.validate_controller(meta['controller_state'],47)
    checkpoint.validate_material_history_policy(meta,meta['runtime_bindings']);checkpoint.validate_sweep(meta['moving_bucket_sweep_state'])
    meta['source_identity']=checkpoint.checkpoint_identity(path);meta['source_metadata_identity']=checkpoint.checkpoint_identity(str(path)+'.json')
    return arrays,meta
class FinishRuntime(CandidateRuntime):
    def __init__(self,source,output,**kw):
        original=checkpoint.load
        try:
            checkpoint.load=load_endpoint
            super().__init__(source,output,freeze_passive_jp=True,**kw)
        finally:checkpoint.load=original
    def save(self,path,**kw):
        result=super().save(path,**kw);side=Path(str(path)+'.json');meta=read(side)
        meta.update(schema='layered_47_full_response_second_finish_unload_v1',
            tail_bindings={str(p):sha(p) for p in W.glob('*.py')},tail_protocol_sha256=sha(W/'protocol.json'),
            highlevel_state=getattr(self,'highlevel_state',None),continuation_qualified=False)
        side.write_text(json.dumps(meta,indent=2)+'\n');return result
