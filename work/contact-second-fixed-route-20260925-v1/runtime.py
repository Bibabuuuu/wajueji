"""Restore only the normal, source-bound second curl handoff; no dig replay."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
RUN=ROOT/'runs/contact-second-fixed-route-20260925-v1'
SOURCE=ROOT/'runs/contact-second-finish-unload-20260925-v4/physical/handoff.npz'
SOURCE_SHA='f750e55864d6189ff6cd149cc1574894c236a334e6817a4b88f66493fd33c84d'
SOURCE_FRAME=3164
MAX_FRAMES=5561;MAX_SECONDS=1830
sys.path.insert(0,str(ROOT/'work/contact-full-response-20260924-v1'))
from candidate_runtime import CandidateRuntime
import layered_47_cycle_checkpoint as checkpoint
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_endpoint(path,*,allow_training_reset=False):
    path=Path(path).resolve()
    if path!=SOURCE or sha(path)!=SOURCE_SHA or allow_training_reset:raise ValueError('fixed_route_source')
    meta=read(str(path)+'.json');report=read(path.parent/'report.json');state=meta['highlevel_state']
    if (meta['schema']!='layered_47_full_response_second_finish_unload_v1' or meta['stage']!='hold'
        or meta['cycle_index']!=2 or meta.get('training_reset') is not None or not meta['truck_enabled']
        or meta['checkpoint_sha256']!=SOURCE_SHA or meta['controller_state']['native_state_kind']!='full'
        or (meta['coupling_hz'],meta['control_hz'],meta['rigid_hz'])!=(60,60,4800)
        or state['frame']!=SOURCE_FRAME or state['carry']['stage']!='CLEAR_PILE'
        or state['carry']['handoff_frame']!=SOURCE_FRAME or state['gate']['last_frame']!=SOURCE_FRAME
        or not report['handoff']['ready'] or report['handoff_checkpoint']['sha256']!=SOURCE_SHA):
        raise ValueError('normal_handoff_contract')
    for field in ('candidate_bindings','tail_bindings'):
        for p,h in meta[field].items():
            if sha(p)!=h:raise ValueError('changed_handoff_code:'+p)
    prior_protocol=ROOT/'work/contact-second-finish-unload-20260925-v4/protocol.json'
    if sha(prior_protocol)!=meta['tail_protocol_sha256']:raise ValueError('changed_prior_protocol')
    for p,h in read(prior_protocol)['inputs'].items():
        if sha(p)!=h:raise ValueError('changed_prior_input:'+p)
    for p,h in meta['runtime_bindings'].items():
        if sha(ROOT/'src/v4'/p)!=h:raise ValueError('changed_handoff_runtime:'+p)
    with np.load(path,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    checkpoint.sparse._validate(arrays,256441,47);checkpoint.validate_controller(meta['controller_state'],47)
    checkpoint.validate_material_history_policy(meta,meta['runtime_bindings']);checkpoint.validate_sweep(meta['moving_bucket_sweep_state'])
    meta['source_identity']=checkpoint.checkpoint_identity(path);meta['source_metadata_identity']=checkpoint.checkpoint_identity(str(path)+'.json')
    return arrays,meta

class TransportRuntime(CandidateRuntime):
    def __init__(self,source,output,*,authorize_steps=False,**kw):
        self.authorize_steps=authorize_steps
        original=checkpoint.load
        try:
            checkpoint.load=load_endpoint
            super().__init__(source,output,freeze_passive_jp=True,**kw)
        finally:checkpoint.load=original
    def step(self,command):
        if not self.authorize_steps:raise ValueError('preflight_cannot_step')
        return super().step(command)
    def save(self,path,**kw):
        info=super().save(path,**kw);side=Path(str(path)+'.json');meta=read(side)
        meta.update(schema='layered_47_full_response_second_fixed_route_v1',
            fixed_route_bindings={str(p):sha(p) for p in W.glob('*.py')},protocol_sha256=sha(W/'protocol.json'),
            highlevel_state=getattr(self,'highlevel_state',None),continuation_qualified=False,
            observation_scope='fixed truck route; no particle terrain map; physical particles untouched')
        side.write_text(json.dumps(meta,indent=2)+'\n');return info
