"""Dedicated D152 loader: retains candidate identity, reuses original validators."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
W=Path(__file__).resolve().parent;ROOT=W.parents[1];PRIOR_W=ROOT/'work/contact-full-response-20260924-v1';PRIOR=ROOT/'runs/contact-full-response-20260924-v1/candidate60';SOURCE=PRIOR/'final.npz'
sys.path.insert(0,str(PRIOR_W))
from candidate_runtime import CandidateRuntime as FullResponseRuntime
import layered_47_cycle_checkpoint as checkpoint
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SOURCE_SHA='37cb63c32f9070bcf119fac3a729be64ad6cde31593fa9f5ac8a17a6b5c6a0f2'

def load_candidate(path,*,allow_training_reset=False):
    path=Path(path).resolve()
    if path!=SOURCE or sha(path)!=SOURCE_SHA:raise ValueError('D153_source_not_authorized_endpoint')
    meta=json.loads(Path(str(path)+'.json').read_text())
    if meta['schema']!='layered_47_full_response_candidate_checkpoint_v1' or meta['base_schema']!=checkpoint.SCHEMA:raise ValueError('D153_wrong_physics_schema')
    if meta['checkpoint_sha256']!=SOURCE_SHA or meta['stage']!='hold' or meta['controller_state']['native_state_kind']!='full':raise ValueError('D153_source_not_complete')
    if (meta['coupling_hz'],meta['control_hz'],meta['rigid_hz'])!=(60,60,4800):raise ValueError('D153_physics_rate_changed')
    if not allow_training_reset:raise ValueError('D153_training_reset_boundary_required')
    checkpoint.validate_training_reset(meta)
    for p,digest in meta['candidate_bindings'].items():
        if sha(p)!=digest:raise ValueError('D153_candidate_code_changed:'+p)
    for name,digest in meta['runtime_bindings'].items():
        if sha(ROOT/'src/v4'/name)!=digest:raise ValueError('D153_original_runtime_changed:'+name)
    protocol=json.loads((PRIOR_W/'protocol.json').read_text())
    if not all(sha(x['path'])==x['sha256'] for x in protocol['inputs']):raise ValueError('D153_prior_dependency_changed')
    with np.load(path,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    checkpoint.sparse._validate(arrays,256441,47)
    checkpoint.validate_material_history_policy(meta,meta['runtime_bindings']);checkpoint.validate_controller(meta['controller_state'],47);checkpoint.validate_sweep(meta['moving_bucket_sweep_state'])
    if meta['cycle_index']!=3 or type(meta['truck_enabled']) is not bool:raise ValueError('D153_source_domain')
    meta['source_identity']=checkpoint.checkpoint_identity(path);meta['source_metadata_identity']=checkpoint.checkpoint_identity(str(path)+'.json')
    return arrays,meta

class ContinuationRuntime(FullResponseRuntime):
    def __init__(self,source,output,**kw):
        original=checkpoint.load
        try:checkpoint.load=load_candidate;super().__init__(source,output,**kw)
        finally:checkpoint.load=original
    def save(self,path,**kw):
        info=super().save(path,**kw);sidecar=Path(str(path)+'.json');m=json.loads(sidecar.read_text())
        m.update(schema='layered_47_full_response_lift_candidate_checkpoint_v1',physics_parent_schema='layered_47_full_response_candidate_checkpoint_v1',control_candidate_bindings={str(p):sha(p) for p in W.glob('*.py')},continuation_qualified=False)
        sidecar.write_text(json.dumps(m,indent=2)+'\n');return info
