"""Rebuild causal control state from immutable D152 logs, without any physics."""
from collections import deque
from pathlib import Path
import copy,json,time
import numpy as np
from continuation_runtime import W,ROOT,PRIOR_W,PRIOR,SOURCE,SOURCE_SHA,sha
from contact_bc_inference import structured_observation
from contact_phase_bc_inference import OnlinePhaseFeatures,PhaseInferenceWorker,with_phase
from contact_phase_live import LivePhaseObserver
from contact_phase_candidate import ContactPhaseConfig
from contact_phase_completion import CompletionConfig
from contact_geometric_feedback import ContactGeometryFeedback,FeedbackConfig
from contact_cut_reference import GroundSoilCutReference
from contact_bc_ground_guard import BucketGroundGuard
from layered_excavation_approach_planner import _load_kinematics
from run_layered_47_sparse_manual_contact_episode import ContactCommandProjector
OUT=ROOT/'runs/contact-preview-lift-20260924-v1'
DYNAMIC=('contact','cut_target','last_phase','last_frame','soil_plan','root_ready','root_ready_count','no_down_latched','no_down_frame','previous_command')
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()]

def encode(v):
    if isinstance(v,np.ndarray):return dict(array=v.tolist(),dtype=str(v.dtype))
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,deque):return dict(deque=[encode(x) for x in v],maxlen=v.maxlen)
    if isinstance(v,dict):return {k:encode(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [encode(x) for x in v]
    return v

def decode(v):
    if isinstance(v,dict):
        if set(v)=={'array','dtype'}:return np.array(v['array'],dtype=v['dtype'])
        if set(v)=={'deque','maxlen'}:return deque([decode(x) for x in v['deque']],maxlen=v['maxlen'])
        return {k:decode(x) for k,x in v.items()}
    if isinstance(v,list):return [decode(x) for x in v]
    return v

def construct(feedback_class=ContactGeometryFeedback,assembly=None,chain=None):
    report=read(PRIOR/'report.json');contract=read(PRIOR_W/'protocol.json')
    with np.load(report['source']['path']) as f:initial={k:f[k].copy() for k in ('particle_q','mechanism_qpos')}
    if assembly is None:
        k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');assembly,chain=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
    proxy=read(ROOT/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json');guard=BucketGroundGuard(assembly,initial['mechanism_qpos']);ref=GroundSoilCutReference(initial['particle_q'],proxy)
    feedback=feedback_class(assembly,chain,guard,report['selected_goal'],initial['mechanism_qpos'],FeedbackConfig(**contract['feedback_parameters']),ref)
    pose=assembly.assemble(initial['mechanism_qpos'],include_meshes=False)[1]['bucket'];polygon=read(contract['cavity_config'])['bucket']['polygon_xz_m']
    phases=LivePhaseObserver(proxy,polygon,ContactPhaseConfig(**read(contract['contact_config'])['parameters']),CompletionConfig(**read(contract['completion_config'])['parameters']),pose)
    return feedback,phases,OnlinePhaseFeatures(),report,contract

def restore(feedback,phases,features,path):
    data=read(path)
    if data['source_sha256']!=SOURCE_SHA or sha(SOURCE)!=SOURCE_SHA:raise ValueError('control_history_source')
    for item in data['inputs']:
        if sha(item['path'])!=item['sha256']:raise ValueError('control_history_input_changed')
    for k,v in decode(data['feedback']).items():setattr(feedback,k,v)
    for k,v in decode(data['phase']).items():setattr(phases.router,k,v)
    for k,v in decode(data['contact']).items():setattr(phases.router.contact,k,v)
    features.rows=decode(data['features']);features.last_frame=data['last_frame'];features.last_sequence=1
    if feedback.last_frame!=2400 or phases.router.last_frame_end!=2400 or features.last_frame!=2400:raise ValueError('control_history_frame')
    return data

def main():
    OUT.mkdir(exist_ok=True);started=time.monotonic();feedback,phases,features,report,contract=construct();p=ContactCommandProjector(feedback.chain)
    actions=rows(PRIOR/'actions.ndjson');physical=rows(PRIOR/'physics/frames.ndjson');traces=rows(PRIOR/'phase-trace.ndjson');terrain={x['frame']:x['terrain'] for x in rows(PRIOR/'terrain.ndjson')};exact_bc=[]
    with PhaseInferenceWorker('/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python',report['policy']['path'],report['policy']['sha256'],stderr_path=OUT/'history-inference-stderr.log') as worker:
        for e,row,t in zip(actions,physical,traces,strict=True):
            f=e['frame'];req,cmd,detail=feedback.command(frame=f,qpos=e['before_state']['mechanism_qpos'],phase=e['phase'],bc_action=e['action_normalized'],projector=p)
            if req.tolist()!=e['requested_command_si'] or cmd.tolist()!=e['applied_command_si'] or detail!=e['feedback']:raise ValueError('history_command:'+str(f))
            state=row['after_state'];pose=feedback.assembly.assemble(state['mechanism_qpos'],include_meshes=False)[1]['bucket']
            phase=phases.observe_end(frame=f,state=state,pose=pose,soil_force_n=t['bucket_mpm_force_norm_n'],rock_contact_substeps=row['rock_bucket_contact_substeps'],particles=lambda:(_ for _ in ()).throw(ValueError('unexpected_historical_geometry')))
            if phase['phase']!=t['phase']:raise ValueError('history_phase:'+str(f))
            if f>=2388:
                observation=structured_observation(frame=f,state=e['before_state'],terrain=terrain[e['terrain_anchor_frame']],terrain_frame=e['terrain_anchor_frame'],goal=report['selected_goal'],previous_action=e['previous_applied_action_si'],previous_load=e['previous_drive_load'])
                x,digest=features.update(with_phase(observation,e['phase'],f-1))
                if f>=2398:
                    predicted,_=worker.predict(x,digest)
                    if predicted.tolist()!=e['action_normalized']:raise ValueError('history_bc:'+str(f))
                    exact_bc.append(f)
            if f%300==0:print('HISTORY',f,'seconds',time.monotonic()-started,flush=True)
    inputs=[SOURCE,Path(str(SOURCE)+'.json'),PRIOR/'actions.ndjson',PRIOR/'physics/frames.ndjson',PRIOR/'phase-trace.ndjson',PRIOR/'terrain.ndjson',PRIOR/'physics/native-halves.ndjson',PRIOR_W/'protocol.json',Path(report['policy']['path']),Path(__file__)]
    data=dict(schema='D153_causal_control_history_v1',source_sha256=SOURCE_SHA,last_frame=2400,feedback=encode({k:getattr(feedback,k) for k in DYNAMIC}),phase=encode({k:v for k,v in vars(phases.router).items() if k not in ('config','contact')}),contact=encode({k:v for k,v in vars(phases.router.contact).items() if k!='config'}),features=encode(features.rows),last_action=actions[-1]['applied_command_si'],last_load=rows(PRIOR/'physics/native-halves.ndjson')[-1]['equivalent_drive_load'],exact_bc_frames=exact_bc,all_2400_commands_and_phases_exact=True,wall_seconds=time.monotonic()-started,physics_steps_added=0,inputs=[dict(path=str(p.resolve()),sha256=sha(p)) for p in inputs])
    path=OUT/'control-history.json';path.write_text(json.dumps(data,indent=2)+'\n')
    f2,p2,x2,_,_=construct();restore(f2,p2,x2,path)
    assert encode({k:getattr(f2,k) for k in DYNAMIC})==data['feedback'];assert encode(x2.rows)==data['features'];assert encode({k:v for k,v in vars(p2.router).items() if k not in ('config','contact')})==data['phase']
    review=dict(history_path=str(path),history_sha256=sha(path),source_sha256=SOURCE_SHA,commands_and_phases_exact=True,feature_serialization_exact=True,last_three_bc_actions_exact=exact_bc,physics_steps_added=0,wall_seconds=data['wall_seconds'])
    (W/'history-review.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps(review),flush=True)
if __name__=='__main__':main()
