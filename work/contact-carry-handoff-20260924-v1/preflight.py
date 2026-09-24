"""D155 history repair, bounded kinematic preview and zero-step restore."""
import copy
import json
import time
from pathlib import Path
import numpy as np
from manual_key4_review import ROOT,W,sha
from carry_feedback import CarryFeedback
from continuous_preview import preview
from control_history import encode,decode
from contact_transport_handoff import CarryHandoffGate
from contact_bc_ground_guard import BucketGroundGuard
from contact_bc_inference import structured_observation,LIMITS
from contact_phase_bc_inference import OnlinePhaseFeatures,with_phase
from layered_excavation_approach_planner import _load_kinematics
from run_layered_47_sparse_manual_contact_episode import ContactCommandProjector
from restore_runtime import HandoffRuntime,SOURCE,SOURCE_SHA

PRIOR=SOURCE.parent
OUT=ROOT/'runs/contact-carry-handoff-20260924-v1/verified-preflight'
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]

def repair_history():
    original=read(PRIOR/'control-state.json');report=read(PRIOR/'report.json')
    actions=rows(PRIOR/'actions.ndjson');terrain={r['frame']:r['terrain'] for r in rows(PRIOR/'terrain.ndjson')}
    features=OnlinePhaseFeatures()
    for action in actions[-10:]:
        f=action['frame'];anchor=action['terrain_anchor_frame']
        row=structured_observation(frame=f,state=action['before_state'],terrain=terrain[anchor],terrain_frame=anchor,
            goal=report['selected_goal'],previous_action=action['previous_applied_action_si'],previous_load=action['previous_drive_load'])
        _,digest=features.update(with_phase(row,action['phase'],f-1))
        assert digest==action['feature_spec_sha256']
    assert features.last_frame==original['frame']==actions[-1]['frame']==2327
    assert original['feature_last_frame']==2328
    saved_rows=decode(original['features'])
    assert encode(list(features.rows)[1:])==encode(list(saved_rows)[:-1])
    repaired=copy.deepcopy(original);repaired.update(schema='D155_repaired_contact_history_v1',features=encode(features.rows),feature_last_frame=features.last_frame)
    root_count=0
    for action in actions:
        excess=action['feedback']['root_above_surface_m']
        if excess is not None:root_count=root_count+1 if excess<=0 else 0
    assert repaired['feedback']['last_frame']==2327
    repaired['feedback']['root_ready_count']=root_count
    repaired['source_checkpoint_sha256']=SOURCE_SHA
    repaired['repair_inputs']={str(p):sha(p) for p in (PRIOR/'control-state.json',PRIOR/'actions.ndjson',PRIOR/'terrain.ndjson',PRIOR/'report.json')}
    path=OUT/'repaired-control-state.json';path.write_text(json.dumps(repaired,indent=2)+'\n')
    return dict(path=str(path),sha256=sha(path),old_feature_frame=2328,new_feature_frame=2327,
                common_nine_feature_rows_exact=True,root_ready_count_rebuilt=root_count),actions[-1]

def kinematic_preview(q,previous,gate_snapshot,assembly,chain,ground):
    gate=CarryHandoffGate.restore(gate_snapshot)
    control=CarryFeedback(assembly,chain,ground,q,previous,frame_end=2327,initial_soil_top_m=gate.initial_soil_top_m)
    projector=ContactCommandProjector(chain);first=None;handoff=None;trace=[]
    for offset in range(1,601):
        frame=2327+offset
        command,detail=control.command(frame=frame,qpos=q,projector=projector)
        if first is None:
            first=command.copy();assert np.array_equal(first,previous)
        chain.q=q.copy();reference=chain.residual(q)
        scale=max(1.,float(np.max(abs(command)/np.array([.12,.0225,.0225,.0225]))))
        q,_=preview(projector,command/scale,scale/60,reference)
        pose=assembly.assemble(q,include_meshes=False)[1]['bucket']
        observation=gate.observe_end(frame=frame,closure_fraction=(q[15]-.37)/(.852-.37),
            origin_z_m=float(pose[2,3]),minimum_bucket_z_m=ground.measure(q)['minimum_bucket_z_m'],contact_established=True)
        trace.append(dict(frame=frame,applied_command_si=command.tolist(),qpos=q.tolist(),control=detail,gate=observation))
        if observation['ready'] and handoff is None:
            control.accept_handoff(readiness=observation,qpos=q);handoff=copy.deepcopy(observation)
        if handoff and frame>=handoff['frame_end']+30:break
        if offset%120==0:print('KINEMATIC',offset,'closure',observation['closure_held'],'rise',observation['sustained_actual_rise'],flush=True)
    (OUT/'kinematic-preview.ndjson').write_text(''.join(json.dumps(t)+'\n' for t in trace))
    deltas=[float(np.max(abs((np.asarray(trace[i]['applied_command_si'])-trace[i-1]['applied_command_si'])/LIMITS))) for i in range(1,len(trace))]
    result=dict(steps=len(trace),first_command_exact=True,handoff=handoff,final=trace[-1],
                maximum_normalized_command_step=max(deltas,default=0.),physics_steps_added=0,
                scope='Kinematic preview only: no soil forces, load response, spill or unload prediction')
    (W/'kinematic-review.json').write_text(json.dumps(result,indent=2)+'\n')
    if handoff is None:raise ValueError('D155_no_kinematic_handoff_within_600_frames')
    return result

def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=False)
    history,last=repair_history()
    cached=read(W/'cache-review.json')['episodes'][0]
    assert cached['inputs'][str(SOURCE)]==SOURCE_SHA
    with np.load(SOURCE) as source:q=source['mechanism_qpos'].copy()
    previous=np.asarray(last['applied_command_si'])
    k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
    assembly,chain=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
    ground=BucketGroundGuard(assembly,q)
    result=dict(status='preflight_in_progress',physics_steps_added=0,history=history,source_sha256=SOURCE_SHA)
    try:
        result['kinematic']=kinematic_preview(q.copy(),previous,cached['gate_snapshot'],assembly,chain,ground)
        with HandoffRuntime(SOURCE,OUT/'zero') as runtime:
            before=runtime.capture();native=copy.deepcopy(runtime.bridge.state);pending=copy.deepcopy(runtime.sweep.diagnostic_state())
            exact={key:bool(np.array_equal(value,runtime.arrays[key])) for key,value in before.items()}
            if not all(exact.values()) or max(runtime.restored['cycle_errors'].values())>1e-12:
                raise ValueError('D155_zero_restore_not_exact')
            feedback=CarryFeedback(runtime.assembly,runtime.chain,ground,q,previous,frame_end=2327,initial_soil_top_m=cached['final']['initial_soil_top_m'])
            command,detail=feedback.command(frame=2328,qpos=np.array(runtime.bridge.state['mechanism_qpos']),projector=runtime.projector)
            assert np.array_equal(command,previous)
            after=runtime.capture()
            assert all(np.array_equal(before[key],after[key]) for key in before)
            assert runtime.bridge.state==native and runtime.sweep.diagnostic_state()==pending
            saved=runtime.save(OUT/'zero.npz',stage='hold')
            from layered_47_cycle_checkpoint import load
            try:load(OUT/'zero.npz')
            except ValueError as error:assert str(error)=='cycle_checkpoint_schema'
            else:raise AssertionError('ordinary_loader_accepted_D155_candidate')
            result.update(arrays_exact=exact,controller_errors=runtime.restored['cycle_errors'],
                live_state_unchanged=True,pending_reaction_unchanged=True,first_command_exact=True,
                first_preview=detail,zero_checkpoint=saved)
        result['status']='D155_preflight_passed'
    except BaseException as error:
        result.update(status='preflight_failed',failure=repr(error));raise
    finally:
        result.update(wall_seconds=time.monotonic()-started,source_unchanged=sha(SOURCE)==SOURCE_SHA,
                      candidate_code={str(p):sha(p) for p in W.glob('*.py')})
        (W/'preflight-review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('kinematic','candidate_code')},indent=2),flush=True)

if __name__=='__main__':main()
