"""Source-bound, frozen three-way evaluation; no training or physics changes."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path('/home/qunwang/wajueji')
WORK=ROOT/'work/roadshow-three-way-20261014-v1'
OUT=ROOT/'runs/roadshow-three-way-20261014-v1'
sys.path[:0]=[str(ROOT/'local-data/runtime/sb3-2.9.0-py312'),str(ROOT/'src/v4'),str(ROOT),str(ROOT/'work/contact-preview-lift-20260924-v1')]
import numpy as np
from manual_residual_sac_env import SOURCE,SOURCE_SHA,INPUT_SHA,Budget,ManualResidualEnv,sha,write,environment_contract
from manual_trajectory_residual import compare_arrays,LIMITS_SI
from layered_47_cycle_runtime import CycleRuntime
from contact_bc_ground_guard import BucketGroundGuard
import run_layered_47_sparse_manual_contact_episode as manual
from live_bucket_fill import MixedFillEstimator,load_config
from view_contact_bc_probe import ProbeDisplayCache


class Metrics:
    def __init__(self,out):
        self.out=out;self.pending=None;self.rows=[];self.observation_finished=None
    def begin(self,frame,phase):
        if self.pending is not None:raise ValueError('unsubmitted_request')
        self.pending=dict(frame=frame,phase=phase,capture_ns=time.perf_counter_ns(),decision_ns=None,
                          cold=frame<=50,accepted_ns=None)
    def observed(self):
        self.pending['decision_ns']=time.perf_counter_ns();self.observation_finished=self.pending['decision_ns']
    def submit(self,frame):
        if self.pending is None:return None
        row=self.pending;self.pending=None
        row.update(submit_ns=time.perf_counter_ns(),physical_dispatch_frame=frame)
        row['capture_to_submit_ms']=(row['submit_ns']-row['capture_ns'])/1e6
        row['decision_and_guard_ms']=(row['submit_ns']-row['decision_ns'])/1e6
        if min(row['capture_to_submit_ms'],row['decision_and_guard_ms'])<0:raise ValueError('clock_order')
        self.rows.append(row);return row
    def persist(self,row):
        if row is not None:manual.append_json(self.out/'response.ndjson',row)
    def summary(self):
        def describe(rows):
            values=[x['capture_to_submit_ms'] for x in rows]
            if not values:return dict(n=0)
            qs=np.quantile(values,[.5,.95,.99,1.])
            return dict(n=len(values),p50_ms=float(qs[0]),p95_ms=float(qs[1]),p99_ms=float(qs[2]),max_ms=float(qs[3]),over_100ms=int(np.sum(np.array(values)>100)))
        return dict(scope='simulation-state acquisition to software runtime step submission; no actuator acknowledgment or human cognitive timing',
                    all=describe(self.rows),warm=describe([x for x in self.rows if not x['cold']]),
                    phases={p:describe([x for x in self.rows if x['phase']==p]) for p in sorted({x['phase'] for x in self.rows})})


def instrument(r,metrics,cache=None,compact=None):
    original=r.step
    compact=manual.compact_state if compact is None else compact
    if cache is not None:cache.capture(0,compact(r.bridge.state),r.fixture.state.particle_q.numpy())
    def step(command):
        timing=metrics.submit(r.frames+1)
        row=original(command)
        if cache is not None:cache.capture(r.frames,row['after_state'],r.fixture.state.particle_q.numpy())
        metrics.persist(timing)
        return row
    r.step=step


def estimator(r):
    return MixedFillEstimator(load_config(ROOT/'configs/v4/live-bucket-fill-local-v1.json'),
        [x['volume_m3'] for x in r.records],manual._cache_rock_vertices(r.root,r.records))


def measure(r,est):
    q=np.array(r.bridge.state['mechanism_qpos']);pose=r.assembly.assemble(q,include_meshes=False)[1]['bucket']
    return est.measure_snapshot(dict(frame=r.frames,soil=r.fixture.state.particle_q.numpy(),pose=pose,
                              rocks=np.array([x['qpos'] for x in r.bridge.state['rocks']])))


def summarize_actions(path):
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    values=np.array([x['applied_si'] for x in rows])
    delta=np.diff(values/LIMITS_SI,axis=0)
    return dict(frames=len(rows),normalized_total_variation=float(np.sum(np.abs(delta))),
        normalized_total_variation_per_sim_s=float(np.sum(np.abs(delta))/(len(rows)/60)),
        projection_frames=sum(bool(x.get('projection_reason')) for x in rows),
        max_axis_speed_fraction=np.max(np.abs(values/LIMITS_SI),axis=0).tolist())


def self_check():
    from types import SimpleNamespace
    cache_rows=[];commands=[]
    class R:
        frames=0
        bridge=SimpleNamespace(state={'value':0})
        fixture=SimpleNamespace(state=SimpleNamespace(particle_q=SimpleNamespace(numpy=lambda:[[1.,2.,3.]])))
        def step(self,cmd):
            commands.append(cmd);self.frames+=1;return {'after_state':{'value':self.frames}}
    m=Metrics(Path('/tmp'));m.persist=lambda row:None
    c=SimpleNamespace(capture=lambda f,s,p:cache_rows.append(f))
    r=R();instrument(r,m,c,lambda x:x.copy());m.begin(0,'follow');m.observed()
    assert r.step([.1])['after_state']['value']==1
    r.step([.1]);r.step([.1]);assert len(m.rows)==1
    m.begin(3,'follow');m.observed();r.step([.2])
    assert len(m.rows)==2 and cache_rows==[0,1,2,3,4] and commands==[[.1],[.1],[.1],[.2]]
    assert m.summary()['all']['n']==2
    print('metrics and dispatch instrumentation checks passed; physical_steps=0')


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['manual','geometry','residual']);p.add_argument('--frames',type=int,default=3000)
    p.add_argument('--label',required=False);p.add_argument('--capture',action='store_true');p.add_argument('--self-check',action='store_true');a=p.parse_args()
    if a.self_check:self_check();return
    if not 0<=a.frames<=3000:raise ValueError('shared_task_frame_budget')
    budget=Budget(ROOT/'runs/roadshow-three-way-20261014-budget.json',seconds=2700,purpose='fair_three_way_evaluation');budget.check()
    if budget.remaining()<420+180:raise TimeoutError('reserve_before_case')
    prep=json.loads((WORK/'preparation.json').read_text());assert prep['status']=='cpu_preparation_passed'
    assert sha(SOURCE)==SOURCE_SHA
    out=OUT/(a.label or a.case);assert not out.exists();out.mkdir(parents=True)
    started=time.monotonic();metrics=Metrics(out);cache=None;env=None;r=None
    report=dict(status='initializing',case=a.case,source=dict(path=str(SOURCE),sha256=SOURCE_SHA),
        max_physics_frames=a.frames,common_task_ceiling_frames=3000,common_task_ceiling_sim_s=50,
        domain='CycleRuntime_D165_motion_consistency',material_history_policy='zero_hardening_frozen_jp_v1',
        training_updates=0,common_ground_guard_m=.005,soil_gap_gate=False,samples=[],events=[],
        score_scope='same uncalibrated bucket occupancy display, three samples over 0.5 seconds; not truck volume',
        endpoint_rules='each method completes its own control objective; all reported scores require 3 valid positive short-hold samples. Goal/extra constraints and control Hz disclosed, no statistical generalization')
    write(out/'report.json',report)
    try:
        if a.case=='residual':
            import torch
            from stable_baselines3 import SAC
            from run_manual_residual_sac import episode
            torch.set_num_threads(1)
            model_path=ROOT/'work/manual-strength-overnight-20261013-v1/anchor-grid-v1/seed44-weight100.zip'
            assert sha(model_path)=='84aade9a3d8819368afb72223b8452ebfbc5e9ac9ad190818207d1142692aab1'
            model=SAC.load(model_path,device='cpu');updates=model._n_updates
            env=ManualResidualEnv(out/'policy',budget,waiting_action_mode='error_strength_v1',max_frames=a.frames,collect=True)
            original_observation=env.observation
            def observation():
                metrics.begin(env.runtime.frames,'hold' if env.hold is not None else 'waiting' if getattr(env,'cursor',0)>=1170 else 'follow')
                obs=original_observation()
                metrics.pending['phase']='hold' if env.hold is not None else 'waiting' if obs[-1]>.5 else 'follow'
                metrics.observed();return obs
            env.observation=observation;original_reset=env.reset
            def reset(**kwargs):
                nonlocal r,cache
                value=original_reset(**kwargs);r=env.runtime
                if a.capture:cache=ProbeDisplayCache(out/'display',a.frames,r.arrays['particle_q'].shape,
                    dict(records=r.records,root=str(r.root),selected_cut=r.metadata['selected_cut'],particle_spacing_m=r.base['mpm']['particle_spacing_m']))
                instrument(r,metrics,cache);return value
            env.reset=reset
            def policy(obs):
                if budget.remaining()<420:raise TimeoutError('evaluation_reserve_reached')
                return model.predict(obs,deterministic=True)[0]
            result=episode(env,policy)
            report.update(status=result['status'],result=result,frames=result['physical_frames'],samples=result['scores'],
                control_hz=20,model_sha256=sha(model_path),runtime_bindings=result['runtime_bindings'],
                zero_step_exact=result['zero_step_exact'],simulation_seconds=result['simulation_seconds'])
            assert model._n_updates==updates and sha(model_path)==report['model_sha256']
            if result['success']:
                report['retained_display_percent']=result['retained_fill_fraction']*100
                report['completed_short_hold']=True
            previous=ROOT/'runs/manual-strength-overnight-20261014/anchor-sweep-v1/seed44-weight100-independent-three/repetition-1/episode-0001'
            if result['status']=='completed':
                new=[json.loads(x) for x in (env.directory/'actions.ndjson').read_text().splitlines()]
                old=[json.loads(x) for x in (previous/'actions.ndjson').read_text().splitlines()]
                report['actions_equal_previous']=new==old
                with np.load(result['checkpoint']['path']) as x,np.load(previous/'endpoint.npz') as y:
                    report['endpoint_equal_previous']={k:bool(np.array_equal(x[k],y[k])) for k in x.files}
                assert report['actions_equal_previous'] and all(report['endpoint_equal_previous'].values())
                report['motion']=summarize_actions(env.directory/'actions.ndjson')
        else:
            r=CycleRuntime(SOURCE,out/'physics',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True);r.__enter__()
            exact,errors=compare_arrays(r.capture(),r.arrays)
            assert all(v for k,v in exact.items() if k!='simulation_time_s') and errors['simulation_time_s']<=1e-12
            report.update(zero_step_exact=exact,zero_step_errors=errors,runtime_bindings=r.bindings,control_hz=60)
            ground=BucketGroundGuard(r.assembly,r.bridge.state['mechanism_qpos']);est=estimator(r)
            if a.capture:cache=ProbeDisplayCache(out/'display',a.frames,r.arrays['particle_q'].shape,
                dict(records=r.records,root=str(r.root),selected_cut=r.metadata['selected_cut'],particle_spacing_m=r.base['mpm']['particle_spacing_m']))
            instrument(r,metrics,cache)
            hold=None
            if a.case=='manual':
                selected=prep['selected_manual'];path=Path(selected['run'])/'input.ndjson'
                assert sha(path)==selected['input_sha256']
                inputs=[json.loads(x) for x in path.read_text().splitlines()]
                report.update(manual_record=selected,manual_selection_rule=prep['manual_selection_rule'],
                    human_reaction_time_measured=False,manual_replay_is_not_live_human=True)
                n=min(len(inputs),a.frames)
            else:
                from roadshow_geometry_baseline import geometric_feedback_class
                from roadshow_adaptive_lift_feedback_v2 import LiftFeedback
                from contact_geometric_feedback import FeedbackConfig
                from contact_cut_reference import GroundSoilCutReference
                from contact_phase_live import LivePhaseObserver
                from contact_phase_candidate import ContactPhaseConfig
                from contact_phase_completion import CompletionConfig
                from contact_bc_inference import structured_observation
                from contact_phase_bc_inference import OnlinePhaseFeatures,with_phase
                from run_layered_47_sparse_contact_handoff import wrench_summary
                contract=json.loads((ROOT/'work/contact-multisource-lift-20260924-v1/first-protocol.json').read_text())
                cfg=FeedbackConfig(**contract['feedback_parameters']);face=r.metadata['selected_cut'].get('contact',r.metadata['selected_cut'])
                goal=dict(entry_xy_m=face.get('contact_xy_m',face.get('entry_xy_m')),surface_z_m=face.get('contact_surface_z_m',face.get('surface_z_m')))
                pose=r.assembly.assemble(r.arrays['mechanism_qpos'],include_meshes=False)[1]['bucket']
                polygon=json.loads(Path(contract['cavity_config']).read_text())['bucket']['polygon_xz_m']
                phases=LivePhaseObserver(r.proxy,polygon,ContactPhaseConfig(**json.loads(Path(contract['contact_config']).read_text())['parameters']),
                    CompletionConfig(**json.loads(Path(contract['completion_config']).read_text())['parameters']),pose)
                feedback=geometric_feedback_class(LiftFeedback)(r.assembly,r.chain,ground,goal,r.arrays['mechanism_qpos'],cfg,GroundSoilCutReference(r.arrays['particle_q'],r.proxy))
                features=OnlinePhaseFeatures();vertices=manual._cache_rock_vertices(r.root,r.records);terrain=None;anchor=0;previous_action=[0.]*4;previous_load=None
                adaptive=ROOT/'src/v4/roadshow_adaptive_lift_feedback_v2.py'
                adaptive_receipt=json.loads((ROOT/'runs/roadshow-measurements-20261011/geometry-adaptive-v2-120/integration-review.json').read_text())
                assert adaptive_receipt['status']=='passed' and sha(adaptive)==adaptive_receipt['adaptive_preview_sha256']
                report.update(geometric_parameters=contract['feedback_parameters'],learned_weights_used=False,extra_internal_ground_margin_m=cfg.ground_clearance_m,
                    geometry_code={**prep['geometry_cpu']['geometry_code'],str(adaptive):sha(adaptive)},adaptive_preview='roadshow_A_direct_proportional_probe_v2')
                n=a.frames
            if a.frames==0:
                report.update(status='zero_restore_passed',frames=0)
            for frame in range(1,n+1):
                budget.check()
                if budget.remaining()<420:raise TimeoutError('evaluation_reserve_reached')
                if not ground.measure(r.bridge.state['mechanism_qpos'])['safe']:raise ValueError('common_ground_guard')
                phase='manual_record' if a.case=='manual' else 'hold' if hold is not None else phases.router.phase_for_decision(frame)
                metrics.begin(frame-1,phase);before=manual.compact_state(r.bridge.state)
                if a.case=='manual':
                    metrics.observed();requested=np.array(inputs[frame-1]['applied_command_si']);applied=r.projector.project(before['mechanism_qpos'],requested)
                    detail=dict(recorded_frame=frame,recorded_requested=inputs[frame-1]['requested_command_si'])
                elif hold is not None:
                    metrics.observed();requested=applied=np.zeros(4);detail=dict(short_hold=True)
                else:
                    if terrain is None or (frame-1)%3==0:
                        terrain=manual.terrain_observation(r.fixture,r.bridge.state,vertices);anchor=frame-1
                    structured=with_phase(structured_observation(frame=frame,state=before,terrain=terrain,terrain_frame=anchor,goal=goal,
                        previous_action=previous_action,previous_load=previous_load),phase,frame-1)
                    features.update(structured);metrics.observed()
                    requested,applied,detail=feedback.command(frame=frame,qpos=before['mechanism_qpos'],phase=phase,bc_action=np.zeros(4),projector=r.projector)
                manual.append_json(out/'actions.ndjson',dict(frame=frame,phase=phase,applied_si=applied.tolist(),requested_si=requested.tolist(),
                    projection_reason=r.projector.reason,detail=detail,before_state=before))
                row=r.step(applied)
                if not(row['hard_guard'] and row['hencky_material_guard']['safe'] and row['moving_bucket_sweep']['safe']):raise ValueError('common_execution_guard')
                if not ground.measure(r.bridge.state['mechanism_qpos'])['safe']:raise ValueError('common_ground_guard_after')
                report['frames']=r.frames
                if a.case=='manual':
                    if a.frames>=len(inputs) and frame in (len(inputs)-30,len(inputs)-15,len(inputs)):
                        report['samples'].append(measure(r,est))
                elif hold is not None:
                    hold+=1
                    if hold in (15,30):report['samples'].append(measure(r,est))
                    if hold==30:report['status']='completed';break
                else:
                    pose=r.assembly.assemble(row['after_state']['mechanism_qpos'],include_meshes=False)[1]['bucket']
                    event=phases.observe_end(frame=frame,state=row['after_state'],pose=pose,
                        soil_force_n=wrench_summary(r.captured['wrenches']['bucket'])['bucket_mpm_force_norm_n'],
                        rock_contact_substeps=row['rock_bucket_contact_substeps'],particles=lambda:r.fixture.state.particle_q.numpy())
                    manual.append_json(out/'phases.ndjson',event)
                    if event['phase']['curl_boundary_unresolved']:raise ValueError('geometry_curl_boundary_unresolved')
                    if event['phase']['event']:report['events'].append(event['phase']['event'])
                    previous_action=applied.tolist();previous_load=list(r.bridge.state['equivalent_drive_load'])
                    if phases.router.phase in ('lift','off_soil'):
                        sample=measure(r,est)
                        if sample['valid_estimate'] and sample['fill_percent']>0:
                            report['samples']=[sample];hold=0;report['hold_started_frame']=r.frames
                        else:report['last_invalid_lift_measurement']=sample
                if frame%60==0:
                    print('CASE_FRAME',a.case,frame,phase,'remaining',round(budget.remaining()),flush=True)
                    write(out/'report.json',report)
                if frame==120:
                    report['prefix_checkpoint']=r.save(out/'prefix-120.npz',stage='hold')
                    if a.label=='geometry-full':
                        prefix=OUT/'geometry-prefix'
                        old=[json.loads(x) for x in (prefix/'actions.ndjson').read_text().splitlines()]
                        new=[json.loads(x) for x in (out/'actions.ndjson').read_text().splitlines()]
                        fields=['frame','phase','applied_si','requested_si','before_state']
                        report['prefix_actions_exact']=len(old)==len(new)==120 and all(all(x[k]==y[k] for k in fields) for x,y in zip(old,new,strict=True))
                        with np.load(report['prefix_checkpoint']['path']) as x,np.load(prefix/'prefix-120.npz') as y:
                            report['prefix_arrays_exact']={k:bool(np.array_equal(x[k],y[k])) for k in x.files}
                        assert report['prefix_actions_exact'] and all(report['prefix_arrays_exact'].values())
                        write(out/'report.json',report)
            if a.frames and report['status']=='initializing':
                report['status']='completed' if a.case=='manual' and len(inputs)<=a.frames else 'bounded_window_completed'
            if a.case=='geometry':report['final_phase']=phases.router.phase
            report['final_checkpoint']=r.save(out/'endpoint.npz',stage='hold' if report['status']=='completed' else 'transport')
            report['simulation_seconds']=r.frames/60
            if len(report['samples'])==3 and all(x['valid_estimate'] for x in report['samples']):
                score=min(x['fill_percent'] for x in report['samples']);report['retained_display_percent']=score
                report['completed_short_hold']=report['status']=='completed' and score>0
            else:report['completed_short_hold']=False
            if r.frames:report['motion']=summarize_actions(out/'actions.ndjson')
            if a.label=='geometry-full':
                prefix=OUT/'geometry-prefix'
                old=[json.loads(x) for x in (prefix/'actions.ndjson').read_text().splitlines()]
                new=[json.loads(x) for x in (out/'actions.ndjson').read_text().splitlines()][:len(old)]
                fields=['frame','phase','applied_si','requested_si','before_state']
                report['prefix_actions_exact']=all(all(x[k]==y[k] for k in fields) for x,y in zip(old,new,strict=True))
                assert report['prefix_actions_exact']
                if r.frames>=120:
                    with np.load(report['prefix_checkpoint']['path']) as x,np.load(prefix/'prefix-120.npz') as y:
                        report['prefix_arrays_exact']={k:bool(np.array_equal(x[k],y[k])) for k in x.files}
                    assert all(report['prefix_arrays_exact'].values())
    except BaseException as e:
        report.update(status='failed' if not isinstance(e,TimeoutError) else 'budget_stopped',error=type(e).__name__+':'+str(e))
        if r is not None and r.bridge is not None:
            report['frames']=r.frames
            try:report['failure_checkpoint']=r.save(out/'failure.npz',stage='failure')
            except Exception as err:report['failure_save_error']=str(err)
        raise
    finally:
        if cache is not None:
            capture=cache.finish(report['status']);capture['label']=a.case.upper()+' / SAME-DOMAIN THREE-WAY EVALUATION'
            write(out/'display/reconstruction.json',capture);report['display']=capture
        if env is not None:env.close()
        elif r is not None:r.close()
        report.update(response=metrics.summary(),wall_seconds=time.monotonic()-started,remaining_seconds=budget.remaining(),source_unchanged=sha(SOURCE)==SOURCE_SHA)
        write(out/'report.json',report)
        print('CASE_RESULT',json.dumps({k:report.get(k) for k in ['case','status','frames','retained_display_percent','completed_short_hold','wall_seconds','remaining_seconds','error']}),flush=True)


if __name__=='__main__':main()
