"""Read-only checks of the actual three-way runs, timing and frozen predictions."""
from pathlib import Path
import json
import sys
import time

ROOT=Path('/home/qunwang/wajueji');WORK=ROOT/'work/roadshow-three-way-20261014-v1';RUN=ROOT/'runs/roadshow-three-way-20261014-v1'
sys.path[:0]=[str(ROOT/'local-data/runtime/sb3-2.9.0-py312'),str(ROOT/'src/v4'),str(ROOT)]
import numpy as np
import torch
from stable_baselines3 import SAC
from manual_residual_sac_env import SOURCE,SOURCE_SHA,sha


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def quantiles(a):
    a=np.asarray(a,float)
    if not len(a):return dict(n=0)
    q=np.quantile(a,[.5,.95,.99,1.])
    return dict(n=len(a),p50_ms=float(q[0]),p95_ms=float(q[1]),p99_ms=float(q[2]),max_ms=float(q[3]),over_100ms=int(np.sum(a>100)))


torch.set_num_threads(1)
model_path=ROOT/'work/manual-strength-overnight-20261013-v1/anchor-grid-v1/seed44-weight100.zip'
model=SAC.load(model_path,device='cpu')
checks=dict(initial_source_unchanged=sha(SOURCE)==SOURCE_SHA,
            frozen_model_unchanged=sha(model_path)=='84aade9a3d8819368afb72223b8452ebfbc5e9ac9ad190818207d1142692aab1',
            no_saved_model_updates=model._n_updates==1000)
cases={};summary=[]
bindings=None
for label in ['manual','residual-v3','geometry-full']:
    out=RUN/label;report=read(out/'report.json');case=report['case'];n=report.get('frames',0)
    ep=out/'policy/episode-0001' if case=='residual' else out
    physical=rows(ep/'physics/frames.ndjson' if case=='residual' else out/'physics/frames.ndjson')
    actions=rows(ep/'actions.ndjson');timing=rows(out/'response.ndjson')
    c=dict(source_sha=report['source']['sha256']==SOURCE_SHA,source_preserved=report['source_unchanged'],
        no_training=report['training_updates']==0,domain=report['domain']=='CycleRuntime_D165_motion_consistency',
        initial_restore=all(v for k,v in report['zero_step_exact'].items() if k!='simulation_time_s'),
        physical_count=len(physical)==len(actions)==n,
        frame_order=[x['frame'] for x in physical]==[x['frame'] for x in actions]==list(range(1,n+1)),
        hard_guard=all(x['hard_guard'] for x in physical),material_guard=all(x['hencky_material_guard']['safe'] for x in physical),
        sweep_guard=all(x['moving_bucket_sweep']['safe'] for x in physical),
        dispatched_exact=all(np.array_equal(x['applied_si'],y['applied_command_si']) for x,y in zip(actions,physical,strict=True)),
        timed_requests_sorted=all(x['physical_dispatch_frame']<y['physical_dispatch_frame'] for x,y in zip(timing,timing[1:])),
        clock_order=all(x['capture_ns']<=x['decision_ns']<=x['submit_ns'] for x in timing),
        no_fake_acceptance=all(x['accepted_ns'] is None for x in timing),
        all_response_recomputed=report['response']['all']==quantiles([(x['submit_ns']-x['capture_ns'])/1e6 for x in timing]),
        warm_response_recomputed=report['response']['warm']==quantiles([(x['submit_ns']-x['capture_ns'])/1e6 for x in timing if not x['cold']]),
        cold_definition=all(x['cold']==(x['frame']<=50) for x in timing))
    if bindings is None:bindings=report['runtime_bindings']
    c['common_runtime']=report['runtime_bindings']==bindings
    if case=='residual':
        policy=rows(ep/'policy.ndjson')
        c['frozen_prediction_recomputed']=all(np.array_equal(model.predict(np.array(x['observation'],dtype=np.float32),deterministic=True)[0],np.array(x['action'])) for x in policy)
        c['neural_decision_count']=len(policy)==len(timing)==402
        c['actions_equal_verified_run']=report['actions_equal_previous']
        c['endpoint_equal_verified_run']=all(report['endpoint_equal_previous'].values())
        endpoint=Path(report['result']['checkpoint']['path'])
    else:
        endpoint=Path(report.get('final_checkpoint',report.get('failure_checkpoint'))['path'])
        cache=np.load(out/'display/particles.npy',mmap_mode='r')
        with np.load(SOURCE) as source:
            c['initial_particles_exact']=np.array_equal(cache[0],source['particle_q'])
        c['cache_count']=len(cache)==n+1
        with np.load(endpoint) as end:c['cache_endpoint_exact']=np.array_equal(cache[-1],end['particle_q'])
    meta=read(Path(str(endpoint)+'.json'))
    c['endpoint_file_sha']=sha(endpoint)==meta['checkpoint_sha256']
    c['motion_policy']=meta['bucket_motion_policy']=='bucket_origin_predicted_endpoint_v1'
    c['material_policy']=meta['material_history_policy']=='zero_hardening_frozen_jp_v1'
    with np.load(endpoint) as end:
        c['particle_identity_count']=end['particle_q'].shape==(256441,3) and len(end['rigid_ids'])==47
    samples=report['samples']
    if report.get('completed_short_hold'):
        c['three_valid_samples']=len(samples)==3 and all(x['valid_estimate'] and x['fill_percent']>0 for x in samples)
        c['minimum_score']=report['retained_display_percent']==min(x['fill_percent'] for x in samples)
        c['sample_spacing']=[samples[i+1]['frame']-samples[i]['frame'] for i in [0,1]]==[15,15]
        c['capacity_common']=all(abs(x['capacity_liters']-255.164729950989)<1e-9 for x in samples)
        c['threshold_common']=all(x['occupancy_level']==.2 and x['display_only'] for x in samples)
    else:c['no_invalid_zero_score']='retained_display_percent' not in report
    if case=='geometry':
        c['no_neural_weights']=not report['learned_weights_used']
        c['strong_adaptive_baseline']=report['adaptive_preview']=='roadshow_A_direct_proportional_probe_v2'
        c['prefix_commands']=report['prefix_actions_exact']
        c['prefix_arrays']=all(report['prefix_arrays_exact'].values())
    if case=='manual':
        record=report['manual_record'];c['manual_input_unchanged']=sha(Path(record['run'])/'input.ndjson')==record['input_sha256']
        c['human_timing_not_invented']=not report['human_reaction_time_measured']
    cycles=quantiles([(y['capture_ns']-x['capture_ns'])/1e6 for x,y in zip(timing,timing[1:])])
    final_state=physical[-1]['after_state'];z=[x['after_state']['bucket_origin_world_m'][2] for x in physical]
    cases[case]=dict(checks=c,source_report=str(out/'report.json'),source_report_sha256=sha(out/'report.json'),
        observed_cycle=cycles,cycle_scope='consecutive request capture timestamps; final cycle right-censored; includes physics/logging/capture, not a common actuator latency',
        rise_from_episode_minimum_m=z[-1]-min(z),final_bucket_origin_m=final_state['bucket_origin_world_m'])
    for k,v in c.items():checks[case+'_'+k]=v
    response=report['response']['all'];motion=report.get('motion',{})
    if not motion:
        values=np.array([x['applied_si'] for x in actions]);limits=np.array([.12,.0225,.0225,.0675])
        motion=dict(normalized_total_variation_per_sim_s=float(np.sum(np.abs(np.diff(values/limits,axis=0)))/(n/60)),projection_frames=sum(bool(x.get('projection_reason')) for x in actions))
    summary.append(dict(case=case,status=report['status'],completed_short_hold=bool(report.get('completed_short_hold')),
        held_display_percent=report.get('retained_display_percent'),simulation_seconds=n/60,physical_frames=n,
        control_hz=report['control_hz'],software_response_p50_ms=response.get('p50_ms') if case!='manual' else None,
        software_response_p95_ms=response.get('p95_ms') if case!='manual' else None,
        software_response_p99_ms=response.get('p99_ms') if case!='manual' else None,
        software_response_max_ms=response.get('max_ms') if case!='manual' else None,
        software_over100_count=response.get('over_100ms') if case!='manual' else None,
        software_response_n=response.get('n') if case!='manual' else None,
        replay_forward_p95_ms=response.get('p95_ms') if case=='manual' else None,
        human_reaction_ms=None,observed_cycle_p95_ms=cycles.get('p95_ms'),
        projection_frames=motion.get('projection_frames'),normalized_total_variation_per_sim_s=motion.get('normalized_total_variation_per_sim_s'),
        all_executed_frames_safe=c['hard_guard'] and c['material_guard'] and c['sweep_guard'],
        response_scope='state acquisition to software runtime submission; not human cognition or actuator response'))
result=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,cases=cases,summary=summary,
            shared_source_sha256=SOURCE_SHA,source_files={str(p):sha(p) for p in WORK.glob('*.py')},
            original_best_manual_display_percent=145.38132234025687,training_updates=0)
write(WORK/'independent-audit.json',result)
print('AUDIT',result['status'],'checks',len(checks),'failed',[k for k,v in checks.items() if not v],flush=True)
print(json.dumps(summary,ensure_ascii=False),flush=True)
assert all(checks.values())
