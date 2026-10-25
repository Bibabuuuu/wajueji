"""Inspect same-source manual candidates and reuse pure geometry intent."""
import json
from pathlib import Path
import sys
import time

ROOT=Path('/home/qunwang/wajueji')
WORK=ROOT/'work/roadshow-three-way-20261014-v1'
sys.path[:0]=[str(ROOT/'local-data/runtime/sb3-2.9.0-py312'),str(ROOT/'src/v4'),str(ROOT),str(ROOT/'work/contact-preview-lift-20260924-v1')]
import numpy as np
from layered_47_cycle_runtime import CycleRuntime
from manual_residual_sac_env import SOURCE,SOURCE_SHA,Budget,sha,environment_contract
from live_bucket_fill import MixedFillEstimator,load_config
import run_layered_47_sparse_manual_contact_episode as manual
from roadshow_geometry_baseline import geometric_feedback_class,self_check
from lift_feedback import LiftFeedback
from contact_geometric_feedback import FeedbackConfig
from contact_bc_ground_guard import BucketGroundGuard
from contact_cut_reference import GroundSoilCutReference

budget=Budget(ROOT/'runs/roadshow-three-way-20261014-budget.json',seconds=2700,purpose='fair_three_way_evaluation');budget.check()
r=CycleRuntime(SOURCE,WORK/'cpu-runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
vertices=manual._cache_rock_vertices(r.root,r.records)
estimator=MixedFillEstimator(load_config(ROOT/'configs/v4/live-bucket-fill-local-v1.json'),[x['volume_m3'] for x in r.records],vertices)
result=dict(status='preparing',new_physics_steps=0,source_sha256=SOURCE_SHA,environment_contract=environment_contract('error_strength_v1'),manual_candidates=[])
for name in ['manual-contact-20260925-134737-2439624','manual-contact-20260925-141348-2458450']:
    budget.check()
    path=ROOT/'runs/layered-47-sparse-manual-contact-v1'/name
    report=json.loads((path/'report.json').read_text());inputs=[json.loads(x) for x in (path/'input.ndjson').read_text().splitlines()]
    n=report['episode_summary']['frames'];assert report['source_checkpoint']['sha256']==SOURCE_SHA
    assert [x['frame'] for x in inputs]==list(range(1,n+1))
    assert len(inputs)>=30 and all(x['input_device']=='system_safety_hold' for x in inputs[-30:])
    cache=path/'display';states=json.loads((cache/'states.json').read_text());particles=np.load(cache/'particles.npy',mmap_mode='r')
    samples=[]
    for frame in [n-30,n-15,n]:
        state=states[frame];pose=r.assembly.assemble(np.array(state['mechanism_qpos']),include_meshes=False)[1]['bucket']
        sample=estimator.measure_snapshot(dict(frame=frame,soil=particles[frame],pose=pose,rocks=np.array([x['qpos'] for x in state['rocks']])))
        samples.append(sample)
    q=np.array(states[-1]['mechanism_qpos'])
    item=dict(run=str(path),frames=n,recording_status=report['status'],stop_reason=report['stop_reason'],input_sha256=sha(path/'input.ndjson'),source_sha256=SOURCE_SHA,original_physics_domain='historical_manual_not_D165_replay',samples=samples,
        retained_display_percent=min(x['fill_percent'] for x in samples) if all(x['valid_estimate'] for x in samples) else None,
        original_end_bucket_origin=states[-1]['bucket_origin_world_m'],all_recorded_frames_safe=report['checks']['all_executed_frames_hard_safe'],material_history_policy=report.get('material_history_policy'))
    result['manual_candidates'].append(item)
    print('MANUAL_CPU',name,item['retained_display_percent'],[(s['valid_estimate'],s['quality_reasons']) for s in samples],flush=True)
eligible=[x for x in result['manual_candidates'] if x['retained_display_percent'] is not None and x['retained_display_percent']<145.38132234025687]
result['selected_manual']=eligible[0] if eligible else result['manual_candidates'][0]
result['manual_selection_rule']='First chronological same-source user_finished recording with complete terminal 30-frame hold and valid historical lower display; if none qualify, first record retained as an unqualified case, never assigned zero.'
contract=json.loads((ROOT/'work/contact-multisource-lift-20260924-v1/first-protocol.json').read_text())
cfg=FeedbackConfig(**contract['feedback_parameters'])
face=r.metadata['selected_cut'].get('contact',r.metadata['selected_cut'])
goal=dict(entry_xy_m=face.get('contact_xy_m',face.get('entry_xy_m')),surface_z_m=face.get('contact_surface_z_m',face.get('surface_z_m')))
ground=BucketGroundGuard(r.assembly,r.arrays['mechanism_qpos'])
cut=GroundSoilCutReference(r.arrays['particle_q'],r.proxy)
feedback=geometric_feedback_class(LiftFeedback)(r.assembly,r.chain,ground,goal,r.arrays['mechanism_qpos'],cfg,cut)
t=time.perf_counter();requested,applied,detail=feedback.command(frame=1,qpos=r.arrays['mechanism_qpos'],phase='approach_contact',bc_action=np.zeros(4),projector=r.projector)
result['geometry_cpu']=dict(status='pure_geometry_first_intent_passed',requested=requested.tolist(),applied=applied.tolist(),detail=detail,wall_seconds=time.perf_counter()-t,feedback_parameters=contract['feedback_parameters'],geometry_code={f:sha(ROOT/f) for f in ['src/v4/roadshow_geometry_baseline.py','work/contact-preview-lift-20260924-v1/lift_feedback.py','src/v4/contact_geometric_feedback.py','work/contact-preview-lift-20260924-v1/continuous_preview.py']})
self_check();result.update(status='cpu_preparation_passed',remaining_seconds=budget.remaining())
(WORK/'preparation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print('PREPARED',result['selected_manual']['run'],result['remaining_seconds'])
