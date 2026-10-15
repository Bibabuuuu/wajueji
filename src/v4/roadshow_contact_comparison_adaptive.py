"""Bounded roadshow A entry on original D154 runtime with isolated preview.

0/120 preparation gate; 900-frame window requires the passed preparation
receipt. A 900-frame window is not a complete scoop or task.
"""
from pathlib import Path
import argparse,ast,hashlib,json,sys,subprocess,datetime,platform

ROOT=Path('/home/qunwang/wajueji')
WORK=ROOT/'work/contact-multisource-lift-20260924-v1'
BASE=WORK/'run_candidate.py'
BASE_SHA='63a84592718a68d60bad7ee6d518193e67b8c0c78e48bb9775f19c00469c949a'
LIFT=ROOT/'work/contact-preview-lift-20260924-v1/lift_feedback.py'
LIFT_SHA='a127962c1595565c38bde9d8b4747ec336866502aa796db0ed2eeed294227b83'
ADAPTIVE=ROOT/'src/v4/roadshow_adaptive_lift_feedback.py'
ADAPTIVE_SHA='4ac2f4ec3410fb29adaabf147befb8cc86ac9f97bf928d2cd0f5d325cb361d90'
TOOLS=Path('/home/qunwang/roadshow-tools-20261011/experiment-tools')
GEOMETRY=ROOT/'src/v4/roadshow_geometry_baseline.py'
SOURCE=ROOT/'runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz'
POLICY=ROOT/'runs/contact-phase-bc-pilot-v1/pilot.pt'
GEOMETRY_900=ROOT/'runs/roadshow-measurements-20261011/paired-driver580178-20261012/geometry/physics'
GEOMETRY_900_ACTIONS_SHA='e71287d9b41ab5cbfb5c555cef4fa90b8a598082def783bf2c172cf918a532ce'
GEOMETRY_900_FINAL_SHA='a494e0c9de89d9a2189cf964cbd387c48c3570588efc641b19e16b5526cec72d'
TRAIN='/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare_source(original,mode,frames=120):
    allowed='(0,120,900)'
    wall_budget=360 if frames==900 else 180
    edits=[('if args.frames not in (0,3000):',f'if args.frames not in {allowed}:'),('wall_budget=900',f'wall_budget={wall_budget}')]
    if mode=='geometry':
        edits += [('from candidate_multisource import LiftFeedback as ContactGeometryFeedback,save_control',
                   'from candidate_multisource import save_control\nfrom roadshow_adaptive_lift_feedback import LiftFeedback\nfrom roadshow_geometry_baseline import geometric_feedback_class,NoPolicyWorker\nContactGeometryFeedback=geometric_feedback_class(LiftFeedback)\nInferenceWorker=NoPolicyWorker'),
                  ("source='geometry_feedback_then_guarded_phase_bc'","source='geometric_feedback_without_policy'")]
    else:
        edits += [('from candidate_multisource import LiftFeedback as ContactGeometryFeedback,save_control',
                   'from candidate_multisource import save_control\nfrom roadshow_adaptive_lift_feedback import LiftFeedback as ContactGeometryFeedback')]
    for old,new in edits:
        if original.count(old)!=1:raise ValueError('comparison_source_anchor_changed:'+old)
        original=original.replace(old,new)
    # Existing collection hooks only; do not change step, guard or phase logic.
    from desktop_contact_pilot import annotated
    return annotated(original)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['geometry','bc'],required=True)
    p.add_argument('--check',action='store_true');p.add_argument('--frames',type=int,choices=[0,120,900],default=0)
    p.add_argument('--capture-display',action='store_true')
    p.add_argument('--output',type=Path);p.add_argument('--metrics',type=Path)
    args=p.parse_args()
    sys.path[:0]=[str(ROOT),str(ROOT/'src/v4'),str(TOOLS),str(WORK),str(ROOT/'work/night-dual-start-preparation-20260925-v2')]
    assert sha(BASE)==BASE_SHA and sha(LIFT)==LIFT_SHA and sha(ADAPTIVE)==ADAPTIVE_SHA
    contract=json.loads((WORK/'first-protocol.json').read_text())
    for item in contract['inputs']:
        if sha(item['path'])!=item['sha256']:raise ValueError('comparison_input_changed:'+item['path'])
    if sha(SOURCE)!=contract['source_sha256'] or sha(POLICY)!=contract['policy_sha256']:raise ValueError('comparison_initial_or_policy_changed')
    if args.frames>=900:
        receipt=json.loads((ROOT/'runs/roadshow-measurements-20261011/geometry-adaptive-120/integration-review.json').read_text())
        assert receipt['status']=='passed' and receipt['shared_prefix_actual_controls_equal']
        prior=json.loads((ROOT/'runs/roadshow-measurements-20261011/geometry-adaptive-120/metrics/comparison-binding.json').read_text())
        assert prior['geometric_intent_sha256']==sha(GEOMETRY) and prior['base_sha256']==sha(BASE) and prior['adaptive_preview_sha256']==sha(ADAPTIVE)
    code=prepare_source(BASE.read_text(),args.mode,args.frames);compile(code,str(BASE),'exec')
    if args.check:
        from roadshow_geometry_baseline import self_check
        self_check();print('comparison source checks passed; mode='+args.mode+'; physics_steps=0');return
    if args.output is None or args.metrics is None:raise ValueError('new_output_and_metrics_required')
    if args.output.exists() or args.metrics.exists():raise ValueError('immutable_output_already_exists')
    environment={'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'nvidia_kernel':Path('/proc/driver/nvidia/version').read_text().splitlines()[0],
        'gpu_driver':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip(),
        'python':sys.version,'machine':platform.machine(),
        'sampling_module_sha256':sha(TOOLS/'desktop_contact_pilot.py'),'collector_sha256':sha(TOOLS/'collector.py')}
    sys.path[:0]=[str(ROOT),str(ROOT/'src/v4'),str(WORK),str(ROOT/'work/night-dual-start-preparation-20260925-v2')]
    from desktop_contact_pilot import Logging
    class WindowLogging(Logging):
        def __init__(self,directory):
            super().__init__(directory)
        def begin(self,frame):
            self.request=self.rec.begin(str(frame),batch=args.mode+'-contact-window',warmup=frame<=50)
        def initialize(self,r):
            import numpy as np
            from audit_layered_47_loaded_handoff import cavity_mask
            self.np=np;self.mask=cavity_mask
            cfg=json.loads((ROOT/'configs/v4/live-bucket-fill-local-v1.json').read_text())
            self.contract=json.loads(Path(cfg['contract']).read_text())
            with np.load(cfg['front_rim']) as data:self.front=data['front'].copy()
            self.capacity=json.loads(Path(cfg['calibration']).read_text())['capacity_liters']
            self.mass=r.fixture.model.particle_mass.numpy().copy()
            self.rec.event('measurement_scope',time_s=float(r.arrays['simulation_time_s'][0]),cycle='pilot',
                scope=str(args.frames)+'-frame '+args.mode+' development window; no complete scoop/task or formal volume qualification',
                latency_start='current proprioception/terrain collection',latency_end='software command submission; no acceptance acknowledgment',
                terrain_refresh_frames=3,reference_capacity_only=True,receiver_enabled=r.truck_enabled)
    logger=WindowLogging(args.metrics)
    binding={'mode':args.mode,'runner_sha256':sha(__file__),'base_sha256':sha(BASE),'lift_sha256':sha(LIFT),
             'adaptive_preview_sha256':sha(ADAPTIVE),
             'geometric_intent_sha256':sha(GEOMETRY),
             'source_sha256':sha(SOURCE),'candidate_policy_sha256':sha(POLICY),'learned_weights_used':args.mode=='bc',
             'frames_requested':args.frames,'scope':'900-frame adaptive preview probe' if args.frames==900 else '0/120-frame adaptive preview preparation',
             'physics_steps_completed':None,'not_formal_task_comparison':True,
             'first_120_preparation_reused':args.frames>=900}
    (args.metrics/'comparison-binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    (args.metrics/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    sys.argv=[str(BASE),'--source',str(SOURCE),'--policy',str(POLICY),'--policy-sha256',sha(POLICY),
              '--output',str(args.output),'--frames',str(args.frames),'--training-python',TRAIN,
              '--coupling-hz','60','--phase-contract',str(WORK/'first-protocol.json')]
    if args.capture_display:sys.argv.append('--capture-display')
    try:exec(compile(code,str(BASE),'exec'),{'__name__':'__main__','__file__':str(BASE),'roadshow':logger})
    finally:
        logger.rec.close()
        report_path=args.output/'report.json'
        if report_path.exists():
            report=json.loads(report_path.read_text())
            report['comparison_controller']=binding
            report['policy_used_for_control']=args.mode=='bc'
            if args.mode=='geometry':
                report['candidate_policy_reference']=report.pop('policy')
                report['policy']={'used':False,'controller':'position feedback, no learned weights'}
                report['baseline_placeholder_local_seconds_quantiles']=report.pop('inference_roundtrip_seconds_quantiles',[])
                display_path=args.output/'display/reconstruction.json'
                if display_path.exists():
                    display=json.loads(display_path.read_text())
                    display['label']='GEOMETRY CONTACT WINDOW REPLAY'
                    display['capture_origin']='live_geometry_rollout'
                    display_path.write_text(json.dumps(display,indent=2)+'\n')
                    report['display_cache']=display
            report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
