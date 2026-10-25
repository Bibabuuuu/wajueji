"""Desktop-only 0/120-frame instrumentation of the SHA-frozen first BC loop.

Executes the original loop with its original __file__ and only explicit logging
insertions. No controller/physics/guard change, batch, training, or retry.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
from collector import Recorder

ROOT = Path('/home/qunwang/wajueji')
BASE = ROOT/'work/night-dual-start-preparation-20260925-v2/first_loop.py'
BASE_SHA = '41440e7aa412718ad38c4a0fdc84f52b4ef7c1b3385ab7261302bd421fc29fc6'
FIRST_CODE = ROOT/'work/contact-multisource-lift-20260924-v1'
SOURCE = ROOT/'runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz'
POLICY = ROOT/'runs/contact-phase-bc-pilot-v1/pilot.pt'
POLICY_SHA = 'b4bbc98cf472bcad23f89e014a025cfc933c36091ada91ced4c4cc9680d94095'
TRAIN = '/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def annotated(source):
    edits = [
      ('                initial=r.capture();exact=', '                roadshow.initialize(r)\n                initial=r.capture();exact='),
      ('                    before=manual.compact_state(r.bridge.state)', '                    roadshow.begin(frame)\n                    before=manual.compact_state(r.bridge.state)'),
      ('                    x,digest=features.update(structured);action,timing=worker.predict(x,digest)',
       "                    x,digest=features.update(structured)\n                    roadshow.ready(frame, anchor)\n                    action,timing=worker.predict(x,digest)"),
      ('                    event=dict(frame=frame,before_state=before', '                    roadshow.decision()\n                    event=dict(frame=frame,before_state=before'),
      ('                        row=r.step(applied)', '                        roadshow.send()\n                        row=r.step(applied)\n                        roadshow.sample(r, phase_before)'),
    ]
    for old,new in edits:
        if source.count(old)!=1: raise ValueError('instrumentation_anchor_changed:'+old)
        source=source.replace(old,new)
    compile(source,str(BASE),'exec')
    return source

class Logging:
    def __init__(self, directory):
        self.rec = Recorder(directory,end_boundary='cmd_send',material_scope='soil_only')
        self.request=None
    def initialize(self,r):
        import numpy as np
        from audit_layered_47_loaded_handoff import cavity_mask
        self.np=np; self.mask=cavity_mask
        cfg=json.loads((ROOT/'configs/v4/live-bucket-fill-local-v1.json').read_text())
        self.contract=json.loads(Path(cfg['contract']).read_text())
        with np.load(cfg['front_rim']) as data: self.front=data['front'].copy()
        self.capacity=json.loads(Path(cfg['calibration']).read_text())['capacity_liters']
        self.mass=r.fixture.model.particle_mass.numpy().copy()
        self.rec.event('measurement_scope',time_s=float(r.arrays['simulation_time_s'][0]),cycle='pilot',
            scope='120-frame integration only; no acquisition/transport/unload qualification',
            latency_start='start of current proprioception/terrain observation collection',
            latency_end='software command submission before original runtime.step; no acceptance acknowledgment',
            terrain_refresh_frames=3,reference_capacity_only=True,receiver_enabled=r.truck_enabled)
    def begin(self,frame):
        self.request=self.rec.begin(str(frame),batch='BC-contact-pilot',warmup=frame<=50)
    def ready(self,frame,anchor):
        self.rec.mark(self.request,'obs_ready')
        self.rec.event('terrain_age',time_s=(frame-1)/60,cycle='pilot',frame=frame,
            terrain_anchor_frame=anchor,age_sim_s=((frame-1)-anchor)/60,
            fresh_terrain=anchor==frame-1)
    def decision(self): self.rec.mark(self.request,'decision_end')
    def send(self): self.rec.mark(self.request,'cmd_send'); self.request=None
    def sample(self,r,stage):
        if r.frames%6: return
        start=time.perf_counter_ns();np=self.np
        q=r.fixture.state.particle_q.numpy()
        pose=r.assembly.assemble(r.bridge.state['mechanism_qpos'],include_meshes=False)[1]['bucket']
        bucket=self.contract['bucket'];local=(q-pose[:3,3])@pose[:3,:3]
        mask=self.mask(local,np.asarray(bucket['polygon_xz_m']),bucket['y_limits_m'],self.front)
        if mask.shape!=self.mass.shape or not np.isfinite(q).all():raise ValueError('particle_mass_identity')
        t=float(r.arrays['simulation_time_s'][0])+r.frames/60
        self.rec.inventory_sample(time_s=t,cycle='pilot',stage=stage,bucket_capacity_l=float(self.capacity),
            bucket_mass_kg=float(self.mass[mask].sum()),measurement_valid=False,
            invalid_reason='early contact pilot; no acquired cohort, receiver or support/volume qualification')
        self.rec.event('observer_cost',time_s=t,cycle='pilot',
            wall_ms=(time.perf_counter_ns()-start)/1e6,includes='GPU readback, CAD mask and log write')

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true')
    p.add_argument('--frames',type=int,choices=(0,120),default=0)
    p.add_argument('--output',type=Path);p.add_argument('--metrics',type=Path)
    args=p.parse_args();source=BASE.read_text()
    if sha(BASE)!=BASE_SHA:raise ValueError('base_loop_changed')
    protocol=json.loads((FIRST_CODE/'first-protocol.json').read_text())
    for item in protocol['inputs']:
        if sha(item['path'])!=item['sha256']:raise ValueError('protocol_input_changed:'+item['path'])
    if sha(SOURCE)!=protocol['source_sha256'] or sha(POLICY)!=POLICY_SHA:raise ValueError('input_identity')
    code=annotated(source)
    if args.check: print('STATIC_CHECK_PASSED; physics_steps=0');return
    if args.output is None or args.metrics is None:raise ValueError('new_output_and_metrics_required')
    if args.output.exists() or args.metrics.exists():raise ValueError('immutable_output_already_exists')
    sys.path[:0]=[str(ROOT),str(ROOT/'src/v4'),str(FIRST_CODE),str(ROOT/'work/night-dual-start-preparation-20260925-v2')]
    from sources import first_source
    first_source()
    logging=Logging(args.metrics)
    identity={'runner_sha256':sha(__file__),'base_sha256':sha(BASE),'source_sha256':sha(SOURCE),
      'policy_sha256':sha(POLICY),'frames_requested':args.frames,'collection_only':True,
      'not_algorithm_comparison':True,'physics_steps_completed':None}
    (args.metrics/'pilot-binding.json').write_text(json.dumps(identity,indent=2)+'\n')
    sys.argv=[str(BASE),'--source',str(SOURCE),'--policy',str(POLICY),'--policy-sha256',POLICY_SHA,
      '--output',str(args.output),'--frames',str(args.frames),'--training-python',TRAIN,
      '--coupling-hz','60','--phase-contract',str(FIRST_CODE/'first-protocol.json')]
    try:exec(compile(code,str(BASE),'exec'),{'__name__':'__main__','__file__':str(BASE),'roadshow':logging})
    finally:logging.rec.close()

if __name__=='__main__':main()
