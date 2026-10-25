from pathlib import Path
import sys,time,ast,numpy as np
R=Path('/home/qunwang/wajueji');W=R/'work/edge-first-two-scoop-20261015-v1';sys.path.insert(0,str(W))
import run_edge_pair as m
b=m.Budget();b.check();cpu=m.read(W/'cpu-protocol.json');checks={}
checks['correct_new_work_scope']=m.W==W
checks['no_prior_closure'] = not (W/'closure.json').exists()
checks['cpu_bindings_unchanged']=all(m.sha(p)==h for p,h in cpu['bindings_sha256'].items())
old=m.read(R/'work/second-scoop-qerror-long-20261014-v1/protocol.json')
checks['reused_controller_and_physics_unchanged']=all(m.sha(p)==h for p,h in old['code_sha256'].items())
checks['new2h_and15min_reserve']=(b.data['seconds'],b.data['reserve_seconds'])==(7200,900)
checks['new_source_stage']=m.read(str(b.data['source'])+'.json')['stage']=='transport'
checks['source_not_relabelled_qualified']=m.read(Path(b.data['source']).parent/'safe_start.control.json')['continuation_qualified'] is False
checks['frozen_model_binding']=m.sha(m.MODEL)==m.MODEL_SHA
checks['original_contract']=m.require_environment_contract(m.read(m.RECEIPT),'error_strength_v1') is None
checks['geometry_and_adapter_cpu_passed']=m.read(R/'work/edge-first-strips-20261015-v1/reference-edge-v2/report.json')['status']=='cpu_rotation_geometry_passed'
fake=m.BoundRuntime.__new__(m.BoundRuntime);fake.budget=b;fake.frames=0;fake.step_authorized=False
try:fake.step(np.zeros(4))
except ValueError as e:checks['zero_gate_rejects_early_step']=str(e)=='native_zero_gate_required'
fake.step_authorized=True;fake.frames=3;fake.prefix_verified=False
try:fake.step(np.zeros(4))
except ValueError as e:checks['prefix_gate_rejects_frame4']=str(e)=='prefix3_gate_required'
fake.prefix_verified=True;fake.frames=30000
try:fake.step(np.zeros(4))
except ValueError as e:checks['whole_frame_ceiling']=str(e)=='whole_pair_frame_ceiling'
class Expired:
 data={'reserve_seconds':900}
 def remaining(self):return 900
try:m.Budget.check(Expired())
except TimeoutError:checks['reserve_stops_new_physics']=True
prior=R/'runs/continuous-single-scoop-20261014-v1'
rows=[m.json.loads(s) for s in (prior/'dispatch.ndjson').read_text().splitlines()[2813:2816]]
checks['saved_prefix_global_frame_binding']=[x['global_frame'] for x in rows]==[2814,2815,2816]
assert all(checks.values()),checks
bindings=dict(cpu['bindings_sha256']);bindings.update(old['code_sha256'])
for p in [W/'run_edge_pair.py',W/'preflight_edge_pair.py',W/'budget.json',Path(b.data['source']+'.json'),Path(b.data['source']).parent/'safe_start.control.json']:
 bindings[str(p)]=m.sha(p)
m.write(W/'live-protocol.json',dict(status='new2h_live_scope_preflight_passed',cpu_protocol_sha256=m.sha(W/'cpu-protocol.json'),bindings_sha256=bindings,
 action_mapping_version=m.WAIT_VERSION,scope=b.data,preflight_checks=checks,physical_steps=0,training_updates=0,
 launch='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python -u work/edge-first-two-scoop-20261015-v1/run_edge_pair.py',
 estimated_physical_wall_minutes='60-100 if all gates pass; not guaranteed within budget',
 log=str(W/'console.log'),output=str(m.OUT),interruption='Per-step budget/failure guards save recovery and terminate. Never resume failed state or restart automatically.'))
m.write(W/'preflight.json',dict(status='passed',checks=checks,passed=sum(checks.values()),physical_steps=0,training_updates=0,remaining_seconds=b.remaining()))
print('PREFLIGHT',sum(checks.values()),'remaining',round(b.remaining()),flush=True)
