from pathlib import Path
import sys,time,ast
R=Path('/home/qunwang/wajueji');W=R/'work/edge-first-two-scoop-20261015-v1';sys.path.insert(0,str(W));import run_edge_pair_attempt2 as m
b=m.Budget();b.check();old=m.read(W/'live-protocol.json');prior=R/'runs/edge-first-two-scoop-20261015-v1';audit=m.read(prior/'read-only-audit.json');cpu=m.read(W/'cpu-direct-bridge-v2/report.json')
assert audit['record_integrity_passed'] and audit['run_status']=='failed' and audit['frames']==4654
assert cpu['status']=='cpu_direct_bridge_passed' and all(cpu['checks'].values())
assert m.OUT==R/'runs/edge-first-two-scoop-20261015-attempt2'
fake=m.BoundRuntime.__new__(m.BoundRuntime);fake.budget=b;fake.frames=30000-4654;fake.step_authorized=True;fake.prefix_verified=True
try:fake.step(m.np.zeros(4))
except ValueError as e:assert str(e)=='whole_pair_frame_ceiling'
else:raise AssertionError('cumulative_frame_limit_failed')
bindings=dict(old['bindings_sha256'])
for p in [W/'run_edge_pair_attempt2.py',W/'direct_reference_bridge.py',W/'inplace_cache.py',W/'cpu-direct-bridge-v2/report.json',prior/'report.json',prior/'read-only-audit.json']:
 bindings[str(p)]=m.sha(p)
assert all(m.sha(k)==v for k,v in bindings.items())
protocol=dict(old,status='attempt2_direct_bridge_cpu_admitted_same_budget',bindings_sha256=bindings,
 prior_attempt=dict(run=str(prior),frames=4654,status='failed',reason='actual32cm approach gate; no digging or fill value',immutable_report_sha256=m.sha(prior/'report.json')),
 repair_scope='Routine geometric endpoint repair within authorized2h implementation/verification scope. Earlier geometry and all policy/physical/protection interfaces unchanged. Old failure not reclassified. Fresh original untouched source only; never failure recovery.',
 cache_scope='In-place trim of own display output avoids double disk allocation;4 shape/data equality cases passed. No physical/source particle values are written.',
 cumulative_max_frames=30000,attempt2_max_frames=25346,attempts_max=2,no_further_physical_restart=True,
 budget_path=str(b.path),budget_sha256=m.sha(b.path),budget_deadline_unchanged=b.data['epoch_deadline'],
 launch='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python -u work/edge-first-two-scoop-20261015-v1/run_edge_pair_attempt2.py',output=str(m.OUT),log=str(W/'console-attempt2.log'))
m.write(W/'live-protocol-attempt2.json',protocol)
receipt=m.read(W/'record-start-receipt.json');updated={}
note='\n\nD-190接入内修复：首实例在4654帧触发原32cm接近门（实际0.319909283m），没有进入挖掘；原失败与14GB缓存保留，独立记录审计通过。根因是最后名义预备点仅0.107mm裕度，实际跟踪偏差使其越门。CPU验证把下降末点直接接到已确认参考＋1mm动臂缸交接点，前缀几何/模型/命令/所有保护不变，下降段最小间距0.327489215m；四项缓存原地截尾检查验证已用数据字节不变，避免额外完整复制。按夜间授权的常规接入/修复验证，在原2小时截止内只安排一次新实例，从原未开挖源严格零步/首3步重验，不从失败源续跑；两实例物理帧累计最多30000，第二实例最多25346，之后任何失败不再重试。预算未延长，旧预算未使用。当前第二实例尚未运行/验证。\n'
for name,x in receipt.items():
 p=R/name;assert m.sha(p)==x['after'],('record_conflict',name);before=m.sha(p);text=p.read_text();pos=text.index('\n\n',text.index('D-190'))
 # Append to the current D190 section before the next heading.
 end=text.find('\n## ',pos+2)
 if end<0:end=len(text)
 text=text[:end]+note+text[end:];p.write_text(text);updated[name]=dict(before=before,after=m.sha(p))
m.write(W/'record-attempt2-receipt.json',updated)
m.write(W/'preflight-attempt2.json',dict(status='passed',old_failure_audited=True,cpu_direct_bridge=True,cumulative_frame_ceiling_test=True,all_frozen_bindings_unchanged=True,original_deadline_unchanged=True,physical_steps=0,remaining_seconds=b.remaining()))
print('ATTEMPT2_PREFLIGHT_PASSED remaining',round(b.remaining()),flush=True)
