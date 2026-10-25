from pathlib import Path
import sys,time,ast
R=Path('/home/qunwang/wajueji');W=R/'work/edge-first-two-scoop-20261015-v1';sys.path.insert(0,str(W));import run_prepoint_continuation as m
b=m.Budget();b.check();q=m.read(W/'qualified-prepoint-source.json');assert q['status']=='completed_phase_prepoint_restore_qualified' and q['physical_frames']==6 and all(q['replica_end24_exact'].values())
assert m.OUT.name=='edge-first-two-scoop-20261015-continuation'
old=m.read(W/'live-protocol-attempt2.json');bindings=dict(old['bindings_sha256'])
for p in [W/'run_prepoint_continuation.py',W/'qualified-prepoint-source.json',W/'qualify_prepoint.py',Path(q['source']),Path(q['source']+'.json'),Path(q['source_control']),W/'prepoint-qualification/replica-a/particles.npy',W/'prepoint-qualification/replica-a/states.json',Path(q['source_prior_run'])/'report.json']:
 bindings[str(p)]=m.sha(p)
assert all(m.sha(k)==v for k,v in bindings.items())
fake=m.BoundRuntime.__new__(m.BoundRuntime);fake.budget=b;fake.frames=20550;fake.step_authorized=True;fake.prefix_verified=True
try:fake.step(m.np.zeros(4))
except ValueError as e:assert str(e)=='whole_pair_frame_ceiling'
else:raise AssertionError('cumulative_frame_cap')
protocol=dict(old,status='qualified_completed_prepoint_continuation_admitted',bindings_sha256=bindings,source=q,
 prior_attempts=[dict(frames=4654,reason='original32cm approach stop'),dict(frames=4790,reason='logging read inactive policy fields; completed prepoint saved before exception; no dig'),dict(frames=6,reason='twin native zero-plus3 qualification')],
 cumulative_before_production=9450,cumulative_max_frames=30000,production_max_frames=20550,
 lifecycle_fix='Only log hold/cursor after policy has attached; no actions/progress/controller/physics change. Recovery uses completed-phase checkpoint equal to last cached state, not failed checkpoint.',
 continuity_scope='Both actual digs and intervening unload/return share one new native instance from qualified first prepoint. Earlier air approach is a prior saved phase, not claimed uninterrupted.',
 attempts_max='No further physical continuation or restart after this production instance',
 launch='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python -u work/edge-first-two-scoop-20261015-v1/run_prepoint_continuation.py',output=str(m.OUT),log=str(W/'console-continuation.log'),budget_deadline_unchanged=b.data['epoch_deadline'])
m.write(W/'live-protocol-continuation.json',protocol);m.write(W/'preflight-continuation.json',dict(status='passed',exact_completed_phase_source=True,twin_restore_first3=True,logging_lifecycle_guard_corrected=True,cumulative_frame_cap=True,deadline_unchanged=True,remaining_seconds=b.remaining(),new_physical_steps=0))
receipt=m.read(W/'record-attempt2-receipt.json');new={};note='\n\nD-190合格阶段续接：第二实例完成4790帧安全接近后，记录器在策略尚未attach时读取hold/cursor未初始化字段而停止；不是物理/走廊保护失败，首铲未开始。生命周期日志检查已修复，控制律和保护不变。选取刚成功完成、与最后缓存逐值一致的预备点快照dc6b1f23…，保留transport原stage及未资格标记，不用failure/recovery源。两个独立原生实例零步24数组/底层恢复严格通过，各3步零指令后24数组、4帧状态及粒子全等，共新增6帧；这不是与未保存原连续后缀的比较。按既有完整快照续接合同，从该合格阶段起让两个实际挖掘及中间运卸返回共用一个新实例；此前空中接近不冒称全程不中断。所有物理累计起点9450帧，生产实例最多20550帧，累计仍≤30000；截止和最后15分钟预留不变，不训练、不改保护、之后不再续接/重试。\n'
for name,v in receipt.items():
 p=R/name;assert m.sha(p)==v['after'];before=m.sha(p);text=p.read_text();start=text.index('D-190');end=text.find('\n## ',text.index('\n\n',start)+2);end=len(text) if end<0 else end;p.write_text(text[:end]+note+text[end:]);new[name]=dict(before=before,after=m.sha(p))
m.write(W/'record-continuation-receipt.json',new);print('CONTINUATION_PREFLIGHT_PASSED remaining',round(b.remaining()),flush=True)
