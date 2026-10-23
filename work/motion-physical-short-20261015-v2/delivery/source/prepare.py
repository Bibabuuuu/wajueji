from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,time,numpy as np
R=Path('/home/qunwang/wajueji');W=R/'work/motion-physical-short-20261015-v2';A=R/'runs/edge-first-two-scoop-20261015-continuation';C=R/'work/motion-review-20261015-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert read(R/'work/motion-physical-short-20261015-v1/closure.json')['physics_steps']==0
assert read(C/'audit-v1/report.json')['status']=='cpu_candidate_audit_passed'
t=time.time();m=time.monotonic();budget=dict(purpose='D196_motion_short_pair',seconds=1800,epoch_start=t,epoch_deadline=t+1800,monotonic_deadline=m+1800,reserve_seconds=300,max_loaded_frames=1000,max_return_frames=3300,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),training_updates_allowed=0,authorization='User: 你继续推进，启动仿真; same previously confirmed30min/two-short-case bounds; old budget stays closed')
(W/'budget.json').write_text(json.dumps(budget,ensure_ascii=False,indent=2))
states=read(A/'display/states.json');cloud=np.load(A/'display/particles.npy',mmap_mode='r');dispatch={x['global_frame']:x['command'] for x in map(json.loads,(A/'dispatch.ndjson').open())}
cases=[]
for name,source,index,cap in [('loaded',A/'policy-1/episode-0001/endpoint.npz',3860,1000),('return',A/'dump.npz',6549,3300)]:
 out=W/(name+'-prefix');out.mkdir();metadata=read(str(source)+'.json')
 with np.load(source) as z:
  assert np.array_equal(z['particle_q'],cloud[index]) and np.array_equal(z['mechanism_qpos'],states[index]['mechanism_qpos'])
 assert metadata['stage'] in ['hold','transport','unloaded'] and metadata['truck_enabled'] and not metadata.get('training_reset')
 np.save(out/'particles.npy',cloud[index:index+4]);(out/'states.json').write_text(json.dumps(states[index:index+4]));commands=[dispatch[index+i] for i in [1,2,3]]
 cases.append(dict(name=name,source=str(source),source_sha256=sha(source),metadata_sha256=sha(str(source)+'.json'),original_run=str(A),original_frame=index,prefix=str(out),commands=commands,max_frames=cap))
bindings={str(p):sha(p) for p in [W/'run.py',C/'motion_candidate.py',C/'return-candidate-v3/report.json',C/'return-candidate-v3/return-plan.npz',C/'candidate-v1/carry-reference.npz',C/'audit-v1/report.json']}
bindings.update(read(R/'work/edge-adjacent-59um-20261015-v1/live-protocol-continuation.json')['bindings_sha256'])
assert all(sha(p)==h for p,h in bindings.items())
(W/'preflight.json').write_text(json.dumps(dict(status='preflight_passed',cases=cases,bindings_sha256=bindings,budget_sha256=sha(W/'budget.json')),ensure_ascii=False,indent=2))
zone=ZoneInfo('Asia/Shanghai');stamp=lambda v:datetime.fromtimestamp(v,zone).isoformat()
note='## 2026-10-15 D-196：两组短段物理已重新授权\n\n用户明确要求继续推进并启动仿真。沿用此前已确认的新增30分钟上限和两组范围，新预算'+stamp(t)+'至'+stamp(t+1800)+'；旧D195预算保持关闭。最后5分钟保存审计，两源各最多一次原生恢复与原连续首3帧核验；保斗缸举升/回转入口最多1000帧，卸料后合并接近最多3300帧；任一门失败则整轮停止，不自动重试，不训练或挖掘。源与CPU路径身份、前缀缓存及绑定已预检；尚未物理验证。代码work/motion-physical-short-20261015-v2/run.py，cuda:0，日志console.log，产物runs/motion-physical-short-20261015-v2/；物理期间按原截止检查并保存失败状态。\n\n'
for rel in ['docs/CURRENT_STATUS.md','docs/plans/S07-truck-reproduction-submission.md','docs/decisions.md']:
 p=R/rel;text=p.read_text();assert 'D-196' not in text;first,_,rest=text.partition('\n');text=first+'\n\n'+note+rest
 if rel.endswith('decisions.md'):text=text.replace('## 当前决策索引（2026-10-15）','## 当前决策索引（2026-10-15）\n\n- `D-196`：用户重新要求启动仿真；新30分钟，两源最多1000/3300帧，原生零步/首3帧门，任一失败整轮停止，旧预算不重开。',1)
 p.write_text(text)
print('PREFLIGHT_PASSED',stamp(t),stamp(t+1800),flush=True)
