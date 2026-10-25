"""CPU-only admission and existing-record update for the user-authorized scope."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib,shutil,py_compile
ROOT=Path('/home/qunwang/wajueji');W=ROOT/'work/second-scoop-short-20261014-v1';sys.path.insert(0,str(W))
import run_short as m
from types import SimpleNamespace
import numpy as np
checks={}
def check(n,v):checks[n]=bool(v);assert v,n
b=m.Budget();check('new45min_budget_open',b.remaining()>420)
py_compile.compile(str(W/'run_short.py'),doraise=True)
base=m.read(m.A/'adapter-record-update-receipt.json')
for n,h in base.items():check('record_unchanged_'+n,m.sha(ROOT/n)==h['after'])
r=m.CpuRuntime(m.SOURCE2,W/'preflight-runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
r.bridge=SimpleNamespace(state={'mechanism_qpos':r.arrays['mechanism_qpos']})
with np.load(m.A/'cpu-v4/terrain.npz') as f:t={k:f[k] for k in f.files}
with np.load(m.A/'cpu-v4/next-prepoint-plan.npz') as f:q=f['qpos'].copy()
p=m.read(m.A/'cpu-v4/report.json');off=0
for s in p['next_prepoint']['segments']:
 n=s['frames'];r.bridge.state['mechanism_qpos']=q[off]
 c=m.LiveClearance(r,t,rise=s['name'].startswith('RETURN_RISE'))
 for i in sorted({off,off+n,*range(off,off+n+1,120)}):c.check(q[i],physical=True)
 check('live_checker_CPU_'+s['name'],True);off+=n
candidate=m.PhysicalYawTransferEnv.__new__(m.PhysicalYawTransferEnv)
candidate.runtime=None
try:candidate.step(np.zeros(5))
except ValueError:check('physical_adapter_rejects_unbound',True)
else:raise AssertionError('unbound accepted')
candidate.runtime=SimpleNamespace(core=m.BoundRuntime.__new__(m.BoundRuntime),frames=120);candidate.budget=b
try:candidate.step(np.zeros(5))
except ValueError as e:check('physical_adapter_rejects_above120','120_residual' in str(e))
else:raise AssertionError('frame limit accepted')
raw=m.BoundRuntime.__new__(m.BoundRuntime);raw.budget=b;raw.frames=0;raw.step_authorized=False
try:raw.step(np.zeros(4))
except ValueError as e:check('runtime_rejects_without_zero_gate','zero_restore' in str(e))
else:raise AssertionError('missing zero gate accepted')
check('nominal_path_with_hold_within5000',4198+81+8*60+3+120<=5000)
check('source_stage_and_no_reset',r.metadata['stage']=='returned' and r.metadata['cycle_index']==1 and not r.metadata.get('training_reset'))
old=m.read(ROOT/'work/continuous-single-scoop-20261014-v1/protocol.json')['code_sha256']
for n,h in old.items():assert m.sha(n)==h,n
adapter=m.read(m.A/'adapter-independent-audit.json')
for n,h in adapter['code_sha256'].items():assert m.sha(m.A/n)==h,n
code=dict(old)
for p in [W/'run_short.py',W/'prepare_short.py',m.A/'yaw_transfer.py',m.A/'next_prepoint_candidate.py']:
 code[str(p)]=m.sha(p)
check('frozen_model_unchanged',m.sha(m.MODEL)==m.MODEL_SHA)
m.write(W/'preflight.json',dict(status='passed',check_count=len(checks),checks=checks,physics_steps=0,native_initializations=0,training_updates=0))
m.write(W/'protocol.json',dict(code_sha256=code,budget_path=str(b.path),budget_sha256=m.sha(b.path),source_sha256=m.sha(m.SOURCE2),source_metadata_sha256=m.sha(str(m.SOURCE2)+'.json'),
 source_audit_sha256=m.sha(m.SOURCE2.parent/'read-only-audit.json'),adapter_cpu_sha256=m.sha(m.A/'adapter-cpu-v3/report.json'),adapter_audit_sha256=m.sha(m.A/'adapter-independent-audit.json'),preflight_sha256=m.sha(W/'preflight.json')))
budget=b.data
note=f"""
### 新增45分钟物理短检：已授权实施

用户选择新增45分钟，仅原生恢复、返回上抬、第二铲接近及最多120帧冻结残差短检；不训练、不跑完整第二铲，旧预算不重开。预算自{budget['started_at']}至{budget['deadline_at']}，总2700秒，含准备/物理/保存/审计，最后420秒仅保存和CPU审计；全局最多5000帧，残差最多120帧。恢复绑定6fe40910…真实首铲返回末态，24数组/原生控制/扫掠/时钟严格核验；同一新原生续接实例内接近和残差不重置，不能称为从初始场景连续三铲。

设备台式机cuda:0物理、CPU冻结预测；命令`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python -u work/second-scoop-short-20261014-v1/run_short.py`。预计物理20–30分钟、加准备/保存/审计不超过45分钟；日志`work/second-scoop-short-20261014-v1/console.log`，产物`runs/second-scoop-short-20261014-v1/`。每步检查墙钟和帧上限，失败或保留时间到即保存recovery/周期阶段末态并停止；中断只保留诊断，不自动从失败源续跑。新入口在原CPU适配器外显式绑定此预算及零步门，原CPU类仍拒绝物理。

CPU预检{len(checks)}项通过，无新物理/训练。复用旧控制器、力/速度路由、材料/扫掠保护；RETURN延长保留静态保护并要求继承土堆间距不减/斗体不下降，APPROACH仍32厘米。固定路线细CAD检查及机构—土堆2厘米不放宽；新末态的黏附允许，不以汇总接触计数要求铲底零接触。当前为已授权待物理结果，不能写成短检已通过。
"""
backup=W/'record-before-short';backup.mkdir(exist_ok=False)
for n,h in base.items():
 dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dst)
texts={n:(ROOT/n).read_text() for n in base}
n='docs/plans/S05-imitation-and-rl.md';needle='### 本轮A观测转换：已实现、CPU已验证';assert needle in texts[n];texts[n]=texts[n].replace(needle,note+'\n'+needle,1)
n='docs/CURRENT_STATUS.md';head='## 2026-10-14 当前：第二铲CPU准备完成，观测转换CPU已验证，待新增物理预算';assert head in texts[n];texts[n]=texts[n].replace(head,'## 2026-10-14 当前：第二铲45分钟恢复/接近/残差短检已授权',1)
line=f"\n用户已新增45分钟，{budget['started_at']}至{budget['deadline_at']}，仅恢复、返回上抬/接近及最多120帧残差，最后7分钟预留保存审计；不训练、不跑完整第二铲。正在准备/执行，结果未验证。W=work/second-scoop-short-20261014-v1，R=runs/second-scoop-short-20261014-v1。旧待预算描述仅为上一CPU阶段历史，当前授权以此为准。\n"
pos=texts[n].index('\n',texts[n].index('## '));texts[n]=texts[n][:pos+1]+line+texts[n][pos+1:]
for n in ['docs/PROJECT_PLAN.md','docs/plans/INDEX.md']:
 pos=texts[n].index('\n',texts[n].index('## '));texts[n]=texts[n][:pos+1]+line+texts[n][pos+1:]
for n,h in base.items():assert m.sha(ROOT/n)==h['after'],n
for n,s in texts.items():(ROOT/n).write_text(s)
m.write(W/'record-start-receipt.json',{n:dict(before=h['after'],after=m.sha(ROOT/n)) for n,h in base.items()})
print('CPU preflight passed',len(checks),'budget remaining',round(b.remaining()),flush=True)
