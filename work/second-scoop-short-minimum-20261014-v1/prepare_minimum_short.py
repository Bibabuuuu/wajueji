"""CPU admission and existing-plan update for the newly authorized minimum run."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,shutil,py_compile
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-short-minimum-20261014-v1';sys.path.insert(0,str(W))
import run_minimum_short as m
from types import SimpleNamespace
import numpy as np
b=m.Budget();checks={}
def check(n,v):checks[n]=bool(v);assert v,n
check('new45min_budget_open',b.remaining()>420)
py_compile.compile(str(W/'run_minimum_short.py'),doraise=True)
base=m.read(m.OLD_SHORT/'record-final-receipt.json')
for n,h in base.items():check('record_unchanged_'+n,m.sha(R/n)==h['after'])
cpu=m.read(m.C/'cpu-v1/report.json')
check('28_CPU_candidate_checks_pass',cpu['status']=='cpu_candidate_passed' and cpu['check_count']==28 and all(cpu['checks'].values()))
for n,h in cpu['code_sha256'].items():check('tested_candidate_sha_'+n,m.sha(m.C/n)==h)
check('old_failed_budget_remains_closed',(m.OLD_SHORT/'closure.json').is_file())
r=m.CpuRuntime(m.SOURCE2,W/'preflight-runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
r.bridge=SimpleNamespace(state={'mechanism_qpos':r.arrays['mechanism_qpos']})
with np.load(m.A/'cpu-v4/terrain.npz') as f:t={k:f[k] for k in f.files}
with np.load(m.A/'cpu-v4/next-prepoint-plan.npz') as f:qs=f['qpos'].copy()
off=0
for s in m.read(m.A/'cpu-v4/report.json')['next_prepoint']['segments']:
 n=s['frames'];r.bridge.state['mechanism_qpos']=qs[off];c=m.LiveClearance(r,t,rise=s['name'].startswith('RETURN_RISE'))
 for i in sorted({off,off+n,*range(off,off+n+1,120)}):c.check(qs[i],physical=True)
 check('live_CPU_'+s['name'],True);off+=n
e=m.PhysicalYawTransferEnv.__new__(m.PhysicalYawTransferEnv);e.runtime=None
try:e.step(np.zeros(5))
except ValueError:check('policy_unbound_rejected',True)
else:raise AssertionError('unbound accepted')
e.runtime=SimpleNamespace(core=m.BoundRuntime.__new__(m.BoundRuntime),frames=120);e.budget=b
try:e.step(np.zeros(5))
except ValueError as exc:check('policy120_cap','120_residual' in str(exc))
else:raise AssertionError('cap bypassed')
raw=m.BoundRuntime.__new__(m.BoundRuntime);raw.budget=b;raw.frames=0;raw.step_authorized=False
try:raw.step(np.zeros(4))
except ValueError as exc:check('native_zero_gate_enforced','zero_restore' in str(exc))
else:raise AssertionError('no zero gate')
check('source_qualified_original_return',r.metadata['stage']=='returned' and r.metadata['cycle_index']==1 and not r.metadata.get('training_reset'))
check('frozen_model_unchanged',m.sha(m.MODEL)==m.MODEL_SHA)
check('planned_frames_within_scope',4198+81+8*60+3+120<=5000)
code=dict(m.read(m.OLD_SHORT/'protocol.json')['code_sha256'])
for n,h in code.items():assert m.sha(n)==h,n
for p in [W/'run_minimum_short.py',W/'prepare_minimum_short.py',m.C/'minimum_guard.py',m.C/'check_minimum_guard.py']:
 code[str(p)]=m.sha(p)
m.write(W/'preflight.json',dict(status='passed',check_count=len(checks),checks=checks,physics_steps=0,native_initializations=0,training_updates=0))
m.write(W/'protocol.json',dict(code_sha256=code,budget_path=str(b.path),budget_sha256=m.sha(b.path),source_sha256=m.sha(m.SOURCE2),source_metadata_sha256=m.sha(str(m.SOURCE2)+'.json'),
 minimum_guard_cpu_sha256=m.sha(m.C/'cpu-v1/report.json'),preflight_sha256=m.sha(W/'preflight.json'),prior_failed_report_sha256=m.sha(m.OUT.parent/'second-scoop-short-20261014-v1/report.json')))
budget=b.data
note=f'''## 2026-10-14 最低点版本45分钟短检：已授权，启动前CPU检查通过

用户新增45分钟仅验证最低点版本的原生恢复、返回上抬、第二铲接近及最多120帧冻结残差；含接入、保存与审计，不训练、不跑完整第二铲，失败/到期即保存停止。预算{budget['started_at']}至{budget['deadline_at']}，2700秒、5000总帧上限，最后420秒保留保存审计；旧预算封存。当前{len(checks)}项启动前CPU检查通过，尚无本版本物理通过结论。

复用原返回源6fe40910…、冻结seed44模型、原控制器和物理；仅接入用户确认的最低点高度门，1微米容差、继承间距基线、32厘米接近和其他原保护保持。旧371帧失败不改判。设备cuda:0物理/CPU冻结预测；命令`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python -u work/second-scoop-short-minimum-20261014-v1/run_minimum_short.py`，预计物理20–30分钟。日志work/second-scoop-short-minimum-20261014-v1/console.log，产物runs/second-scoop-short-minimum-20261014-v1/。每步预算检查；失败保存recovery并退出，不自动从失败源续跑。阶段末保存和原生控制日志用于恢复，未审计前不授予续跑资格。恢复24数组/原生控制/扫掠/时钟门先通过；同一续接实例不重置进入接近/残差。新增记录决策前世界力矩以供CPU坐标审计，不改变物理或输入。

'''
backup=W/'record-before-start';backup.mkdir(exist_ok=False)
texts={n:(R/n).read_text() for n in base}
for n in texts:
 dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/n,dst)
n='docs/plans/S05-imitation-and-rl.md';pos=texts[n].index('## ');texts[n]=texts[n][:pos]+note+texts[n][pos:]
brief=f'用户已新增45分钟，{budget["started_at"]}至{budget["deadline_at"]}，本次仅最低点版本恢复/上抬/第二铲接近和最多120帧残差；最后7分钟保存审计，不训练。{len(checks)}项CPU预检通过，物理结果未验证。旧371帧失败及封存预算保持。运行目录work/second-scoop-short-minimum-20261014-v1，产物runs/second-scoop-short-minimum-20261014-v1。范围和命令由S05维护。'
for n in ['docs/CURRENT_STATUS.md','docs/PROJECT_PLAN.md','docs/plans/INDEX.md']:
 s=texts[n];a=s.index('## ');z=s.index('\n## ',a+3);texts[n]=s[:a]+'## 2026-10-14 当前：最低点版本45分钟恢复/接近/残差短检已授权\n\n'+brief+'\n'+s[z:]
for n,h in base.items():assert m.sha(R/n)==h['after'],n
for n,s in texts.items():(R/n).write_text(s)
m.write(W/'record-start-receipt.json',{n:dict(before=h['after'],after=m.sha(R/n)) for n,h in base.items()})
print('Minimum run CPU admission passed',len(checks),'remaining seconds',round(b.remaining()),flush=True)
