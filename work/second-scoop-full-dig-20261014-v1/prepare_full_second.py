import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,shutil,py_compile
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-full-dig-20261014-v1';sys.path.insert(0,str(W))
import run_full_second as m
import numpy as np
from types import SimpleNamespace
from stable_baselines3 import SAC
b=m.Budget();checks={}
def check(n,v):checks[n]=bool(v);assert v,n
check('new30min_budget_open',b.remaining()>300)
py_compile.compile(str(W/'run_full_second.py'),doraise=True)
base=m.read(m.PRIOR/'record-final-receipt.json')
for n,h in base.items():check('record_unchanged_'+n,m.sha(R/n)==h['after'])
check('prior_short_budget_closed',(m.PRIOR/'closure.json').is_file())
audit=m.read(m.SOURCE2.parent/'read-only-audit.json');check('41_prior_checks_passed',audit['status']=='passed' and audit['physical_short_passed'] and all(audit['checks'].values()))
r=m.CpuRuntime(m.SOURCE2,W/'preflight-runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
check('correct_actual_prepoint_source',r.metadata['stage']=='prepoint' and r.metadata['cycle_index']==2 and not r.metadata.get('training_reset'))
check('source24_arrays_and_original_inventory',len(r.arrays)==24 and len(r.arrays['particle_q'])==256441 and len(r.records)==47)
check('NPZ_sidecar_control_identity',m.sha(m.SOURCE2)==b.data['source_sha256'] and m.sha(str(m.SOURCE2)+'.json')==b.data['source_metadata_sha256'] and m.sha(m.SOURCE2.parent/'reference_align.control.json')==b.data['source_control_sha256'])
e=m.FullYawEnv.__new__(m.FullYawEnv);e.runtime=None
try:e.step(np.zeros(5))
except ValueError:check('unbound_policy_rejected',True)
else:raise AssertionError('unbound policy accepted')
raw=m.BoundRuntime.__new__(m.BoundRuntime);raw.budget=b;raw.frames=0;raw.step_authorized=False;raw.prefix_verified=False
try:raw.step(np.zeros(4))
except ValueError as exc:check('zero_gate_enforced','zero_gate' in str(exc))
else:raise AssertionError('zero gate bypass')
raw.step_authorized=True;raw.frames=3
try:raw.step(np.zeros(4))
except ValueError as exc:check('prefix_gate_before4th_frame','prefix_gate' in str(exc))
else:raise AssertionError('prefix gate bypass')
raw.prefix_verified=True;raw.frames=1560
try:raw.step(np.zeros(4))
except ValueError as exc:check('1560_frame_cap','1560_frame' in str(exc))
else:raise AssertionError('frame cap bypass')
env=m.FullYawEnv(W/'preflight-reference',b,collect=False,waiting_action_mode='error_strength_v1',max_frames=1560,yaw_offset_rad=m.CANDIDATE['yaw_offset_rad'])
check('full_env_original221_5_1170_contract',env.observation_space.shape==(221,) and env.action_space.shape==(5,) and env.commands.shape==(1170,4) and env.max_frames==1560)
first=m.read(m.PRIOR/'aligned-source-candidate.json');check('source_checkpoint_frame_bound',first['actual_frame']==4762 and first['source_sha256']==m.sha(m.SOURCE2))
decision=__import__('json').loads((m.SOURCE2.parent/'policy-decisions.ndjson').read_text().splitlines()[0]);model=SAC.load(m.MODEL,device='cpu')
check('frozen_first_prediction_recomputes',np.array_equal(model.predict(np.asarray(decision['observation'],np.float32),deterministic=True)[0],decision['action']))
check('frozen_model_unchanged',m.sha(m.MODEL)==m.MODEL_SHA and model._n_updates==1000)
code=dict(m.read(m.PRIOR/'protocol.json')['code_sha256'])
for n,h in code.items():assert m.sha(n)==h,n
for f in [W/'run_full_second.py',W/'prepare_full_second.py']:code[str(f)]=m.sha(f)
inputs={str(f):m.sha(f) for f in [m.PRIOR/'aligned-source-candidate.json',m.PRIOR/'closure.json',m.SOURCE2,m.SOURCE2.parent/'reference_align.control.json',Path(str(m.SOURCE2)+'.json'),m.SOURCE2.parent/'read-only-audit.json',m.SOURCE2.parent/'policy-decisions.ndjson',m.SOURCE2.parent/'policy/episode-0001/actions.ndjson']}
m.write(W/'preflight.json',dict(status='passed',check_count=len(checks),checks=checks,physics_steps=0,native_initializations=0,training_updates=0,CPU_frozen_predictions=1))
m.write(W/'protocol.json',dict(code_sha256=code,input_sha256=inputs,budget_path=str(b.path),budget_sha256=m.sha(b.path),preflight_sha256=m.sha(W/'preflight.json'),prefix_rule='initial221 observation and first action exact; first3 actual particle positions, compact states and applied commands exactly equal saved prefix; reject before frame4 if not passed'))
note=f'''## 2026-10-14 对齐快照恢复与完整第二铲：新增30分钟已授权

用户明确选择新增30分钟，从本轮已审计挖掘前对齐快照f31c6497…恢复；先严格零步24数组/原生控制/扫掠/时钟门，再比对初始221维输入/冻结动作、首3帧实际粒子/compact state/发令与已保存SAC前缀完全一致，通过后在同一新实例继续1170参考及30帧保载。前缀帧计入总残差帧，不重复/重置。最多1560物理帧，不训练、不运输卸料，失败或到期保存停止。终点允许黏附，记录新点装载，不承诺改善。

预算{b.data['started_at']}至{b.data['deadline_at']}，1800秒含接入/CPU预检/物理/保存/独立审计，最后300秒仅保存审计，旧预算关闭。启动前{len(checks)}项CPU检查通过，物理恢复/完整挖掘尚未验证。设备cuda:0物理/CPU冻结预测；命令`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python -u work/second-scoop-full-dig-20261014-v1/run_full_second.py`，预计5–10分钟物理；日志work/second-scoop-full-dig-20261014-v1/console.log，产物runs/second-scoop-full-dig-20261014-v1/。每步检查墙钟/帧上限，任一恢复/前缀/原保护/偏离/超时失败保存failure并退出，不从失败源自动续跑。原模型/控制/物理源码及奖励不改，完整第二铲不等于运输、第三铲或从初始场景同实例连续三铲。

'''
backup=W/'record-before-start';backup.mkdir(exist_ok=False);texts={n:(R/n).read_text() for n in base}
for n in texts:
 dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/n,dst)
n='docs/plans/S05-imitation-and-rl.md';i=texts[n].index('## ');texts[n]=texts[n][:i]+note+texts[n][i:]
brief=f'用户新增30分钟，{b.data["started_at"]}至{b.data["deadline_at"]}，仅对齐快照原生恢复/首3帧复现、完整第二铲挖掘及0.5秒保载；最多1560残差帧，最后5分钟保存审计，不训练/运输。{len(checks)}项CPU预检通过，物理结果尚未验证。旧短检4882帧/41项通过及全部封存预算保持。工作目录work/second-scoop-full-dig-20261014-v1，产物runs/second-scoop-full-dig-20261014-v1；详见S05。'
for n in ['docs/CURRENT_STATUS.md','docs/PROJECT_PLAN.md','docs/plans/INDEX.md']:
 s=texts[n];i=s.index('## ');j=s.index('\n## ',i+3);texts[n]=s[:i]+'## 2026-10-14 当前：完整第二铲30分钟验证已授权\n\n'+brief+'\n'+s[j:]
for n,h in base.items():assert m.sha(R/n)==h['after'],n
for n,s in texts.items():(R/n).write_text(s)
m.write(W/'record-start-receipt.json',{n:dict(before=h['after'],after=m.sha(R/n)) for n,h in base.items()})
print('Full second-dig CPU preflight passed',len(checks),'remaining',round(b.remaining()),flush=True)
