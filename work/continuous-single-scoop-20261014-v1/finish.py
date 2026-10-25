"""Freeze successful continuous-scoop evidence, close budget, update existing records."""
from pathlib import Path
import json,time,datetime,hashlib,subprocess,shutil,re,zipfile
ROOT=Path('/home/qunwang/wajueji');W=ROOT/'work/continuous-single-scoop-20261014-v1';RUN=ROOT/'runs/continuous-single-scoop-20261014-v1'
read=lambda p:json.loads(Path(p).read_text());write=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=read(RUN/'read-only-audit.json');report=read(RUN/'report.json');assert audit['status']=='passed' and all(audit['checks'].values()) and report['status']=='continuous_single_scoop_completed'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
bp=Path(str(RUN)+'-budget.json');b=read(bp);now=time.time();z=datetime.timezone(datetime.timedelta(hours=8));stamp=lambda t:datetime.datetime.fromtimestamp(t,z).isoformat()
closure=dict(status='closed_early' if now<b['epoch_deadline'] else 'closed',budget_sha256=sha(bp),start_cst=stamp(b['epoch_start']),fixed_deadline_cst=stamp(b['epoch_deadline']),closed_cst=stamp(now),
    physical_frames=report['frames'],neural_updates=0,native_initializations=1,audit_checks=audit['check_count'],unused_seconds_sealed=max(0,b['epoch_deadline']-now),old_budgets_reopened=False,gpu_compute_processes=[])
assert not Path(str(bp)+'.closed.json').exists();write(RUN/'closure.json',closure);write(str(bp)+'.closed.json',closure)
out=RUN/'summary';out.mkdir(exist_ok=False);score=audit['retained_fill_fraction']*100
brief=f'''## 2026-10-14 统一残差连续单铲：已实现并验证，本轮计算关闭

用户授权的三点已完成：统一入口/阶段计时/CPU交接检查、当前工况同实例连续单铲、完整日志和全帧缓存回放。一个CycleRuntime、一个原生物理实例，共{report['frames']}帧（{report['frames']/60:.4f}仿真秒）：合法预备点真实退至安全高位→自动接近→冻结SAC残差挖掘及保载→几何举升/回转/卸料→退出→安全高位返回。中间未关闭/重建物理、恢复旧末态、重置土石/机构/底层目标或物理时钟。首段真实准备用于兼容合法起点，不能宣称从更早旧格式初始点直接续跑。

{audit['check_count']}项独立CPU审计通过；全部几何发令复算最大绝对差{audit['geometry_max_abs_difference_si']:.3g}（数值容差1e-15），载荷偏置及{audit['policy_decisions']}次冻结模型预测精确一致，残差在全局第{audit['residual_start_global_frame']}帧进入、使用阶段内帧号，共{audit['residual_frames']}帧完成1170参考及30帧保载；短保载显示{score:.4f}%为本次未标定挖掘段诊断，旧直接预备点恢复的148.2321%及人工145.3813%显示均未保持，差异原因未查明；不扩写为性能复现、装车或泛化优势。每次交接24数组、原生控制/扫掠状态与时钟无改写，全局缓存和实际发令贯穿各阶段，原机械/非接触间距/跟踪/材料/防漏保护保留。0训练，原训练代码/模型及正式物理源码未改，未Git提交。

代理按自主授权设新增90分钟上限，{closure['start_cst']}至{closure['fixed_deadline_cst']}，实际{closure['closed_cst']}关闭，剩余{closure['unused_seconds_sealed']:.1f}秒封存；旧预算不重开。代码work/continuous-single-scoop-20261014-v1，证据runs/continuous-single-scoop-20261014-v1/read-only-audit.json、closure.json、display/及summary/README.md；Windows deliverables/continuous-single-scoop-20261014/。实际装载量归因按用户要求暂缓。本次仅当前固定铲位第一铲，不授予第二/第三铲、新目标泛化或正式比赛验收；决策D-182。

'''
table='\n'.join(f"| {p['name']} | {p['start_frame']} | {p['end_frame']} | {p['end_frame']-p['start_frame']} |" for p in report['phases'])
body=f'''# 冻结残差策略统一连续单铲

状态：统一入口已实现，当前工况连续单铲已验证，全帧回放已保存。

这次不是把挖掘、运卸和退出三个旧运行拼接：只有一次原生初始化，整段{report['frames']}帧都来自同一个物理实例和时间线。残差模型实际控制挖掘，后续几何控制从实时末态接管并安全返回高位。

## 执行路线

| 阶段 | 开始全局帧 | 结束全局帧 | 实际帧数 |
|---|---:|---:|---:|
{table}

合法输入是训练/评价使用的700f76be完整预备点；更早接近保存点使用旧格式。本次先真实退至高位，再自动接近原铲位，随后完整挖运卸返回；起点准备始终在同一物理实例，不手动摆姿态、不恢复料堆。最后返回安全高位，未开展下一铲。

## 模型与交接

冻结模型seed44-weight100，SHA256 `84aade9a3d8819368afb72223b8452ebfbc5e9ac9ad190818207d1142692aab1`，仍为原1000次离线SAC更新版本，本次0新更新。原221维观测/5维动作、20Hz决策、四轴及节奏、等待纠偏、模式切换和走廊语义保持。原训练源码与环境合同未改。

PhaseView只提供阶段内时钟和入口快照，始终借用同一活动环境。挖掘起于全局{audit['residual_start_global_frame']}帧，却从阶段内0开始，避免接近帧数污染模型时间特征和超时判断。仅初始化本阶段策略动作/观测历史；实际土石状态、机构速度、原生目标/积分/热启动和反力历史不改写。运输按实际姿态与原生目标初始化载荷偏置，保留已发指令连续性。每次交接有完整数组/原生控制/扫掠/帧号摘要的前后相等证据。

残差段实际{audit['residual_frames']}帧，{audit['policy_decisions']}次预测，完成1170参考和30帧保载；本次最低短保载显示{score:.4f}%。本次实际连续运行与旧直接恢复预备点评价148.2321%存在差异，也未保持旧人工基准145.3813%的显示值；差异来源尚未查明。本轮验证的是流程与控制交接，未证明旧装载表现能够原样迁移。该显示未标定，不宣称本轮装车量提升、泛化或官方95%成功率。实际装载量专题按用户要求降优先级。

## 验证与边界

{audit['check_count']}项独立审计全部通过：单实例、连续全局帧/时钟、无改写交接、真实模型预测、阶段内计时、全部几何发令/载荷偏置复算、实际控制分发一致、全硬门及材料/防漏、源/模型/代码/缓存身份、完整结束状态和回放覆盖。几何复算最大绝对差{audit['geometry_max_abs_difference_si']:.3g} SI；{audit['geometry_nonexact_commands']}条非逐位相等。原审计v1因浮点逐位比较失败已完整封存，v2用绝对容差1e-15、rtol=0逐条复核，实际发令与原生日志、载荷偏置及模型预测仍精确相等；物理保护阈值没有变化。审计0新增物理。CPU v1的闭链端点拒绝保留；v2复用既有被动投影求解后全路径通过，未放宽阈值。

只覆盖当前固定参考/铲位第一铲。后续铲位的人工参考适用性及自动重新选点接入尚待另验；完整三铲、正式响应/效率/成功率和真实堆积体积验收未授予。正式默认训练入口不变，新的统一运行入口在本交付源码目录。

## 完整回放和复核

台式机完整缓存：`/home/qunwang/wajueji/runs/continuous-single-scoop-20261014-v1/display/`，包含全局0至{report['frames']}的particles.npy、states.json、scene.json、phases.json和SHA元数据。replay.py只允许`--view-only`，已在台式机实际桌面做有界显示检查并保存viewer-ready.png；显示检查新增物理0步，随后关闭。播放器按原样展示缓存，不调用策略、不推进新物理。Space播放/暂停，R回首，左右逐帧，Home/End首尾，X退出；默认0.5倍播放并显示真实阶段名。

```bash
cd /home/qunwang/wajueji
env DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority PYTHONPATH=src/v4:.:work/continuous-single-scoop-20261014-v1 \\
  /home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python \\
  work/continuous-single-scoop-20261014-v1/replay.py --view-only \\
  --output runs/continuous-single-scoop-20261014-v1/display
```

运行入口run.py需新的已授权预算和全新输出目录；本次预算已闭锁，不能复用原文件重跑。protocol.json记录实际执行版本、原控制依赖、模型/起点/合同和CPU报告SHA；source-snapshot为本次新入口/接入/审计/回放源码，依赖原项目运行环境。

完整动作、保护和物理帧日志、各阶段检查点及大型缓存均保存在台式机运行目录，不将数十GB缓存重复下载。轻量包保留报告/审计/协议/预算/关闭/阶段日志/源码身份，manifest.json逐文件校验。

## 预算

开始{closure['start_cst']}，原截止{closure['fixed_deadline_cst']}，关闭{closure['closed_cst']}，剩余{closure['unused_seconds_sealed']:.1f}秒封存。物理与训练均结束，旧预算未重开。回放是独立的只读显示。
'''
(out/'README.md').write_text(body)
for n in ['report.json','read-only-audit.json','closure.json','audit-execution.json','read-only-audit-v1.json','audit-execution-v1.json','audit-command-diagnostic.json']:shutil.copyfile(RUN/n,out/n)
for n,s in [('budget.json',bp),('protocol.json',W/'protocol.json'),('cpu-preflight.json',W/'cpu-v2/report.json')]:shutil.copyfile(s,out/n)
for n in ['viewer-ready.png','viewer-ready.json','viewer-closed.json']:shutil.copyfile(RUN/'display'/n,out/n)
write(out/'timeline.json',report['phases']);shutil.copyfile(RUN/'display/phases.json',out/'frame-phases.json')
(out/'source-snapshot').mkdir()
for n in ['continuous_scoop.py','preflight.py','run.py','audit.py','audit_launcher.py','audit_v2.py','audit_launcher_v2.py','replay.py','finish.py']:shutil.copyfile(W/n,out/'source-snapshot'/n)
pattern=r'^## 2026-10-14 统一残差连续单铲：CPU已验证，物理已授权\n.*?(?=^## |\Z)'
for rel in ['docs/CURRENT_STATUS.md','docs/PROJECT_PLAN.md','docs/plans/S05-imitation-and-rl.md','docs/plans/S07-truck-reproduction-submission.md','docs/acceptance.md']:
    p=ROOT/rel;t=p.read_text()
    if re.search(pattern,t,re.M|re.S):t=re.sub(pattern,lambda _:brief,t,count=1,flags=re.M|re.S)
    else:
        pos=re.search(r'^## ',t,re.M).start()
        if rel=='docs/acceptance.md':pos=t.index('\n',t.index('## 当前能力摘要'))+1
        t=t[:pos]+'\n'+brief+t[pos:]
    if rel=='docs/CURRENT_STATUS.md':t=t.replace('S05进行中；统一单铲CPU通过；连续物理准备启动','S05进行中；统一残差连续单铲已验证；物理与训练关闭',1)
    if rel=='docs/acceptance.md':t=re.sub(r'^## 当前能力摘要.*$','## 当前能力摘要（2026-10-14）：统一残差连续单铲已验证；固定工况，全程回放已保存',t,count=1,flags=re.M)
    p.write_text(t)
p=ROOT/'docs/plans/INDEX.md';t=p.read_text();first=re.search(r'^## 2026-10-14 .*?(?=^## |\Z)',t,re.M|re.S)
if first:t=t[:first.start()]+brief+t[first.end():]
t=t.replace('冻结策略达标；D181两组退出静置4880帧及72项审计通过；正式体积未标定',f'冻结策略及D182同实例连续单铲已验证；{audit["check_count"]}项审计通过，全帧回放已保存');p.write_text(t)
p=ROOT/'docs/decisions.md';t=p.read_text();pattern=r'^## D-182：.*?(?=^## |\Z)';assert re.search(pattern,t,re.M|re.S)
t=re.sub(pattern,lambda _:brief.replace('## 2026-10-14 统一残差连续单铲：已实现并验证，本轮计算关闭','## D-182：当前工况统一残差连续单铲（2026-10-14，已验证）'),t,count=1,flags=re.M|re.S)
t=re.sub(r'^- `D-182`：.*$',f'- `D-182`：同实例统一残差连续单铲{report["frames"]}帧、{audit["check_count"]}项审计通过；0训练，全程缓存，预算关闭，仅当前固定铲位。',t,count=1,flags=re.M);p.write_text(t)
write(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
archive=RUN/'continuous-single-scoop-summary-20261014.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in out.rglob('*'):
        if p.is_file():z.write(p,'continuous-single-scoop-20261014/'+p.relative_to(out).as_posix())
print(json.dumps(dict(status='closed_packaged',frames=report['frames'],audit_checks=audit['check_count'],archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),unused_seconds_sealed=closure['unused_seconds_sealed']),ensure_ascii=False),flush=True)
