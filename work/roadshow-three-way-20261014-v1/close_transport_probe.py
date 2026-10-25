from pathlib import Path
import json,datetime,time,re,hashlib,shutil,subprocess
ROOT=Path('/home/qunwang/wajueji');WORK=ROOT/'work/roadshow-three-way-20261014-v1'
RUN=ROOT/'runs/d165-transport-native-probe-20261014-v1'
read=lambda p:json.loads(Path(p).read_text())
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=read(RUN/'read-only-audit.json');report=read(RUN/'report.json')
assert audit['status']=='passed' and audit['check_count']==60 and all(audit['checks'].values())
assert report['status']=='completed' and report['physical_frames']==240
budget_path=ROOT/'runs/d165-transport-native-probe-20261014-v1-budget.json';budget=read(budget_path)
completed=Path(RUN/'report.json').stat().st_mtime
for name in ['better_manual','frozen_residual']:
    p=RUN/name/'display/reconstruction.json';original=read(p)
    write(p.with_name('reconstruction.original-generic.json'),original)
    corrected=dict(original,capture_origin='native_transport_short_probe',
        label='D165 TRANSPORT SHORT PROBE / '+name,
        metadata_note='Corrected generic ProbeDisplayCache BC label; physical cache contents and their SHA unchanged')
    write(p,corrected)
    case=read(RUN/name/'report.json');case['display']=corrected
    write(RUN/name/'report.json',case);report['cases'][name]=case
write(RUN/'report.json',report)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True)
assert not gpu.stdout.strip(),gpu.stdout
now=time.time();assert now<budget['epoch_deadline']
zone=datetime.timezone(datetime.timedelta(hours=8))
stamp=lambda x:datetime.datetime.fromtimestamp(x,zone).isoformat()
closure=dict(status='closed_early',new_budget_seconds=1200,budget_sha256=sha(budget_path),
    start_cst=stamp(budget['epoch_start']),fixed_deadline_cst=stamp(budget['epoch_deadline']),closed_cst=stamp(now),
    batch_report_completed_cst=stamp(completed),elapsed_seconds=now-budget['epoch_start'],
    unused_seconds_sealed=budget['epoch_deadline']-now,physical_frames=240,training_updates=0,
    gpu_compute_processes=[],full_transport_started=False,old_budgets_reopened=False,
    final_audit='read-only-audit.json',display_metadata_normalized=True)
write(RUN/'closure.json',closure)
out=RUN/'handover';out.mkdir(exist_ok=False)
table=[]
for name,label in [('better_manual','较好人工'),('frozen_residual','冻结残差')]:
    case=audit['cases'][name];a,b=case['diagnostic_display']
    table.append(f"| {label} | 120 | {100*case['origin_delta_m'][2]:.4f}厘米 | {1000*case['max_axis_error_m']:.4f}毫米 | {a['fill_percent']:.4f}% | {b['fill_percent']:.4f}% |")
body='''# D165原生恢复与120帧保斗短检

状态：两组原生零步恢复及短段保斗已验证；本轮新增20分钟提前关闭；完整运输/净入车量尚未验证。

## 结果

用户只授权新增1200秒、每组最多120物理帧、不训练、不跑完整运输。较好人工与冻结残差各从原hold独立恢复，同D165物理及既有固定保斗控制；两组24类数组、原生控制器/时钟、材料/防漏内部态逐值一致。全部240条实际命令与独立CPU复算逐值相同，60项审计全部通过，机械/材料/防漏/跟踪保护未放宽。

| 来源 | 物理帧 | 斗原点上升 | 最大线性轴跟踪误差 | 入口瞬时显示 | 2秒举升末瞬时显示 |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

最大线性轴误差分别发生在斗缸；回转角最大误差分别约0.00006395/0.00007456弧度，不混写为毫米。两组实际斗缸最低约0.85245米，高于原保斗下限，控制指令未主动开斗；实际运动由原受载物理求解，未强制锁死物理位置。

## 显示与结论边界

上述装载值是运动前后两个时刻、原0.2占据阈值/原容量/含石与堆高的诊断显示，不是新增0.5秒静止保载或正式稳定入车量。人工原短保载基准145.3813%不改；入口末帧145.4257%是不同采样时刻。短段末残差比人工高约3.0474个百分点，仅这一固定场景/这一2秒短段，不能外推完整运输收益、显著性或泛化。

实际本轮只执行CLEAR_PILE前120帧，两组均未回转/卸料；CPU已过的4厘米斗—车及2厘米机构间距要求仍保留，但本轮没有授予实际完整回转路径通过。没有净空硬门，没有删除黏附土或改变任何物料。没有SAC更新或新策略推理；残差指本次挖掘末态来源，运输使用相同几何保斗控制。

## 预算和来源

开始：'''+closure['start_cst']+'''；固定截止：'''+closure['fixed_deadline_cst']+'''。实际提前关闭：'''+closure['closed_cst']+'''，剩余约'''+str(round(closure['unused_seconds_sealed'],1))+'''秒封存，不用于完整运输、不重开旧预算。GPU已空闲。

输入、预算和代码身份见protocol.json、budget.json、closure.json与source-snapshot；read-only-audit.json含60项核验和每组显示诊断。各组report/actions/clearance/physics帧日志随轻量包提供，display-cache目录仅含元数据。

大数据及完整checkpoint保存在台式机原运行目录：/home/qunwang/wajueji/runs/d165-transport-native-probe-20261014-v1/{better_manual,frozen_residual}/。包括zero.npz、short-120.npz、final.npz及display/particles.npy；最终与120帧保存态24数组一致、初态与原源24数组一致。缓存只读播放无需增加物理步。

## 下一步

下一门是新增预算内的两组完整举升、回转、卸料对照；保留已确认D165闭斗端点及3.35米举升点。正式稳定混合净入车体积仍需定义/标定，开发接收土粒计数不得冒充正式体积。当前短段保存点含独立控制状态记录但continuation_qualified=false，不能未经恢复审计直接续跑或伪造连续性。

正式控制/物理源码、冻结模型及原PPT主表未改，未Git提交。文档差异检查通过；原S07四项元数据检查错误仍保留，不属于本轮新增问题。
'''
(out/'README.md').write_text(body)
for name in ['report.json','read-only-audit.json','closure.json']:shutil.copyfile(RUN/name,out/name)
shutil.copyfile(budget_path,out/'budget.json');shutil.copyfile(WORK/'transport-native-probe-protocol-v1.json',out/'protocol.json')
(out/'source-snapshot').mkdir()
for name in ['transport_physical_probe.py','audit_transport_probe.py']:shutil.copyfile(WORK/name,out/'source-snapshot'/name)
for name in ['better_manual','frozen_residual']:
    dest=out/name;dest.mkdir()
    for item in ['report.json','actions.ndjson','clearance.ndjson','zero.npz.json','short-120.npz.json','final.npz.json','final.npz.control.json']:
        shutil.copyfile(RUN/name/item,dest/item)
    (dest/'physics').mkdir();shutil.copyfile(RUN/name/'physics/frames.ndjson',dest/'physics/frames.ndjson')
    (dest/'display-cache').mkdir()
    shutil.copyfile(RUN/name/'display/reconstruction.json',dest/'display-cache/reconstruction.json')
brief='''## 2026-10-14 D165原生恢复及保斗短检：已验证，本轮20分钟提前关闭

用户新增1200秒，仅两来源零步恢复及各最多120帧；已完成240物理帧，0训练/0完整运输，60项独立审计全过。两组24类初态及原生控制/材料/防漏内部态精确恢复，240条实际命令独立复算逐值相同。实际各2秒，斗原点上升约10.47厘米；最大线性轴误差人工1.0718毫米、残差1.0305毫米；原保斗/机械/材料/防漏/跟踪保护通过，未主动开斗。

入口→举升末瞬时装载诊断：人工145.4257→144.2588%、残差148.2321→147.3062%，不是新静止保载或净入车量，不改人工145.3813%原基准/PPT主表。本轮只执行CLEAR_PILE前120帧，未回转/卸料。预算提前关闭、剩余封存，GPU空闲，旧预算不重开。下一门为另授权预算的两组完整运输卸料；稳定混合净入车体积仍待定义/标定，短段checkpoint未经进一步恢复审计不授予自动连续资格。

证据runs/d165-transport-native-probe-20261014-v1/read-only-audit.json、closure.json、handover/README.md；Windows deliverables/transport-native-short-20261014/。原控制/物理源码及冻结模型未改，未Git提交。决策D-179。

'''
old='#### D165闭斗端点CPU验证已完成，物理尚未启动'
for rel in ['docs/CURRENT_STATUS.md','docs/plans/S05-imitation-and-rl.md','docs/plans/S06-three-scoop-objective-evaluation.md','docs/PROJECT_PLAN.md','docs/acceptance.md']:
    p=ROOT/rel;t=p.read_text()
    t=re.sub(r'^#### D165原生恢复与短段保斗：新增20分钟已授权并运行准备\n.*?(?=^#### |\Z)','',t,flags=re.M|re.S)
    t=t.replace(old,'#### D165闭斗端点CPU验收历史（后续原生短段已验证）')
    t=t.replace('### 2026-10-14 后续离线诊断已验证（0新增物理/训练）','### 2026-10-14 离线诊断与运输分阶段验证')
    if rel=='docs/acceptance.md':
        pos=t.index('\n',t.index('## 当前能力摘要'))+1
    else:pos=re.search(r'^## ',t,re.M).start()
    p.write_text(t[:pos]+'\n'+brief+t[pos:])
p=ROOT/'docs/CURRENT_STATUS.md';t=p.read_text();t=t.replace('S05进行中；整晚目标及争取项已验证；本轮计算已关闭','S05进行中；原生恢复及两组保斗短段已验证；本轮物理已关闭',1);p.write_text(t)
p=ROOT/'docs/plans/INDEX.md';t=p.read_text();t=t.replace('冻结策略达标；D178运输闭斗端点CPU路径/5394指令通过；下一门原生恢复/物理预算','冻结策略达标；D179原生恢复/两组120帧及60项审计通过；下一门完整运输预算');p.write_text(t)
p=ROOT/'docs/decisions.md';t=p.read_text();assert '## D-179' not in t
decision='''## D-179：新增20分钟，仅原生恢复和保斗短段（2026-10-14）

用户选择新增1200秒，仅较好人工/冻结残差各原生零步恢复和最多120帧，不训练、不跑完整运输，旧预算不重开。固定起止及SHA见runs/d165-transport-native-probe-20261014-v1-budget.json；预留180秒保存/审核。

已验证并提前关闭：240物理帧、60项审计通过，初态/内部态/240条实际发令精确核验，原保护保留。两组各上抬约10.47厘米，末时刻显示144.2588/147.3062%，仅运动中诊断。原基准/PPT评分不改，不授予回转卸料或稳定混合入车量。剩余预算封存，不自动转作完整运输。源码未改、模型未更新，证据见运行目录closure.json、read-only-audit.json及handover/README.md。

'''
at=t.index('## D-178');t=t[:at]+decision+t[at:]
t=t.replace('## 当前决策索引（2026-10-14）','## 当前决策索引（2026-10-14）\n\n- `D-179`：新增20分钟仅两组原生恢复/120帧短检；240帧、60项审计通过，剩余封存，完整运输未做。',1);p.write_text(t)
manifest={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
write(out/'manifest.json',manifest)
print(json.dumps({'status':'closed_early','audit_checks':60,'physical_frames':240,'unused_seconds_sealed':closure['unused_seconds_sealed'],'handover_files':len(manifest)}))
