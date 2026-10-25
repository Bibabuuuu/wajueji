"""Close the completed bounded run; preserve evidence and update existing records."""
from pathlib import Path
import json,hashlib,time,datetime,subprocess,re,shutil,csv,zipfile
ROOT=Path('/home/qunwang/wajueji');WORK=ROOT/'work/roadshow-three-way-20261014-v1'
RUN=ROOT/'runs/d165-exit-settle-pair-20261014-v1';CPU=WORK/'exit-settle-cpu-v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
audit=read(RUN/'read-only-audit.json');report=read(RUN/'report.json');assert audit['status']=='passed' and all(audit['checks'].values()) and report['status']=='completed'
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert not gpu.strip(),gpu
bp=Path(str(RUN)+'-budget.json');budget=read(bp);now=time.time();zone=datetime.timezone(datetime.timedelta(hours=8));stamp=lambda t:datetime.datetime.fromtimestamp(t,zone).isoformat()
closure=dict(status='closed_early' if now<budget['epoch_deadline'] else 'closed',budget_sha256=sha(bp),start_cst=stamp(budget['epoch_start']),fixed_deadline_cst=stamp(budget['epoch_deadline']),closed_cst=stamp(now),
    physical_frames=report['physical_frames'],main_frames=4520,prefix_validation_frames=360,training_updates=0,audit_physical_frames=0,audit_checks=audit['check_count'],
    unused_seconds_sealed=max(0,budget['epoch_deadline']-now),gpu_compute_processes=[],old_budgets_reopened=False,full_return_or_new_excavation_started=False,formal_bulk_volume_qualified=False)
assert not Path(str(bp)+'.closed.json').exists()
write(RUN/'closure.json',closure);write(str(bp)+'.closed.json',closure)
out=RUN/'summary';out.mkdir(exist_ok=False)
metrics=[]
for name,label in [('better_manual','较好人工来源'),('frozen_residual','冻结残差来源')]:
    c=audit['cases'][name];q=c['quantities'];s=q['soil'];obs=c['final_soil_observation'];groups=c['initial_above_rim_cohorts']
    metrics.append(dict(case=name,label=label,exit_frames=2260,settle_sim_seconds=5.,stable_window_seconds=3.,final_core_soil_count=obs['partition']['core'],final_above_soil_count=obs['partition']['above_rim'],
        stable_core_soil_count=s['core_stable_count'],stable_above_soil_count=s['above_rim_stable_count'],stable_soil_count=s['stable_count'],
        stable_soil_mass_kg=s['stable_mass_kg'],stable_soil_reference_liters=s['stable_original_reference_volume_liters'],
        stable_rock_reference_liters=q['stable_rock_reference_volume_liters'],stable_rock_sample_mass_kg=q['stable_rock_sample_mass_kg'],
        stable_mixed_reference_liters=q['stable_mixed_reference_volume_liters'],stable_mixed_sample_mass_kg=q['stable_mixed_sample_mass_kg'],
        initial_near_bucket_soil_count=groups['near50']['initial_count'],initial_near_group_final_near_bucket=groups['near50']['final_near_bucket_count'],
        final_bucket_opening_gap_m=obs['bucket_outside_opening_aabb_gap_m'],formal_bulk_volume_qualified=False))
write(out/'metrics.json',metrics)
with (out/'metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
a,b=metrics;diff=audit['comparison']['stable_reference_volume_difference_liters'];pct=audit['comparison']['stable_reference_volume_difference_percent'];massdiff=audit['comparison']['stable_sample_mass_difference_kg']
conclusion=(f'本次开发静稳参考量残差较人工多{diff:.4f}升（{pct:+.4f}%），限同场景单次配对，不能扩写为正式混合净装车体积提升。' if diff>0 else f'本次开发静稳参考量残差较人工少{-diff:.4f}升（{pct:+.4f}%），未证明残差提高最终留存量。挖掘段满斗显示优势不能直接外推为装车优势。')
brief=f'''## 2026-10-14 退出静置及接收量核验：两项已完成开发验证，预算已关闭

用户要求自主完成缓存核查和两组相同退出/静置。缓存中车沿上方土粒人工5571/残差4612，其中265/324贴近斗壳，其余多数几何上邻接车内物料；邻接不单独证明受力支承。两组均从原卸料末态精确恢复，重放末177帧复现原末态并通过独立恢复首3步连续性门，实际CLEAR_TRUCK/回转离车/5秒静置各2260帧；连同两组180帧恢复验证，共4880新物理帧、0训练。{audit['check_count']}项独立CPU审计全过，原机械/材料/防漏/跟踪保护保留。

末3秒按冻结1cm位移/1cm每秒筛选，稳定土粒人工{a['stable_soil_count']}/残差{b['stable_soil_count']}；土原始参考体积{a['stable_soil_reference_liters']:.4f}/{b['stable_soil_reference_liters']:.4f}升，石块参考采样体积{a['stable_rock_reference_liters']:.4f}/{b['stable_rock_reference_liters']:.4f}升，合计{a['stable_mixed_reference_liters']:.4f}/{b['stable_mixed_reference_liters']:.4f}升。{conclusion}这些是稳定观察窗的开发质量/原始物料参考体积，未标定孔隙和真实堆积体积，不授予官方95%或三铲；仅从hold来源的后续开发验证。终点允许黏附，退出后随斗物料不计入车。

本次代理按用户自主授权设置45分钟上限（非用户指定时长）：{closure['start_cst']}至{closure['fixed_deadline_cst']}，实际{closure['closed_cst']}关闭，剩余{closure['unused_seconds_sealed']:.1f}秒封存，GPU空闲、旧预算不重开。未返回挖掘预备点，未新挖/复训/改PPT主表。证据runs/d165-exit-settle-pair-20261014-v1/read-only-audit.json、closure.json、summary/README.md；Windows deliverables/exit-settle-pair-20261014/。决策D-181。

'''
table='\n'.join(f"| {m['label']} | {m['stable_soil_count']} | {m['stable_soil_reference_liters']:.4f} | {m['stable_rock_reference_liters']:.4f} | {m['stable_mixed_reference_liters']:.4f} | {m['stable_mixed_sample_mass_kg']:.4f} |" for m in metrics)
cohorts='\n'.join(f"| {label} | {group} | {v['initial_count']} | {v['final_receipt_count']} | {v['final_stable_receipt_count']} | {v['final_near_bucket_count']} | {v['median_world_displacement_m']:.4f} |" for name,label in [('better_manual','人工'),('frozen_residual','残差')] for group,v in audit['cases'][name]['initial_above_rim_cohorts'].items())
body=f'''# 缓存归属核查与配对退出静置结果

状态：第1项缓存核查和第2项退出静置均已完成，并通过本轮开发验证。{conclusion}

## 结果与口径

| 挖掘来源 | 静稳土粒数 | 土原始参考体积/L | 石参考采样体积/L | 混合参考合计/L | 混合采样质量/kg |
|---|---:|---:|---:|---:|---:|
{table}

残差－人工：参考合计{diff:+.4f}升（{pct:+.4f}%），采样质量{massdiff:+.4f}千克。每组仅一次，不能据此推断泛化或统计显著性。前次“车厢CAD内部土粒+7.17%”只描述低于车沿部分的分布，未包括上方堆料；不能用作整体装车增益。

评价规则在物理前冻结：退出后参考保持5秒，取末3秒；土粒每帧均处于原CAD核心或开口上方区域，相对窗口起点最大位移≤0.01m，末速度≤0.01m/s。石块用既有体内参考样本，持续处于接收域，并要求最大样本位移≤0.01m，线/角速度分别≤0.01m/s及0.01rad/s。原边界不确定区分开，不把外部土堆计入撒漏；未静稳和域外物料不删点。

土质量读取并核验原生model.particle_mass，土原始参考体积=数量×0.025³立方米；石体积/质量按原参考样本在域内的比例计算。二者相加是明确的原始物料参考量，不是孔隙校准后的真实堆积体积，也不是赛事正式体积。原挖掘满斗显示估计采用另一算法，不能拿两种量直接计算运输效率。开发土石质量用于核查，不替换体积优化目标。

## 车沿上方物料去了哪里

初始上方人工5571、残差4612，几何上贴近斗壳的265/324个单独冻结ID跟踪，剩余5306/4288个为远离斗壳组。25/50mm体素邻接敏感性保留在CPU报告中，不将邻接本身当作接触支承证明。

| 来源 | 初始分组 | 初始数量 | 退出后在接收域 | 其中静稳 | 退出后近斗壳 | 世界位移中位数/m |
|---|---|---:|---:|---:|---:|---:|
{cohorts}

near50/far50以初始距20mm间距斗壳代理采样是否≤50mm分组，是近壳诊断，不是接触力所有权标签。实际离车与逐粒运动为“疑似黏附是否随斗走”的证据；允许少量黏附，但不把随斗物料算入车。整斗CAD投影包围盒已与车口分离，人工/残差间距分别{a['final_bucket_opening_gap_m']:.4f}/{b['final_bucket_opening_gap_m']:.4f}米，接收域没有50mm内近斗壳土粒。

## 恢复、执行与独立核验

两组均从上一轮2697帧正常卸料末态开发续接；先恢复2520帧点，重放原177帧DUMP命令，物理23数组逐字节、原生/高层控制及防漏历史精确复现；仿真时钟按预先固定1e-10秒容差比较并记录是否位级相同。随后连续执行退出首3步，与独立恢复原2697末态的首3步逐字段比较；没有重置物料、底层目标或高层载荷偏置，没有冒充从完整挖掘初态连续复现。

每组同一几何配方：CLEAR_TRUCK1190帧＋30帧保持，WITHDRAW_SWING710帧＋30帧保持，最后300帧静置，共2260帧（37.6667仿真秒）。同一目标/限速/控制器按各自实际状态反馈，实际命令不强求两组逐字相同。各增加180帧前缀与连续性验证，合计4880物理帧。全部命令和载荷偏置独立CPU复算，原机械、车体4cm/机构2cm间距、闭链、材料、防漏及实际跟踪保护保持；不增加终点净空硬门。

{audit['check_count']}项审计通过；审计0新增物理、0训练。新增脚本只在work候选目录，正式控制/物理源码、冻结模型未变，无Git提交。check_docs现有S07元数据问题单列，不将文档预检描述为全部通过。

## 预算与交付

{brief}
本地摘要包括CPU报告、独立审计、metrics、预算关闭、协议与执行源码快照；manifest.json为逐文件SHA256。完整原始动作、保护日志、所有物理帧、控制侧记录、checkpoint及全帧显示缓存保存在台式机 `/home/qunwang/wajueji/runs/d165-exit-settle-pair-20261014-v1/`。上一轮输入不覆盖。

任务2完成的是离车并静置验证，没有返回挖掘预备点、开展后续铲次或正式体积标定。后续先按本次结果决定是否整理PPT证据或离线定位保料损失，不自动启动新物理/训练。
'''
(out/'README.md').write_text(body)
for n in ['report.json','read-only-audit.json','closure.json']:shutil.copyfile(RUN/n,out/n)
shutil.copyfile(bp,out/'budget.json');shutil.copyfile(WORK/'exit-settle-protocol-v1.json',out/'protocol.json')
shutil.copyfile(CPU/'report.json',out/'cpu-preflight.json');shutil.copyfile(CPU/'dependency-sha256.json',out/'dependency-sha256.json')
(out/'source-snapshot').mkdir()
for n in ['exit_common.py','prepare_exit_settle.py','run_exit_settle.py','audit_exit_settle.py','finish_exit_settle.py']:shutil.copyfile(WORK/n,out/'source-snapshot'/n)
for name in ['better_manual','frozen_residual']:
    d=out/name;d.mkdir();shutil.copyfile(RUN/name/'report.json',d/'report.json');shutil.copyfile(RUN/name/'prefix-continuity/report.json',d/'prefix-continuity-report.json')
pattern=r'^## 2026-10-14 退出静置及接收量核验：已授权实施\n.*?(?=^## |\Z)'
for rel in ['docs/CURRENT_STATUS.md','docs/plans/S05-imitation-and-rl.md','docs/plans/S06-three-scoop-objective-evaluation.md','docs/plans/S07-truck-reproduction-submission.md','docs/PROJECT_PLAN.md','docs/acceptance.md']:
    p=ROOT/rel;t=p.read_text()
    if re.search(pattern,t,re.M|re.S):t=re.sub(pattern,lambda _:brief,t,count=1,flags=re.M|re.S)
    else:
        pos=re.search(r'^## ',t,re.M).start()
        if rel=='docs/acceptance.md':pos=t.index('\n',t.index('## 当前能力摘要'))+1
        t=t[:pos]+'\n'+brief+t[pos:]
    if rel=='docs/CURRENT_STATUS.md':t=t.replace('S05进行中；缓存核查/退出CPU已过；两组退出静置准备运行','S05进行中；两组退出静置开发验证完成；物理已关闭',1)
    if rel=='docs/acceptance.md':t=re.sub(r'^## 当前能力摘要.*$', '## 当前能力摘要（2026-10-14）：配对退出静置开发验证完成；正式体积未标定',t,count=1,flags=re.M)
    p.write_text(t)
p=ROOT/'docs/plans/INDEX.md';t=p.read_text();t=t.replace('冻结策略达标；D180两组运输卸料5394帧及63项审计通过；正式净入车量待验证',f'冻结策略达标；D181两组退出静置4880帧及{audit["check_count"]}项审计通过；正式体积未标定')
first=re.search(r'^## 2026-10-14 当前：.*?(?=^## |\Z)',t,re.M|re.S)
if first:t=t[:first.start()]+brief+t[first.end():]
p.write_text(t)
p=ROOT/'docs/decisions.md';t=p.read_text();pattern=r'^## D-181：.*?(?=^## |\Z)';assert re.search(pattern,t,re.M|re.S)
t=re.sub(pattern,lambda _:brief.replace('## 2026-10-14 退出静置及接收量核验：两项已完成开发验证，预算已关闭','## D-181：自主完成缓存核查和配对退出静置（2026-10-14，已验证）'),t,count=1,flags=re.M|re.S)
t=re.sub(r'^- `D-181`：.*$',f'- `D-181`：两项已完成开发验证；4880帧、{audit["check_count"]}项审计全过；参考量差{diff:+.4f}升，正式混合堆积体积未标定，预算关闭。',t,count=1,flags=re.M);p.write_text(t)
write(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
archive=RUN/'exit-settle-pair-summary-20261014.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in out.rglob('*'):
        if p.is_file():z.write(p,'exit-settle-pair-20261014/'+p.relative_to(out).as_posix())
print(json.dumps(dict(status='closed_packaged',audit_checks=audit['check_count'],unused_seconds_sealed=closure['unused_seconds_sealed'],reference_volume_difference_liters=diff,reference_volume_difference_percent=pct,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive)),ensure_ascii=False),flush=True)
