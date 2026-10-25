from pathlib import Path
import json, hashlib, shutil, re, csv, zipfile

ROOT=Path('/home/qunwang/wajueji'); WORK=ROOT/'work/roadshow-three-way-20261014-v1'
RUN=ROOT/'runs/d165-transport-full-pair-20261014-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
audit=read(RUN/'read-only-audit.json'); report=read(RUN/'report.json'); closure=read(RUN/'closure.json')
assert audit['status']=='passed' and audit['check_count']==63 and audit['full_pair_task_complete']
assert report['physical_frames']==5394 and closure['training_updates']==0
out=RUN/'handover';out.mkdir(exist_ok=False)
metrics=[]
for name,label in [('better_manual','较好人工来源'),('frozen_residual','冻结残差来源')]:
    c=audit['cases'][name]; last=c['observations'][-1]; inv=c['source_cavity_inventory_recomputed']
    metrics.append(dict(case=name,label=label,frames=c['frames'],simulation_seconds=c['frames']/60,
        wall_seconds=report['cases'][name]['wall_seconds'],
        source_cavity_soil_count=inv['source_particles'],source_cavity_soil_in_receiver_core=inv['received_source_particles'],
        source_cavity_soil_above_rim_unverified=c['source_cavity_receiver_partition']['above_rim_unverified'],
        source_cavity_soil_outside_receiver=c['source_cavity_receiver_partition']['outside'],
        source_cavity_soil_remaining_in_bucket=inv['remaining_source_particles'],
        all_soil_in_receiver_core=last['all_soil_centers_in_truck'],
        all_soil_above_rim_unverified=last['soil_receiver_partition']['above_rim_unverified'],
        rock_centers_in_receiver_core=last['rock_centers_in_truck'],
        rock_reference_sample_volume_liters_in_core=last['rock_reference_sample_volume_liters_in_truck'],
        bucket_final_display_percent=last['bucket_display']['fill_percent'],
        maximum_tracking_error_m_or_rad=c['observed_max_tracking_error_m_or_rad'],formal_mixed_volume_qualified=False))
write(out/'metrics.json',metrics)
with (out/'metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
gain=(metrics[1]['all_soil_in_receiver_core']/metrics[0]['all_soil_in_receiver_core']-1)*100
brief=f'''## 2026-10-14 两组完整运输卸料：已验证开发门，45分钟预算提前关闭

较好人工与冻结残差原始hold各独立恢复，均完成CLEAR_PILE/ALIGN/DUMP、2697帧（44.95仿真秒），共5394物理帧、0训练。63项独立审计通过，5394条实际命令和载荷偏置逐值复算一致；原生24数组/控制与扫掠内部态、原机械/材料/防漏/跟踪保护和冻结模型均核验。没有退出/返回或三铲结论。

最终车厢CAD内部土粒：人工14345，残差15374（+{gain:.4f}%）；全土粒车沿上方未核验5571/4612。起始斗腔土粒来源不同：人工12394、残差11877，最终车内7032/7457，车沿上方5362/4420、车体外0/0、原斗腔余0/0；不含起始堆高及石块，不能将这些比例称为整体装车率。石块CAD内部参考采样体积56.1329/55.7566升，残差未在这一分项占优。以上均为开发分类，不能证明正式稳定混合净入车量提升；车沿上方低速不等于支承/归属已验证。终点仍允许黏附，无净空硬门。

开始{closure['start_cst']}，原截止{closure['fixed_deadline_cst']}，提前关闭{closure['closed_cst']}；剩余{closure['unused_seconds_sealed']:.1f}秒封存，GPU空闲，旧预算不重开。下一步先用现有缓存离线核验车沿上方支承/归属与稳定接收口径，不能直接把上方物料全加进车内；退出返回的物理验证需另确认范围/预算。现有PPT挖掘三方主表不改，几何缺失值保留。

证据runs/d165-transport-full-pair-20261014-v1/read-only-audit.json、closure.json、handover/README.md；Windows deliverables/transport-full-pair-20261014/。正式控制/物理源码、冻结模型未改，未Git提交。决策D-180。

'''
table='\n'.join('| '+m['label']+' | '+str(m['all_soil_in_receiver_core'])+' | '+str(m['all_soil_above_rim_unverified'])+' | '+f"{m['rock_reference_sample_volume_liters_in_core']:.4f}"+' | '+f"{m['bucket_final_display_percent']:.4f}%"+' |' for m in metrics)
body='''# 两组运输卸料开发对照（2026-10-14）

状态：已实现并通过本轮开发验证；正式稳定混合净入车量未验证。

两来源分别为较好人工挖掘末态和冻结残差挖掘末态，运输均使用相同几何保斗路线，不能把本轮称为“运输阶段SAC对几何控制”。不是较低装载人工/纯几何/残差三方挖掘对照；原Stage03三方PPT主表保留。

两组各2697帧、44.95仿真秒，实际计算550.21/527.12秒。两组各一次，不能推断泛化或统计显著优势。完成举升脱堆、回转对准和卸料，含末段零指令保持；未执行退出/返回、第二铲或完整三铲。模型没有更新，运输不调用策略推理。63项CPU独立证据审计通过，0新增审计物理帧；原生24数组恢复、全部5394条命令/载荷偏置、保护、缓存和来源SHA精确核验。实际最大跟踪误差人工0.00447745、残差0.00426215（日志为米或弧度混合轴量，不能统一当成毫米）。

| 挖掘来源 | 车厢CAD内部土粒数 | 车沿上方未核验土粒数 | 石块车内参考采样体积/升 | 末态斗内显示诊断 |
|---|---:|---:|---:|---:|
'''+table+f'''

残差车内土粒数多1029（+{gain:.4f}%），是一个分项正向信号，不是“净装车量提高{gain:.2f}%”。车内石块参考体积分项略低；土石堆积体积、堆高、孔隙及归属尚未统一标定，不将分项强行相加。原土堆中的全部外部土粒不是撒料计数。

只跟踪原始斗腔土粒的开发门：人工7032/12394在车内，5362在车沿上方；残差7457/11877在车内，4420在车沿上方；两组此来源子集车体外0、原斗腔余0。它不覆盖起始堆高/石块，不能作为整斗效率。两组原开发门（车内原斗腔土粒至少50%，斗腔剩余不超过max(50,10%)）均通过，不等价于95%正式验收。

车沿上方不是自动丢失，也不是自动有效载荷。原斗腔上方子集速度P95约人工0.000604/残差0.002613米/秒，末0.5秒位移P95约0.000250/0.000500米；只说明多数运动较小，未确认车体/料堆支承、是否外部黏附或真实混合堆积体积。少量底部黏附按已确认要求允许，未增加净空门槛。

## 预算、交付和下一步

'''+brief+'''
protocol.json、budget.json、closure.json固定身份和时限；source-snapshot保留真实执行runner与独立审计脚本。report/actions/clearance/physics日志及checkpoint控制/元数据随包提供，manifest.json记录逐文件SHA256。

大数据保留台式机 `/home/qunwang/wajueji/runs/d165-transport-full-pair-20261014-v1/{better_manual,frozen_residual}/`：zero/short-120/周期/clear_pile/align/dump/final.npz和display/particles.npy及states.json，轻量包不含大缓存和NPZ。只读回放不需新增物理。保存态不自动授予进一步续接资格。

下一步优先用当前缓存审计车沿上方物料的归属和稳定性，形成可解释的接收量口径；需要机器退出后才可确认的部分明确保留为待验证，不启动新物理或训练。原PPT可增加“运输卸料已打通、接收量为开发分项”的证据页，不能据此声称最终净入车量提升。
'''
(out/'README.md').write_text(body)
for n in ['report.json','read-only-audit.json','closure.json']:shutil.copyfile(RUN/n,out/n)
shutil.copyfile(Path(str(RUN)+'-budget.json'),out/'budget.json')
shutil.copyfile(WORK/'transport-full-pair-protocol-v1.json',out/'protocol.json')
(out/'source-snapshot').mkdir()
for n in ['transport_full_pair.py','audit_transport_full.py','close_transport_full.py']:shutil.copyfile(WORK/n,out/'source-snapshot'/n)
for name in ['better_manual','frozen_residual']:
    d=out/name;d.mkdir()
    for p in (RUN/name).iterdir():
        if p.is_file() and (p.suffix in ['.json','.ndjson']):shutil.copyfile(p,d/p.name)
    (d/'physics').mkdir();shutil.copyfile(RUN/name/'physics/frames.ndjson',d/'physics/frames.ndjson')
    (d/'display-cache').mkdir();shutil.copyfile(RUN/name/'display/reconstruction.json',d/'display-cache/reconstruction.json')
for rel in ['docs/CURRENT_STATUS.md','docs/plans/S05-imitation-and-rl.md','docs/plans/S06-three-scoop-objective-evaluation.md','docs/PROJECT_PLAN.md','docs/acceptance.md']:
    p=ROOT/rel;t=p.read_text()
    pattern=r'^## 2026-10-14 两组完整运输卸料：新增45分钟已授权并启动准备\n.*?(?=^## |\Z)'
    if re.search(pattern,t,re.M|re.S):t=re.sub(pattern,lambda _:brief,t,count=1,flags=re.M|re.S)
    else:
        pos=re.search(r'^## ',t,re.M).start()
        if rel=='docs/acceptance.md':pos=t.index('\n',t.index('## 当前能力摘要'))+1
        t=t[:pos]+'\n'+brief+t[pos:]
    if rel=='docs/CURRENT_STATUS.md':t=t.replace('S05进行中；两组完整运输卸料对照执行中；新增45分钟','S05进行中；两组运输卸料开发验证完成；本轮物理已关闭',1)
    if rel=='docs/acceptance.md':t=t.replace('Stage 03三方对照已归档，几何量测有缺口','三方挖掘与两组运输开发对照已归档；正式净入车量待验证',1)
    p.write_text(t)
p=ROOT/'docs/plans/INDEX.md';t=p.read_text().replace('冻结策略达标；D179原生恢复/两组120帧及60项审计通过；下一门完整运输预算','冻结策略达标；D180两组运输卸料5394帧及63项审计通过；正式净入车量待验证');p.write_text(t)
p=ROOT/'docs/decisions.md';t=p.read_text();assert '## D-180' not in t
decision='''## D-180：新增45分钟，两组完整运输卸料开发对照（2026-10-14）

用户确认新增2700秒，较好人工及冻结残差各最多2697帧，原始hold独立恢复，零步/120帧门失败即停止；不训练、不改变保护，预算到期保存，旧预算不重开。固定起止及代码身份见runs/d165-transport-full-pair-20261014-v1-budget.json和协议。

已完成并验证：两组各2697帧，63项独立审计全过，0训练，全部实际命令逐值复算。原保护及模型保持，开发接收门均通过。车内土粒14345/15374，车沿上方5571/4612尚未核验，不能宣布正式混合净装车量优势。未退出返回、未三铲。提前关闭并封存剩余预算；不借剩余时间扩大物理范围。证据见运行目录read-only-audit.json、closure.json及handover/README.md，下一步优先缓存离线量测核验，新增物理需另授权。

'''
at=t.index('## D-179');t=t[:at]+decision+t[at:]
t=t.replace('## 当前决策索引（2026-10-14）','## 当前决策索引（2026-10-14）\n\n- `D-180`：新增45分钟两组完整运输卸料；5394帧、63项审计通过，预算封存；正式混合净入车量待验证。',1);p.write_text(t)
write(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
archive=RUN/'transport-full-pair-20261014.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for p in out.rglob('*'):
        if p.is_file():z.write(p,'transport-full-pair-20261014/'+p.relative_to(out).as_posix())
print(json.dumps(dict(status='packaged',files=len(read(out/'manifest.json')),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),core_soil_count_gain_percent=gain)))
