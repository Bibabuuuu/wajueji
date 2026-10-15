from pathlib import Path
import json,hashlib,shutil
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-short-phased-20261014-v1';D=R/'runs/second-scoop-short-phased-20261014-v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=read(D/'report.json');a=read(D/'read-only-audit.json');c=read(W/'closure.json');source=read(W/'aligned-source-candidate.json')
assert p['status']=='second_scoop_short_completed' and a['status']=='passed' and a['check_count']==41 and all(a['checks'].values())
base=read(W/'record-start-receipt.json')
for n,v in base.items():assert sha(R/n)==v['after'],'concurrent:'+n
texts={n:(R/n).read_text() for n in base}
heading='2026-10-14 第二铲接近与冻结残差120帧短检通过';anchor='2026-10-14-第二铲接近与冻结残差120帧短检通过'
result='''模式切换版本已完成本轮授权短检：从真实首铲返回源6fe40910…精确恢复24数组、原生控制/扫掠和时钟，一个新原生续接实例内执行4882帧（81.3667仿真秒），中间不重置土石/机构/控制/时钟。2151帧主动上抬后在全局2154帧通过名义完成、原端点误差和32厘米间距准入，原子切为高位保持；60帧跟踪/零指令保持通过。继续高位转向、伸展、张斗、清障、外移、降低第二铲预备点及参考对齐，随后40次冻结预测执行120个残差物理帧。主动上抬最低点1微米门、保持位置/静态门和原硬界/碰撞/材料/扫掠均保留。'''
evidence='''41项独立CPU审计全部通过：4519条几何发令复算最大差0，2151个主动上抬帧及60个保持帧独立核验；全部交接/切换前后物理和原生摘要不变。40次冻结动作预测、从实际姿态/土石缓存和决策前世界力矩重建的221维策略输入差均0；局部20Hz决策、60Hz物理及模式作废语义保持。源/协议源码/模型SHA未变，训练更新0，GPU空闲。原371帧及2159帧失败不追溯改判，新通过证据只属于本模式切换版本。'''
limit=f'''**验收边界**：残差仅执行120帧，参考进度{p['reference_cursor']:.0f}/1170，未完成整条参考或0.5秒保载，没有新满斗率结果。策略子报告的closed_early/未success是授权短段结束状态，不是完整挖掘的完成/失败判据。新点短段接口、动作和安全通过，不授予完整第二铲、第三铲、同实例连续三铲或泛化装载优势。终点允许黏附的用户边界保持。当前是从首铲末态原生恢复后的一个续接实例，不冒称从初始场景不间断三铲。'''
closed=f'''新增45分钟预算已完成并关闭：20:34:16开始，{c['closed_at']}关闭，总耗时{c['elapsed_seconds']:.1f}秒、余{c['sealed_remaining_seconds']:.1f}秒封存；不将余时扩展为完整挖掘。全帧缓存及各段原生快照保留，无Git提交。'''
candidate=f'''**下一步来源候选，CPU已核对/物理恢复未验证**：挖掘前对齐快照`runs/second-scoop-short-phased-20261014-v1/reference_align.npz`，SHA`{source['source_sha256']}`，全局{source['actual_frame']}帧、时钟{source['simulation_time_s']:.6f}秒，stage=prepoint/cycle=2，无training_reset。11项CPU来源核验通过：24数组、土粒/岩石/机构/底层目标及实际冻结策略入口与缓存一致。控制侧的continuation_qualified仍False，不能自动作为已验证恢复；不从短段120帧末态盲续策略。'''
next_='''**待确认下一轮范围/预算**：推荐从上述挖掘前快照原生恢复，先零步24数组/底层状态门，再将同一实例首3个真实残差帧与本轮保存前缀对照；通过后继续同一实例完成1170参考及30帧短保载，记录新点装载，不训练、不运输卸料。建议新增30分钟含接入/恢复/完整挖掘/保存/独立审计；不保证完成或装载改善，恢复/前缀/偏离/超时门失败即保存停。旧预算不重开。当前仅为可评审候选，尚未授权新物理。S05进行中，后续完整第二铲/第三铲待验证。'''
links='''证据：[物理结果](../../runs/second-scoop-short-phased-20261014-v1/report.json)、[41项独立审计](../../runs/second-scoop-short-phased-20261014-v1/read-only-audit.json)、[预算关闭](../../work/second-scoop-short-phased-20261014-v1/closure.json)、[下一来源CPU核验](../../work/second-scoop-short-phased-20261014-v1/aligned-source-candidate.json)。'''
n='docs/plans/S05-imitation-and-rl.md';pos=texts[n].index('## ');texts[n]=texts[n][:pos]+f'## {heading}\n\n'+result+'\n\n'+evidence+'\n\n'+limit+'\n\n'+closed+'\n\n'+candidate+'\n\n'+next_+'\n\n'+links+'\n\n'+texts[n][pos:]
texts[n]=texts[n].replace('## 2026-10-14 模式切换版本45分钟短检：已授权，启动前CPU检查通过','## 历史：模式切换版本45分钟短检获授权，现已完成并关闭',1)
def replace_first(n,head,body):
 s=texts[n];i=s.index('## ');j=s.index('\n## ',i+3);texts[n]=s[:i]+f'## {head}\n\n{body}\n'+s[j:]
brief='模式切换版本4882帧完成真实返回恢复、上抬/高位保持、第二铲接近/参考对齐及120帧冻结残差；41项独立审计通过，模型/物理源码未改、0训练。本轮预算关闭。参考进度76/1170，完整第二铲及保载尚未完成。下一轮候选为原生恢复已审计的挖掘前对齐快照，先零步/首3帧门，再完整第二铲；来源CPU核验11项通过，但新恢复和预算尚未授权。'
replace_first('docs/CURRENT_STATUS.md','2026-10-14 当前：第二铲接近与120帧残差通过，预算关闭',result+'\n\n'+evidence+'\n\n'+limit+'\n\n'+closed+'\n\n'+candidate+'\n\n'+next_+f'\n\n实施由[S05](plans/S05-imitation-and-rl.md#{anchor})维护。')
replace_first('docs/PROJECT_PLAN.md','2026-10-14 当前推进：第二铲短段通过，完整挖掘待验证',brief+f' 仍仅S05推进当前新点接口/评价，原连续首铲结论保持。下一门见[S05](plans/S05-imitation-and-rl.md#{anchor})；D-186。')
replace_first('docs/plans/INDEX.md','2026-10-14 当前实施S05：第二铲接近与残差短检通过',brief+f' 其他小节不因此升级验收，范围见[S05](S05-imitation-and-rl.md#{anchor})。')
n='docs/acceptance.md';i=texts[n].index('## 当前能力摘要');j=texts[n].index('\n## ',i+3);texts[n]=texts[n][:i]+'## 当前能力摘要（2026-10-14）：残差连续首铲已验证；第二铲接近及120帧残差短检通过；完整第二/第三铲未完成\n'+texts[n][j:]
pos=texts[n].index('\n## ',texts[n].index('## 当前能力摘要')+3);texts[n]=texts[n][:pos]+f'\n## {heading}\n\n'+result+'\n\n'+evidence+'\n\n'+limit+'\n\n'+closed+f'\n\n后续来源和范围见[S05](plans/S05-imitation-and-rl.md#{anchor})。\n'+texts[n][pos:]
n='docs/decisions.md';assert '## D-186' not in texts[n];pos=texts[n].index('## D-185');texts[n]=texts[n][:pos]+'''## D-186：模式切换短检通过后，先准备完整第二铲来源候选（2026-10-14）

'''+brief+' '+limit+'\n\n'+candidate+'\n\n'+next_+f'\n\n本决策只记录已通过短检和待确认候选，不授予新预算或新的原生恢复资格；见[S05](plans/S05-imitation-and-rl.md#{anchor})。\n\n'+texts[n][pos:]
backup=W/'record-before-final';backup.mkdir(exist_ok=False)
for n,v in base.items():
 assert sha(R/n)==v['after'];dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/n,dst)
for n,s in texts.items():(R/n).write_text(s)
(W/'record-final-receipt.json').write_text(json.dumps({n:dict(before=base[n]['after'],after=sha(R/n)) for n in texts},indent=2))
print('Recorded completed short scope and pending full-second-dig proposal in six existing records.')
