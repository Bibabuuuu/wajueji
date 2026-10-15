from pathlib import Path
import json,hashlib,shutil
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-qerror-long-20261014-v1';D=R/'runs/second-scoop-qerror-long-20261014-v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=read(D/'report.json');a=read(D/'read-only-audit.json');c=read(W/'closure.json');diag=read(D/'wait-timeline-diagnosis.json');base=read(W/'record-start-receipt.json')
assert p['status'] in ('failed','full_second_dig_completed') and p['prefix_verified'] and 0<p['frames']<=7200
assert a['status'] in ('passed','diagnostic_audit_passed') and all(a['checks'].values()) and a['new_law_max_error']<=1e-12
assert sha(D/'report.json')==c['report_sha256'] and sha(D/'read-only-audit.json')==c['audit_sha256']
success=bool(a['full_second_dig_passed']);frames=p['frames'];cursor=p['reference_cursor'];fill=a.get('retained_fill_fraction')
for n,v in base.items():assert sha(R/n)==v['after'],'concurrent:'+n
texts={n:(R/n).read_text() for n in base}
heading='2026-10-14 第二铲7200帧长时验证：'+('完整挖掘与保载通过' if success else '未完成，保存停止')
anchor='2026-10-14-第二铲7200帧长时验证'+('完整挖掘与保载通过' if success else '未完成保存停止')
result=f'''用户确认新增30分钟、最多7200物理帧，最后5分钟仅保存审计，接受时间输入仍按frame/1560而可能超过训练范围的风险。23项CPU预检通过后，从真实挖掘前对齐源f31c6497…在一个新原生续接实例严格恢复24数组/原生目标/扫掠/时钟，并通过初始221维输入、冻结首动作及首3帧粒子/compact state/发令完全复现门。只扩大隔离运行器的帧上限；实际关节等待误差映射、冻结模型、221维编码与时间归一化、原进度算法/走廊/10%轴速/0%停止/跨模式作废及所有保护均保留。'''
if success:
 outcome=f'''**本次来源上的完整第二铲挖掘及0.5秒保载已验证**：共{frames}物理帧（{frames/60:.6f}仿真秒），参考1170/1170，保载30帧，三个有效量测点的最低短保载显示值为{fill*100:.6f}%。该值是当前仿真显示口径，不是已标定净入车量；不与不同初态的人工/几何或原独立冻结148.2321%直接作优劣结论。本轮没有运输卸料或第三铲，不授予同实例连续三铲成功率或跨场景泛化。来源为首铲真实末态经已验证第二铲接近后保存的对齐点，再原生恢复续接，不能冒称从初始场景不间断三铲。'''
 next_='''下一步候选：将本次合格保载末态作为第二铲运输源，先做来源与控制侧/几何端点CPU核验，再讨论原生零步/运输短检及完整运输卸料预算；本轮不自动启动运输、不从未核验源直接继续第三铲。第三铲仍需从新的真实返回末态重新选点/转向接近/挖掘验证，不能重复初始土堆。S05继续收口连续多铲接口，其他小节不因本单次结果升级验收。'''
else:
 outcome=f'''**完整第二铲仍未完成**：执行{frames}物理帧（{frames/60:.6f}仿真秒），参考{cursor:.6f}/1170，状态{p['status']}，失败{p.get('failure')}，保存recovery。未达到完整1170参考加30帧保载，没有合格新保载满斗率、运输或第三铲结果。不得用长时部分推进代替整铲成功；旧1560帧上限289/1170和旧映射76的失败记录均保留，不追溯改判。'''
 next_='''下一步只做已有缓存/原生目标与实际跟踪的CPU诊断，区分持续慢响应、反复等待及无法恢复的位置；不继续盲增帧数、不自动改进度匹配或控制律。任何控制接口、帧上限、模型输入/奖励或新物理预算变化需另行讨论确认。S05继续进行中，完整第二/第三铲及连续三铲尚未验收。'''
evidence=f'''{a['check_count']}项独立CPU审计全部通过：{a['policy_decisions']}次冻结动作预测最大复算误差{a['maximum_prediction_error']}、221维输入由实际土石/姿态/原始世界力矩重建最大误差{a['maximum221_input_error']}，逐物理帧新等待发令/节奏复算最大误差{a['new_law_max_error']}。所有原硬界/碰撞/材料/扫掠门通过，绑定模型/源码/输入SHA不变、训练更新0、GPU已空闲。审计记录成功或失败的实际边界，不以记录审计通过冒充业务成功。'''
timeline=f'''等待时间线：{diag['waiting_frames']}/{frames}帧等待（{100*diag['waiting_frames']/frames:.3f}%）、{diag['follow_frames']}帧跟随、{diag['hold_frames']}帧保载，实际匹配进度{diag['actual_progress']}，最大参考领先{diag['max_reference_lead']}帧，未放宽原6帧领先和走廊。全部真实物理快照/粒子缓存/发令/状态/策略输入输出保留，可用于后续媒体与诊断，但任何视频须标注真实运行来源及完整/失败边界。终点允许黏附的用户口径保持。'''
closed=f'''新增预算{read(W/'budget.json')['started_at']}开始，原截止{c['original_budget_expired_at']}。仿真在{c['physical_finished_at']}于原预算内完成并停止，物理运行耗时{p['wall_seconds']:.3f}秒；会话中断使只读CPU审计与收尾记录延后至{c['closed_at']}补齐，审计耗时{a['wall_seconds']:.3f}秒，未新增物理或训练。closure的elapsed_seconds包含中断间隔，不冒充计算用时或在30分钟内回传；物理结束时余时{c['unused_seconds_at_physical_completion']:.3f}秒未再使用，预算已过期封存。本轮不运输、不做第三铲、不重开旧预算，不自动重试，无Git提交。'''
links='''证据：[物理结果](../../runs/second-scoop-qerror-long-20261014-v1/report.json)、[独立审计](../../runs/second-scoop-qerror-long-20261014-v1/read-only-audit.json)、[等待时间线](../../runs/second-scoop-qerror-long-20261014-v1/wait-timeline-diagnosis.json)、[预算关闭](../../work/second-scoop-qerror-long-20261014-v1/closure.json)、[协议](../../work/second-scoop-qerror-long-20261014-v1/protocol.json)。'''
body='\n\n'.join([result,outcome,evidence,timeline,closed,next_])
n='docs/plans/S05-imitation-and-rl.md';i=texts[n].index('## ');texts[n]=texts[n][:i]+f'## {heading}\n\n{body}\n\n{links}\n\n'+texts[n][i:];texts[n]=texts[n].replace('## 2026-10-14 实际关节等待纠偏7200帧：新增30分钟已授权','## 历史：第二铲7200帧/30分钟授权，现已执行关闭',1)
def first(n,title,body):
 s=texts[n];i=s.index('## ');j=s.index('\n## ',i+3);texts[n]=s[:i]+f'## {title}\n\n{body}\n'+s[j:]
brief=f'''7200帧上限/新增30分钟检验已执行，实际{frames}物理帧、参考{cursor}/1170，'''+('完整第二铲与30帧保载通过，最低短保载显示'+f'{fill*100:.6f}%。' if success else '完整第二铲未完成，失败保存，无合格保载值。')+f'''{a['check_count']}项审计通过，模型/221维编码/时间归一化/原保护保持、训练0，预算已关闭、余时封存。本轮没有第二铲运输或第三铲。'''
first('docs/CURRENT_STATUS.md',heading,body+f'\n\n当前由[S05](plans/S05-imitation-and-rl.md#{anchor})维护。')
first('docs/PROJECT_PLAN.md',heading,brief+f' 仅S05推进当前连续多铲接口，D-188；下一门见[S05](plans/S05-imitation-and-rl.md#{anchor})。')
first('docs/plans/INDEX.md',heading,brief+f' 其他小节不自动升级验收，见[S05](S05-imitation-and-rl.md#{anchor})。')
n='docs/acceptance.md';i=texts[n].index('## 当前能力摘要');j=texts[n].index('\n## ',i+3);cap='残差连续首铲已验证；'+('第二铲完整挖掘与保载已验证、运输/第三铲未完成' if success else '第二铲长时验证仍未完成、连续三铲未完成');texts[n]=texts[n][:i]+f'## 当前能力摘要（2026-10-14）：{cap}\n'+texts[n][j:];i=texts[n].index('\n## ',texts[n].index('## 当前能力摘要')+3);texts[n]=texts[n][:i]+f'\n## {heading}\n\n{body}\n\n证据见[S05](plans/S05-imitation-and-rl.md#{anchor})。\n'+texts[n][i:]
n='docs/decisions.md';assert '## D-188' not in texts[n];i=texts[n].index('## D-187');texts[n]=texts[n][:i]+'''## D-188：保持映射和时间归一化，以7200帧/新增30分钟验证完整第二铲（2026-10-14）

'''+result+'\n\n'+outcome+'\n\n'+evidence+'\n\n'+closed+'\n\n'+next_+f'\n\n证据见[S05](plans/S05-imitation-and-rl.md#{anchor})。\n\n'+texts[n][i:];idx='## 当前决策索引（2026-10-14）';i=texts[n].index(idx)+len(idx);texts[n]=texts[n][:i]+f'\n\n- `D-188`：{brief}\n'+texts[n][i:]
backup=W/'record-before-final';backup.mkdir(exist_ok=False)
for n,v in base.items():
 assert sha(R/n)==v['after'];dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/n,dst)
for n,s in texts.items():(R/n).write_text(s)
(W/'record-final-receipt.json').write_text(json.dumps({n:dict(before=base[n]['after'],after=sha(R/n)) for n in texts},indent=2))
print('Updated six existing records with actual outcome, not a pending plan.')
