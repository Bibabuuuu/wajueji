from pathlib import Path
import json,hashlib,shutil
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-qerror-dig-20261014-v1';D=R/'runs/second-scoop-qerror-dig-20261014-v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=read(D/'report.json');a=read(D/'read-only-audit.json');c=read(W/'closure.json');diag=read(D/'wait-timeline-diagnosis.json');base=read(W/'record-start-receipt.json')
assert p['status']=='failed' and p['frames']==1560 and p['reference_cursor']==289 and p['prefix_verified']
assert a['status']=='diagnostic_audit_passed' and a['check_count']==31 and all(a['checks'].values()) and a['new_law_max_error']==0
assert diag['waiting_frames']==1260 and diag['follow_frames']==300 and diag['hold_frames']==0
assert sha(D/'report.json')==c['report_sha256'] and sha(D/'read-only-audit.json')==c['audit_sha256']
for n,v in base.items():assert sha(R/n)==v['after'],'concurrent:'+n
texts={n:(R/n).read_text() for n in base}
heading='2026-10-14 实际关节等待纠偏物理恢复通过；完整第二铲仍超时'
anchor='2026-10-14-实际关节等待纠偏物理恢复通过完整第二铲仍超时'
result='''新增20分钟已执行并关闭。22项接入CPU预检通过后，从已验证挖掘前对齐快照f31c6497…在一个新原生实例严格恢复24数组/底层目标/扫掠/时钟；初始221维输入、冻结动作和首3帧实际粒子/compact state/发令完全复现。接入原CPU候选相同SHA的实际关节等待误差映射，模型/原221维编码/原保护/进度算法不变，无复训。原参考76处等待在物理78–109帧共32帧后解除，后续也从172/173处长等待恢复并继续。'''
failure='''**局部恢复已验证，完整第二铲未通过**：1560物理帧（26仿真秒）达到授权上限后timeout保存；最终参考289/1170、实际匹配283，比旧映射固定76有推进，但仍未完成1170参考和30帧保载。没有第二铲保载满斗率、运输/第三铲或连续三铲结果。不得将本轮当作装载改善、整铲成功或新地形泛化优势；原76帧卡点版本的失败记录保持。终点允许黏附边界保持。'''
evidence='''31项独立CPU审计全通过：521次冻结动作重算差0、221维输入由实际土石/姿态/世界力矩缓存重建差0，1560条新映射发令和节奏逐物理帧复算差0；等待方向/0%停止/10%轴速及跨模式作废核验通过，所有原硬界/碰撞/材料/扫掠保护通过，模型和绑定源码SHA不变、训练更新0、GPU已空闲。CPU候选类仍拒绝物理step/reset；本轮仅由绑定新预算和单实例门的隔离wrapper授权调用原env.step，动作版本沿用不可变候选标识`wait_actual_joint_error_strength_v2_cpu`，物理准入在wrapper单独标记，旧经验仍不可直接混训。'''
diagnosis='''等待耗时是本轮主要限制：1260/1560帧（80.769%）等待、300帧跟随、0帧保载。参考172固定等待207–850帧共644帧，参考285也有长等待；在172段第三轴实际关节误差曾达约10.92毫米，命令到应用差0，但原生目标移动约12.03毫米时实际关节仅移动约3.73毫米，显示原生目标和受载实际响应的跟踪滞后。部分小误差轴反复变号，尚不能仅凭符号次数判定整体控制不稳定。1560帧失败不再是旧映射的零发令吸收态；也不能据此保证增加帧数即可完成。'''
closed=f'''预算{read(W/'budget.json')['started_at']}开始，{c['closed_at']}关闭，耗时{c['elapsed_seconds']:.3f}秒，余{c['sealed_remaining_seconds']:.3f}秒封存。物理执行约258.31秒，审计约13.69秒；封存后无新物理或训练，不从failure快照自动续跑。原始NPZ/粒子缓存/物理日志保留远端，无Git提交。'''
next_='''后续仍待讨论/授权：可保持当前映射，单独增加第二铲帧上限以验证慢恢复最终能否完成；代价是更多物理计算，且冻结模型的局部时间输入`frame/1560`会超过训练时范围，不能无说明地改归一化或宣称可复用旧评价。另一条是先做CPU等待控制与原生目标/实际跟踪诊断，再讨论限幅内的反馈节奏或防累积候选；需要新的控制接口确认和物理验证，离线不能证明受载结果。推荐先利用现有日志收窄这一选择，当前不实施新的控制律、不改变上限或进度门，不授予新物理预算。S05仍进行中，完整第二/第三铲未完成。'''
links='''证据：[物理结果](../../runs/second-scoop-qerror-dig-20261014-v1/report.json)、[31项独立审计](../../runs/second-scoop-qerror-dig-20261014-v1/read-only-audit.json)、[等待时间线](../../runs/second-scoop-qerror-dig-20261014-v1/wait-timeline-diagnosis.json)、[预算关闭](../../work/second-scoop-qerror-dig-20261014-v1/closure.json)、[绑定隔离wrapper](../../work/second-scoop-qerror-dig-20261014-v1/run_qerror_second.py)。'''
body='\n\n'.join([result,failure,evidence,diagnosis,closed,next_])
n='docs/plans/S05-imitation-and-rl.md';i=texts[n].index('## ');texts[n]=texts[n][:i]+f'## {heading}\n\n{body}\n\n{links}\n\n'+texts[n][i:];texts[n]=texts[n].replace('## 2026-10-14 实际关节等待纠偏接入：新增20分钟已授权','## 历史：实际关节纠偏20分钟授权，现已到帧上限保存并关闭',1)
def first(n,title,body):
 s=texts[n];i=s.index('## ');j=s.index('\n## ',i+3);texts[n]=s[:i]+f'## {title}\n\n{body}\n'+s[j:]
brief='实际关节等待纠偏已在真实受载第二铲恢复原76及172/173卡点；但1560帧上限最终参考289/1170、timeout保存，1260帧（约81%）用于等待，无保载/新满斗率。31项独立审计通过，521次预测/221维输入及逐帧发令重算差0，原保护和权重不变，训练0。20分钟预算关闭，余时封存。后续先基于现有等待/原生目标跟踪证据讨论扩大帧上限或控制节奏候选，尚未授权新物理。'
first('docs/CURRENT_STATUS.md','2026-10-14 当前：等待恢复有效，完整第二铲289/1170超时',body+f'\n\n由[S05](plans/S05-imitation-and-rl.md#{anchor})维护。')
first('docs/PROJECT_PLAN.md','2026-10-14 当前推进：第二铲等待恢复已验证，整铲仍未完成',brief+f' 仍仅S05进行当前工作；D-187后续验证见[S05](plans/S05-imitation-and-rl.md#{anchor})。')
first('docs/plans/INDEX.md','2026-10-14 当前实施S05：等待恢复通过，完整第二铲超时',brief+f' 其他小节不升级验收，见[S05](S05-imitation-and-rl.md#{anchor})。')
n='docs/acceptance.md';i=texts[n].index('## 当前能力摘要');j=texts[n].index('\n## ',i+3);texts[n]=texts[n][:i]+'## 当前能力摘要（2026-10-14）：残差连续首铲已验证；第二铲等待恢复有效、整铲289/1170超时；连续三铲未完成\n'+texts[n][j:];i=texts[n].index('\n## ',texts[n].index('## 当前能力摘要')+3);texts[n]=texts[n][:i]+f'\n## {heading}\n\n{body}\n\n证据见[S05](plans/S05-imitation-and-rl.md#{anchor})。\n'+texts[n][i:]
n='docs/decisions.md';i=texts[n].index('\n\n',texts[n].index('## D-187'))
texts[n]=texts[n][:i]+f'\n\n**后续已授权物理验证结论**：{brief} {closed} 完整证据见[S05](plans/S05-imitation-and-rl.md#{anchor})。以下保留先前CPU决策与当时的待确认范围，后续授权及结果以本段和S05最新节为准。'+texts[n][i:]
idx='## 当前决策索引（2026-10-14）';i=texts[n].index(idx)+len(idx);texts[n]=texts[n][:i]+'''

- `D-187`：实际关节等待误差映射CPU48项/原回归8项通过；新增20分钟物理恢复卡点有效，完整第二铲289/1170超时，31项审计通过，预算关闭。
- `D-186`：第二铲接近/残差120帧短检4882帧、41项审计通过；随后完整尝试1560帧超时在76，保留失败。
- `D-185`：主动上抬→高位保持模式切换CPU22项通过，后续已接入并通过短检。
- `D-184`：最低点保护CPU28项通过，物理在高位保持微量回落停止，失败保留。
- `D-183`：第二铲转向参考与221维观测转换CPU已核验，完整三铲仍待完成。
'''+texts[n][i:]
backup=W/'record-before-final';backup.mkdir(exist_ok=False)
for n,v in base.items():
 assert sha(R/n)==v['after'];dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/n,dst)
for n,s in texts.items():(R/n).write_text(s)
(W/'record-final-receipt.json').write_text(json.dumps({n:dict(before=base[n]['after'],after=sha(R/n)) for n in texts},indent=2))
print('Recorded wait recovery and incomplete full second scoop; sealed budget and unchanged next-decision boundary.')
