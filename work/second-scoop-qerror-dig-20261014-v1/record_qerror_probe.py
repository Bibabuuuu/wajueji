from pathlib import Path
import json,hashlib,shutil
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-qerror-dig-20261014-v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=read(W/'record-final-receipt.json');probe=read(W/'cached-time-input-probe.json');assert probe['case_count']==20 and probe['maximum_absolute_action_change']<.0005 and not probe['frame_cap_changed']
for n,v in base.items():assert sha(R/n)==v['after'],n
texts={n:(R/n).read_text() for n in base}
detail='''追加只读CPU时间输入敏感性检查：取本轮5个已记录状态，将第85号字段由原`frame/1560`分别替换为1560/3120/6000/7200帧的值，共20次冻结预测，动作最大变化0.00046117，权重/物理/帧上限未改。只测试已有状态，不能证明未来状态分布安全或一定完成。证据`work/second-scoop-qerror-dig-20261014-v1/cached-time-input-probe.json`。'''
proposal='''**待讨论、未授权的新方案**：优先保持现有映射与时间归一化，仅将独立第二铲验证的帧上限提高至7200（120仿真秒），另给30分钟总预算，最后5分钟保存/审计；原生零步与首3帧门后，在同一新实例验证完整1170参考和30帧保载。原源/模型/221维编码/0%停止/10%速度/走廊/所有保护不变，无训练、运输或第三铲，失败或到期保存停止，旧预算不重开。根据当前26仿真秒只推进289参考，7200是有限的慢恢复检验上限，不保证完成；冻结模型的时间输入将超出训练时范围，此风险不能由上述20次CPU预测消除。另一方案先讨论等待控制节奏/防目标累积的CPU候选，需要新控制接口确认和后续物理验证，可能增加返工。推荐先维持已验证映射做一次有界完整时长检验，以区分“只是当前上限太短”和“还有后段卡点”，用户选择前不启动。'''
for n in ['docs/CURRENT_STATUS.md','docs/PROJECT_PLAN.md','docs/plans/INDEX.md','docs/plans/S05-imitation-and-rl.md','docs/acceptance.md']:
 s=texts[n];i=s.index('## ');j=s.index('\n## ',i+3);texts[n]=s[:j]+'\n\n'+detail+'\n\n'+proposal+'\n'+s[j:]
n='docs/decisions.md';i=texts[n].index('## D-187');j=texts[n].index('\n\n',i);texts[n]=texts[n][:j]+'\n\n'+detail+'\n\n'+proposal+texts[n][j:]
backup=W/'record-before-probe';backup.mkdir(exist_ok=False)
for n,v in base.items():
 assert sha(R/n)==v['after'];dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/n,dst)
shutil.copyfile(W/'record-final-receipt.json',W/'record-result-receipt.json')
for n,s in texts.items():(R/n).write_text(s)
(W/'record-final-receipt.json').write_text(json.dumps({n:dict(before=base[n]['after'],after=sha(R/n)) for n in texts},indent=2))
print('Recorded cached sensitivity and reviewable7200-frame/30min proposal, not physical authorization.')
