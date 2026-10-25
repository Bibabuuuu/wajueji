from pathlib import Path
import json,hashlib,shutil
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-full-dig-20261014-v1';C=R/'work/second-scoop-wait-qerror-20261014-v1';D=R/'runs/second-scoop-full-dig-20261014-v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=read(D/'report.json');a=read(D/'read-only-audit.json');c=read(W/'closure.json');cpu=read(C/'cpu-validation.json');diag=read(D/'stall-diagnosis.json')
assert p['status']=='failed' and p['frames']==1560 and p['reference_cursor']==76 and p['prefix_verified']
assert a['status']=='diagnostic_audit_passed' and a['check_count']==27 and all(a['checks'].values())
assert cpu['status']=='cpu_candidate_passed' and cpu['check_count']==48 and all(cpu['checks'].values())
assert cpu['candidate_sha256']==sha(C/'actual_joint_wait.py') and cpu['checker_sha256']==sha(C/'check_actual_joint_wait.py')
assert sha(D/'report.json')==c['report_sha256'] and sha(D/'read-only-audit.json')==c['audit_sha256'] and sha(D/'stall-diagnosis.json')==c['diagnosis_sha256']
base=read(W/'record-start-receipt.json')
for n,v in base.items():assert sha(R/n)==v['after'],'concurrent:'+n
texts={n:(R/n).read_text() for n in base}
heading='2026-10-14 完整第二铲超时保存；实际关节等待纠偏CPU通过'
anchor='2026-10-14-完整第二铲超时保存实际关节等待纠偏cpu通过'
result='''本轮新增30分钟范围已执行并关闭：从真实挖掘前对齐快照f31c6497…在一个新原生续接实例恢复，24数组/控制目标/扫掠/时钟零步门通过，初始221维输入与首动作、首3个实际物理帧的粒子位置/compact state/发令均与上轮缓存完全一致。随后不重置地继续冻结残差，最终1560帧到上限超时保存，参考进度仍76/1170、实际匹配70；未完成完整第二铲和30帧保载，没有第二铲满斗率。该结果保留为失败，不用后续CPU候选追溯改判。'''
evidence='''27项独立CPU审计全部通过，结论是失败记录完整和输入/动作/保护可信，不是完整第二铲成功。521次冻结预测与221维输入重建差均0，首3帧严格复现通过；原硬限位/碰撞/材料/扫掠检查保留，模型及协议绑定源码未变，训练更新0。真实原生恢复能力在本来源已验证，完整新点挖掘能力尚未验证。'''
cause='''CPU诊断显示等待1483帧；最后120帧策略力度均值约50%，而参考原生驱动设定值与当前设定值误差约1e-15，各轴旧发令最大约1.27e-13，不能补偿实际机构偏差。参考76处实际斗原点误差约3.001毫米，四轴“参考关节−实际关节”误差约[8.7521e-5弧度,-2.3166e-5米,5.9404e-4米,2.3866e-4米]；参考70→76驱动目标完全不变。该证据定位旧等待动作映射卡点，不推断模型通过复训即可解决，也不自动改平台段进度准则。'''
candidate='''用户已确认并授权CPU候选：等待发令误差改为当前参考时刻插值后的四轴实际关节目标减实际关节位置，版本`wait_actual_joint_error_strength_v2_cpu`；非等待动作沿用旧映射，冻结模型和221维观测编码保持，观测中原设定值误差字段仍是原口径。仅交给command_at的对齐副本提供实际误差，原last_alignment/进度/可信度/走廊/端点门不变。沿用0%～100%力度、0%主动停止、10%轴速上限、60Hz剩余误差限幅、20Hz决策和跨模式旧动作作废；保载阶段仍为零指令。旧动作映射经验不能直接混训；本轮无训练或新物理。'''
verified='''CPU候选48项检查及8项原有真实env.step状态替身回归通过，包括缓存卡点0/50/100%、插值/转向不变性、符号/限幅/剩余误差、非法输入拒绝、跨模式多帧作废、下一决策可恢复、模型观测源不被替换及物理step/reset拒绝。原卡点50%力度发令为[0.0052512,-0.001125,0.001125,0.003375]（首轴弧度/秒，其余米/秒），不再是数值零；100%同样受剩余误差与轴速限制。理想关节积分替身只证明CPU命令计算，不证明受载物理恢复、稳定性或装载改善。首两版检查脚本自身的列表abs及小误差测试设定问题已修复，失败脚本留存；候选算法文件未因检查脚本问题改动。'''
closed=f'''物理预算{c['epoch_start']:.6f}起（21:05:02），{c['closed_at']}关闭，耗时{c['elapsed_seconds']:.3f}秒，余{c['sealed_remaining_seconds']:.3f}秒封存；不重开或转给新物理。当前CPU候选没有原生初始化/物理步/训练更新。未提交Git；原有工作区改动保留。'''
next_='''下一步仅为待确认物理方案：另给20分钟，接入候选后从已验证挖掘前对齐源恢复，先做零步及首3帧输入/发令/粒子严格前缀门（新映射在最初非等待段应等同旧版）；通过后在同一实例验证等待纠偏能否越过76帧，最多1560个残差物理帧，若完成再做30帧保载。包含接入CPU预检/保存/独立审计，任一原保护、恢复、前缀或超时失败即保存停止。无训练、运输卸料、第三铲或进度匹配算法修改。当前只完成CPU候选，本段不授予新物理预算；项目AGENTS.md第10.3节要求新范围和高成本物理运行明确授权。终点允许黏附边界保持。S05继续进行中，尚无完整第二/第三铲或同实例连续三铲验收。'''
links='''证据：[完整尝试失败记录](../../runs/second-scoop-full-dig-20261014-v1/report.json)、[27项审计](../../runs/second-scoop-full-dig-20261014-v1/read-only-audit.json)、[卡点诊断](../../runs/second-scoop-full-dig-20261014-v1/stall-diagnosis.json)、[预算关闭](../../work/second-scoop-full-dig-20261014-v1/closure.json)、[CPU候选检查](../../work/second-scoop-wait-qerror-20261014-v1/cpu-validation.json)、[候选代码](../../work/second-scoop-wait-qerror-20261014-v1/actual_joint_wait.py)。'''
body='\n\n'.join([result,evidence,cause,candidate,verified,closed,next_])
n='docs/plans/S05-imitation-and-rl.md';i=texts[n].index('## ');texts[n]=texts[n][:i]+f'## {heading}\n\n{body}\n\n{links}\n\n'+texts[n][i:]
texts[n]=texts[n].replace('## 2026-10-14 对齐快照恢复与完整第二铲：新增30分钟已授权','## 历史：完整第二铲30分钟授权，现已超时保存并关闭',1)
def replace_first(n,head,body):
 s=texts[n];i=s.index('## ');j=s.index('\n## ',i+3);texts[n]=s[:i]+f'## {head}\n\n{body}\n'+s[j:]
brief='真实挖掘前快照原生恢复/首3帧严格复现已通过；完整第二铲冻结尝试在1560帧超时、参考76/1170，无保载或新装载结果。27项审计通过只确认失败证据可信。已定位旧等待映射的设定值误差为零而实际机构仍偏离；用户确认的实际关节误差映射CPU候选48项及8项原回归通过，221维输入/冻结权重/保护不变，新映射经验版本隔离。30分钟预算已关闭、余时封存；新候选物理恢复未验证，新20分钟方案待确认。'
replace_first('docs/CURRENT_STATUS.md','2026-10-14 当前：第二铲等待卡点已定位，实际关节纠偏CPU通过',body+f'\n\n由[S05](plans/S05-imitation-and-rl.md#{anchor})维护。')
replace_first('docs/PROJECT_PLAN.md','2026-10-14 当前推进：第二铲实际关节等待纠偏CPU通过',brief+f' 仅S05进行当前接口/验证，D-187；详见[S05](plans/S05-imitation-and-rl.md#{anchor})。')
replace_first('docs/plans/INDEX.md','2026-10-14 当前实施S05：等待纠偏CPU候选已验证',brief+f' 其他小节不升级验收，见[S05](S05-imitation-and-rl.md#{anchor})。')
n='docs/acceptance.md';i=texts[n].index('## 当前能力摘要');j=texts[n].index('\n## ',i+3);texts[n]=texts[n][:i]+'## 当前能力摘要（2026-10-14）：残差连续首铲已验证；第二铲接近/恢复已验证、完整挖掘超时；新等待纠偏仅CPU通过\n'+texts[n][j:]
pos=texts[n].index('\n## ',texts[n].index('## 当前能力摘要')+3);texts[n]=texts[n][:pos]+f'\n## {heading}\n\n'+body+f'\n\n证据由[S05](plans/S05-imitation-and-rl.md#{anchor})链接。\n'+texts[n][pos:]
n='docs/decisions.md';assert '## D-187' not in texts[n];pos=texts[n].index('## D-186');texts[n]=texts[n][:pos]+'''## D-187：等待改用实际关节误差，保持模型观测，先CPU验证（2026-10-14）

'''+cause+'\n\n'+candidate+'\n\n'+verified+'\n\n'+next_+f'\n\n完整尝试失败和预算关闭记录见[S05](plans/S05-imitation-and-rl.md#{anchor})。\n\n'+texts[n][pos:]
backup=W/'record-before-final';backup.mkdir(exist_ok=False)
for n,v in base.items():
 assert sha(R/n)==v['after'];dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/n,dst)
for n,s in texts.items():(R/n).write_text(s)
(W/'record-final-receipt.json').write_text(json.dumps({n:dict(before=base[n]['after'],after=sha(R/n)) for n in texts},indent=2))
(C/'protocol.json').write_text(json.dumps(dict(version=cpu['version'],candidate_sha256=cpu['candidate_sha256'],checker_sha256=cpu['checker_sha256'],validation_sha256=sha(C/'cpu-validation.json'),observation_contract=cpu['observation_contract'],old_replay_direct_mixing_allowed=False,physical_admission=False,training_admission=False,frozen_sources=cpu['source_hashes']),indent=2))
print('Recorded failed full dig and CPU-only candidate in six existing records; no new physical admission.')
