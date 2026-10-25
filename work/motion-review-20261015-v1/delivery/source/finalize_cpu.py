"""Freeze CPU evidence and update the existing project records; no physical run."""
from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile
R=Path('/home/qunwang/wajueji');W=R/'work/motion-review-20261015-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=read(W/'audit-v1/report.json');route=read(W/'return-candidate-v3/report.json');control=read(W/'control-handoff-cpu-v2/report.json')
assert audit['status']=='cpu_candidate_audit_passed' and audit['checks_passed']==26 and all(x['passed'] for x in audit['checks'])
assert route['physical_admission'] is False and all(control['checks'].values())
body='''**实施结果：CPU候选已实现并验证，物理尚未准入。** 26项保存路径/原保护/来源复核及9项控制交接检查通过。卸料末态新观测重新选中向内贴边点(5.75,0.15)m；新路线只保留小幅退出车厢和原合格外侧航点两个中间点，随后进入冻结人工参考预备点，删除旧RETURN_SAFE→再次上抬循环。卸空退出允许抬升25cm时同步小幅卷斗0.15rad，这是避开原车厢4cm门的必要候选；与满载零主动斗缸不是同一阶段。全程沿用原CAD4cm/2cm、限位/闭链门，对当前土堆加32cm检查；运输区外只复用已有已知刚性地面语义，最终接近段逐帧通过原完整ROI32cm门。60Hz保存姿态复核最小斗-土堆间距0.344299m、机构-土堆0.0211202m；运输规划120Hz采样，均非连续碰撞或真实动力学证明。

新返回路径2660参考帧、44.3333名义秒；旧卸料后至次铲开挖147.0833是物理日志仿真时长，二者不构成实测提速比。候选没有额外插入各航点保持帧，航点处参考速度/加速度为零，终点后保持与闭环收敛需物理实测。

保斗缸平滑举升候选735参考帧/12.25名义秒（旧588帧/9.8秒），斗缸参考位置固定0.849378692m、主动速度严格0；起止加速度为零，速度及参考加速度不超原发令变化率。复用existing exit_common.controller继承上一帧命令/负载偏置，9项CPU检查通过；未改保持刚度、冻结挖掘模型或物理。尚未证明减振或保料改善。

被拒的直接/纯开斗退出/缺外侧航点候选及首次检查器输出格式错误日志全部保留。新增原生初始化/物理/策略推理/训练均0，所有旧预算关闭，未创建物理运行器或Git提交。证据：work/motion-review-20261015-v1/audit-v1/report.json、return-candidate-v3/、candidate-v1/carry-reference.npz、control-handoff-cpu-v2/report.json、delivery/。下一门为另获明确范围和预算后的原生零步/前缀核验、保斗缸举升短段与卸料后合并接近的物理对照；在此之前维持CPU资格，旧已验证流程和媒体包保留。

'''
receipts=[]
for rel in ['docs/CURRENT_STATUS.md','docs/plans/S07-truck-reproduction-submission.md','docs/decisions.md']:
 p=R/rel;t=p.read_text();before=sha(p);heading='## 2026-10-15 D-194：返回接近合并与保斗缸运输减振，CPU已授权';assert heading in t
 t=t.replace(heading,heading.replace('CPU已授权','CPU候选已验证、物理未准入')+'\n\n'+body,1)
 if rel.endswith('decisions.md'):
  t=t.replace('- `D-194`：卸料后重观测选点、返回接近合并；满载保斗缸并优化其余轴与交接。仅CPU候选已授权，无新物理/训练。','- `D-194`：返回合并候选44.33名义秒、2中间航点；保斗缸平滑举升12.25名义秒，26项路径/来源＋9项交接CPU检查通过。物理/减振/提速未验证，无新预算。',1)
 p.write_text(t);receipts.append(dict(file=rel,before_sha256=before,after_sha256=sha(p)))
for rel,text in [('docs/plans/S05-imitation-and-rl.md','D194运输返回候选已通过26项路径/来源与9项交接CPU检查，返回44.33名义秒、2中间航点；保斗缸平滑举升12.25名义秒。见[S07](S07-truck-reproduction-submission.md)，不改变冻结残差挖掘。物理与减振/提速未验证，无新物理/训练预算。'),('docs/PROJECT_PLAN.md','D194返回接近合并及保斗缸运输减振候选已通过CPU：26项路径/来源、9项交接；新返回44.33名义秒、两个中间航点。下一门是另获范围/预算后的短段物理对照；当前仅CPU资格，实际减振/提速未验证，S05仍进行中。详见[S07](plans/S07-truck-reproduction-submission.md)。')]:
 p=R/rel;t=p.read_text();before=sha(p);first,_,rest=t.partition('\n');p.write_text(first+'\n\n## 2026-10-15 D-194 CPU结果\n\n'+text+'\n\n'+rest);receipts.append(dict(file=rel,before_sha256=before,after_sha256=sha(p)))
p=R/'docs/acceptance.md';t=p.read_text();before=sha(p);first,_,rest=t.partition('\n');p.write_text(first+'\n\n## 2026-10-15 D-194：仅授予CPU候选资格\n\n'+body+rest);receipts.append(dict(file=str(p.relative_to(R)),before_sha256=before,after_sha256=sha(p)))
p=R/'docs/plans/INDEX.md';t=p.read_text();old='D191：59微米容差CPU通过，第二铲物理验证及新预算待确认；D190两铲未完成';new='D192两铲149.96%/109.94%完成；D194返回/减振CPU候选已验证，物理未准入、无新预算';assert old in t;before=sha(p);p.write_text(t.replace(old,new,1));receipts.append(dict(file=str(p.relative_to(R)),before_sha256=before,after_sha256=sha(p)))
run=subprocess.run(['git','diff','--check','--',*[x['file'] for x in receipts]],cwd=R,capture_output=True,text=True);assert run.returncode==0,run.stdout+run.stderr
write=W/'final-record-receipt.json';write.write_text(json.dumps(dict(status='cpu_records_updated',files=receipts,git_diff_check=True,no_commit=True),ensure_ascii=False,indent=2))
D=W/'delivery';D.mkdir(exist_ok=False);(D/'source').mkdir();(D/'reports').mkdir();(D/'paths').mkdir()
for src,dst in [('analysis.json','reports/motion-analysis.json'),('loaded-transfer.csv','loaded-transfer.csv'),('geometry-probe.json','reports/initial-geometry-probe.json'),('candidate-v1/report.json','reports/first-candidate.json'),('return-candidate-v2/report.json','reports/second-candidate.json'),('return-candidate-v3/report.json','reports/return-candidate.json'),('control-handoff-cpu-v2/report.json','reports/control-handoff.json'),('audit-v1/report.json','reports/cpu-audit.json'),('final-record-receipt.json','reports/records-receipt.json'),('candidate-v1/carry-reference.npz','paths/carry-reference.npz'),('return-candidate-v3/return-plan.npz','paths/return-plan.npz')]:shutil.copy2(W/src,D/dst)
for name in ['probe_geometry.py','motion_candidate.py','return_candidate_v2.py','return_candidate_v3.py','control_handoff_cpu.py','audit_candidate.py','finalize_cpu.py']:
 shutil.copy2(W/name,D/'source'/name)
readme='''# 卸料后合并接近与运输减振：CPU候选

用户选择：满载铲斗缸不主动开合，优化其余轴与交接；本轮仅CPU，不运行物理或训练。

## 问题核对

旧流程卸料后先卷斗退出、返回固定姿态，再重新上抬、转向、伸展与张斗。旧卸料后到第二铲开挖147.0833仿真秒，其中RETURN_SAFE32.25秒、次铲RETURN_RISE36.85秒。现有流程已用IK，冗余来自路线衔接。

原抬斗CLEAR_PILE实际斗姿对规划最大偏差4.11005度；该段斗缸命令全零，实际斗缸对名义锁定位最大偏差3.2104mm。是实际机构与计划之间的摆动，不是运输时SAC开合斗。具体动力学激振因果未证明。

## 已实现且CPU验证

新路线：卸料末态新观测与贴边选点 → 抬升25cm并小幅卷斗0.15rad退出车厢 → 同步回转与伸展到原外侧接近点 → 原人工参考预备点。只有两个中间航点；卸空时的小幅卷斗不等于放开满载斗缸。

保存路径2660参考帧，44.3333名义秒。原CAD车斗4cm、机构2cm、限位/闭链保留；当前土堆32cm检查、最终完整ROI32cm和原参考走廊通过。运输区外沿用已有已知刚性地面模型，不冒称新增传感观测。最小保存姿态斗-土堆0.344299m、机构-土堆0.0211202m。规划120Hz采样，另对全部60Hz保存姿态逐帧复核；均不等于连续碰撞或动态安全证明。

保斗缸举升采用五次平滑参考，起止速度/加速度为零，斗缸位置0.849378692m固定、速度严格零；参考从原9.8秒增加到12.25秒。复用已有助手继承上一帧命令与负载偏置，不重设底层位置目标。26项路径/身份审计＋9项交接CPU检查通过。

44.33秒是新名义轨迹，147.08秒是旧物理过程，不能换算为已实测提速百分比。减振、保料、物理节拍及新路线后的挖掘完成能力仍未验证。三个早期失败候选及输出格式问题日志保留在台式机工作目录，不能把被拒路径投入执行。

## 数据与复核

- `reports/cpu-audit.json`：26项CPU复核及最小间距。
- `reports/return-candidate.json`：新观测选点、完整参考准入、各段时长与约束。
- `reports/control-handoff.json`：9项交接检查，只是一帧同状态控制比较，不代表后续动力学已复现。
- `loaded-transfer.csv`：原实际运输参考/斗姿/发令完整序列。
- `paths/return-plan.npz`：qpos为N+1×19，velocity为N×4，参考60Hz；reference_q为原冻结参考的转向副本。
- `paths/carry-reference.npz`：同样的qpos/velocity格式，斗缸保持。
- `source/`：生成及复核源码，使用原Linux依赖与资源；原始大型缓存仍留台式机，身份见报告。

新原生初始化、物理、策略推理、训练均0。现有正式源码、模型、原运行和媒体包未修改；旧预算均关闭。下一步须另确认范围和物理预算，再做保存源原生零步/连续前缀核验、举升减振及卸料后合并接近短段对照。当前没有物理运行器或自动续跑资格。未Git提交。
'''
(D/'README.md').write_text(readme)
manifest='\n'.join(f'{sha(p)}  {p.relative_to(D).as_posix()}' for p in sorted(D.rglob('*')) if p.is_file())+'\n';(D/'MANIFEST.sha256').write_text(manifest)
archive=W/'motion-optimization-cpu-20261015.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p in D.iterdir():tar.add(p,arcname=p.name)
print('CPU_DELIVERY_COMPLETE',archive,archive.stat().st_size)
