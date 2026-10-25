"""Update the existing CPU-stage records after the authorized adapter passes."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path('/home/qunwang/wajueji');W=ROOT/'work/second-scoop-continuation-20261014-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((W/'record-update-receipt.json').read_text())
for n,v in receipt.items():assert sha(ROOT/n)==v['after'],'Concurrent record edit: '+n
p=json.loads((W/'adapter-cpu-v3/report.json').read_text());audit=json.loads((W/'adapter-independent-audit.json').read_text())
assert p['status']=='cpu_adapter_passed' and audit['status']=='passed'
texts={n:(ROOT/n).read_text() for n in receipt}
def change(n,old,new):
 assert old in texts[n],(n,old)
 texts[n]=texts[n].replace(old,new,1)
for n in texts:
 texts[n]=texts[n].replace('第二铲续接：CPU几何已验证，观测转换已授权','第二铲续接：CPU几何与观测转换已验证')
 texts[n]=texts[n].replace('2026-10-14-第二铲续接cpu几何已验证观测转换已授权','2026-10-14-第二铲续接cpu几何与观测转换已验证')
change('docs/PROJECT_PLAN.md','策略观测转换和三铲尚未完成。','观测转换已实现并通过CPU；第二铲物理及三铲尚未完成。')
change('docs/PROJECT_PLAN.md','输入坐标转换已授权、待CPU实现与验证','输入坐标转换CPU已验证，待另确认物理预算')
change('docs/PROJECT_PLAN.md','CPU候选可行，输入坐标转换CPU已验证，待另确认物理预算；','CPU候选及观测转换已验证，待另确认物理范围/预算；')
change('docs/CURRENT_STATUS.md','观测转换已授权，待CPU验证','观测转换CPU已验证，待新增物理预算')
change('docs/CURRENT_STATUS.md','只转向人工参考的CPU候选可行，尚未修改策略观测、进行新点推理或第二铲开挖。','只转向人工参考及观测转换已接入独立CPU候选，105项CPU检查和17项适配器独立审计通过；12个原方向真实缓存观测逐字节一致，24组虚拟旋转/模式/裁剪案例通过，48次冻结网络CPU预测用于接口等价检查。没有在真实第二铲物理中推理或开挖，物理入口仍禁用。')
n='docs/plans/S05-imitation-and-rl.md'
change(n,'**已确认/已授权实施，尚未实现**：用户已选择A先转向参考','**已确认方案**：用户已选择A先转向参考')
change(n,'目前尚未接入观测转换或新原生续跑器。','观测转换现已接入独立CPU候选，原生续跑器仍未准入。')
change(n,'新增物理步/策略推理/训练均0。早期CPU-v1/v2/v3','此前几何和参考转向子阶段新增物理步/策略推理/训练均0；随后观测适配CPU另有48次冻结网络预测，仍无物理/训练。早期CPU-v1/v2/v3')
pos=texts[n].index('\n## ',texts[n].index('## 2026-10-14 第二铲续接')+4)
block=f'''
### 本轮A观测转换：已实现、CPU已验证

独立候选`work/second-scoop-continuation-20261014-v1/yaw_transfer.py`保留221/5接口，参考只改回转角及对应驱动目标，三缸参考/原速度指令/时序不变。关节回转角归回训练坐标，关节速度/参考差保持各自标量坐标；斗世界位置、线/角速度、力/力矩和8×8局部土石高度图统一转回训练方向，在转换后执行原裁剪。物理状态、底层目标、土石与模型文件不改。运行和策略观测坐标分别标注；`.reset()`及`.step()`均拒绝实际物理，当前只授予CPU观测适配资格。

105项检查通过：4个真实缓存时刻×跟随/等待/保持共12个原方向221维观测与旧编码逐字节一致；正负7.31°/37°、三种模式、受力有无及超裁剪量的24组虚拟旋转案例通过，最大观测差{p['maximum_observation_error']:.9g}、48次冻结CPU预测的最大动作差{p['maximum_frozen_action_error']:.9g}；原速度/等待/节奏命令保持，非法角度/输入与物理步拒绝。17项独立适配审计复算汇总并核验源码/模型，GPU空闲，新增物理0/训练0。虚拟旋转仅为CPU测试，不是旋转真实场景或第二铲实测。使用既有RL隔离Python/PyTorch，未新增依赖。

首版CPU测试误用紧凑回放字段名，v2测试环境缺PyTorch；失败及源码快照保留，未运行物理。候选编码在检查前已按真实19维qpos/18维qvel的221维布局修正。v3为当前通过证据：[CPU检查](../../work/second-scoop-continuation-20261014-v1/adapter-cpu-v3/report.json)、[独立审计](../../work/second-scoop-continuation-20261014-v1/adapter-independent-audit.json)。

**下一门：待物理范围/预算确认**。建议先原生恢复核验、返回高位伸展/第二铲接近及至多120帧残差短检；不训练、不以本CPU结果授予真实新点效果或完整第二/三铲。若授权更大预算，仍先通过短门，再完整第二铲和重新观测准备第三铲。恢复须绑定原6fe40910…来源并保留24类物理/控制/材料状态，终点黏附允许，原间距/限位/材料/扫掠保护不变，旧预算不重开。
'''
texts[n]=texts[n][:pos]+block+texts[n][pos:]
change('docs/acceptance.md','观测坐标转换尚未接入，第二/第三铲和同实例三铲均未验证。','观测坐标转换已实现并通过105项CPU检查和17项独立审计，48次冻结CPU预测只验证接口旋转等价；没有新增物理或训练，第二/第三铲和同实例三铲均未验证。')
# Qualify earlier zero-inference statement to the initial geometry substage.
change('docs/acceptance.md','没有执行物理步、策略推理或训练；','此前几何子阶段没有执行物理步、策略推理或训练；')
change('docs/decisions.md','B未采用；D-183不授予物理续跑或三铲验收。','B未采用；A已实现并通过105项CPU检查和17项独立适配审计（48次CPU冻结预测、0物理/训练），D-183仍不授予物理续跑或三铲验收。')
change('docs/plans/INDEX.md','观测坐标转换已授权、待CPU接入和验证','观测坐标转换已实现并通过CPU')
change('docs/plans/INDEX.md','观测转换已授权，CPU验证待完成','观测转换CPU已验证，物理预算待确认')
backup=W/'record-before-adapter-finish';backup.mkdir(exist_ok=False)
for n in texts:
 dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dst)
for n,v in receipt.items():assert sha(ROOT/n)==v['after'],'Concurrent record edit before write: '+n
for n,s in texts.items():(ROOT/n).write_text(s)
(W/'adapter-record-update-receipt.json').write_text(json.dumps({n:dict(before=receipt[n]['after'],after=sha(ROOT/n)) for n in texts},indent=2))
print('Recorded CPU adapter completion in',len(texts),'existing records; physical budget remains pending.')
