"""Record only the current CPU stage; reject concurrent baseline changes."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path('/home/qunwang/wajueji');W=ROOT/'work/second-scoop-continuation-20261014-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected=json.loads((W/'record-baseline.json').read_text())
for name,value in expected.items():
 assert sha(ROOT/name)==value,'Concurrent edit: '+name
audit=json.loads((W/'independent-cpu-audit.json').read_text());assert audit['status']=='passed'
text={n:(ROOT/n).read_text() for n in expected}
def insert_after_title(n,block):
 head,rest=text[n].split('\n',1);text[n]=head+'\n\n'+block+'\n\n'+rest.lstrip('\n')
brief='''## 2026-10-14 当前推进：第二铲CPU接通，参考输入适配待确认

用户确认推进当前残差策略的连续三铲，先第二铲、后第三铲，先完成再优化装载。当前仍在S05内处理新铲位的参考/策略接口，复用S02/S03/S07已有能力。已取得真实首铲返回后选点、接近路径及人工参考转向的CPU候选，17项回归、28项CPU审计通过；第二铲物理、策略观测转换和三铲尚未完成。路线候选、来源及下一门见[S05](plans/S05-imitation-and-rl.md#2026-10-14-第二铲续接cpu阶段已验证观测坐标适配待确认)。旧预算封存；新物理/训练预算未授予。D-183。'''
insert_after_title('docs/PROJECT_PLAN.md',brief)
old_summary=next(p for p in text['docs/PROJECT_PLAN.md'].split('\n\n') if p.startswith('当前实施S05：'))
text['docs/PROJECT_PLAN.md']=text['docs/PROJECT_PLAN.md'].replace(old_summary,'当前实施S05：第二铲参考及冻结策略接口接通。CPU候选可行，输入坐标方案待确认；原首铲已验证结论保持，尚未执行第二铲/三铲或新训练。详见S05和CURRENT_STATUS。',1)
old_partition='用户已确认：Windows当前会话只负责PPT与路演表达；台式独立GPT负责技术路线、采集和实验，维护台式技术规划与现状。'
assert old_partition in text['docs/PROJECT_PLAN.md']
text['docs/PROJECT_PLAN.md']=text['docs/PROJECT_PLAN.md'].replace(old_partition,old_partition+'2026-10-14本轮用户进一步授权现会话推进连续三铲，本次承担第二铲CPU接通和接口方案准备；新物理/训练及后续关键接口仍按本轮确认范围执行。',1)

n='docs/CURRENT_STATUS.md';s=text[n];first=s.index('## ');end=s.index('\n## ',first+3)
text[n]=s[:first]+'''## 2026-10-14 当前：第二铲CPU准备完成，观测转换方案待确认

本轮无物理/训练任务；GPU计算进程核验为空，全部旧预算保持关闭。用户已确认继续推进第二铲→第三铲→同实例三铲。CPU从本次真实返回点6fe40910…重新观测选点并规划，旧源/冻结模型及正式源码未变；17项回归、28项独立CPU审计通过。只转向人工参考的CPU候选可行，尚未修改策略观测、进行新点推理或第二铲开挖。具体结果与下一门由[S05](plans/S05-imitation-and-rl.md#2026-10-14-第二铲续接cpu阶段已验证观测坐标适配待确认)维护。

待讨论：先转向参考＋观测转回训练坐标的CPU适配，或精确位置/高度适配并补样复训。新物理范围/预算尚未确认；返回末态有bucket MPM接触汇总样本，尚未判定黏附/连地归属，不授予严格零接触续跑或三铲资格。终点允许黏附，实际装载量标定维持低优先级。PPT单铲媒体已交付，原首铲结论保持。工作包`work/second-scoop-continuation-20261014-v1/`，无Git提交。
'''+s[end:]

section='''## 2026-10-14 第二铲续接：CPU阶段已验证，观测坐标适配待确认

**已确认/已授权实施**：用户要求继续当前残差策略的第二铲接通、验证新点适配，再接第三铲；优先三铲完成，各铲装载先记录，不要求均达到首铲，不立即复训。先以CPU准备推进，旧预算不重开，新物理/训练范围和预算尚未确认。工作区仍为台式`/home/qunwang/wajueji`，分支`codex/roadshow-metrics-20261011`，HEAD`867bb546…`，保留原未提交修改。

**已实现/CPU已验证**：从D182本次真实返回末态`runs/continuous-single-scoop-20261014-v1/final.npz`（SHA`6fe4091069bdbedb2ea99e5a44475047f8f4367957a1fb0c5745f0045137ef8e`）重新观察，256441土粒/47石、原库存及277.83333333333223秒时钟保留。原选点器生成8候选，选定接触点(5.8472,0.6974,0.6774)m，下一条带仅为机会估计，第三铲仍须真实第二铲后重选。CPU-v4得到4198帧名义接近路径；继承返回点仅26.495cm几何间距，故先延长RETURN向高位伸展约52.16cm，再进入仍需32cm的APPROACH。返回扩展保持原静态运输保护并额外限制继承间距不减/斗体采样点不下降。车体AABB保守拒绝复用既有CAD细化，4/2cm车体、2cm机构地形、硬界及闭链门保留。

原参考直接接入被原15cm走廊拒绝，位置误差72.103cm；一次独立CPU探针将1171帧人工参考只转向7.30835°，三缸参考、原速度和时序保持，增加81帧名义末端对齐后入口匹配误差0.6407mm。转向参考切入点相对观测点差4.723cm，没有径向/高度变形。全部参考姿态通过原静态/硬限位和1mm物理闭链门，最低斗体距地39.99cm；不证明新土堆动态或策略泛化。原返回末帧187个bucket MPM接触汇总样本尚不能区分黏附/连地，不能宣称严格零接触；用户允许终点黏附的边界保持。

**验证/证据**：17项现有CPU回归、28项独立CPU审计通过；CPU-v4约99.80秒，转向探针v2约14.96秒。独立审计复核4199个记录姿态的闭链/硬界、速度积分及147个边界/每30帧几何状态（确切数量以审计JSON为准），原D182协议绑定的源码/模型/输入/报告不变，GPU空闲。新增物理步/策略推理/训练均0。早期CPU-v1/v2/v3与rotation-probe-v1失败保留，不能以修正版追溯改判。工作包及产物说明：[README](../../work/second-scoop-continuation-20261014-v1/README.md)，[接近及兼容性](../../work/second-scoop-continuation-20261014-v1/cpu-v4/report.json)，[参考转向](../../work/second-scoop-continuation-20261014-v1/rotation-probe-v2/report.json)，[独立审计](../../work/second-scoop-continuation-20261014-v1/independent-cpu-audit.json)。

**待讨论、未实现**：候选A先转向参考，将模型输入中的转角、斗位姿/速度/受力和局部地形一致地转回训练坐标，保留冻结权重，先CPU恒等/旋转等价、单位和模式交接检查；通过后另定物理预算。候选B精确适配位置和高度、重建参考并补样复训，成本更高。用户选择尚未返回，不能把A写成已确认；本次没有接入观测转换或新原生续跑器。两种路线都需真实新点评价，不能靠坐标变换宣称适应性已通过。D-183只确认本轮推进范围，不预授予这些接口变更、物理预算或三铲能力。'''
# Preserve the header metadata; add this meaningful stage before historical results.
n='docs/plans/S05-imitation-and-rl.md';pos=text[n].index('## ');text[n]=text[n][:pos].rstrip()+'\n\n'+section+'\n\n'+text[n][pos:]
# Use the actual independently audited sample count, not a guessed prose count.
text[n]=text[n].replace('147个边界/每30帧几何状态',str(audit['independent_geometry_states'])+'个边界/每30帧几何状态')

n='docs/acceptance.md';lines=text[n].splitlines();lines[2]='## 当前能力摘要（2026-10-14）：残差连续首铲已验证；第二铲仅CPU几何/参考可行，物理与三铲未验证';text[n]='\n'.join(lines)+'\n'
pos=text[n].index('\n## 2026-10-14 统一残差连续单铲')
evidence='''
## 2026-10-14 第二铲CPU前置证据

17项现有CPU回归、28项独立CPU审计通过：本次真实返回状态的选点/接近候选及人工参考只转向候选可行，原源/模型/协议源码不变。没有执行物理步、策略推理或训练；观测坐标转换尚未接入，第二/第三铲和同实例三铲均未验证。原返回末帧bucket接触汇总未作黏附/连地分类，不授予严格零接触。具体实施和证据链接由[S05](plans/S05-imitation-and-rl.md#2026-10-14-第二铲续接cpu阶段已验证观测坐标适配待确认)维护。
'''
text[n]=text[n][:pos]+evidence+text[n][pos:]

n='docs/decisions.md';assert '## D-183' not in text[n];pos=text[n].index('## D-182')
decision='''## D-183：推进当前残差连续三铲，先收口第二铲CPU接口（2026-10-14）

用户接受“第二铲接通→新点参考适配→第三铲→同实例连续三铲”，先以完成为目标，装载率记录后优化，不立即复训；本轮已完成CPU草案及审计，来源和具体实现由S05维护。旧计算预算均保持关闭，新物理/训练预算尚未确认。

新证据：原固定参考不能直接接新铲位；只转向参考具有CPU几何可行性，但冻结策略含固定坐标输入，不能只旋转参考后直接推理。A“参考转向＋观测转回训练坐标”与B“完整位置/高度适配及补样复训”已进入Grill讨论，选择待返回。D-183不将A/B写成用户已确认，也不授予物理续跑或三铲验收。见[S05](plans/S05-imitation-and-rl.md#2026-10-14-第二铲续接cpu阶段已验证观测坐标适配待确认)。

'''
text[n]=text[n][:pos]+decision+text[n][pos:]
n='docs/plans/INDEX.md';pos=text[n].index('## 2026-10-14 统一残差连续单铲')
text[n]=text[n][:pos]+'''## 2026-10-14 当前实施S05：第二铲CPU准备已验证

用户已确认继续连续三铲主线，先第二铲接通。当前仅S05承担参考/策略新点接口工作；复用S02/S03/S07代码作为依赖。CPU候选和参考转向通过，观测坐标方案待确认，未运行第二铲/三铲。结果和下一门见[S05](S05-imitation-and-rl.md#2026-10-14-第二铲续接cpu阶段已验证观测坐标适配待确认)，其他阶段验收不因CPU候选而升级。

'''+text[n][pos:]
text[n]=text[n].replace('冻结策略及D182同实例连续单铲已验证；56项审计通过，全帧回放已保存','D182首铲保留；第二铲CPU候选通过，参考/观测适配待确认')

# The user selected A while the initial CPU record was being assembled.
for name in text:
 text[name]=text[name].replace('第二铲续接：CPU阶段已验证，观测坐标适配待确认','第二铲续接：CPU几何已验证，观测转换已授权')
 text[name]=text[name].replace('2026-10-14-第二铲续接cpu阶段已验证观测坐标适配待确认','2026-10-14-第二铲续接cpu几何已验证观测转换已授权')
 text[name]=text[name].replace('参考输入适配待确认','参考输入适配已授权')
 text[name]=text[name].replace('参考/策略接口，复用S02/S03/S07已有能力。','参考/策略接口，复用S02/S03/S07已有能力。')
 text[name]=text[name].replace('策略观测转换和三铲尚未完成。路线候选、来源及下一门','策略观测转换和三铲尚未完成。用户已选择A转向迁移，授权观测转换接入与CPU验证；来源及下一门')
 text[name]=text[name].replace('输入坐标方案待确认；原首铲已验证结论保持','输入坐标转换已授权、待CPU实现与验证；原首铲已验证结论保持')
 text[name]=text[name].replace('观测转换方案待确认','观测转换已授权，待CPU验证')
 text[name]=text[name].replace('待讨论：先转向参考＋观测转回训练坐标的CPU适配，或精确位置/高度适配并补样复训。','已确认并授权：A先转向参考＋观测转回训练坐标，先接入和CPU验证；不采用B精确位置/高度适配及补样复训作为本轮实施范围。')
 text[name]=text[name].replace('**待讨论、未实现**：候选A先转向参考','**已确认/已授权实施，尚未实现**：用户已选择A先转向参考')
 text[name]=text[name].replace('候选B精确适配位置和高度、重建参考并补样复训，成本更高。用户选择尚未返回，不能把A写成已确认；本次没有接入观测转换或新原生续跑器。','B精确适配位置和高度、重建参考并补样复训保留为未采用备选。用户已确认A；目前尚未接入观测转换或新原生续跑器。')
 text[name]=text[name].replace('D-183只确认本轮推进范围，不预授予这些接口变更、物理预算或三铲能力。','D-183现同时确认A观测转换的CPU实施，不授予新物理预算或三铲能力。')
 text[name]=text[name].replace('已进入Grill讨论，选择待返回。D-183不将A/B写成用户已确认，也不授予物理续跑或三铲验收。','已完成Grill选择，用户明确选择A，并授权接观测转换及CPU验证。B未采用；D-183不授予物理续跑或三铲验收。')
 text[name]=text[name].replace('观测坐标方案待确认，未运行第二铲/三铲。','观测坐标转换已授权、待CPU接入和验证，未运行第二铲/三铲。')
 text[name]=text[name].replace('参考/观测适配待确认','观测转换已授权，CPU验证待完成')
backup=W/'record-before-cpu-stage';backup.mkdir(exist_ok=False)
for name in expected:
 dst=backup/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
for name in expected:assert sha(ROOT/name)==expected[name],'Concurrent edit before write: '+name
for name,s in text.items():(ROOT/name).write_text(s)
receipt={name:dict(before=expected[name],after=sha(ROOT/name)) for name in expected}
(W/'record-update-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Updated',len(text),'existing authoritative records; no source/model/physical changes.')
