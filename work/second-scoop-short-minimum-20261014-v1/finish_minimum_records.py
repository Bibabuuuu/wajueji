from pathlib import Path
import json,hashlib,shutil
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-short-minimum-20261014-v1';C=R/'work/second-scoop-rise-hold-20261014-v1';D=R/'runs/second-scoop-short-minimum-20261014-v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=read(D/'report.json');a=read(D/'read-only-audit.json');d=read(D/'stop-diagnosis.json');closed=read(W/'closure.json');cpu=read(C/'cpu-v1/report.json')
assert p['status']=='failed' and p['frames']==2159 and a['status']=='diagnostic_audit_passed' and cpu['status']=='cpu_phase_candidate_passed' and all(cpu['checks'].values())
base=read(W/'record-start-receipt.json')
for n,v in base.items():assert sha(R/n)==v['after'],'concurrent:'+n
texts={n:(R/n).read_text() for n in base}
heading='2026-10-14 最低点短检在高位保持停止，阶段切换CPU已验证';anchor='2026-10-14-最低点短检在高位保持停止阶段切换cpu已验证'
physical='''最低点版本已执行2159帧（35.9833仿真秒）并失败停止。原返回源6fe40910…的24数组/原生控制/扫掠/时钟恢复通过，同一续接实例完成2151帧名义主动上抬（全局2154帧），斗尖抬高约0.52146米；随后第5个高位跟踪保持帧，最低点回落0.00137373毫米，超过保留的1微米单向门。累计离峰值约0.00238760毫米，最低点距地2.55193米、观测土堆间距1.21826米；原硬界/碰撞/材料/扫掠均通过。尚未转向、第二铲接近或执行残差，策略预测/残差帧/训练均0，不归因于SAC策略表现。'''
audit=f'''29项独立CPU记录审计通过，2156条实际几何发令复算最大差0；独立重查2156个上抬/保持实际帧，确认单向门在准确失败帧触发。物理短检仍失败，不能以审计通过改判。已保存recovery、控制/扫掠快照及全2159帧缓存。新45分钟自20:07:48开始，{closed['closed_at']}关闭，实际总耗时{closed['elapsed_seconds']:.1f}秒、余{closed['sealed_remaining_seconds']:.1f}秒封存；GPU空闲，无Git提交，旧预算不重开。'''
candidate='''用户通过Grill确认先做“上抬→高位保持”CPU模式切换，未选择保留跨保持单向门并重做四轴协调。候选RiseHoldGuard中，主动上抬保留最低点1微米单向门；仅在名义2151步全部完成、实际轴误差通过原0.015端点门且静态/机构土堆/32厘米间距通过后，原子切到高位位置保持。保持继续原间距/限位/闭链/跟踪检查，并维持0.015端点误差门，不再要求高度逐帧只增；原物理材料/扫掠保护由未来原运行器继续执行，候选本身不调用物理。'''
verified='''22项CPU检查通过：2151个真实主动上抬帧仍通过单向门，同一失败回落在rise模式仍拒绝；准入后5个真实缓存保持帧通过。提前/重复切换、未到位、32厘米不足、保持漂移、非有限状态等被拒绝，失败准入不改变mode。仅5个真实保持帧可用，更长保持、真实原子切换、后续转向/接近/残差仍未物理验证。原2159帧失败和此前371帧失败均保留；此CPU阶段新增物理/初始化/推理/训练均0，模型及正式源码不改。'''
next_='''**下一步：待新增物理预算**。建议接入已确认阶段切换，从原合格返回源重新恢复，验证完整上抬与高位保持，再转向/张斗/第二铲接近及至多120帧冻结残差；不训练、不跑完整第二铲，失败或到期保存停止。所有旧预算关闭。S05仍进行中，第二/第三铲和同实例三铲未完成。'''
links='''证据：[2159帧物理结果](../../runs/second-scoop-short-minimum-20261014-v1/report.json)、[29项记录审计](../../runs/second-scoop-short-minimum-20261014-v1/read-only-audit.json)、[高位保持诊断](../../runs/second-scoop-short-minimum-20261014-v1/stop-diagnosis.json)、[预算关闭](../../work/second-scoop-short-minimum-20261014-v1/closure.json)、[22项阶段切换CPU](../../work/second-scoop-rise-hold-20261014-v1/cpu-v1/report.json)。'''
n='docs/plans/S05-imitation-and-rl.md';pos=texts[n].index('## ');texts[n]=texts[n][:pos]+f'## {heading}\n\n'+physical+'\n\n'+audit+'\n\n'+candidate+'\n\n'+verified+'\n\n'+next_+'\n\n'+links+'\n\n'+texts[n][pos:]
texts[n]=texts[n].replace('## 2026-10-14 最低点版本45分钟短检：已授权，启动前CPU检查通过','## 历史：最低点版本45分钟短检获授权，现已失败停止并关闭',1)
def replace_first(n,head,body):
 s=texts[n];i=s.index('## ');j=s.index('\n## ',i+3);texts[n]=s[:i]+f'## {head}\n\n{body}\n'+s[j:]
brief='最低点版完成2151帧主动上抬，在高位保持第5帧触发1微米单向门，2159帧停止；尚未第二铲接近/残差。29项记录审计通过，失败不改判，预算已关闭。用户另确认上抬→高位保持切换，22项CPU验证通过；真实切换和后续路径未验证。'
replace_first('docs/CURRENT_STATUS.md','2026-10-14 当前：高位保持触发停止；阶段切换CPU通过，物理预算关闭',physical+'\n\n'+audit+'\n\n'+candidate+'\n\n'+verified+'\n\n'+next_+f'\n\n权威实施见[S05](plans/S05-imitation-and-rl.md#{anchor})。')
replace_first('docs/PROJECT_PLAN.md','2026-10-14 当前推进：阶段切换CPU通过，待物理准入',brief+f' 当前仅S05推进新点接口/准入，原连续首铲结论保持；下一门由[S05](plans/S05-imitation-and-rl.md#{anchor})维护。D-185。')
replace_first('docs/plans/INDEX.md','2026-10-14 当前实施S05：上抬/保持切换CPU通过',brief+f' S02/S03/S07为复用依赖，不因此升级验收；下一门见[S05](S05-imitation-and-rl.md#{anchor})。')
n='docs/acceptance.md';i=texts[n].index('## 当前能力摘要');j=texts[n].index('\n## ',i+3);texts[n]=texts[n][:i]+'## 当前能力摘要（2026-10-14）：残差连续首铲已验证；第二铲上抬后保持停止，阶段切换仅CPU通过；第二/三铲未完成\n'+texts[n][j:]
pos=texts[n].index('\n## ',texts[n].index('## 当前能力摘要')+3);texts[n]=texts[n][:pos]+f'\n## {heading}\n\n'+physical+'\n\n'+audit+'\n\n'+verified+f'\n\n详见[S05](plans/S05-imitation-and-rl.md#{anchor})。\n'+texts[n][pos:]
n='docs/decisions.md';assert '## D-185' not in texts[n];pos=texts[n].index('## D-184');texts[n]=texts[n][:pos]+'''## D-185：区分主动上抬与高位位置保持，先验证CPU切换（2026-10-14）

'''+brief+' '+candidate+'\n\n'+verified+f'\n\n这项确认仅授予CPU实现/验证，物理重跑另需新增预算；原两轮预算和失败记录不重开/不改判。实施见[S05](plans/S05-imitation-and-rl.md#{anchor})。\n\n'+texts[n][pos:]
backup=W/'record-before-final';backup.mkdir(exist_ok=False)
for n,v in base.items():
 assert sha(R/n)==v['after'];dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/n,dst)
for n,s in texts.items():(R/n).write_text(s)
(W/'record-final-receipt.json').write_text(json.dumps({n:dict(before=base[n]['after'],after=sha(R/n)) for n in texts},indent=2))
print('Recorded failed physical run and confirmed CPU-only phase candidate in six existing records.')
