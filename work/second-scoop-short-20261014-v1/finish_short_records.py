"""Record the failed physical attempt and separately confirmed CPU candidate."""
from pathlib import Path
import json,hashlib,shutil
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-short-20261014-v1';C=R/'work/second-scoop-minimum-guard-20261014-v1';D=R/'runs/second-scoop-short-20261014-v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=read(D/'report.json');a=read(D/'read-only-audit.json');d=read(D/'stop-diagnosis.json');c=read(W/'closure.json');cpu=read(C/'cpu-v1/report.json')
assert p['status']=='failed' and a['status']=='diagnostic_audit_passed' and cpu['status']=='cpu_candidate_passed'
base=read(W/'record-start-receipt.json')
for n,v in base.items():assert sha(R/n)==v['after'],'concurrent:'+n
texts={n:(R/n).read_text() for n in base}
section='2026-10-14 第二铲短检停止，最低点保护CPU候选已验证'
anchor='2026-10-14-第二铲短检停止最低点保护cpu候选已验证'
summary=f'''恢复/上抬短检已执行并失败停止：真实首铲返回源6fe40910…的24数组、原生控制/扫掠/时钟恢复通过，一个原生续接实例执行371帧（6.1833仿真秒），随后在RETURN_RISE_0触发新增“所有采样点逐帧不下降”门。上部角点单帧下降{d['minimum_vertex_vertical_step_m']*1000:.7f}毫米，同时斗体最低点上升{d['bucket_minimum_z_last_step_m']*1000:.7f}毫米、斗尖上升、继承土堆间距从26.4524厘米增到30.9747厘米；原硬界/碰撞/材料/扫掠保护均通过。不把这次停止归为SAC失败：第二铲接近尚未开始，策略预测/残差帧/训练均0。25项独立CPU记录审计通过，368条几何发令复算最大差0，物理短检仍为失败。'''
closed=f'''新增45分钟预算已按失败即停关闭：{c['closed_at']}，总耗时{c['elapsed_seconds']:.1f}秒，剩余{c['sealed_remaining_seconds']:.1f}秒封存；GPU空闲，不重开旧预算，不从failure源自动续跑。恢复/失败快照及371帧全帧缓存保留。未Git提交，模型及正式物理源码不改。'''
candidate=f'''用户随后通过Grill明确确认“先做最低点保护的CPU候选”。独立候选仅将RETURN新增高度门改为`min(z_current)-min(z_previous)>=-1e-6米`，保留土堆间距不低于继承基线（原1e-8米数值容差）、APPROACH 32厘米及静态/硬界/材料/扫掠门。28项CPU检查通过：368个真实上抬缓存转移全部通过，旧版仅371帧额外高度门拒绝；超1微米最低点下降、间距不足、非有限输入等反例拒绝，37个接近姿态与旧判定一致。最低点整个缓存上抬段每帧均上升。原失败不追溯改判；该候选无物理运行器/准入，新增物理/预测/训练均0，不能证明后续接近或第二铲完成。'''
next_='''**下一步：待新增物理范围/预算确认**。建议仍从原合格返回源重新恢复，接入最低点候选、原保护保持，完成返回上抬/第二铲接近及至多120帧冻结残差短检；不训练、不跑完整第二铲。原45分钟已封存，CPU候选授权不含物理重跑。当前S05继续进行中，其他小节不升级验收。'''
links='''证据：[原短检](../../runs/second-scoop-short-20261014-v1/report.json)、[25项记录审计](../../runs/second-scoop-short-20261014-v1/read-only-audit.json)、[停止诊断](../../runs/second-scoop-short-20261014-v1/stop-diagnosis.json)、[关闭预算](../../work/second-scoop-short-20261014-v1/closure.json)、[最低点CPU](../../work/second-scoop-minimum-guard-20261014-v1/cpu-v1/report.json)。原审计初版漏设CPU视图的空采集历史，第二版已修复并通过；旧审计错误日志保留，无新物理。'''
n='docs/plans/S05-imitation-and-rl.md';pos=texts[n].index('## ');texts[n]=texts[n][:pos]+f'## {section}\n\n'+summary+'\n\n'+closed+'\n\n'+candidate+'\n\n'+next_+'\n\n'+links+'\n\n'+texts[n][pos:]
texts[n]=texts[n].replace('### 新增45分钟物理短检：已授权实施','### 历史：新增45分钟物理短检获授权，本轮现已失败停止并关闭',1)
def replace_first_section(n,heading,body):
 s=texts[n];start=s.index('## ');end=s.index('\n## ',start+3);texts[n]=s[:start]+f'## {heading}\n\n{body}\n'+s[end:]
replace_first_section('docs/CURRENT_STATUS.md','2026-10-14 当前：第二铲上抬短检停止；最低点CPU候选通过，物理预算关闭',summary+'\n\n'+closed+'\n\n'+candidate+'\n\n'+next_+f'\n\n权威细节见[S05](plans/S05-imitation-and-rl.md#{anchor})。')
brief='第二铲短检在371帧返回上抬阶段触发新增逐点不下降门并停止；最低点持续上升，原硬保护通过，尚未接近或执行残差。25项记录审计通过，失败结果保持。用户另确认最低点保护CPU候选，28项检查通过，但未做新物理。原45分钟已关闭，余时封存。'
replace_first_section('docs/PROJECT_PLAN.md','2026-10-14 当前推进：第二铲短检停止，最低点保护CPU通过',brief+f' 当前仍由S05推进；下一步需要新增恢复/接近/120帧短检预算，不训练。详见[S05](plans/S05-imitation-and-rl.md#{anchor})；连续首铲旧结论不变，第二/三铲未完成。D-184。')
replace_first_section('docs/plans/INDEX.md','2026-10-14 当前实施S05：最低点保护CPU通过，物理待准入',brief+f' 仍仅S05进行接口与准入工作，S02/S03/S07作为依赖复用。下一门见[S05](S05-imitation-and-rl.md#{anchor})。')
n='docs/acceptance.md';start=texts[n].index('## 当前能力摘要');end=texts[n].index('\n## ',start+3);texts[n]=texts[n][:start]+'## 当前能力摘要（2026-10-14）：残差连续首铲已验证；第二铲上抬短检失败停止，最低点CPU候选通过；第二/三铲未完成\n'+texts[n][end:]
pos=texts[n].index('\n## ',texts[n].index('## 当前能力摘要')+3);texts[n]=texts[n][:pos]+f'\n## {section}\n\n'+summary+'\n\n'+closed+'\n\n'+candidate+f'\n\n具体来源见[S05](plans/S05-imitation-and-rl.md#{anchor})。\n'+texts[n][pos:]
n='docs/decisions.md';assert '## D-184' not in texts[n];pos=texts[n].index('## D-183');texts[n]=texts[n][:pos]+'''## D-184：第二铲上抬短检停止后，先验证最低点保护CPU候选（2026-10-14）

'''+brief+''' 用户已确认以“实际斗体最低点逐帧不下降”替代新增的“每个角点逐帧不下降”，沿用1微米容差，继承间距和原32厘米接近等门保持。此确认只授予CPU候选，不授予物理重跑；保留逐点门并重做四轴协调方案未采用。本次原短检25项记录审计通过，但结果仍失败；新候选28项CPU通过不能追溯改判。下一次物理预算需单独确认。

'''+f'实施与完整证据见[S05](plans/S05-imitation-and-rl.md#{anchor})。\n\n'+texts[n][pos:]
backup=W/'record-before-final';backup.mkdir(exist_ok=False)
for n,v in base.items():
 assert sha(R/n)==v['after'];dst=backup/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/n,dst)
for n,s in texts.items():(R/n).write_text(s)
(W/'record-final-receipt.json').write_text(json.dumps({n:dict(before=base[n]['after'],after=sha(R/n)) for n in texts},indent=2))
print('Updated six existing records: failed physical run and separate CPU candidate; no physical budget active.')
