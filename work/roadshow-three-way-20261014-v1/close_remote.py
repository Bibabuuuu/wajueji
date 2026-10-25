from pathlib import Path
import json, hashlib, re, datetime, subprocess

repo = Path('/home/qunwang/wajueji')
work = repo/'work/roadshow-three-way-20261014-v1'
run = repo/'runs/roadshow-three-way-20261014-v1'
budget = json.loads((repo/'runs/roadshow-three-way-20261014-budget.json').read_text())
protocol = json.loads((work/'protocol.json').read_text())
protocol.update(status='evaluation_closed', training_updates=0, physical_frames_total=5840,
    closed_for_compute=True, packaging_only_after_deadline=True,
    case_runners={'manual':'compare-v3.py','residual-v3':'compare-v3.py','geometry-full':'compare-v4.py'},
    results={'manual_display_percent':97.06992230854607,'residual_display_percent':148.23212303789444,
             'geometry_display_percent':None,'geometry_reason':'original lift height budget reached; endpoint measurement boundary invalid',
             'audit_checks_passed':96},
    post_budget_scope='Read-only output audit, chart QA, document update and archive; no simulation or training')
(work/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True)
assert gpu.returncode == 0 and not gpu.stdout.strip(), gpu.stdout
closure={'status':'closed','budget_original':budget,'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'physical_frames_total':5840,'training_updates':0,'gpu_compute_processes':[],
         'budget_reopened':False,'all_physical_runs_completed_before_deadline':True,
         'physical_end_budget_remaining_seconds':413.730818,
         'after_deadline':'Only chart repairs/QA, source/data copying, documentation and archive; no new experiment',
         'audit':'work/roadshow-three-way-20261014-v1/independent-audit.json',
         'geometry':'Original 0.35m lift cap reached; score unknown, not zero. Raw exception preserved.'}
(work/'closure.json').write_text(json.dumps(closure,ensure_ascii=False,indent=2)+'\n')
block='''## 2026-10-14 三方同域对照Stage 03：实验已关闭，结果与边界已核验

用户新增2700秒仅三方对照、不训练：08:50:56开始、09:35:56固定截止。全部物理运行在截止前完成（最后尚余413.73秒），总5840物理帧，包括120帧几何短检；首次残差计时接口失败发生在0物理帧并保留。96项独立核验通过。预算不重开；截止后只修复图表标签、复核、更新记录和归档。无GPU计算任务、无新训练更新、未改正式控制/物理源码、未提交Git，已有未提交改动保留。

三者同700f76be…完整第一初态、D165 CycleRuntime及共同原执行保护；各自原任务目标/额外内部约束和60/60/20Hz仿真调度保留。短保载显示口径为相同容量/占据阈值0.2，末0.5秒三点有效值取最低，含土石/堆高、未标定，不是正式稳定净入车体积，终点不设净空门槛。

| 方案 | 短保载满斗显示 | 软件采集至提交P95 | 仿真执行/观察时长 |
|---|---:|---:|---:|
| 较低装载人工示例 | 97.0699% | 人类认知反应未测 | 26.7333秒 |
| 当前示范参考＋残差 | 148.2321% | 59.4620ms | 20.10秒 |
| 纯几何＋既有优化预览v2 | 无有效值 | 76.9896ms | 48.50秒后停止 |

几何2910帧后达到原0.35米抬升上限，末态量测提示cavity-connected material touches measurement boundary；保留failure.npz和原异常。所有实际执行帧硬保护通过，已进入lift，不解释成空斗、0%或完全不能挖掘。缺有效量测，不能做完整三方装载量排名。

当前整体策略相对预先选定的人工例高51.1622个百分点、仿真轨迹短24.81%；该人工例是时间顺序首条同源合格较低记录，不是人工平均/最低例。其历史40.0476%仅用于选择，同域重放后正式得97.0699%。当前模型参考另一条较好人工轨迹145.3813%、20.00秒，实际残差增量2.8508个百分点、增加0.10秒，不能把51.16个百分点全部归功于SAC。

本次软件采集至提交P95较几何低22.77%；残差>100ms为0/402、几何4/2910。包含预处理/原保护，不是单网络forward；各阶段、冷暖和原始慢值保留。人工反应/执行器接受未测。请求周期含物理/日志且不同步数，仿真60/20Hz不是墙钟实时性。归一化指令总变差每仿真秒：几何1.3080、人工2.3875、残差3.9663，当前不宣称平顺性最佳。

素材：`runs/roadshow-handover-20261012/stage-03/`及Windows `deliverables/roadshow-three-way-stage03-20261014/`；三张300dpi PNG/文字可编辑SVG/PDF、指标表、原始响应、源码/协议快照、96项审计和图表QA。新残差动作/24类末态与Stage 02原验收逐值相同，可复用该视频。原物理日志/大缓存留在`runs/roadshow-three-way-20261014-v1/`，来源及模型SHA绑定。

下一步优先解决几何末态量测缺口，再补正式运输卸料的同域接口和净入车量测；这些尚未实现或授权新计算，本轮不追加运行。S05/S06总体状态不因一次素材对照改成整体完成。
'''
for rel in ['docs/plans/S05-imitation-and-rl.md','docs/plans/S06-three-scoop-objective-evaluation.md','docs/CURRENT_STATUS.md']:
    path=repo/rel
    old=path.read_text()
    pattern=r'^## 2026-10-14 三方同域对照：已授权并实施\n.*?(?=^## |\Z)'
    new,count=re.subn(pattern,block+'\n',old,flags=re.M|re.S)
    assert count == 1, (rel,count)
    path.write_text(new)
for rel in ['docs/PROJECT_PLAN.md','docs/roadshow/PPT_DATA_REQUEST.md']:
    path=repo/rel
    old=path.read_text()
    assert '## 2026-10-14 三方同域对照Stage 03' not in old
    first_section=re.search(r'^## ',old,re.M)
    pos=first_section.start() if first_section else old.find('\n')+1
    path.write_text(old[:pos]+block+'\n'+old[pos:])
path=repo/'docs/acceptance.md'
old=path.read_text()
pos=old.find('\n')+1
current=block.replace('## 2026-10-14 三方同域对照Stage 03：实验已关闭，结果与边界已核验','## 当前能力摘要（2026-10-14）：Stage 03三方对照已归档，几何量测有缺口')
path.write_text(old[:pos]+'\n'+current+'\n'+old[pos:].replace('## 当前能力摘要（2026-10-14）：PPT素材Stage 02','## 2026-10-14 历史素材Stage 02',1))
print(json.dumps({'closed':True,'physical_frames':5840,'training_updates':0,'docs_updated':6,'gpu_idle':True}))
