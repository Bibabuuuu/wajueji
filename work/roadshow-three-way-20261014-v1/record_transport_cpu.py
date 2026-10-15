from pathlib import Path
import json,shutil,subprocess
ROOT=Path('/home/qunwang/wajueji');WORK=ROOT/'work/roadshow-three-way-20261014-v1';out=WORK/'offline-boundary-v2'
handoff=json.loads((WORK/'transport-cpu-candidate-v3/report.json').read_text())
assert handoff['status']=='cpu_handoff_passed'
assert all(all(x['checks'].values()) for x in handoff['sources'].values())
reach=json.loads((WORK/'transport-fixed-route-reachability-v1/report.json').read_text())
assert all(x['status']=='first_lift_pose_rejected' and x['physical_state_unchanged'] for x in reach['sources'].values())
for source,name in [(WORK/'transport-cpu-candidate-v3/report.json','transport-cpu-handoff.json'),(WORK/'transport-nominal-cpu-v1/report.json','transport-nominal-failure.json'),(WORK/'transport-fixed-route-reachability-v1/report.json','transport-reachability.json')]:
    shutil.copyfile(source,out/name)
for name in ['transport_cpu_candidate_v3.py','transport_nominal.py','transport_plan_cases.py']:
    shutil.copyfile(WORK/name,out/'source-snapshot'/name)
note='''
## 进一步CPU接通：已实现局部，完整路线未通过

D165来源绑定的CPU候选已实现，强制禁止原生入口和step。较好人工及冻结残差各8项交接检查通过（共16项）：24类来源数组未变，实际末条零指令继承，首条保斗指令为零，负载偏置继承、限速及原150ms预览通过。前两次仅报告组装因NumPy布尔值JSON序列化失败，未推进物理；v3复用转换并加可执行断言修复，失败源码/目录保留。

完整原名义路径未通过，在首个CLEAR_PILE目标求解前终止，未执行任何名义轨迹指令或物理。两来源独立复核均失败：较好人工闭链误差3.9250mm、位置误差2.4069mm；冻结残差3.9006mm、位置误差2.3961mm；活动边界均为15号斗缸。旧solve_pose对名义关节界两侧各保留8mm，斗缸规划上界0.852m，而两个真实已验收末态分别0.8535216/0.8534835m。当前机械/原物理保护通过不代表该姿态能直接符合更紧的旧规划边界。未放宽求解、几何或机械阈值。

新的关键选择（待讨论）：推荐复用人工控制已有“实际姿态靠近端点时可保持/内收、禁止继续向外”的处理，建立来源绑定运输规划候选，保留当前实际斗缸位置、不主动松斗，原硬限位/地面/车体/闭链精度不放宽；先CPU验证。备选为先微量开斗回到名义规划区间再规划，较易复用旧求解但改变现有不主动松斗原则，可能损失载荷。尚未实施两者，没有授权新物理预算，不把CPU动作交接通过写成完整路线或真实恢复成功。

新增证据：transport-cpu-handoff.json（16项局部通过）、transport-nominal-failure.json、transport-reachability.json（两源同一边界拒绝）。加此前26项，合计42项局部CPU检查通过，完整路径门仍未通过。下一步先确认已存在端点姿态的规划处理，再完成CPU整路径与名义指令预检；原生零步恢复和有界运输之后另确认预算。
'''
p=out/'README.md';text=p.read_text();text=text.replace('状态：离线诊断已验证；D165运输接通方案待具体实施；原生恢复/物理运输尚未验证。','状态：离线诊断及D165局部CPU交接已验证；完整名义路线拒绝，端点规划边界处理待讨论；原生恢复/物理运输尚未验证。')
p.write_text(text+note)
brief='''
#### D165运输CPU接通增量：局部通过，路线边界待讨论

来源绑定候选已实现，原生入口/物理step显式禁止；较好人工和冻结残差各8项动作交接检查通过，实际末零指令/负载偏置继承和原预览通过。完整名义路线首次举升逆解失败：两源闭链误差约3.925/3.901mm，活动边界斗缸15。旧solve_pose含8mm名义规划余量、斗缸上界0.852m，真实已验收末态0.853522/0.853483m；不是新物理故障，未放宽阈值。

待讨论：推荐沿用现有端点姿态可保持/向内而不继续向外的规则，保留实际闭斗位置构造来源绑定运输规划候选；备选先微量开斗回名义区间，但会改变不主动松斗原则。尚未实施边界改动；0新增物理/训练，无原生恢复/完整路径通过结论。补充证据`work/roadshow-three-way-20261014-v1/offline-boundary-v2/transport-cpu-handoff.json`、`transport-nominal-failure.json`、`transport-reachability.json`。此前26项加本16项，共42项局部CPU检查通过，完整路线门失败保留。

'''
for rel in ['docs/plans/S05-imitation-and-rl.md','docs/plans/S06-three-scoop-objective-evaluation.md','docs/CURRENT_STATUS.md','docs/acceptance.md','docs/PROJECT_PLAN.md']:
    p=ROOT/rel;t=p.read_text();anchor='### 2026-10-14 后续离线诊断已验证（0新增物理/训练）'
    assert anchor in t
    t=t.replace(anchor+'\n',anchor+'\n'+brief,1)
    p.write_text(t)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True)
assert not gpu.stdout.strip()
print(json.dumps({'cpu_handoff_checks':16,'full_route':'rejected_at_first_goal_before_dispatch','new_physical_frames':0,'training_updates':0,'gpu_idle':True}))
