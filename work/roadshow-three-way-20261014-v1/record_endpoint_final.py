from pathlib import Path
import json,re,hashlib,shutil,subprocess
ROOT=Path('/home/qunwang/wajueji');WORK=ROOT/'work/roadshow-three-way-20261014-v1'
out=WORK/'transport-endpoint-final-cpu-v1'
report=json.loads((out/'report.json').read_text());audit=json.loads((out/'read-only-audit.json').read_text())
assert report['status']=='endpoint_nominal_cpu_passed' and audit['status']=='passed' and all(audit['checks'].values())
sources=out/'source-snapshot';sources.mkdir(exist_ok=False)
for name in ['transport_endpoint_candidate_v2.py','transport_endpoint_preflight_v2.py','transport_endpoint_preflight_v3.py','transport_cpu_candidate_v3.py','aggregate_endpoint.py','audit_endpoint_final.py']:
    shutil.copyfile(WORK/name,sources/name)
dependencies=['work/contact-second-fixed-route-20260925-v1/plan.py','work/contact-second-fixed-route-20260925-v1/controller.py',
    'work/contact-second-fixed-route-20260925-v1/clearance.py','work/contact-second-finish-unload-20260925-v4/transport_feedback.py',
    'planning/approach.py','planning/cycle_geometry.py','src/v4/run_layered_47_sparse_manual_contact_episode.py',
    'src/v4/layered_47_cycle_runtime.py','src/v4/layered_47_cycle_checkpoint.py']
manifest={str(ROOT/rel):hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in dependencies}
(out/'dependency-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
table=[]
for name,label in [('better_manual','较好人工'),('frozen_residual','冻结残差')]:
    case=report['sources'][name];p=case['plan'];d=p['endpoint_solver_calls'][0]
    minimum=min(s['minimum_clearance']['bucket_truck_m'] for s in p['stages'])
    table.append(f"| {label} | {case['nominal_commands']} | {minimum*100:.4f}厘米 | {d['closure_m']:.3g}米 | {d['cylinder_m']:.10f}米 |")
readme='''# D165运输闭斗端点CPU验证

状态：用户已确认、CPU候选已实现、两组完整名义路径/指令已验证；原生恢复和真实运输尚未验证。

用户选择保留实际闭斗位置，接入既有端点处理，只授权CPU。本轮使用较好人工基准与冻结残差末态，0新增物理帧、0训练更新，CUDA禁用，候选原生入口与step显式拒绝。旧预算不重开。

## 结果

| 来源 | 完整名义指令 | 路径最小斗—车间距 | 举升逆解闭链误差 | 保持的实际斗缸位置 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

两组均完成CLEAR_PILE/ALIGN/DUMP，共5394条名义指令；29项只读产物审计全部通过。CLEAR_PILE/ALIGN的斗缸位置逐帧保持、斗缸名义速度为零，指令速率/原150ms预览通过，物理数组24类、来源和冻结模型SHA不变。不是实际受载闭环跟踪或稳定净入车量结论，也不包含退出/返回。

## 最小改动与保留边界

新代码只放work候选目录，未改正式规划、控制、物理和模型。复用既有来源绑定CPU构造、固定路线、保斗控制、回转预览及拒绝回滚。

原位姿求解器把斗缸截回0.852米名义规划上界，无法保留约0.8535米的真实已验收姿态。候选将该既有斗缸值作为锁定状态，明确检查仍在原0.37–0.86米机械界内，禁止进一步外推；其余关节8毫米名义余量和原闭链精度不变。锁斗缸后，斗角由机构耦合运动决定并记录，不再额外强求原入段斗角，以免把自由度重复约束。

初版已解决端点逆解，但原3.30米举升点在回转时被原4厘米车体间距拒绝（约3.941厘米），失败保留。最终候选只把举升点提高5厘米至3.35米，平面位置、车位及原卸料目标保持，4厘米斗—车、2厘米机构、地面/机械及闭链保护未放宽。固定运输阶段继续沿用已确认的静态检查策略，土体地形图不作此路线通过依据；真实粒子没有删除或修改。

## 失败与验证来源

- transport-endpoint-preflight-v1：旧高度的回转间距拒绝。
- transport-endpoint-preflight-v2：两组几何路径通过；人工组2697指令完成，残差组逐条检查时触发脚本自设180秒CPU保护。
- transport-endpoint-preflight-v3：SHA绑定复用已通过的残差几何路径，单独重做该组完整2697条CPU名义指令，61.57秒完成；不重算人工组。
- report.json：按来源聚合已完整通过的结果，原超时保留，不把原失败报告改为成功。
- read-only-audit.json：29项只读核验，物理/训练计数0。
- 两个*-plan/：NPZ名义路径和计划报告；source-snapshot及dependency-sha256.json记录实际候选与依赖版本。

## 下一道门

从相同来源做原生零步恢复、控制/材料/扫掠内部态精确检查，再做短受载保斗及有界配对运输；新增物理预算须另确认。稳定混合净入车体积仍待定义/标定，不拿TruckReceiver几何计数冒充正式体积。CPU成功不授予物理成功、正式装车收益、完整循环或泛化。
'''
(out/'README.md').write_text(readme)
summary='''#### D165闭斗端点CPU验证已完成，物理尚未启动

用户已确认保留实际闭斗并接入端点处理，仅CPU。候选已实现：锁定0.8535米左右实际斗缸状态并检查原机械界，保持/向内而禁止继续外推；其余关节余量及机械/碰撞/闭链精度不放宽。斗缸锁定后斗角随机构耦合变化并记录，不再额外强求原斗角。初版回转间距约3.941厘米被原4厘米门拒绝；最终只把举升点3.30→3.35米，平面位置、车位与卸料目标保持。

较好人工与冻结残差两组均完成CLEAR_PILE/ALIGN/DUMP的CPU路径及2697条名义指令，总5394条；29项只读审计全部通过，保斗/指令速率/原预览、来源/24类物理数组及冻结模型SHA不变。180秒CPU预检超时保留，人工组已完成不重算，残差复用已过几何路径单独完整复核。正式控制/物理源码未改，0新增物理/训练，原生入口及step仍禁止，GPU空闲。

下一门为原生零步恢复和短受载/配对运输，新增物理预算未确认，不授予实际运输、稳定净入车量、退出返回或完整循环结论。证据`work/roadshow-three-way-20261014-v1/transport-endpoint-final-cpu-v1/README.md`、`report.json`、`read-only-audit.json`；旧端点及间距失败保留。决策见D-178。

'''
old_heading='#### D165运输CPU接通增量：局部通过，路线边界待讨论'
for rel in ['docs/plans/S05-imitation-and-rl.md','docs/plans/S06-three-scoop-objective-evaluation.md','docs/CURRENT_STATUS.md','docs/acceptance.md','docs/PROJECT_PLAN.md']:
    p=ROOT/rel;t=p.read_text();assert old_heading in t
    start=t.index(old_heading);end=t.index('现有缓存CPU复算9项检查通过',start)
    t=t[:start]+summary+t[end:]
    p.write_text(t)
decision='''## D-178：运输保留实际闭斗端点，仅CPU验证（2026-10-14）

用户明确选择保留闭斗、接入端点处理，仅授权CPU实现和验证，不新增物理或训练预算。来源是已验收的较好人工和冻结残差第一铲hold，实际斗缸约0.8535米，原8毫米规划余量的0.852米上界不能直接套用。保留该已存在状态，允许保持或内收而不继续外推；机械限位、碰撞和闭链精度不放宽，不先主动开斗。

工作区候选已实现，并通过两组完整举升/回转/卸料路径和5394条名义指令、29项只读审计。举升点上调5厘米以通过原4厘米车体间距；同位置斗缸锁定后姿态由机构耦合决定。旧不可达/间距失败及180秒CPU检查超时保留。0新增物理/训练，无原生恢复、实际运输或净入车量结论；进入物理前需另确认预算。正式源码未改，模型和原证据不变。证据见`work/roadshow-three-way-20261014-v1/transport-endpoint-final-cpu-v1/`，实施归S05/S06。

'''
p=ROOT/'docs/decisions.md';t=p.read_text();assert not re.search(r'^## D-178',t,re.M)
at=t.index('## D-177');t=t[:at]+decision+t[at:]
anchor='## 当前决策索引（2026-10-13）';assert anchor in t
t=t.replace(anchor, '## 当前决策索引（2026-10-14）\n\n- `D-178`：保留实际闭斗端点，仅CPU；两来源5394条名义指令及29项审计通过，物理未运行。',1)
p.write_text(t)
p=ROOT/'docs/plans/INDEX.md';t=p.read_text();t=t.replace('本轮冻结3次达标且提高2.8508个百分点；计算关闭；下一步运输卸料对照','冻结策略达标；D178运输闭斗端点CPU路径/5394指令通过；下一门原生恢复/物理预算');p.write_text(t)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True);assert not gpu.stdout.strip()
files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
(out/'manifest.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps({'cpu_case_count':2,'nominal_commands':5394,'audit_checks':29,'gpu_idle':True,'physical_steps':0,'decision':'D-178'}))
