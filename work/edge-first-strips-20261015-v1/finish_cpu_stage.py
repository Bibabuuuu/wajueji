"""Bind CPU evidence and update existing authoritative records only."""
from pathlib import Path
import sys,json,hashlib,datetime
R=Path('/home/qunwang/wajueji');W=R/'work/edge-first-strips-20261015-v1'
sys.path.insert(0,str(R/'work/continuous-single-scoop-20261014-v1'))
from continuous_scoop import read,write,sha
previous=read(R/'work/second-scoop-qerror-long-20261014-v1/record-final-receipt.json')
for name,entry in previous.items():
    assert sha(R/name)==entry['after'],('record_changed',name)
approach=read(W/'cpu-edge-v2/report.json');ref=read(W/'reference-edge-v2/report.json');adj=read(W/'adjacent-preview.json')
assert approach['status']=='edge_first_approach_cpu_passed' and ref['status']=='cpu_rotation_geometry_passed' and adj['status']=='cpu_adjacent_preview_passed'
assert all(ref['checks'].values()) and all(adj['checks'].values())
assert ref['entry_alignment']['trusted'] and ref['entry_minimum_clearance']['bucket_observed_terrain_m']>=.32
model=R/'work/manual-strength-overnight-20261013-v1/anchor-grid-v1/seed44-weight100.zip'
assert sha(model)=='84aade9a3d8819368afb72223b8452ebfbc5e9ac9ad190818207d1142692aab1'
wait=R/'work/second-scoop-wait-qerror-20261014-v1/actual_joint_wait.py'
assert sha(wait)=='063bd4415a0e83c12865f60af33c7047d460c45a9ad832312f8e8d83e5d03007'
protocol=dict(status='cpu_verified_physical_budget_pending',decision='D-189',source=approach['source'],source_sha256=approach['source_sha256'],
 model=str(model),model_sha256=sha(model),wait_mapping=str(wait),wait_mapping_sha256=sha(wait),
 cpu_checks=dict(rotated_reference=ref['checks'],adjacent_preview=adj['checks']),
 cpu_physics_steps=0,cpu_training_updates=0,physical_budget_authorized=False,budget_started=False,
 physical_proposal=dict(total_wall_minutes=120,save_and_audit_reserve_minutes=15,total_physical_frame_ceiling=30000,
   per_dig_frame_ceiling=7200,reference_frames_per_dig=1170,hold_frames=30,device='desktop cuda:0 physics; CPU frozen inference',
   sequence=['fresh native zero/first3 continuity gate from untouched safe_start','4358-frame edge approach and 71-frame guarded handoff, fresh physical tracking gates','frozen first dig and short hold','existing geometric transport, unload, exit and high return in same instance','fresh ground-connected terrain and recorded actual first cut entry/sweep','inward adjacent ranking, full-width material and original approach/reference checks; no candidate means stop','frozen second dig and short hold','save, offline command/input audit and per-scoop comparison'],
   failure_rule='Any source/continuity/protection/corridor/budget/adjacency infeasibility saves and stops. No failed-source automatic restart. Any meaningful interface change returns for discussion.',
   exclusions=['no training','no third scoop','no second-scoop transport','no changed material/limits/protection','no old budget reuse'],
   acceptance='Report each scoop minimum 0.5s hold display, completion/time/wait fraction and actual cut-footprint gap/overlap. No promise of improved fill.',
   comparison='Existing central-first 142.6193/68.175044 is historical trend reference only: first-run wait mapping differs. Clean placement causality needs matched current-controller central/edge branches under additional scope.'),
 limitations=['Safe_start native restoration not freshly validated; its continuation_qualified=false is preserved','Adjacent preview uses untouched stock, not an observed post-first state','Entry gap alone does not prove no swept overlap or no soil slumping','Third-strip room not qualified','Live wrapper/source and policy-input preflight remains necessary before any physical step'])
bindings={}
for p in [Path(approach['source']),model,wait,R/'work/second-scoop-continuation-20261014-v1/yaw_transfer.py',R/'work/continuous-single-scoop-20261014-v1/continuous_scoop.py',R/'src/v4/manual_residual_sac_env.py',R/'src/v4/manual_trajectory_alignment.py',R/'src/v4/layered_47_cycle_selection.py',R/'planning/contact_footprint.py',W/'adjacent_selection.py',W/'cpu-edge-v2/edge-first-approach.npz',W/'reference-edge-v2/reference-candidate.npz',W/'cpu-edge-v2/report.json',W/'reference-edge-v2/report.json',W/'adjacent-preview.json']:
    bindings[str(p)]=sha(p)
protocol['bindings_sha256']=bindings;write(W/'protocol.json',protocol)

heading='2026-10-15 D-189：边缘首铲＋向内贴边次铲，CPU候选已验证'
detail='''用户明确要求先尝试“首铲从土堆边缘开始、次铲边界贴首铲边界”，检验满斗率是否提高。该布局路线已确认并授权CPU实施；新物理预算尚未确定，所有旧预算保持封存。

已完成：使用真实未开挖safe_start（8d0e5abc…）只读重建土堆；旧“必须留两条未来条带”筛选实际只留下中部候选，首次探查记录保留且不作边缘成功证据。修正本次实验排序为优先现有合法候选最外侧、至少一条后续空间，选中锚点[5.75,0.65]、接触点[5.849367,0.661233,0.653455]；原整斗宽0.60m×0.90m几何评分和未知区域门保留，观测带材料覆盖97.22%是几何诊断，不是满斗率预测。

原32cm离土、机械/碰撞/闭链保护下，4358帧CPU安全接近通过。强行到精确参考起点会低于32cm门，初始安全预备点也在策略关节走廊外；两项失败原样保留。新交接只在参考起点动臂缸多保留1mm，71帧接近段最小间距32.0107cm、终点32.7489cm，原策略匹配可信、斗位姿误差约1.073cm。1171姿态转向参考完整静态检查通过（7项检查），三个缸参考及原命令不变，仅航向转6.95657°；参考接触点距观测选点3.662cm，未做径向/高度扭曲。CPU通过不证明受载恢复或满斗率改善。

新增隔离贴边排序候选沿首铲向土堆内部找下一条，使用新地图和上次实际入口，优先最小名义斗宽边界间距，再保留整斗宽材料评分。初始土堆预演6项检查通过，第二锚点[5.45,0.05]的名义入口边界间隔约7.17mm；这只是未开挖地图预演，不是实际第二铲选点，也不是实际扫掠无重叠证据。实际试验须先完成首铲→原几何运卸/返回，再由新观测和首铲实测入口/切入轨迹重新算次铲，检查全段重叠/余隙；没有合格贴边候选即停，不跳回固定中点。

当前物理步0、推理调用0、训练0，冻结模型、等待映射、正式源码和旧证据不变。S05继续进行中。建议另确认120分钟总物理预算，最后15分钟保存审计；一个原生实例完成边缘首铲、运输卸料/返回、贴边次铲挖掘及各0.5秒保载，总物理帧最多30000、每铲残差最多7200。失败/到期保存停止，不训练、不第三铲、不次铲运卸。旧中央首铲142.6193%及第二铲68.175044%仅作历史趋势参考，首铲等待接口不同，不能当纯布局因果配对；严格因果对照需另扩为同当前控制的中央/边缘两分支。物理运行器接入、源原生零步/首3步、策略观测预检与真实贴边有效性仍待验证，未启动预算。
'''
short='用户已确认先试边缘首铲＋向内贴边次铲。外侧点4358帧安全接近、71帧参考交接及完整转向参考CPU检查通过；贴边排序仅初始土堆预演通过，须首铲实际运卸返回后以新观测重算。新物理步/训练均0；真实保载改善未验证、新预算待确认、旧预算封存。S05继续进行中，D-189。'
backup=W/'records-before-edge-stage';backup.mkdir(exist_ok=False)
receipt={}
for name in previous:
    p=R/name;text=p.read_text();before=sha(p);dst=backup/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    if name=='docs/decisions.md':
        insert=text.index('## D-188：');text=text[:insert]+'## D-189：先试边缘首铲与向内贴边次铲（2026-10-15）\n\n'+detail+'\n'+text[insert:]
        insert=text.index('- `D-188`');text=text[:insert]+'- `D-189`：边缘首铲/贴边次铲路线已确认，接近/交接/转向参考及贴边预演CPU通过；物理预算待确认，未有新满斗率。\n'+text[insert:]
    elif name=='docs/plans/S05-imitation-and-rl.md':
        insert=text.index('## 2026-10-14 第二铲');text=text[:insert]+'## '+heading+'\n\n'+detail+'\n证据：[CPU协议及绑定](../../work/edge-first-strips-20261015-v1/protocol.json)、[接近](../../work/edge-first-strips-20261015-v1/cpu-edge-v2/report.json)、[转向参考](../../work/edge-first-strips-20261015-v1/reference-edge-v2/report.json)、[贴边预演](../../work/edge-first-strips-20261015-v1/adjacent-preview.json)。\n\n'+text[insert:]
    else:
        insert=text.index('\n')+1;text=text[:insert]+'\n## '+heading+'\n\n'+short+'\n\n'+text[insert:]
    text=text.replace('最后复核：2026-10-14（Asia/Shanghai）','最后复核：2026-10-15（Asia/Shanghai）',1)
    if name=='docs/PROJECT_PLAN.md':
        text=text.replace('当前实施S05：第二铲参考及冻结策略接口接通。CPU候选及观测转换已验证，待另确认物理范围/预算；原首铲已验证结论保持，尚未执行第二铲/三铲或新训练。详见S05和CURRENT_STATUS。','当前实施S05：原第二铲完整挖掘与保载已验证；按用户要求先试边缘首铲＋向内贴边次铲。布局CPU候选通过，新物理预算待确认；三铲未完成，新训练未启动。详见S05和CURRENT_STATUS。')
    p.write_text(text);receipt[name]=dict(before=before,after=sha(p))
write(W/'record-receipt.json',receipt)
write(W/'cpu-stage-summary.json',dict(status='cpu_stage_verified_physics_pending',reference_checks_passed=sum(ref['checks'].values()),adjacent_checks_passed=sum(adj['checks'].values()),source_and_model_bindings=bindings,physical_steps=0,training_updates=0))
print('CPU_STAGE_AND_RECORDS_SAVED',flush=True)
