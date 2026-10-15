from pathlib import Path
import json, hashlib, shutil, subprocess
ROOT=Path('/home/qunwang/wajueji')
work=ROOT/'work/roadshow-three-way-20261014-v1'
out=work/'offline-boundary-v2'
report=json.loads((out/'report.json').read_text());preflight=json.loads((out/'transport-source-preflight.json').read_text())
assert report['status']==preflight['status']=='passed'
assert all(report['checks'].values()) and all(preflight['checks'].values())
body='''# Stage 03补充：几何量测离线诊断与运输来源预检

状态：离线诊断已验证；D165运输接通方案待具体实施；原生恢复/物理运输尚未验证。

本轮只用已保存粒子、石块和机构状态在CPU复算，0新增物理帧、0训练更新，CUDA禁用。原模型、末态、量测代码/配置和Stage 03主表均未改变。9项末态诊断检查及17项CPU运输来源检查通过。

## 已验证事实

| 保存末态 | 原0.8米测量框 | 扩大到1.6米的瞬时诊断 |
|---|---:|---:|
| 几何2910帧 | 无效 | 135.3417% |
| 低装载人工1604帧 | 97.0699% | 99.0645% |
| 冻结残差1206帧 | 148.2321% | 149.3039% |

原几何2880/2895/2910三个保存点在0.8米框中均重现边界无效；1.2米末态仍无效，1.6米才有效。扩大后的量测连通区域550536个格子（原358737），世界高度范围约-0.0094至2.0977米，延伸到外部料堆近地面；不能仅凭去掉边界错误认定它就是独立保载。斗腔内有11377个土粒中心，几何不是0%空斗。

扩大框同时改变人工约1.9946个百分点、残差约1.0718个百分点，因此不是可直接套用原主表的等价修复。具体机制尚未分离，未把变化武断归因于一个因素。135.34%仅是改变测量框后的一个瞬时诊断，且几何没有零指令0.5秒末尾保持，不能授予原口径短保载满斗率或正式装车量。

当前结论：现有数据无法补出原口径几何短保载评分。Stage 03主表仍留缺失值，原0.8米人工/残差结果逐值重现。终点铲底少量黏附继续允许，没有重新加入净空硬门。

## 运输接通的具体准备

三来源均通过既有cycle_checkpoint.load显式D165运动策略声明，24类数组逐值相同、full原生历史/hold/cycle_index=1及材料政策完整；来源SHA不变。此为CPU包络读取资格，不等于原生零步恢复或动态运输已通过。

既有run_layered_47_cycle_transport的CycleRuntime构造缺显式bucket_motion_consistency，CPU预检重现cycle_requires_explicit_bucket_motion_consistency；老第二铲full-response loader另外绑定不同schema/来源和carry历史，不能绕过。几何failure仍报cycle_failure_checkpoint_not_resumable，不改标签、不用于续算。

可复用：CycleRuntime完整保存/加载、D165原控制与材料/扫掠保护、既有固定CLEAR_PILE/ALIGN/DUMP几何目标、保斗控制中的负载偏置与回转预览/拒绝回滚。需做最小接通：来源SHA白名单与D165显式开关、沿用实际四轴/控制器参考/累计积分和时间、先零步恢复审计再动作交接；CLEAR_PILE/ALIGN维持实际斗缸位置，不使用会改闭斗姿态的通用pitch=1路线替代。固定车位/地面/机构/车体碰撞和原保斗检查不放宽。

建议先以较好人工基准和当前残差配对，避免把参考轨迹自身优势全归因于学习；低人工例保持补充对照，几何末态暂不具运输来源准入。首先仅做CPU接通/路径与控制检查，达到可评审状态后再确认新增物理预算。稳定混合净入车体积尚需测量定义/标定，TruckReceiver几何/土粒接收计数不冒充正式体积。方案不训练、不重挖、不恢复已封存预算。

## 验证与来源

- report.json：9项CPU诊断、9次已保存状态量测与每次连通范围、原估计器内部量/格尺寸/粒子数；扩大框结果明确仅诊断。
- transport-source-preflight.json：17项检查、3来源SHA/24数组及旧入口拒绝原因。
- source-snapshot/：实际CPU脚本。
- 原主表与96项物理审计保持在runs/roadshow-handover-20261012/stage-03；本补充不覆盖该包。

所有追加物理、实际原生恢复、SAC更新次数均为0。预算文件不改，GPU空闲。正式控制、物理与量测源码未修改，未Git提交。
'''
(out/'README.md').write_text(body)
for name in ['diagnose_boundary_v2.py','transport_sources.py']:
    target=out/'source-snapshot'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(work/name,target)
paragraph='''### 2026-10-14 后续离线诊断已验证（0新增物理/训练）

现有缓存CPU复算9项检查通过：几何2880/2895/2910在原0.8米框仍无效，2910扩大到1.6米返回135.3417%瞬时诊断，但连通区域延伸到近地面外部料堆，且无0.5秒静止保持；不能补进原短保载主表。扩大框使人工97.0699→99.0645%、残差148.2321→149.3039%，也不等价于只修缺失值。原97.0699/148.2321精确复现、原量测与物理/模型不改，终点黏附允许。

17项CPU运输来源预检通过：较好人工/低人工/冻结残差24类包络逐值一致，D165材料/运动策略及full历史完整。原运输入口缺显式运动声明会拒绝；几何failure仍禁止续接。下一门已收敛为D165来源绑定的固定保斗路线接通：保持实际斗缸和原控制内部参考，先CPU路径/交接，再原生零步及有界物理；后三项尚未完成，未增加实验预算。正式稳定混合净入车量测仍待定义/标定。详细证据：`work/roadshow-three-way-20261014-v1/offline-boundary-v2/README.md`、`report.json`、`transport-source-preflight.json`。现有数据无法取得原口径几何短保载评分，主表缺失值保留。

'''
for rel in ['docs/plans/S05-imitation-and-rl.md','docs/plans/S06-three-scoop-objective-evaluation.md','docs/CURRENT_STATUS.md','docs/acceptance.md','docs/PROJECT_PLAN.md']:
    path=ROOT/rel;t=path.read_text()
    assert '### 2026-10-14 后续离线诊断已验证' not in t
    anchor='## 2026-10-14 三方同域对照Stage 03：实验已关闭，结果与边界已核验'
    if rel=='docs/acceptance.md':anchor='## 当前能力摘要（2026-10-14）：Stage 03三方对照已归档，几何量测有缺口'
    assert anchor in t,rel
    t=t.replace(anchor+'\n',anchor+'\n\n'+paragraph,1)
    if rel.startswith('docs/plans/'):
        import re
        t=re.sub(r'^- 最后复核：.*$', '- 最后复核：2026-10-14（Asia/Shanghai）',t,count=1,flags=re.M)
    path.write_text(t)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True)
assert not gpu.stdout.strip()
print(json.dumps({'evidence_checks':len(report['checks'])+len(preflight['checks']),'new_physical_frames':0,'training_updates':0,'docs_updated':5,'gpu_idle':True}))
