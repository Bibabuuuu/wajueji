"""Register the user's confirmed CPU-only motion optimization before implementation."""
from pathlib import Path
import hashlib,json,subprocess
R=Path('/home/qunwang/wajueji');W=R/'work/motion-review-20261015-v1'
body='''## 2026-10-15 D-194：返回接近合并与保斗缸运输减振，CPU已授权

用户指出卸料后收斗、返回旧姿态再伸展的重复动作，并明确选择“先保斗缸，优化其余轴与交接”。本轮只授权CPU候选：卸料末态立即重观测/向内贴边选点，从实际末态复用IK和原静态车/地面/限位/闭链检查，优先直达人工参考预备点；必要时保留最少车厢退出航点，同步调整四轴，不强制回旧返回姿态。进入土堆接近区后保留完整观测ROI与32cm门及原人工参考走廊；运输区外沿用已有已知刚性地面语义，不冒称额外传感观测。失败则保留旧路线，候选不进入物理入口。

满载运输继续保持铲斗缸零主动速度，残差模型/输入/挖掘段/物理/保护阈值不改。先做零端点加速度的协调轨迹，并检查非零实际速度交接、继承负载偏置、反馈积分及原发令slew限制。CPU只验证轨迹和控制接口；不以几何通过证明带载减振有效，不自动降低保持刚度或开放斗缸。

已验证的只读事实：卸料后到第二铲开挖147.08仿真秒，其中旧RETURN_SAFE32.25秒＋次铲RETURN_RISE36.85秒；原返回后才观测选点。CLEAR_PILE铲斗缸命令全零，实际缸相对名义锁定位最大偏差3.2104mm；实际斗姿对规划姿态最大偏差4.11005度。具体激振根因未定。初始直接套APPROACH的CPU探针因观测ROI拒绝，未证明碰撞或新路线不可行。

输入：原D190的dump.npz（SHA26d3928d2e41fc1f5bce5020160be6faddcdca7a936dcaddd7eb65a49e1f94b3）、原真实首铲入口及D192参考。输出：work/motion-review-20261015-v1/。实施顺序：登记范围→CPU候选→原边界/路径/交接复核→保存审计→再讨论物理预算。验收：新观测可追溯、全程原静态门及接近门、冻结参考准入、速度/加速度/连续性不超原限、保斗缸、无原生初始化/物理/推理/训练。旧运行/预算/失败与媒体包不改，未授权新物理或训练。S05仍进行中，S07为本候选接口记录所有者。

'''
files=['docs/CURRENT_STATUS.md','docs/plans/S07-truck-reproduction-submission.md','docs/decisions.md']
receipts=[]
for rel in files:
 p=R/rel;t=p.read_text();assert 'D-194' not in t
 before=hashlib.sha256(p.read_bytes()).hexdigest();first,_,rest=t.partition('\n');t=first+'\n\n'+body+rest
 if rel.endswith('decisions.md'):
  index='## 当前决策索引（2026-10-15）';assert index in t;t=t.replace(index,index+'\n\n- `D-194`：卸料后重观测选点、返回接近合并；满载保斗缸并优化其余轴与交接。仅CPU候选已授权，无新物理/训练。',1)
 p.write_text(t);receipts.append(dict(path=rel,before=before,after=hashlib.sha256(p.read_bytes()).hexdigest()))
p=R/'docs/plans/S05-imitation-and-rl.md';t=p.read_text();first,_,rest=t.partition('\n');p.write_text(first+'\n\n## 2026-10-15 D-194：运输返回CPU候选\n\n用户已确认先保斗缸并优化其余轴与交接，返回接近合并；范围及接口由[S07](S07-truck-reproduction-submission.md)持有。冻结残差挖掘不改；当前仅CPU，无新物理或训练预算。\n\n'+rest)
check=subprocess.run(['git','diff','--check','--',*files,'docs/plans/S05-imitation-and-rl.md'],cwd=R,capture_output=True,text=True);assert check.returncode==0,check.stdout+check.stderr
(W/'scope-record-receipt.json').write_text(json.dumps(receipts,indent=2));print('CPU_SCOPE_RECORDED')
