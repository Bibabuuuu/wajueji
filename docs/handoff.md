# 跨会话交接：第二铲剩余卸料与夜间挖掘参数搜索

## 开始顺序

先读AGENTS.md、总体规划、当前状态、要求、七个活动小节，再读本文件。核对实际分支/HEAD/进程；唯一仓库`/home/qunwang/wajueji`，分支`codex/contact-bc-collection-20260924`。本轮开始HEAD为`6b5a7b1`，不能据此猜下一次实际HEAD。不要覆盖无关未跟踪产物。

动态事实以[CURRENT_STATUS](CURRENT_STATUS.md)为准：第二铲v4已退出，补收交接通过、卸料未完成。夜间仅参数搜索准备，无训练/搜索运行；`automation-2`已暂停。旧回放窗口不推进物理。

## 用户已确认的范围

今晚完成第二铲卸料，不回位、不第三铲；持续运行自动保存，完成或保护退出后统一检查，不自动重跑。直接恢复保存点。10mm累计回落仅限FINISH_CURL，其他阶段不受此新增门限制。原物理保护、原始示范和冻结证据不改。

夜间本机约8h，优先少量挖掘参数搜索，最佳参数及对照回放，不是神经网络RL。学习只到带载抬离，运输/卸料/回位固定。最多6组收斗配合倍率；普通候选不跑卸料，最佳复验才接固定卸料。详见[S05](plans/S05-imitation-and-rl.md)和[准备说明](../work/night-parameter-search-20260925-v1/README.md)。

专项任务`01a0d459-acca-70d3-b21f-a69d646ddece`讨论满斗、晃动、视觉穿模；交接`work/user-three-issues-discussion-20260925-v1/README.md`，未经充分讨论不扩大修复。

## 正确输入与结果

| 用途 | 仓库相对路径 | SHA256 |
| --- | --- | --- |
| 真实第二铲预备点 | runs/contact-next-prepoint-20260925-v1/physical/prepoint.npz | ab4f529501c3991022f6e72a276d4d7b48def9a888fcd85db3148356c73e420e |
| 第二铲2725帧正常接触末态 | runs/contact-next-scoop-20260925-v1/candidate/final.npz | a006213a953947eabd73f975af29dc388fa4ed9dbd3307dc66abf491eec86c1f |
| v4补收439帧正常交接点 | runs/contact-second-finish-unload-20260925-v4/physical/handoff.npz | f750e55864d6189ff6cd149cc1574894c236a334e6817a4b88f66493fd33c84d |
| 阶段BC权重 | runs/contact-phase-bc-pilot-v1/pilot.pt | b4bbc98cf472bcad23f89e014a025cfc933c36091ada91ced4c4cc9680d94095 |

v4交接后规划报`cycle_observation_grid_budget`，同物理数组recovery不代表其元数据与正常handoff相同。优先保留正常交接点，下一续接先核验自身schema/完整控制状态/零步恢复，不重做2725+439帧。完整独立审核尚未完成。

只读诊断[报告](../work/night-parameter-search-20260925-v1/baseline-observation-review.json)：远土粒在接触末态已存在，使密集4cm观察网格21,823,340格超12,000,000预算。不得删粒子、隐藏飞散或覆盖冻结观察器；原物理原因未定位。先做不推进物理的观察表示/来源检查，若扩大控制/物理范围再讨论。

## 夜间准备包与下一道门

`work/night-parameter-search-20260925-v1/`：参数、受约束接触适配、评分、批次预算、回放清单、只读preflight和8项CPU测试。`work/night-search-tail-20260925-v1/`：固定补收/抬离，最佳复验时才接ALIGN/DUMP。源文件未修改，导入检查通过，物理0步。

当前`readiness.json`为launch_blocked。包绑定失败v4，不能运行8h；先解决第二铲观察阻塞、保存连续剩余段并完成审核，再把真实分段基线接入评分/回放、重冻manifest。不要更改failure标签凑出启动条件。之后在总预算内先做120帧全数组恒等短检，未过即停止；再至多6候选，留1h复验/保存。任何失败无自动重试。

候选结果写`runs/night-parameter-search-20260925-v1/`，包含batch/best/comparison、逐帧缓存、检查点及replay清单。未复验候选不作为最终选择；不自动部署。`runs/`和外部依赖只在本机，不进Git。

## 环境与便宜检查

物理Python `/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python`；策略Python `/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python`。`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`，仓库根目录运行。本机RTX4070，策略/MPM60Hz、刚体4800Hz，未改变。

可运行`test_search.py -v`、`batch.py --prepare`、`preflight.py`，最后一条当前预期退出2、零物理。文档检查`python3 scripts/check_docs.py`及`git diff --check`。禁止执行旧冻结run覆盖结果，禁止用旧审核替代新候选。

当前第二铲已打开的旧缓存为接触2725+v2失败343帧，manifest `work/contact-combined-replay-20260924-v1/second-curl-stop-v1.json`；不能称它是v4成功补收回放。v4显示缓存仍在其physical/display。新回放使用已有view.py并另建输出目录；不得控制或关闭用户原窗口。
