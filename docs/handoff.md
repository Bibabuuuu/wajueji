# 跨会话交接：第二铲剩余卸料与夜间挖掘参数搜索

## 开始顺序

先读AGENTS.md、总体规划、当前状态、要求、七个活动小节，再读本文件。核对实际分支/HEAD/进程；唯一仓库`/home/qunwang/wajueji`，分支`codex/contact-bc-collection-20260924`。本轮开始HEAD为`6b5a7b1`，不能据此猜下一次实际HEAD。不要覆盖无关未跟踪产物。

动态事实以[CURRENT_STATUS](CURRENT_STATUS.md)为准：第二铲v4已退出，补收交接通过。D-162固定路线运行已启动，入口work/contact-second-fixed-route-20260925-v1/run.py，输出runs/contact-second-fixed-route-20260925-v1/physical/，当前尚未完成卸料。夜间仅参数搜索准备，无训练/搜索运行；`automation-2`已切换到D-162结束统一审核并启用。旧回放窗口不推进物理。

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

只读诊断[报告](../work/night-parameter-search-20260925-v1/baseline-observation-review.json)：远土粒在接触末态已存在，使密集4cm观察网格21,823,340格超12,000,000预算。不得删粒子、隐藏飞散或覆盖冻结观察器；用户明确不再调查飞散原因，零散土粒不应成为运输障碍。下一候选按纯土固定抬升/回转/卸料目标去除运输对全场土粒地图的依赖，保留静态车体/地面/限位/保斗保护；用户已确认D-162，新包work/contact-second-fixed-route-20260925-v1已实现、4项CPU通过，24数组零步及3398条全路线指令预检通过；已启动剩余卸料，不更改物理碰撞/物料。下一铲选点仍需剩余土堆观测。

## 夜间准备包与下一道门

当前使用`work/night-parameter-search-20260925-v2/`及`work/night-search-tail-20260925-v2/`，原v1保留。v2已接入三段基线评分/回放及D-162固定静态几何，10项CPU测试/导入通过，物理0步。普通候选到CLEAR_PILE，最佳复验才追加固定ALIGN/DUMP。

当前主线仍在运行，readiness为launch_blocked，manifest为prepared_awaiting_baseline；不能冻结运行中的report作为最终证据。结束成功后在`work/contact-second-fixed-route-review-20260925-v1/`执行`prefix_checks.py contact`、`prefix_checks.py carry`、`audit.py`、`replay.py`做一次统一审核。均为零新增物理，已有结果不覆盖。原v4失败不改标签，carry仅核验规划失败前的正常439帧handoff前缀。

全部审核/连续性通过后重新执行v2的`batch.py --prepare`、`preflight.py`冻结最终三段证据。确认GPU没有其他物理实例，再在已授权8h内启动一次`--run`，先120帧全1参数全数组恒等检查，再最多6候选，保留1h复验/保存。失败不自动重跑。输出`runs/night-parameter-search-20260925-v2/`，含batch/best/comparison及基线/最佳回放；不自动部署、不更新神经权重。具体命令见[v2说明](../work/night-parameter-search-20260925-v2/README.md)。

## 环境与便宜检查

物理Python `/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python`；策略Python `/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python`。`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`，仓库根目录运行。本机RTX4070，策略/MPM60Hz、刚体4800Hz，未改变。

可运行`test_search.py -v`、`batch.py --prepare`、`preflight.py`，最后一条当前预期退出2、零物理。文档检查`python3 scripts/check_docs.py`及`git diff --check`。禁止执行旧冻结run覆盖结果，禁止用旧审核替代新候选。

当前第二铲已打开的旧缓存为接触2725+v2失败343帧，manifest `work/contact-combined-replay-20260924-v1/second-curl-stop-v1.json`；不能称它是v4成功补收回放。v4显示缓存仍在其physical/display。新回放使用已有view.py并另建输出目录；不得控制或关闭用户原窗口。
