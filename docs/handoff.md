# 跨会话交接：接续平顺性和离土

2026-09-24接手增量：先看CURRENT_STATUS，当前候选已推进到D-148 v9-handoff；下方v7入口为冻结基线，不执行旧run.sh覆盖结果。v8切深门失败，v9原900窗未交接均保留。新源码、协议及审核见`work/contact-geometric-feedback-20260924-v9-handoff/`；新显式中间窗1200，完整窗仍2400/900秒。第二铲零步/120已审，第一铲新完整初态准备存在时钟/微量接触边界。实际进程仍必须重新核对。

更新：2026-09-24，用户完成1500帧回放人审后。此页只提供接手步骤和运行入口；唯一当前事实源为[CURRENT_STATUS.md](CURRENT_STATUS.md)，总体路线为[PROJECT_PLAN.md](PROJECT_PLAN.md)。旧9月14日纯土交接在Git `fb126fd:docs/handoff.md`保留，不作为现行任务。

## 1. 接手位置与Git

- 本机唯一仓库 `/home/qunwang/wajueji`，不是ChatGPT项目镜像目录或`codex-work`依赖目录。
- 分支 `codex/contact-bc-collection-20260924`；实现基线 `fb126fd72303b46d41a3330087d67edcf8494923`（已推送）。后续交接提交不改运行代码，交接完成标签为`handoff/contact-feedback-v7-20260924`。
- 首先执行下面只读命令，回报位置、分支/HEAD、跟踪改动，再按AGENTS顺序完整阅读核心文档和七份active计划。不要自动切分支、pull、reset、stash或清理文件。

```bash
cd /home/qunwang/wajueji
pwd
git rev-parse --show-toplevel
git branch --show-current
git log -3 --oneline
git status --short --untracked-files=no
```

本次交接前跟踪工作树干净，但有914个未跟踪文件（含用户PDF和实验生成物）；数量会随新工作变化。它们已保留，不能`git add .`或`git clean`批量处理。新修改用明确白名单提交，阶段成果推送当前分支，不重写旧失败历史。

## 2. 用户最新反馈与授权边界

用户认为整体动作和末态装载外观还行；中间一段回收偏快、有速度突变。因为当前是第三铲，尚看不出对后续可挖土堆的影响。用户提出RL是否适合优化，并要求做好交接；**这是路线讨论，不是立即启动RL或双A30长训练的授权**。

已确认：优先在既有混合策略上推进、直接恢复预备点；收斗80%以上不能继续下压；保持真实土石、原保护和≥95%目标。80%本版实现为有效油缸行程比例，不是斗角/满斗率比例。接近/切入为反馈控制，收斗/抬升为受约束阶段BC。不要推倒重做纯RL，也不要说完全没有用模仿学习。

v7：1500帧正常结束，981帧约束触发，1087帧确认持续抬升；根沿触发后最大下沉0.061mm（旧版31.245cm），但局部斗壁仍向下扫5.661cm；末态lift、未离土，没有有效满斗率。用户视觉认可不是95%验收。57测试及局部物理审核通过，详见[证据](../work/contact-geometric-feedback-20260924-v7/README.md)。

## 3. 输入、依赖与不可覆盖对象

完整输入SHA和本地可用性见[核对清单](../work/contact-handoff-20260924-v1/verification.json)；运行绑定见[协议](../work/contact-geometric-feedback-20260924-v7/protocol.json)。

| 用途 | 路径/身份 |
| --- | --- |
| 第三铲开发预备点 | `/home/qunwang/wajueji/runs/fast-contact-collection/20260924-115633-1457317/prepare/prepoint.npz`；SHA `a6fed857a335c95b6fb6222530b33fe73267841d77e527a712731d5edd0d896b` |
| 预备点元数据 | 同名`.npz.json`；SHA `dcd1ed7a975066047336eb93b192b9a521c00a1aa75cc063218c1daafd27dfc8` |
| 阶段BC模型 | `/home/qunwang/wajueji/runs/contact-phase-bc-pilot-v1/pilot.pt`；SHA `b4bbc98cf472bcad23f89e014a025cfc933c36091ada91ced4c4cc9680d94095` |
| 物理Python | `/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python`（有SciPy） |
| 推理/训练Python | `/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python` |
| 几何/场景依赖 | `/home/qunwang/codex-work/small-soil-20260910-v17-performance` |
| 斗腔标定依赖 | `/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json` |
| v7大产物 | `/home/qunwang/wajueji/runs/contact-geometric-probe-20260924-v7/{zero,startup120,contact900,scoop1500}` |
| 精选报告/审核入口 | `/home/qunwang/wajueji/work/contact-geometric-feedback-20260924-v7/` |

`runs/`、模型、显示缓存和外部依赖未上传GitHub；新会话若在另一机器，应先报告缺哪些本地资产，不能仅clone后猜路径或重建示范。原人工数据、源快照、v6/v7协议和运行结果不可覆盖。v6失败快照不能续跑；v7final虽正常，也缺BC短历史/阶段器/反馈锁存的完整恢复合同，不能直接作为精确策略续跑点。

## 4. 第一个工作包：只读定位速度突变

接手后先提交简短诊断，不启动长训练。数据用v7 `scoop1500/actions.ndjson`、`physics/frames.ndjson`、`phase-trace.ndjson`、`report.json`。

1. 定位全程最大原始动作、实际命令与实际速度变化；重点检查423（碰土后）、824（BC接管）、981（80%约束）、1088（lift）附近，不能只看这四个点。
2. 对齐四轴`[swing,boom,arm,bucket]`，区分BC意图、几何求解、原投影和实际运动。qpos轴地址`[7,9,12,15]`；free-base导致qvel对应`[6,8,11,14]`，别错位。保存约束触发、150ms预览、积分/主动助力、载荷的同帧对比。
3. 输出新目录（建议`work/contact-speed-audit-20260924-v1/`；存在则用新名）的摘要JSON和说明，包含源SHA、异常帧段、幅度/单位、原因证据与未知项；不修改原动作/原标签。
4. 若指令突变，提出最小阶段衔接/变化率约束；若指令平滑而机构突变，先查执行器/载荷。不得在最终安全投影之后盲加平滑滤波，从而重新引入向下运动或穿地。
5. 第一/第二铲余料保护、完整离土和有效满斗是下一验收门；RL仅是后续受限残差或低维参数优化候选，先测恢复/吞吐并讨论预算。双A30分别用于独立环境/评价；远端运行必须先读runbook，本轮未核验远端进程。

建议修改白名单（先读后缩小）：`src/v4/contact_geometric_feedback.py`、必要时`src/v4/run_contact_geometric_probe.py`、对应`tests/v4/`、新工作包及现有状态/S05/验收/决策文档。修改底层物理、积分助力、模型架构或训练路线前先给证据和方案，不把这些混进普通平滑修复。

## 5. 便宜验证、回放与物理门

先用现有环境，不升级依赖：

```bash
cd /home/qunwang/wajueji
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:tests/v4:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python -m unittest -v \
 tests.v4.test_contact_geometric_feedback tests.v4.test_contact_cut_reference \
 tests.v4.test_contact_phase_bc_policy tests.v4.test_contact_phase_bc_inference \
 tests.v4.test_contact_bc_ground_guard tests.v4.test_contact_phase_candidate \
 tests.v4.test_contact_phase_completion tests.v4.test_contact_phase_geometry tests.v4.test_contact_phase_data
python3 scripts/check_docs.py
git diff --check
```

本次交接查询时未发现物理、训练及旧回放进程。若用户需要重新看v7，用以下命令打开**只读缓存**（不产生物理步）；空格播放，End末帧，R重置回放，X关闭。先检查是否已有相同窗口，不重复打开。

```bash
cd /home/qunwang/wajueji
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:. \
 /home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python \
 src/v4/view_contact_bc_probe.py --view-only --output runs/contact-geometric-probe-20260924-v7/scoop1500/display
```

真实运行模板在v7 `run.sh`，**不要执行它覆盖现有输出**。新控制候选建立新协议/源码SHA和新输出目录，直接恢复原预备点；依次零步→120审核→900审核→显式有界完整候选（运行器最多2400帧/900秒）。120/900前缀须同版精确比较。v7用1500帧预算正常停下，不是窗口崩溃。末端限位、非有限、保护失败须保存报告/故障态后停止，不放宽门。

本地只用RTX4070 cuda:0、同卡一个物理实例；沿用10cm/60Hz/4800Hz。旧1500帧耗时345.95秒，只有约4.34物理帧/墙钟秒，不能用回放窗口60FPS估算RL吞吐。耗时超10分钟的任务先说明设备、命令、预算、日志和中断方式；本交接没有另行授权批量实验或A30长训练。

离土仍使用距连地土≥3mm连续5帧、证据n仅供决策n+1。不得用截图代替。通过后才用既有`measure_off_soil_fill.py`及0.2占据阈值测含石/堆高满斗；不得调阈值凑95%。当前末帧粒子连通定义也需如实核对，不能因画面看着离土而跳过。

## 6. 新会话回报要求

先说明实际仓库/分支/HEAD、要读哪些数据、是否有运行进程，以及第一个只读诊断；不反复询问已确认的参数。完成一个明确阶段后报告修改、验证、未过门和下一步，白名单提交并推送。跨会话以落盘记录为准，不从旧聊天片段猜进度。
