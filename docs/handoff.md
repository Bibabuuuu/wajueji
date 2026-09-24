# 跨会话交接：第二铲检查完成，夜间双起点挖掘优化

## 开始顺序

先读AGENTS.md、总体规划、当前状态、要求、七个活动小节，再读本文件。核对实际分支/HEAD/进程；唯一仓库`/home/qunwang/wajueji`，分支`codex/contact-bc-collection-20260924`。本轮开始HEAD为`86367e4`，不能据此猜下一次实际HEAD。不要覆盖无关未跟踪产物。

动态事实以[CURRENT_STATUS](CURRENT_STATUS.md)为准。第二铲卸料/127项审核和入口短检均已完成，原结果冻结。D-163新双源参数搜索代码/16项CPU检查已完成，03:58:18已由systemd后台启动，最迟11:56:16截止（Asia/Shanghai）；实际启动/退出时间看`runs/night-dual-search-20260925-v1/{launch,batch}.json`。用户要求不盯跑；automation-2保持暂停。

## 用户已确认的范围

今晚完成第二铲卸料，不回位、不第三铲；持续运行自动保存，完成或保护退出后统一检查，不自动重跑。直接恢复保存点。10mm累计回落仅限FINISH_CURL，其他阶段不受此新增门限制。原物理保护、原始示范和冻结证据不改。

夜间本机约8h，优先满斗率改善：继续参数优化，不更新神经网络权重；当前D152物理，不合入多接触点候选。第一/第二固定预备点成对试验。已知动作保护失败淘汰候选后继续；未知/数值/恢复/保存/基线异常停整批。原失败物理立即结束，下一候选恢复原预备点，不改阈值、不自动重试。普通候选到CLEAR_PILE，最佳双源独立复验接固定卸料；六组候选池按剩余预算截断，预留7200秒复验。详见[S05](plans/S05-imitation-and-rl.md)和[运行包](../work/night-dual-search-20260925-v1/README.md)。

专项任务`01a0d459-acca-70d3-b21f-a69d646ddece`讨论满斗、晃动、视觉穿模；交接`work/user-three-issues-discussion-20260925-v1/README.md`，未经充分讨论不扩大修复。

## 正确输入与结果

| 用途 | 仓库相对路径 | SHA256 |
| --- | --- | --- |
| 真实第二铲预备点 | runs/contact-next-prepoint-20260925-v1/physical/prepoint.npz | ab4f529501c3991022f6e72a276d4d7b48def9a888fcd85db3148356c73e420e |
| 第二铲2725帧正常接触末态 | runs/contact-next-scoop-20260925-v1/candidate/final.npz | a006213a953947eabd73f975af29dc388fa4ed9dbd3307dc66abf491eec86c1f |
| v4补收439帧正常交接点 | runs/contact-second-finish-unload-20260925-v4/physical/handoff.npz | f750e55864d6189ff6cd149cc1574894c236a334e6817a4b88f66493fd33c84d |
| 阶段BC权重 | runs/contact-phase-bc-pilot-v1/pilot.pt | b4bbc98cf472bcad23f89e014a025cfc933c36091ada91ced4c4cc9680d94095 |

旧v4交接后规划报cycle_observation_grid_budget，正常handoff已由D162恢复并完成卸料；127项完整审核已通过，不需重做前缀或恢复。原failure仍保留。最终固定路线快照SHA `620b50d6d69fa2d6c036c8f500284265ffab1978cc4506df655fe7c69179f7a9`。

远土粒使原密集观察网格超预算的[诊断](../work/night-parameter-search-20260925-v1/baseline-observation-review.json)冻结。用户不再调查飞散、不删物料；D162固定运输已去掉全场土粒地图依赖，静态车体/地面/限位/保斗及物理碰撞保留。下一铲选点仍观察余土。

## 已完成审核与下一道门

当前v2及尾段代码已冻结，12项CPU检查通过。统一contact/carry/tail审核均通过，127项；三段6562帧缓存SHA和边界通过。审核在`work/contact-second-fixed-route-review-20260925-v1/`，无需重做。原v4规划失败标签仍保留。

`batch.py --check-only`已运行一次：基线工作满斗率30.21405%；全1倍率120帧、24数组及末态NPZ SHA与原gate-120完全一致。证据`work/night-parameter-search-20260925-v2/short-check-review.json`、`runs/night-parameter-search-20260925-v2/precheck.json`。没有batch.json或任何候选。不要重新prepare或重跑短检；当前同版本--run具备SHA验证后复用能力，扣除66.631579秒。

用户最新双起点方向：第一源`runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz`，SHA `700f76be970c5754bde905837faf642cca0317fef097e31d17cf48f008ba0545`，schema为layered_47_cycle_checkpoint_v1、cycle1、truck_enabled=false；120帧参考`runs/contact-multisource-lift-20260924-v1/first-candidate/gate-120.npz`，SHA `b05c0ecd5fb38ee3e65e154057d093568874ee02dedc54fa80af71e5b042ed09`。第二源见上表。

双源入口准备位于`work/night-dual-start-preparation-20260925-v2/`（注意区别于night-parameter-search-v2）。10项CPU通过；第一源零步复用已通过的v1，120帧37.547386秒、24数组/末态SHA与旧参考完全一致。第二源通过sources.second_evidence复核完整v2 manifest/precheck绑定，执行链未改，不重新跑120。结果见result-review.json。v1参数门拒绝发生在任何物理之前，保留日志；v2只将独立循环帧允许值改0/120、墙时收紧180秒，其余逐字一致。不要重跑两个短检，不改绑定*.py。

D-163已收口当前物理：不合入专项新接口。新双源代码16项CPU检查及两源/尾段加载检查通过，实际完整基线先跑，前120帧嵌入旧参考数组一致性门；没有额外独立短检。第一源仍保留cycle1/无truck接触历史，尾段按既有流程加入truck；第二源保留真实cycle2控制/材料历史，不通过改标签统一状态。

不要重复启动；下一步只在用户请求时查看`wajueji-dual-search-20260925-v1.service`及batch.json。运行中严禁编辑绑定代码或重新prepare；结束只读统一比较baseline、trials、rejections、confirmation和best。不自动恢复停止的批次。候选池六组不保证全部执行；8h含既有检查121.096636秒，留最佳双源复验。没有实证改善就明确保留基线。

## 环境与便宜检查

物理Python `/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python`；策略Python `/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python`。`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`，仓库根目录运行。本机RTX4070，策略/MPM60Hz、刚体4800Hz，未改变。

可以只读核对`preflight.py`；不要重绑已完成短检的manifest。preflight是只读准备检查；实际物理资格看short-check-review.json。文档检查`python3 scripts/check_docs.py`及`git diff --check`。禁止执行旧冻结run覆盖结果，禁止用旧审核替代新候选。

完整第二铲回放manifest为`work/contact-second-fixed-route-review-20260925-v1/second-replay.json`，6562帧。可以使用原view.py另建输出目录观看，播放零物理；旧失败段缓存不能替代完整结果，不控制或关闭用户原窗口。
