# 本机夜间挖掘参数搜索准备

状态：`已实现、CPU检查通过、启动阻塞`。尚未进行本包零步/120帧物理检查，尚未搜索、训练神经网络或生成最佳候选。当前准备的唯一阻塞见 [readiness.json](readiness.json)；不是已获可运行物理资格。

用户已选择本机约8小时、优先明早动作改善的少量参数搜索，并再次明确只优化挖掘。权威路线及评价见 [S05](../../docs/plans/S05-imitation-and-rl.md)、[S06](../../docs/plans/S06-three-scoop-objective-evaluation.md)。原始示范、BC权重、冻结结果及物理/控制保护不改。RTX 4070单物理实例串行运行，不部署A30。

## 范围与候选

同一第二铲完整预备点，每次显式实验重置，不冒充连续三铲证据。从保存点直接开始，无APPROACH重跑。优化仅修改`curl_load`且实际收斗行程<95%时的BC动作意图，再经过原几何反馈、限速、变化率、预览和保护；非直接覆盖实际缸速。接近、切入和95%之后的补收/带载抬离保持原控制。

| 候选 | 大臂意图倍率 | 小臂意图倍率 | 卷斗意图倍率 |
| --- | ---: | ---: | ---: |
| boom_arm_075 | 0.75 | 0.75 | 1 |
| boom_arm_050 | 0.50 | 0.50 | 1 |
| boom_050 | 0.50 | 1 | 1 |
| arm_050 | 1 | 0.50 | 1 |
| curl_115 | 1 | 1 | 1.15 |
| balanced_curl_115 | 0.75 | 0.75 | 1.15 |

普通候选到`CLEAR_PILE`停止。只有筛选出的改善候选另重复一次，接固定运输/卸料做验收；这些后续阶段不学习、不搜索参数。10mm累计回落仍仅限`FINISH_CURL`。

## 输入与预算

- 仓库 `/home/qunwang/wajueji`，分支 `codex/contact-bc-collection-20260924`，本包开始准备的已推送基线 `6b5a7b1`。
- 预备点 `runs/contact-next-prepoint-20260925-v1/physical/prepoint.npz`，SHA `ab4f529501c3991022f6e72a276d4d7b48def9a888fcd85db3148356c73e420e`；原严格零接触资格false保留。
- BC `runs/contact-phase-bc-pilot-v1/pilot.pt`，SHA `b4bbc98cf472bcad23f89e014a025cfc933c36091ada91ced4c4cc9680d94095`。
- 总预算28800墙秒，从批次真正开始计时。至多6候选，预留3600秒用于最佳复验及保存；预算不足不启动下一候选，不承诺用满8h或找到改善。
- 每候选接触≤3000帧/900秒，补收和带载抬离≤6000帧/2000秒；补收自身≤600帧。额外进程看门狗负责卡死退出；给子进程SIGINT后最多等45秒保存，仍不退出再终止。不得把强制终止点冒充完整可恢复检查点。
- 每120帧、交接和阶段结束保存。输入/模型/源码SHA冻结，55GiB空闲空间保留；单批锁，禁止自动重启重新计时。
- 保护、进程、体积测量无效等失败停止整批，不自动重试。候选未完成复验时，`best.json`不作为已验证改进；最终选择回退基线，候选保留在批次记录。

## 比较与产物

先用全1参数做120帧短检查，所有保存数组必须与原基线120帧完全相同。此检查尚未运行。

挖掘评分取抬离后末五帧有效土石/堆高工作装载量中位数（原占据阈值0.2），明显装载量优先，2个百分点内比较用时（2%容差）和归一化动作总变差。无效量测、保护失败不参与排序。这不是正式稳定卸车净增体积，也不是95%满斗认证。最佳候选固定卸料另查开发接收数量，不能用单一开发源证明泛化。

输出根目录 `/home/qunwang/wajueji/runs/night-parameter-search-20260925-v1/`：

- `batch.json`：预算、全部已完成候选、当前/失败阶段和原因。
- `best.json`：参数、分数、是否复验；不自动覆盖生产策略。
- `comparison.md`：比较表、实际选择、未验证边界。
- `baseline-replay.json`、`selected-replay.json`及每个完成候选的`replay.json`：现有零物理回放器入口。
- 每个候选的原日志、动作、阶段、物理帧、120帧检查点、final或recovery、显示缓存均保留；不自动删失败。

回放由 `work/contact-combined-replay-20260924-v1/view.py --manifest <绝对manifest路径> --output <新的回放目录>` 打开，回放器自行校验源SHA和拼接边界。当前还没有夜间候选回放。

## 已做验证及启动阻塞

8项CPU检查通过：全1恒等、阶段/95%作用域、倍率拒绝、评分优先级、全包语法、子进程失败传播、失败结果不选择未复验候选、失败基线拒绝且不启动子进程。接触与尾段模块导入检查通过，未初始化物理。候选物理、评分器真实成功端到端、120帧前缀一致及最佳复验尚未通过。

第二铲v4补收439帧/161.044秒通过交接，但规划抬离路线时报`cycle_observation_grid_budget`。正常`handoff.npz`已保存，SHA `f750e55864d6189ff6cd149cc1574894c236a334e6817a4b88f66493fd33c84d`；不必重挖或重做补收。只读核验见 [baseline-observation-review.json](baseline-observation-review.json)：接触末态已存在一个远处土粒，将4cm密集观察网格扩展至21,823,340格，超过12,000,000预算。原始物料不删、不忽略；飞散起因未建立，尚未完成独立控制审计，也未完成第二铲卸料。

本包显式绑定失败的原v4，`--run`会在任何新物理启动前拒绝。先完成观察问题调查与第二铲剩余段，再为真实合格基线更新/冻结搜索输入；不能覆盖失败报告，或仅改成功标签放行。基线若分段续接，评分/回放要纳入真实连续分段并核验边界，本包还不具备该接线。

## 当前允许执行的准备命令

```bash
cd /home/qunwang/wajueji
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python work/night-parameter-search-20260925-v1/test_search.py -v
/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python work/night-parameter-search-20260925-v1/batch.py --prepare
/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python work/night-parameter-search-20260925-v1/preflight.py
```

最后一条当前预期退出2并报告阻塞，不启动GPU。真实启动前还需核对没有其他物理实例、记录设备/日志/退出方式；搜索硬上限与用户既定8h授权不变。当前不运行`--run`。
