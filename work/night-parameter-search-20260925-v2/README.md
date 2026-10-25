# 夜间8小时挖掘参数搜索：固定路线与三段基线

状态：`第二源基线审核、评分与120帧短检通过；未启动候选搜索`。本包12项CPU测试通过，新增物理仅120帧；原v1和所有失败证据保留。用户仍选择有限参数搜索，只有挖掘参与优化，不更新BC权重。

## 本轮变化

- 候选固定尾段使用D-162同一静态几何检查，取消全场土粒地形图依赖；车体、地面、闭链、限位、保斗及所有原物理保护保留。原运行包`work/contact-second-fixed-route-20260925-v1/`不改，当前物理进程不受本次修改影响。
- 基线显式由接触2725帧、补收439帧、固定路线续段组成。评分累计三段时间和动作变化，带载量从新续段CLEAR_PILE末五帧读取；回放也保留三个连续来源。
- 仅允许原v4正常handoff SHA `f750e55864d6189ff6cd149cc1574894c236a334e6817a4b88f66493fd33c84d`在规划报错前的439帧作为前缀，核对前后来源、步号、哈希及结束审核。原failure标签不更改，不接受通用失败态继续。
- 基线未结束时只冻结代码/静态输入，manifest标记`prepared_awaiting_baseline`；结束审核后重新`--prepare`才冻结完整最终证据。运行中的report不会被误当成冻结结果。

## 已确认范围和预算

本机RTX4070串行执行，近土预备点直接恢复，不重跑APPROACH。六组候选只缩放curl_load且实际收斗<95%的BC意图；原控制/投影/变化率仍执行。非收斗阶段和95%之后的补收/带载抬离保持固定控制。

| 候选 | 大臂 | 小臂 | 卷斗 |
| --- | ---: | ---: | ---: |
| boom_arm_075 | .75 | .75 | 1 |
| boom_arm_050 | .50 | .50 | 1 |
| boom_050 | .50 | 1 | 1 |
| arm_050 | 1 | .50 | 1 |
| curl_115 | 1 | 1 | 1.15 |
| balanced_curl_115 | .75 | .75 | 1.15 |

普通候选止于CLEAR_PILE，不跑回转/卸料；最佳改善候选另重复一次并接固定卸料复验。不是六个神经网络，重复同一预备点不作为连续三铲证据。

总预算28800秒，使用单调时钟；最多6候选，保留3600秒复验/保存。每候选接触≤3000帧/900秒，固定补收和尾段≤6000帧/2000秒；120帧及阶段末保存，另有子进程超时/退出保存缓冲。保护、进程、无效量测失败均停批，不自动重跑；单批锁、不重置计时、不覆盖原结果，保留55GiB磁盘空间。未复验候选不作为最终选择；失败或无预算复验时回退冻结基线，不自动部署。

优化排序：有效CLEAR_PILE末五帧含土石/堆高工作装载中位数（占据阈值0.2）；明显装载量优先，2个百分点内按用时（2%容差）及动作变化比较。读取前核对缓存SHA及帧数；NaN/负数/无效边界不参与排序。该工作估计不是稳定混合卸车净增体积或正式95%资格。最佳复验另检查原开发接收量。

## 输入与输出

仓库 `/home/qunwang/wajueji`，分支`codex/contact-bc-collection-20260924`，本轮起始HEAD `7ac4eb8`。

- 预备点：`runs/contact-next-prepoint-20260925-v1/physical/prepoint.npz`，SHA `ab4f529501c3991022f6e72a276d4d7b48def9a888fcd85db3148356c73e420e`。严格零接触资格false保留。
- BC：`runs/contact-phase-bc-pilot-v1/pilot.pt`，SHA `b4bbc98cf472bcad23f89e014a025cfc933c36091ada91ced4c4cc9680d94095`。
- 三段基线：`runs/contact-next-scoop-20260925-v1/candidate/`、`runs/contact-second-finish-unload-20260925-v4/physical/`、`runs/contact-second-fixed-route-20260925-v1/physical/`。
- 结束统一审核目录：`work/contact-second-fixed-route-review-20260925-v1/`；要求contact-audit、carry-audit、audit-review、replay-review全部通过且输入身份一致。
- 搜索输出：`runs/night-parameter-search-20260925-v2/`，batch.json、best.json、comparison.md、baseline/selected-replay.json，各候选日志/动作/检查点/显示缓存均保留。未运行时这些搜索产物不存在；import-only只是零物理导入检查配置。

## 接下来的执行顺序

用户最新要求先完成统一审核及120帧短验证。本轮使用`batch.py --check-only`：完成真实基线评分和全1参数120帧验证后退出，不启动六组候选，也不创建batch.json。新增入口的12项CPU测试通过。

检查结果保存在`runs/night-parameter-search-20260925-v2/precheck.json`；成功后绑定manifest和评分/短检证据。后续`--run`只在这些SHA完全未变时复用本次短检，不重复跑120帧，并从28800秒预算扣除本次检查用时。检查失败不自动重试；检查开始后禁止重新prepare覆盖输入身份。

固定第二铲预备点只支持同条件参数比较，不证明跨铲泛化。同源最佳复验也只证明复现性。用户已询问此风险，建议最佳参数另在第一铲和第三铲开发预备点验证；这是待讨论的后续扩展，当前未加入六候选预算或启动额外物理。

本轮已完成上述审核、最终prepare、preflight和--check-only，不重新执行。短检证据见[short-check-review.json](short-check-review.json)：24数组与原gate-120全等，120帧43.213983秒，评分＋检查66.631579秒；基线工作满斗30.21405%。

用户新增第一、第二铲双起点方向，本包仍只支持第二源，因此暂不执行旧--run。双源需独立新包，先第一源加载与零步/120验证，再成对比较、减少组合并保留复验预算；详见docs/CURRENT_STATUS.md和S05。

## 当前已做与未做

已完成：127项三段结束审核、6562帧回放SHA/边界、最终证据冻结、12项CPU测试、真实有效基线工作装载评分、第二源全1参数120帧恒等检查。未完成：第一源新接口短检、双源候选闭环、最佳复验或改善。准备检查readiness.json是短检前只读预检，实际物理结论看short-check-review.json。

物理Python `/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python`；策略Python `/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python`。环境`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。禁止修改已经被检查绑定的源码或重新prepare；重复--check-only仅校验并复用成功结果，不重跑物理。
