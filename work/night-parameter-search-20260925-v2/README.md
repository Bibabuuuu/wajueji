# 夜间8小时挖掘参数搜索：固定路线与三段基线

状态：`已实现、10项CPU测试/导入通过、等待基线完成和结束审核`。本包未启动物理/搜索/神经训练，原v1准备包和所有失败证据保留。用户仍选择有限参数搜索，只有挖掘参与优化，不更新BC权重。

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

1. 让当前第二铲固定路线自然结束或保护退出，不并发GPU试验。结束时按用户要求一次统一检查。若失败，不自动重跑或启动搜索。
2. 成功后，在物理Python环境执行审核目录的`prefix_checks.py contact`、`prefix_checks.py carry`、`audit.py`和`replay.py`；全部只读已有轨迹/缓存，不新增物理。已有结果不覆盖。补收审核仅授予正常保存前缀可信性，原规划失败仍保留。
3. 重新运行本包`batch.py --prepare`和`preflight.py`，冻结最终三段证据，核对没有其他物理实例。若仍有blockers不启动。
4. 在已授权8h窗口用`batch.py --run`启动一次；首先做全1参数120帧检查，全部物理数组须与原gate-120全等，失败即止；然后按预算执行候选/最佳复验并自动保存比较和回放入口。正式启动前记录命令/日志/设备/中断方式。

## 当前已做与未做

10项CPU测试通过：全1恒等、阶段/95%范围、倍率拒绝、排名、语法、子进程失败传播、未复验回退、失败基线禁止启动、分段时间/动作边界、三段回放顺序。接触模块和固定几何尾段导入通过，未构造物理实例。

尚未验证：分段基线完整结束审核、新包120帧物理恒等、真实有效评分、候选闭环、最佳复验或改善。当前检查结果以[readiness.json](readiness.json)为准；不将准备工作描述成训练完成。

物理Python `/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python`，策略Python `/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python`。在仓库根设`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。可运行`test_search.py -v`、`batch.py --prepare`、`preflight.py`；preflight当前预期退出2，不启动GPU。
