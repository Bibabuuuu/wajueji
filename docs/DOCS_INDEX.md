# 项目文档索引

- 状态：`active`
- 权威范围：文档角色、阅读条件、状态和替代关系
- 最后复核：2026-09-24（Asia/Shanghai）
- 更新触发：新增权威文档、文档角色变化、替代/归档或目录重构

## 1. 阅读原则

项目Markdown数量较多，不能从日期、`v1/v2`、`next-section`或“最新”字样猜事实来源。

复杂任务的固定阅读链为：

1. 根目录`AGENTS.md`
2. [PROJECT_PLAN.md](PROJECT_PLAN.md)
3. [CURRENT_STATUS.md](CURRENT_STATUS.md)
4. [requirements.md](requirements.md)
5. [plans/INDEX.md](plans/INDEX.md)
6. `plans/INDEX.md`中全部`active`小节方案
7. 与当前结论相关的验收、决策和冻结证据

普通拼写、明确单点修复可以缩短阅读，但不得跳过与修改范围直接相关的小节方案。

## 2. 权威文档

| 路径 | 角色 | 状态 | 阅读条件 |
| --- | --- | --- | --- |
| `AGENTS.md` | AI协作纪律和必读路由 | active | 每次进入项目 |
| `docs/PROJECT_PLAN.md` | 唯一总体规划 | active | 每次复杂任务 |
| `docs/CURRENT_STATUS.md` | 唯一动态状态页 | active | 每次任务 |
| `docs/requirements.md` | 用户要求和比赛硬边界 | active | 方案、实现和验收前 |
| `docs/plans/INDEX.md` | 小节方案注册表 | active | 每次复杂任务 |
| `docs/decisions.md` | 重要选择及原因 | active-log | 涉及相关选择时 |
| `docs/acceptance.md` | 已验证能力和证据边界 | active-log | 宣称通过前 |

## 3. 当前小节方案

七份权威小节方案及状态见[plans/INDEX.md](plans/INDEX.md)。它们保存用户逐项确认的具体方案，不允许用总体规划或聊天摘要代替。

## 4. 支撑文档

| 文档 | 当前角色 | 由谁统领 |
| --- | --- | --- |
| `docs/handoff.md` | 当前跨会话接手步骤、运行入口及本地资产边界 | `CURRENT_STATUS.md`、S05 |
| `docs/autonomous-excavation-plan.md` | 详细架构和研究路线来源 | `PROJECT_PLAN.md` |
| `docs/v4/mixed-material-scene-plan.md` | 混合物理历史设计与R0–R3证据 | S01 |
| `docs/v4/mixed-material-three-cycle-integration-next-section.md` | 连续三铲详细实施来源 | S02、S03、S06、S07 |
| `docs/v4/contact-policy-next-section.md` | 接触策略、数据审计和训练交接来源 | S04、S05 |
| `docs/v4/control-and-observation-contract.md` | 已冻结的动作与观察合同 | S04、S05 |
| `docs/competition-plan-2026-09-10.md` | 截止倒排和历史提交口径 | S07 |
| `docs/environment.md` | 历史环境与运行记录 | `runbooks/a30-platform.md` |
| `.planning/task.md` | 实施流水与历史上下文 | `CURRENT_STATUS.md` |

支撑文档可以比权威方案更详细，但与权威方案冲突时，以权威方案和更新日期更晚的正式决策为准。

## 5. 证据和报告

当前S05闭环候选入口：[v7后段下压约束及1500帧结果](../work/contact-geometric-feedback-20260924-v7/README.md)；新会话先读[交接步骤](handoff.md)。

S05较早数据/模型接线入口：[D-145整理与离线试训](../work/contact-bc-data-preparation-20260924-v1/README.md)。初始地形缺失修正与旧索引适用边界以此证据和S05为准。

当前首批接触记录总入口：[10条清单](../work/contact-collection-progress-20260924-v5/README.md)。v5替代v2–v4的当前数量汇总，旧版保留为历史证据；受S04/S05及CURRENT_STATUS统领，不是新权威规划。

- `reports/`：精选、不可覆盖的报告和审计输出。由`acceptance.md`或小节方案定向引用，不要求启动时遍历。
- `evidence/`：冻结实验源码和证据。不是待修改主线，也不能按目录名猜最新版。
- `results/`：本地运行结果及中间产物。除非被验收记录绑定，否则不能自动升级为事实。
- `local-data/`、`runs/`、`models-local/`：本地大数据和运行产物，不进入默认Git事实来源。

## 6. 历史与被替代文档

- 重复的根目录V0.1总体规划和`docs/START-HERE.md`已经删除；入口统一为`AGENTS.md`及本索引。
- `docs/v4/*next-section.md`：必须查看各自顶部状态；已完成、暂停或已被替代的工作包只按需读取。
- 失败、interrupted、旧版本报告继续保留，不得删除或改写成通过。

## 7. 已知文档债务

当前仓库有较多历史链接指向未迁入主工作树的旧`evidence/`文件。权威入口和七个活动小节必须保持无断链；历史文档的断链在提交后分批修复，不在截止日前批量移动证据目录。

新增文档前先判断能否更新现有权威文件。不得再创建含义重复的“最新版规划”“最终方案2”或无索引的`next-section`。
