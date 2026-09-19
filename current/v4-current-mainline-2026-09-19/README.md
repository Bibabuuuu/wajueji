# v4 当前主线工作包

整理时间：2026-09-19（Asia/Shanghai）

这是从 `/home/qunwang/codex-work` 中筛出的当前四轴模仿学习主线。整理过程只复制文件，没有移动、修改或删除原始实验目录。

## 先读结论

当前断点不是“精确接管刚完成”，而是：

1. 六条训练示范已形成行为克隆基线，另有两条封存盲测轨迹。
2. 自动策略在卷斗阶段出现反向动作，保护逻辑在物理继续前冻结。
3. `v4-handshake-20260918-161214` 完成精确人工接管，记录 282 条人工纠正；3 秒持土窗口的 13 个快速档测量点均超过 95%。该结果不是正式参考档资格。
4. 后续多种微调、阶段重标注和约束拟合均未同时通过全部离线门槛，没有形成新的自主成功策略。
5. 最新物理观察运行 `v4-observation-20260918-210014` 在进场阶段持续下压动臂；234 个决策、11.7 模拟秒后因预测触地时间 1.991 秒触发接管。没有人工继续，墙钟一小时后关闭。
6. 当前没有正在运行的 MPM、训练或观察进程。

最新诊断表明主要问题是进场阶段分布漂移：在相同前进位置，人工轨迹的目标相对高度约为 1.04–1.13 m，自动轨迹约为 0.35 m，最新运行终点只有约 0.17 m。

## 目录导航

| 目录 | 内容 | 使用方式 |
| --- | --- | --- |
| `01-runtime/` | 最新 D071 运行源码与既有预检结果 | 下一轮代码工作的起点 |
| `02-models-and-labels/` | 初始 BC、标签、盲测、D071 observation-only 模型与审计 | 模型和数据口径来源 |
| `03-data/training/` | 2 条人工示范 + 4 条通过审计的 v4 示范 | 当前训练集，只读保留 |
| `03-data/holdout/` | 2 条封存盲测轨迹 | 不得混入训练 |
| `03-data/corrections/` | 精确接管成功运行及 282 条纠正 | 后续 DAgger/残差策略的重要数据 |
| `04-latest-evidence/` | 最新失败运行、漂移诊断和接管预检 | 判断下一步的主要证据 |
| `05-offline-history/` | 接管后的各类离线纠正实验 | 防止重复走已失败路线 |
| `06-new-runs/` | 后续新实验输出位置 | 新结果写到这里 |

## 最重要的文件

- 精确接管：`03-data/corrections/v4-handshake-20260918-161214/report.json`
- 最新运行：`04-latest-evidence/v4-observation-20260918-210014/report.json`
- 最新状态：`04-latest-evidence/v4-observation-20260918-210014/status.json`
- 漂移诊断：`04-latest-evidence/v4-approach-drift-2026-09-18-v2/report.json`
- D071 模型：`02-models-and-labels/v4-constrained-observation-2026-09-18-v1/observation-only-checkpoint.pt`
- D071 独立审计：`02-models-and-labels/v4-constrained-observation-2026-09-18-v1-audit/audit.json`
- 当前运行入口：`01-runtime/launch_v4_takeover_handshake.py`

## 下一步建议

不要直接再跑一次 D071，也不要继续对整网做无约束微调。先解决进场阶段的高度漂移：

1. 对齐训练轨迹与在线运行的观察分布，重点检查目标相对坐标、动臂动作和阶段切换。
2. 在零物理或空载回放中验证轨迹能进入 `relative_x >= -0.3 m` 且 `relative_z = 0.85–1.05 m` 的进场窗口。
3. 确认不会提前触发 TTC 接管后，再做短带土观察运行。
4. 仍保留精确接管，新的人工纠正单独写入 `06-new-runs/`，不得覆盖现有证据。

运行环境及未复制的大型依赖见 `EXTERNAL_DEPENDENCIES.md`；原始目录对应关系见 `SOURCE_MAP.md`。
