# 双起点夜间挖掘参数优化（D-163）

用户已授权本机RTX4070最多8h，以明早满斗率改善为优先；这是参数搜索，不更新神经网络权重。不合入多接触点候选，不修改冻结物理、原始示范或保护。

代码：本目录与`../night-dual-tail-20260925-v1/`。结果：`/home/qunwang/wajueji/runs/night-dual-search-20260925-v1/`。精确代码/输入SHA由manifest冻结。旧单源/双入口包不修改、不重新prepare。

## 执行顺序与预算

第一、第二铲分别从完整保存预备点恢复；保留各源自己的控制、材料及车体历史。第一源接触时无truck，尾段按已有第一铲流程加入静止truck；第二源全程保留truck。每次恢复都检查24数组与原生控制历史。

1. 两个全1基线完整跑到CLEAR_PILE；前120帧嵌入旧参考数组精确比较，不另外重跑短检。
2. 候选池6组：大臂/小臂同时0.75、同时0.50、仅大臂0.50、仅小臂0.50、仅卷斗1.15、双臂0.75＋卷斗1.15。其余倍率为1。
3. 仅curl_load且实际收斗<95%修改BC意图，仍经过原几何反馈、限幅、预览及物理保护。运输/卸料固定，10mm累计回落仅FINISH_CURL。
4. 每组在两源成对评分，任一源装载退步超过2pp即不可选。比较CLEAR_PILE末五帧工作装载中位数，保留原占据阈值0.2。此开发估计不等于稳定卸车体积。
5. 每启动一组前检查最坏预算，预留7200秒给最佳双源独立复验（含固定卸料）。最多6组，不保证预算内全部运行。候选失败不自动重跑，同组第一源失败则跳过该组第二源。
6. 最佳必须两源复验通过、接收原载荷土粒比例≥98%、工作装载不比候选及基线低超过2pp，才写confirmed。否则保留基线。没有改善则如实报告。

总墙钟28800秒扣除先前入口检查121.096636秒；全局定时器保留50秒终止保存余量，子进程收到SIGINT保存恢复信息。systemd为独立外层超时保险。不依赖Codex持续运行或监控。

## 保护失败处理

`failures.py`明确列出可淘汰的动作失败。只有完整恢复/来源未变、已执行数据的数值/材料/闭链等可信、失败检查点存在且SHA匹配，才允许将已知动作保护失败记为rejection并继续下一个候选。原失败场景立即停止，下一组重新恢复原预备点；不从失败状态续挖、不放宽阈值。

NaN、材料状态/闭链异常、未知异常、程序/输入/恢复不一致、保存失败、磁盘余量不足55GiB或基线复现失败均停止批次，保留证据。未知异常不当成负奖励吞掉。基线失败说明对照不成立，不能继续候选并声称改善。

## 启动与查看

从仓库根目录：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python work/night-dual-search-20260925-v1/batch.py --prepare
```

已启动批次禁止prepare/再次run；不可变目录拒绝重试。后台unit为`wajueji-dual-search-20260925-v1.service`，实际启动参数、时间、PID和截止时间记录到运行目录`launch.json`与`batch.json`。

早上先看`batch.json`（状态/退出原因/淘汰记录）、`comparison.md`、`best.json`与`selected-first-replay.json`/`selected-second-replay.json`。每轮保存contact/tail日志、逐帧动作、120帧周期检查点、显示缓存和评分。未完成基线可能没有best.json，不冒充有效结果。

中断：`systemctl --user stop wajueji-dual-search-20260925-v1.service`。停止后保留结果，不自动恢复或重跑。状态检查只按用户后续请求执行；本任务不安排定时监控。

CPU验证：`python -m unittest discover -s work/night-dual-search-20260925-v1 -p 'test_*.py' -v`（使用物理Python/上述环境）。测试含候选失败继续、未知异常停止、双源复验及失败回退。实际满斗改善以运行结果为准。
