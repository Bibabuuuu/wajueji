# AI 协作入口

本文件适用于仓库内全部目录。开始分析或修改前，按以下顺序读取：

1. `docs/START-HERE.md`
2. `docs/v4/README.md`
3. `.planning/task.md`
4. `docs/requirements.md`
5. `docs/acceptance.md`顶部最新记录
6. `docs/decisions.md`顶部最新决策

## 当前事实来源

- `.planning/task.md`保存当前进度、下一步和运行状态。
- `docs/v4/README.md`保存2026-09-19四轴模仿学习主线、数据边界和当前断点。
- `src/v4/`是当前四轴运行源码的唯一Git副本；新版本用提交和分支表达，不再复制“最新版”目录。
- `docs/acceptance.md`只记录有证据支持的能力边界。
- `evidence/autonomous-multiscoop-2026-09-14-v1/`是当前连续挖掘集成源码。
- `evidence/.../results/`中的报告是冻结证据，不是待修改源码。
- 日期更早、名称含`version`、`failed`、`interrupted`的内容只能解释历史，不能覆盖顶部最新状态。

## 修改规则

- 先从明确提交建立自己的分支；不要用“最终版2”“最新版3”复制文件。
- 不修改`evidence/`中的冻结报告、审计和原始结果。运行源码需要调整时，先在新分支形成清晰差异，再以新运行名部署验证。
- 不把未知地形当空地，不删除失败记录，不把开发门槛写成比赛指标。
- `local-data/`、`models-local/`和`runs/`属于本地忽略区；原始证据不覆盖，精选小报告才进入Git。
- 不改390400点冻结工况、MPM物性、账本核心或机构保护，除非任务明确要求并建立独立对照。
- 当前没有给两位队友分配模块；阅读完成后先提交理解摘要，具体分工由团队讨论决定。

## 最小验证

普通Windows环境运行：

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s evidence/autonomous-multiscoop-2026-09-14-v1 -p "test_*.py" -v
```

依赖Warp的测试在未安装Warp时会跳过。完整MPM运行只在已回归的Ubuntu/NVIDIA环境执行；不得在仍运行的目录中覆盖源码。
