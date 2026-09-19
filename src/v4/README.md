# v4运行源码

这里是2026-09-19整理出的四轴模仿学习唯一受版本管理源码。Python文件来自最新D071观察运行时；测试放在 `tests/v4/`，历史报告放在 `reports/v4/`，原始轨迹放在Git忽略的 `local-data/`。

当前源码仍保留历史绝对依赖路径，以维持原始审计哈希。路径迁移必须作为新的候选提交并重新预检，不能静默修改后继续引用旧D071资格。

本地快速测试：

```bash
PYTHONPATH=src/v4 python3 -m unittest discover -s tests/v4 -p "test_*.py" -v
```
