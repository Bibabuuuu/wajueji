# R2最终六场v2装配与烟测报告

本目录对应最终首轮集合：普通场景`26091911/12/13`，稀疏大石场景`26091922/25/26`，每场20块。六个模型均通过`mj_loadXML`结构检查，同一个最大容量20的R2桥完成时间0初始化，并分别完成1秒零命令沉降烟测。

根报告为：

- `report.json`：`r2_six_scene_load_only_passed`
- `bridge-init-report.json`：`r2_six_scene_bridge_init_passed`
- `settling-report.json`：`r2_six_scene_settling_smoke_passed`

这些报告证明最终集合的模型、桥和基础步进契约成立；随机几何初态的稳定资格由同版本稳定化报告单独给出。它们不授予受控进斗、携带、卸料、MPM耦合或满斗率资格。
