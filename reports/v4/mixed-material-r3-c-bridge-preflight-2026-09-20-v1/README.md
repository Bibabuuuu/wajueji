# R3-C双石桥结构预检

状态：`r3_c_two_rock_bridge_preflight_passed`。16/16项结构检查通过，且只授予`bridge_preflight_qualified=true`；`physics_qualified=false`、`training_qualified=false`。

本报告固定了一个恰好两石的小夹具：生成模型契约为`33/30/12`，稳定ID依次为`r1-rock-000`和`r1-rock-001`；桥只接受铲斗和两块石头共三个按名称指定的世界系外力系。状态输出包含逐石世界线/角速度及逐石上一拍MPM载荷回显，铲斗力矩仍使用R3-B的外步起点固定参考。

本次只做两次确定性模型/源码生成、C++编译和时间零协议检查。执行了`0`次`mj_step`、`0`次MPM/GPU步，没有验证双石动力学、碰撞体映射、逐石反力、完整进程内reset、官方包络、材料标定或训练。

正式产物：

- `report.json`：结构预检、时间零状态和冻结输入链。
- `generated-model.xml`与`generated-model-manifest.json`：恰好两石的小夹具模型及清单。
- `generated-bridge.cc`与`generated-bridge-manifest.json`：三路命名外力系桥及源码清单。

下一道门仍限于该小夹具：建立稳定`rock_id → Newton body index → collider ID集合`映射，分别做`rock_000_mpm_only`和`rock_001_mpm_only`单路由检查，再加入clear、MPM-only、rigid-only、combined四工况、逐石力矩参考、反馈、重复和步长对照。刚石步长分项继续只作诊断；不得直接扩到20石或官方`2.2m`包络。
