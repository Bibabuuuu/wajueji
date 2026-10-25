# R3-C双石逐ID路由资格

状态：`r3_c_per_rock_routing_probe_passed`。13/13项通过，授予`routing_physics_qualified=true`；整体R3-C仍为`r3_c_physics_qualified=false`、`training_qualified=false`。

本门沿用已通过结构预检的恰好两石`33/30/12`模型和三路命名力系桥，只运行两个隔离工况：

- `rock_000_mpm_only`：`r1-rock-000`固定映射到Newton body 0和唯一活跃collider，12/12帧产生MPM接触，直接冲量`0.9934312209N·s`；`r1-rock-001`为`0`。
- `rock_001_mpm_only`：`r1-rock-001`固定映射到Newton body 1和唯一活跃collider，12/12帧产生MPM接触，直接冲量`2.3082211983N·s`；`r1-rock-000`为`0`。

两轮均使用1071粒子、目标质量`20.04kg`的局部路由夹具；逐ID施加载荷回显残差为`0`，活跃collider样本全部有唯一所有者，非有限/未映射样本、禁止接触均为`0`，闭链最大约`0.04686mm`。七项MPM历史重置检查通过。

该资格只证明稳定`rock_id → Newton body index → collider ID集合`映射和两条单石MPM路由没有串线。为隔离路由，本门关闭铲斗MPM碰撞，并且每轮只让目标石块具有粒子碰撞；没有运行clear、整体MPM-only、rigid-only、combined四工况，也没有验证完整双石共享基质、铲斗分项、反馈因果、重复、步长对照或完整进程内混合reset。因此不得外推为双石动力学、20石、官方包络、材料标定、铲运或训练资格。

下一门应把两块石头同时放入共享小基质，显式输出各石块冻结力矩参考点，再完成四工况、逐石MPM/刚体/合计载荷、反馈、重复与`1/120→1/240s`步长对照。
