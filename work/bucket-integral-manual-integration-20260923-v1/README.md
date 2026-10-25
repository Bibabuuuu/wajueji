# D-122 积分候选人工试采入口

已实现独立入口并完成短验证。用户已授权在时间有限的情况下暂停反向/抖动深挖，优先人工收斗—抬升试采；不把停止失败改判为通过。

## 当前人工运行

- 运行目录：`/home/qunwang/wajueji/runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-182307-822279`。
- 从D-115原完整预备点6344e74f…b3f4恢复，不执行APPROACH，不使用3帧探针末态。
- [启动观察](manual-ready-observed.json)：实际暂停、0物理帧、精确恢复、Microsoft X-Box 360 pad已识别、积分初值0。
- 运行继续由用户操作；此处启动记录不授予人工带载单铲成功或训练资格。

## 接入

`bash /home/qunwang/wajueji/scripts/run_47_sparse_manual_contact_integral_local.sh APPROACH_OUTPUT_DIRECTORY --manual-input gamepad`

该脚本显式启用`--bucket-integral-candidate`；旧入口默认关闭。只作用铲斗轴，头文件逐字等同D-120，SHA256 226e9421fcae393a6084edb54c43061378d73101f88a29839bbb4be5656ebecc。Ki1.5e6、累积±120kN、30kN/s和0.5mm死区不变，不采用D-121强制反向退出。P/D、60Hz MPM/4800Hz刚体、土石与原硬故障冻结不变。

人工入口从第一步启用零初值积分，增长需要主动铲斗命令；没有D-120对照专用的11秒激活延迟，不能称完整新轨迹已经通过旧加载对照。逐帧、诊断快照元数据保存积分力/控制目标，UI标明BUCKET INTEGRAL PILOT。诊断快照仍不具有完整中途恢复资格。

## 验证

- 34项定向测试通过：原人工保护/输入/优化上下文和新候选作用域、关闭无修改、异常恢复、冻结头文件与锚点拒绝。
- 实际接入头文件C++限幅/变化率/中性/反向/退饱和/死区/非法输入及简化恒载检查通过，简化模型不是带载物理资格。
- 真实3帧零命令入口：`/home/qunwang/wajueji/runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-182209-821575`，9项报告检查全真，源与绑定不变，积分全程0；渲染ready在暂停0帧写出。
- [验证与SHA绑定](validation.json)、[定向测试日志](tests.log)、[C++日志](native-test.log)。

## 操作范围

摇杆回中，先按1标记切入，空格开始；2收斗、3抬升、4确认离土。空格暂停；X结束并保存，正常结束含原30帧零指令尾窗。目标先验证收斗、带载离土与短暂保载。当前硬门仍拒绝石—车接触，不在此入口强行卸车。

原D-120停止/反向门失败仍保留，新的人工操作是否可用待用户实际反馈与日志核验。原默认手操未切换，独立运行明确使用候选；不是正式训练数据。

文件清单不包含仍在写入的manual-launch.log。
