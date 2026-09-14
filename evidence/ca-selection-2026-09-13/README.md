# CA选点代理与冻结MPM对照报告

更新：2026-09-14T00:37:28+08:00。实验目录沿用用户指定的2026-09-13归档入口；部分运行跨至9月14日。

**本轮实验交付已完成；原版CA当前不能取得逐铲选点部署或受力训练资格。** 这里的完成指独立构建、可复现基准、10候选、冻结MPM对照、失败保存与资格裁定已完成，不表示代理通过或项目整体完成。最终进程/关机状态单独由 shutdown-status.json 记录。

## 结果与裁定

CA离散守恒与同种子确定性通过；冻结MPM分配 4 组，实际通过并完成保持 3 组。CA与MPM的完整共同样本Spearman为 **1.000000**。CA前两名保留MPM最佳：**True**。分配样本中CA几何/非零体积预筛未提前识别的物理失败：**C02_near_shallow**。

CA可行候选排序：C06_rear_deep → C04_baseline → C10_late_curl → C03_center → C07_left_edge → C08_right_edge → C01_near → C02_near_shallow。

另有CA给出正斗量、但真实机构几何拒绝的候选：C05_rear。这类样本由前置几何门控截获，不记作已推进的MPM失败，也说明不能单凭CA体积给执行资格。

已完成MPM候选按末态内腔质量排序：C06_rear_deep → C04_baseline → C07_left_edge。失败样本不按未抬离的瞬时库存混入成功排序；几何拒绝明确为0步，不冒充MPM计算。样本少且选自CA前列/边沿/浅切，不能推断总体失败率或95%成功率。

|候选|几何|CA末态斗中土 L|MPM状态/步数|MPM抬离/末态内腔 kg|保持 s|地面RMSE m|
|---|---|---:|---|---:|---:|---:|
|C01_near|passed|8.250|not_selected / 0|— / —|—|—|
|C02_near_shallow|passed|6.375|failed / 2632|— / 1.722|0.000|—|
|C03_center|passed|81.125|not_selected / 0|— / —|—|—|
|C04_baseline|passed|102.375|passed / 6147|62.181 / 62.155|10.000|0.407|
|C05_rear|failed|81.125|geometry_rejected / 0|— / —|—|—|
|C06_rear_deep|passed|149.625|passed / 7038|106.175 / 106.175|10.000|0.436|
|C07_left_edge|passed|20.750|passed / 6013|58.972 / 58.946|10.000|0.362|
|C08_right_edge|passed|17.125|not_selected / 0|— / —|—|—|
|C09_shallow_failure|failed|0.000|geometry_rejected / 0|— / —|—|—|
|C10_late_curl|passed|84.875|not_selected / 0|— / —|—|—|

上表CA L是离散斗上土柱体积，MPM kg是冻结CAD几何内腔的真实粒子质量诊断，二者只分别用于各自排序，没有密度换算或满斗率宣称。所有实际失败保留：

- C02_near_shallow：TimeoutError('stage_tracking_timeout:CURL')；已推进MPM 2632步。
- C05_rear：ValueError('cycle_pose_unresolved: closure_m=0.00174405816, position_error_m=0.00171100595, pitch_error_rad=-0.00266624394, active_bounds=[15]')；已推进MPM 0步。
- C09_shallow_failure：ValueError('cycle_pose_unresolved: closure_m=0.0289666348, position_error_m=0.0660308586, pitch_error_rad=-0.627280441, active_bounds=[12]')；已推进MPM 0步。

## 可复现构建与性能

上游soil_dynamics_cpp v1.1.0，提交7300bb65ed65241254917b41a72e2cc2eec83a3b；原截图模型本地提交4251c862e0b4e6aec25069d48894d5bd9939813d。原版七个核心翻译单元保持不变，以build.sh的g++ -std=c++20 -O2直接构建。源码tar、编译器、库哈希、编译日志、输入和逐调用日志均归档。顶层CMake保留但不要求额外测试/benchmark依赖；没有修改高保真Python环境。

核心为321×321端点包含网格，16×16m、水平/竖向5cm；插件输出320×320，两者口径分开。CA参数休止角0.85rad、最多3次松弛、buffer4、tol1e-5、seed20260913。

- 十候选首个重复合计 25717 次调用，其中真实更新 2002 次。
- 真实更新中位 **0.7423 ms**、P95 **1.3627 ms**；空调用中位 **3.263 μs**、P95 **4.681 μs**。
- 每次完整检查中位 2.0064 ms；三张320×320数组复制微基准中位 0.6866 ms。复制只估计内存写入，未复现MuJoCo归一化、条件分支和GUI开销，不能当生产插件端到端性能。
- 无MPM并发的C04独立核对，三次真实更新中位数的中位为 0.9262 ms，空调用为 3.014 μs；三次末态均与最终主表一致。
- 专门±1mm位移试验20000调用、真实更新0、地形与斗中土变化0；这些调用不作为挖土更新加速证据。
- 各候选完整带检查墙钟、真实/空调用累计耗时分别见candidate-comparison.csv和steps.csv。CA统计部分与独立几何/MPM任务并行，最终另有空闲机器上的isolated-benchmark用于无并发核对；不宣称CA/MPM等精度加速比。

## 守恒、确定性与输入修正

每一步显式执行返回布尔值的CheckVolume与CheckSoil，另按所有地形柱与两层斗中土独立求离散体积，包含空调用。十候选×三独立进程全部通过；最大有符号离散差额绝对值 **0格**。浮点高度积分原差额最大 3.42362572e-06 m³持续保留，不调容差抹除。守恒资格仅限离散土柱，不包括力学、质量/惯性或支撑。

check-probe额外实际删除一格和制造斗层/记录不一致，CheckVolume均拒绝，正常两格通过。三次重跑比较全部地形、斗土四层与斗体四层末态二进制SHA256，相同种子逐字节一致；不外推到不同种子、不同编译器或MPM确定性。

初版浮点floor在3条观测/CA边界行选错相邻格；v1.1改为整比网格的整数索引，改变54个高度柱，初始积分减少约0.057m³，未知掩码未变。原ca/、ca-results.json、terrain.txt和缺SciPy导出日志全部保留；最终主表使用ca-v1.1/与terrain-v1.1.txt。候选、轨迹、MPM及验收阈值未变；修正后前两名是否保持原分配：True。

## 物理与地形证据边界

每组MPM新建完整390400点初态，逐值检查初始粒子、零初速、原物性/接触配置及冻结源。先做全路径几何、空载动力学和独立审计，再带土运行。复用原8mm目标边界、驱动端负载补偿、有限旋转修复、原生土反力、实际到位/速度确认与20kg获取门槛；LIFT后仍保持10秒。没有q/v快照续跑、局部冻结粒子、额外CA阻力或放宽失败阈值。

- C04_baseline：抬离内腔 62.181 kg；10秒后原群体仍在内腔 62.155 kg（比例 0.999580）；末态总内腔 62.155 kg。
- C06_rear_deep：抬离内腔 106.175 kg；10秒后原群体仍在内腔 106.175 kg（比例 1.000000）；末态总内腔 106.175 kg。
- C07_left_edge：抬离内腔 58.972 kg；10秒后原群体仍在内腔 58.946 kg（比例 0.999558）；末态总内腔 58.946 kg。

地形对照在原/末观测共同有效格上，将CA的5cm格取2×2最大值与10cm观测cell-max比较；保留全共同网格RMSE，也补充排除冻结内腔包围盒扩15cm投影后的地面RMSE。高处携土与遮挡可能污染前者，因此补充结果单独列出，不事后替换主判据。所有数组与掩码在paired-terrain.npz。

补充控制试验把每组MPM实际反馈执行的逐步斗姿重放到同一CA，结果见actual-replay-comparison.json，全部实际姿态重放的守恒检查通过。它用于显示参考/实际路径离散差异，不事后调参或替换主排序；若MPM失败，仅重放到其真实停止端点。

每候选CA改变了809～912个观测未知柱。未知处的0只供数值预演，不表示已知空地；这本身阻止部署。原三角斗保留0.68m宽，并将原齿尖对齐当前TIP，未假装等于CAD内腔。初始陡台阶被CA休止角规则大范围重排，不能把这种规则施加到冻结MPM台阶。

预先定义判据逐项结果：

- `ca_conservation`：通过。
- `ca_determinism`：通过。
- `three_completed_mpm`：通过。
- `paired_rank_at_least_point8`：通过。
- `top2_keeps_mpm_best`：通过。
- `no_failure_miss`：未通过。
- `terrain_rmse_at_most_point1`：未通过。
- `no_unknown_affected_columns`：未通过。

因此目前只保留CA为诊断预演和低成本研究输入，不能把“快、守恒、确定”改写成选点代理通过，也不能启动基于该代理资格的连续三铲或受力残差训练。下一项是独立修订观测覆盖/未知区、工具近似和适合黏聚台阶的预筛方法，再按相同冻结MPM协议复核；geometry05/empty04/contact03保持最终物理基线。

## 文件与恢复

- adjudication.json：机器可读判据、排序、每候选结果和限制。
- candidate-comparison.csv：全部候选，含未通过/未分配/0步拒绝。
- INTERFACE.md、interface.json、candidates.json：观测与候选版本、单位和参数语义。
- REPRODUCE.md、build.sh、bench.cpp、upstream-soil.tar：核心与完整独立重跑入口。
- source-runtime-audit.json、archive-manifest.json、archive-record.json：冻结源核验、完整证据清单与本地归档校验。
- shutdown-status.json：进程清理与两台机器关机提交状态，不能用本报告生成时间推断已关机。

工作中间件和全量压缩证据保存在同交付目录的work/ca-selection-v1；项目交接入口为指定工作树的.planning/task.md、docs/handoff.md、docs/acceptance.md与docs/decisions.md。


关机交接（2026-09-14T00:46:12.454545+08:00）：本轮实验交付和801文件本地校验已完成，相关进程已退出。已通过SSH向desktop-tailscale提交正常关机，run-u113.service返回success、退出码0；该回执确认请求提交，不冒称物理断电后观测。接下来按原授权正常关闭当前Windows；最终提交状态见shutdown-status.json。
