# MPM质量账本：只读离线版

2026-09-12：保留当前MPM主线，先复用原高度场方案中明确存量和转移量的设计。代码为`terrain/material_ledger.py`；没有导入原仿真入口、修改原源码或接管GUI。地形规划表示、表面显示优化、不同介质物性和反力控制是后续独立工作。

## 数据契约

一个MaterialLedger实例对应一个固定粒子集合的run。创建时给出唯一整数粒子ID、实际质量kg、可选整数材料ID；质量和材料随粒子ID保留。update允许输入数组重排，但拒绝粒子缺失、重复、新增、无效区域和重复/倒退时间。新增/删除粒子的模型必须另行定义质量转移契约，不能通过重建ID掩盖丢失。

区域是互斥的几何观测：source_region、bucket_cavity、truck_receiver、unassigned、outside_domain、ambiguous。源区域是固定ROI，不是物料来源标签；铲斗和车厢优先于源ROI，两接收区重合记ambiguous，超出明确分析域记outside_domain。界外仍计入总质量，不删除。invalid/nonfinite位置必须由分类器拒绝。

本次几何沿用已审计的实际侧板/斗口内腔判据，**不包含随斗堆高**；车厢为原保守区域，**不包含高于低侧沿的物料**。没有支撑/稳定判据，所以区域归属不能直接叫有效装载。unassigned可能是源ROI外仍在地面的土、堆高、空中或其他沉积，不能直接称撒落或漏失。

## 输出口径

| 字段 | 含义 |
| --- | --- |
| mass_kg / material_mass_kg | 各区域总质量、各材料分区质量 |
| partition_residual_kg | 分区质量之和减初始实际粒子质量 |
| interval_balance_max_error_kg | 存量变化与采样间转移净额的最大误差 |
| current_stock_kg | 当前车厢统计区内质量，包含初始已有物料 |
| sampled_entries_cumulative_kg / sampled_exits_cumulative_kg | 相邻观测间进入/离开统计区的累计质量，同一粒子重新进入会再计一次 |
| first_observed_entries_kg | 初始化后第一次被观测进入的唯一粒子质量；初始就在车里的不计新装入 |
| first_entries_still_inside_kg | 上述新进入粒子中当前仍在统计区内的质量 |
| net_stock_change_kg | 当前存量减初始存量；可以为负 |
| max_sample_gap_s | 最大观测间隔，暴露漏记短时进出的可能 |

每铲必须由操作记录或控制器明确begin_cycle/end_cycle。周期输出车厢存量净变化、观测进入/离开、首次进入质量；不要由铲斗土量减少自动判定装车或完成一铲。没有中间逐粒子状态时，不能恢复间隔内进出，不能用两帧冒充连续装车事件。

```python
from terrain.material_ledger import MaterialLedger

ledger = MaterialLedger(particle_ids, particle_masses_kg, material_ids)
ledger.update(initial_time_s, particle_ids, initial_region_ids)
ledger.begin_cycle('scoop_001')
# 每次只读观测：输入粒子ID和已有分类器输出，不推进物理。
record = ledger.update(next_time_s, particle_ids, next_region_ids)
cycle = ledger.end_cycle()
```

不输出实际松散体积或满斗率；该模块的多材料分账能力不代表三种介质物理模型已经完成。质量/初始密度只能作为另行标明的参考体积。

## 本次验证及限制

9项单元测试通过，覆盖重复进出去重、已有车载物料、混合材料质量、ID重排、增删/重复拒绝、时间/区域错误、周期边界、归属冲突、斗内减少不算装车。

原compact的0、29.5333、39.5333秒快照共390400点：内腔0→194.868120→194.894214kg，车厢保守区0→20.405312→26.667812kg，与既有内腔诊断及库存一致，分区/区间质量残差均0。10秒保持窗口观测进入7.802031kg、离开1.539531kg，净增6.262500kg；原内腔7468点全部保留。两段为分析窗口，不代表两个完整挖装循环。

独立核对v16历史落土试验原日志中的转移矩阵：central区间进入324.971554kg、离开23.197343kg、末存301.774211kg；side从初始310.124211增至315.238586kg，净增5.114375kg。两例各转移矩阵行/列质量与前后库存误差0。历史日志只有聚合矩阵，因此唯一粒子首次进入量标null，不推断。它们不是当前compact版本卸料验收，也不复用后来的车位转换去解释早期坐标。

本机两次完整CPU几何分类+账本分别约0.324/0.316秒，**不是实时100ms达标**。后续接入应复用现有分类输出并单独测量成本，避免重复CPU几何计算；输入几何范围和采样频率仍要验收。

复现命令：

```powershell
python -m unittest discover -s tests -p test_material_ledger.py -v
python evidence/material-ledger-2026-09-12/audit.py
```

详情与输入哈希见[evidence](../evidence/material-ledger-2026-09-12/README.md)。本轮物理步数0，原证据文件哈希前后相同；尚未接入在线UI或控制器。下一步是完整车厢接收几何/支撑口径与独立在线观测适配，不改变原正常物理基线。
