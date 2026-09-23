# 可铲运碎石场景 R0 几何与资产契约

状态：`passed`。本报告只验收几何、资产和确定性清单；物理步数为0，训练步数为0。

## 已冻结输入

- 实际铲斗源网格：`bucket.obj`，SHA256 `e24093436a3eef4e8f4fba21152e6f15ac349a837df8d13ebcdb23df860ca79b`。
- 内腔契约：SHA256 `e9826c5c5aebc30c6551352180ffc2da84ae4e1bbe3ea5227641771dcb285eb4`。
- 坐标系：`bucket_body`；严格复用MJCF中的`refquat`逆参考修正。

## 铲斗通规结果

- 网格3017点、5922三角形、12个连通分量。
- 内侧净宽`0.599490 m`，斗口截面弦长`0.886701 m`，零厚度球规的限制尺寸为`0.599490 m`。
- 两侧板厚度约`0.050616/0.050615 m`。
- 1mm齿尖带识别到5个斗齿；四个齿间净隙为`0.102798/0.104001/0.105192/0.103302 m`。
- 对四种石块原型分别检查24个正交姿态。`0.270 m`下96/96姿态通过；`0.330 m`下`angular_octa`为16/24，其余原型均24/24；`0.599490 m`不作为不规则石块可通行保证。

这里的通规是斗口零厚度几何预检，不等于完成了刚体石块的扫掠进斗验收。逐姿态结果见[bucket-gauge-report.json](bucket-gauge-report.json)。

## 石块资产与场景清单

生成`angular_tetra`、`angular_bipyramid`、`angular_octa`和`angular_block`四种闭合凸原型。每个原型的外观网格与碰撞代理使用同一份归一化OBJ，等效球直径固定为1m；场景缩放后质量按`s³`、质心惯量按`s⁵`重算。首版密度`2600 kg/m³`只是显式仿真假设，不是官方石料真值。

三类确定性清单均通过保守包围球的石块—石块和石块—地面初始无穿插检查：20块主训练分布、4块卡滞边界分布、3块拒绝/重选分布。按用户对满斗率的约束，主清单中80%石块不超过`0.16m`、最大仅`0.23m`、`d95=0.2015m`；`0.285m`以上仍隔离在卡滞边界专项。总质量和逐块惯量均已记录；当前没有合格的机构石块负载包络，不能从这些数值推断机构承载能力。详见[scene-manifest.json](scene-manifest.json)。

## R1前置阻塞

当前MJCF的铲斗刚体接触只有一个`bucket.obj`网格几何。MuJoCo网格碰撞采用凸包，因此该几何会封住凹形斗腔，不能用于“石块真实进斗”资格。R1必须先建立并验证保持开口和斗腔的凸分解或等价非凸接触表示；本R0没有擅自选择或部署该表示。

## 复现

```bash
PYTHONPATH=src/v4 python3 src/v4/mixed_material_r0.py \
  --config configs/v4/mixed-material-r0.json \
  --bucket-obj /home/qunwang/codex-work/small-soil-20260910-v17-performance/scene/input/mesh/bucket.obj \
  --inventory-contract /home/qunwang/codex-work/compact-bank-2026-09-11/inventory-contract.json \
  --asset-dir assets/v4/mixed-material-r0 \
  --output-dir reports/v4/mixed-material-r0-2026-09-19-v1
```

输出目录必须是新目录；如需复验已有报告，应使用新的临时输出目录并比较JSON与资产哈希，不覆盖本报告。

## 验证

- R0针对性测试：5项通过；新增R1几何与粒径策略测试：6项通过。
- 当前`tests/v4`合计56项通过。
- 仓库普通测试：101项执行，93项通过、8项因当前Python缺少Warp按既有条件跳过；该套件需把仓库外的只读`manual-force-tracking-2026-09-13`加入`PYTHONPATH`，否则迁移仓库中已排除的`manual_scene`会在收集阶段导入失败。
- 冻结连续挖掘候选测试：18项通过。
- 独立临时目录重新生成的4个OBJ和3个JSON与本报告逐文件一致。
