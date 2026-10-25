# 可铲运碎石场景 R1 开口铲斗碰撞候选

状态：`r1_full_rock_single_chain_passed_r2_pending`。独立探针、刚体单元、力系，以及实际凸石块全部顶点的真实机构入口、举升后10秒保持和固定接收箱卸料均已通过；R1单石单元完成，多石场景和实际车斗CAD仍未通过。

## 碰撞表示

- 视觉仍使用冻结的实际`bucket.obj`，不改变外观和质量惯量。
- 刚体接触不再使用整斗单网格凸包，而是使用11段曲面斗壳箱体、2块位于内侧净宽之外的凸侧板和5个独立斗齿，共18个凸几何。
- 斗壳截面来自冻结内腔契约的55个点，以`0.01m`容差简化为12点；没有连接首尾的斗口封盖。
- 斗口中心到碰撞代理的最小静态间隙为`0.299740m`，大于`0.27m`探针半径`0.135m`。

完整参数和逐几何数据见[proxy-manifest.json](proxy-manifest.json)。

## 短时MuJoCo探针

使用MuJoCo 2.3.8、零插件和独立固定铲斗夹具，从斗口外`0.08m`释放球体并让重力指向斗腔：

- `0.12m`探针最大进入深度`0.610363m`，5259个时间步检测到斗体接触；
- `0.27m`探针最大进入深度`0.534957m`，5111个时间步检测到斗体接触。

两项均证明碰撞表示没有像整斗凸包那样封住斗口，结果见[probe-results.json](probe-results.json)。末态速度仍为`0.835/0.683m/s`，因此本探针不授予静止保持资格，也不替代不规则石块姿态测试。

## 机构候选与自由落体

在不修改冻结源模型和旧`19/18/12`桥的前提下，独立生成`26/24/12`、零插件候选：保留原机构19个广义位置/18个速度和12个执行器，新增1块自由刚体石头，并把18个开口代理挂到原铲斗body。新桥逐块输出位姿/速度，将石块—地面、石块—铲斗、石块—石块和石块—车斗列为允许接触，其余接触继续判为故障；MPM外力入口目前只预留、必须为零。

`angular_octa`石块等效直径`0.12m`、密度`2600kg/m³`，质量和完整惯量直接复用R0几何积分结果，不采用MuJoCo的近似网格惯量。加载检查得到质量`2.3524245790080363kg`。1秒自由落体中记录1820个石块—地面接触子步，末速度范数`9.52e-11`，最大接触闭合`0.1856mm`，禁止接触为0；冻结源模型前后SHA256一致。结果见[mechanism-freefall-report.json](mechanism-freefall-report.json)。这只通过模型加载、状态协议、接触分类和落地稳定分项；本轮没有石块—铲斗接触，力/力矩字段尚未获得接触用例验证。

同一桥在隔离的15°固定斜面上运行8秒。石块与斜面接触19020个内部子步，沿下坡世界`+x`方向移动`0.503517m`，最低高度较初态下降`0.178141m`，累计姿态转动`1.251528rad`；禁止接触为0、最大闭合仍为`0.1856mm`。末姿相对初姿的净转角只有`0.396753rad`，因此验收采用逐步累计转角，避免翻滚后姿态部分抵消。结果见[mechanism-slope-roll-report.json](mechanism-slope-roll-report.json)。第一次2秒观察只移动约`0.153m`，没有通过门槛；保持15°坡面和摩擦不变，将配置中的观察窗冻结为8秒后才通过。

桥协议随后扩展为一石或两石固定契约。两石候选为`33/30/12`、零插件，使用`angular_octa`和`angular_block`两个`0.12m`不规则石块；下方石块先落地，上方石块继续下落并发生206个石块—石块接触子步。两块稳定ID、各自位姿/速度和总质量均完整输出，总质量`4.7048491580160725kg`与R0逐块质量之和一致；第二块最终产生约`0.1413m`平面位移，禁止接触0、最大闭合`0.1856mm`。结果见[mechanism-rock-collision-report.json](mechanism-rock-collision-report.json)。这只验证两石状态、互撞分类和质量守恒，不代表多石进斗或R2完成。

受控推石夹具把单石放在默认姿态铲斗正侧板外的水平支撑台上，对回转轴施加相同的`+0.06`命令1秒，并配套运行石块移远的无接触基线。接触场景记录1777个斗石接触子步，铲斗沿`+y`移动`0.274125m`，石块沿`+y`移动`0.253392m`；铲斗反作用冲量为`[-0.543469,-29.196369,14.091800]N·s`，其中`y`方向与推石方向相反。关于铲斗原点的角冲量严格等于接触点自身力矩冲量与力臂矩冲量之和，残差为0。无接触基线斗石接触0、斗石冲量0；相同命令下接触使机构末态qpos最大改变`0.129943mm`，低于冻结的`1mm`上限且高于`0.01mm`可观测门槛，证明接触力不仅被记录，也进入了机构动力学。结果见[mechanism-bucket-push-report.json](mechanism-bucket-push-report.json)、[无接触基线](mechanism-bucket-push-clear-report.json)和[对照报告](mechanism-bucket-push-comparison-report.json)。该夹具只验证侧向推石和力系，不代表斗口进料。

## 真实机构入口与10秒保持

入口场景先静置0.25秒，再以`Bucket=+0.05`卷斗1.5秒，最后零命令稳定1秒。首个`y=0`居中尝试的石心虽从斗口外`-0.119926m`进入至`+0.196665m`，但分类斗石接触为0，因此拒绝通过。仅将石块横向初态改为`y=0.25m`后，记录1704个斗石接触子步；石心最终深入斗口`0.166831m`，连续位于冻结斗腔棱柱内`1.891667s`，末平动速度`0.021460m/s`。正式证据见[mechanism-entry-report.json](mechanism-entry-report.json)。

保持场景复用入口，随后以`Boom=+0.02`举升1.5秒并零命令保持10秒。石心相对收斗末态上升`0.046166m`，举升末高出支撑台顶面`0.085863m`；保持期最低高度只回落`0.002064m`，最终斗口深度`0.292567m`、末平动速度`0.019634m/s`，累计26209个斗石接触子步，禁止接触0。第一版门槛相对仿真初始高度计算，忽略0.25秒静置落高而失败；现按收斗末态举升量和台顶净高度验收。正式证据见[mechanism-hold-report.json](mechanism-hold-report.json)。

正式验收进一步使用R0实际`angular_octa`的全部6个顶点、逐步石块四元数和铲斗位姿，并允许与接触闭合门槛一致的`1mm`包络容差。入口末态6/6顶点在斗腔内，最终连续整石内含`1.30s`，最浅顶点斗口深度`0.072817m`；保持末态6/6顶点在斗腔内，最终连续整石内含`11.80s`，最浅顶点深度`0.199739m`。

## 固定接收箱卸料

保持结束后以`Rotation=+0.06`回转2秒至独立固定接收箱上方，稳定0.5秒，再以`Bucket=-0.05`开斗2秒并稳定2秒。石块最终中心`[4.332688,0.795137,2.473075]m`位于接收内界，记录2575个石块—接收箱接触子步；最终斗口深度`-0.359696m`、末平动速度`0.012781m/s`。全部顶点检查得到斗腔内0/6、接收箱内6/6。正式证据见[mechanism-dump-report.json](mechanism-dump-report.json)。接收箱是五块固定碰撞几何，不是实际车斗CAD。

## 大石块稀疏约束

常规场景以`0.08–0.22m`为主，数量上至少80%不超过`0.16m`；每场`0.22–0.27m`大石最多1块，且大石等效实体积不超过全部刚体石料的20%，含`0.27m`石块的常规场景比例不超过25%。`0.285m`以上只进入独立卡滞边界专项，不进入首轮成功示范。R0代表性主清单已调整为20块、80%不超过`0.16m`、最大`0.23m`、`d95=0.2015m`；唯一`0.23m`石块约占刚体等效实体积18.1%。

## 尚未通过

- 尚未完成多石运输/卸料和实际车斗CAD接收；
- 未接MPM、未训练。

## 复现

先生成代理与夹具，再编译临时探针并记录结果：

```bash
PYTHONPATH=src/v4 python3 src/v4/mixed_material_r1.py \
  --config configs/v4/mixed-material-r1.json \
  --inventory /home/qunwang/codex-work/compact-bank-2026-09-11/inventory-contract.json \
  --bucket-obj /home/qunwang/codex-work/small-soil-20260910-v17-performance/scene/input/mesh/bucket.obj \
  --asset-dir assets/v4/mixed-material-r1 \
  --report-dir reports/v4/mixed-material-r1-2026-09-19-v1

python3 src/v4/run_mixed_material_r1_probes.py \
  --repository-root /home/qunwang/wajueji \
  --report-dir reports/v4/mixed-material-r1-2026-09-19-v1 \
  --source src/v4/mixed_material_r1_probe.cc \
  --include-dir /home/qunwang/codex-work/excavator-workspace-20260907/bundle/mujoco-custom/include \
  --library-dir /home/qunwang/codex-work/excavator-workspace-20260907/bundle/lib

PYTHONPATH=src/v4 python3 src/v4/run_mixed_rigid_bridge_smoke.py \
  --repository-root /home/qunwang/wajueji \
  --config configs/v4/mixed-material-r1.json \
  --inventory /home/qunwang/codex-work/compact-bank-2026-09-11/inventory-contract.json \
  --source-model evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/mechanism.xml \
  --mechanism-header-dir evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8 \
  --controller-header-dir /home/qunwang/codex-work/autonomous-acquisition-2026-09-13/mechanism-auto8 \
  --mujoco-include /home/qunwang/codex-work/excavator-workspace-20260907/bundle/mujoco-custom/include \
  --mujoco-library-dir /home/qunwang/codex-work/excavator-workspace-20260907/bundle/lib \
  --report reports/v4/mixed-material-r1-2026-09-19-v1/mechanism-freefall-report.json \
  --parent-report reports/v4/mixed-material-r1-2026-09-19-v1/report.json

# 两石互撞使用同一命令，将场景和输出改为：
# --scenario rock_collision
# --report reports/v4/mixed-material-r1-2026-09-19-v1/mechanism-rock-collision-report.json

# 推石对照还需分别运行bucket_push_clear和bucket_push，随后执行：
# python3 src/v4/compare_mixed_rigid_bucket_push.py --help

PYTHONPATH=src/v4 python3 src/v4/run_mixed_rigid_bridge_smoke.py \
  --repository-root /home/qunwang/wajueji \
  --config configs/v4/mixed-material-r1.json \
  --inventory /home/qunwang/codex-work/compact-bank-2026-09-11/inventory-contract.json \
  --source-model evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/mechanism.xml \
  --mechanism-header-dir evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8 \
  --controller-header-dir /home/qunwang/codex-work/autonomous-acquisition-2026-09-13/mechanism-auto8 \
  --mujoco-include /home/qunwang/codex-work/excavator-workspace-20260907/bundle/mujoco-custom/include \
  --mujoco-library-dir /home/qunwang/codex-work/excavator-workspace-20260907/bundle/lib \
  --scenario slope_roll \
  --report reports/v4/mixed-material-r1-2026-09-19-v1/mechanism-slope-roll-report.json \
  --parent-report reports/v4/mixed-material-r1-2026-09-19-v1/report.json
```

本轮验证还包括：`tests/v4`共66项通过；仓库普通测试101项通过；冻结连续候选18项通过；R1资产、代理清单和两份夹具在独立临时目录重新生成后逐字节一致；自由落体、斜面、互撞、推石基线/接触、真实入口、保持和卸料共8类机构候选确定性通过；10项父报告哈希链全部匹配；`git diff --check`通过。
