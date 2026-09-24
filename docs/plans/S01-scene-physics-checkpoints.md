# S01 场景物理与检查点

- 状态：`active；已授权实施`
- 权威范围：混合土石场景、分层构造、检查点完整性、47/65石资格和独立审计
- 最后复核：2026-09-22（Asia/Shanghai）
- 依赖：无；它是后续所有小节的物理输入
- 更新触发：正式场景、检查点schema、物理参数、资格门或主线候选变化

## 目标

提供一个既稳定又可真实挖掘的“MPM连续基质＋独立动态刚体石块”场景，并能从完整续步检查点精确恢复。场景资格、视觉效果和挖掘能力必须分别验收。

## 已确认路线

- 使用Newton/Warp MPM表示可变形基质，MuJoCo表示机构和动态石块。
- 石块必须真实参与接触、携带和卸料，不能用贴图、静态障碍或不可挖刚性高度场替代。
- 石块采用分层骨架、覆土、受控压入、自由释放和无控制保持的增量构造方式。
- 后续层从最近已验证完整检查点继续，不从第一层重复施工。
- 检查点必须包含全部粒子字段、五类材料历史、刚体qpos/qvel、固定网格参考、grid warm-start、solver场和刚体历史。
- 恢复后必须验证下一步连续性及短窗保持；文件存在、SHA正确或数组可读都不足以授予续跑资格。
- 刚体的MPM接触预测使用本步重力自由速度；MuJoCo仍负责真实重力，不允许重力或冲量重复记账。

## 当前正式输入

### 47石正式主线

- 256441粒子、47石。
- SHA256：`40b98e83bcc1f7b91366612755eee57bc5df21db2a5c53fe2accf4408c987ea5`。
- schema-v2独立恢复审计和完整场景预接触门已验证。
- 它是截止前不受65石支线阻塞的正式挖掘输入。

### 65石增强候选

- 266091粒子、65石。
- SHA256：`22fe16dfecc178257754d6646142de29a231ff91e0f09bee03210a414d64e9b7`。
- 构造和正式写出已通过；v2独立只读审计以退出码0完成，一步连续性和120帧严格保持通过，源文件未改变。
- 只有后续完成完整场景接入，并由用户比较外观与时间成本后，才可以讨论替换47石演示场景。
- `five_layer_hold_start_diagnostic_only`只用于诊断，不能训练、挖掘或对外宣称通过。

## 本地33→47续建结论（非正式支线）

- 冻结输入为本地独立33石schema-v2检查点，SHA256为`d9567c8d6b8036466cc9ec34574802b562193b0355f667413629b47e229ed38a`；它在本次运行前后保持不变，且不替代A30正式47石输入。
- 旧运行在最终汇总读取历史`solver_restore`字段时异常退出。运行器现接受旧`restore.solver_continuation`和新`solver_restore`两种布局；定向测试、编译和文档检查通过。
- 2026-09-22在RTX 4070/cuda:0从冻结33石输入完成558个增量步，物理完成后保存了非正式recovery checkpoint和证据；这证明字段兼容缺陷已消除，但不等于场景资格通过。
- 正式晋级报告状态为`fortyseven_rock_four_layer_failed`：21项检查仅`additional_uncontrolled_hold_all_frames_stable`失败。只读recovery续跑最终观察281帧，其中10帧稀疏瞬态、最长连续7帧、峰值`0.0972281m/s`，超过累计6/连续2预算，故正式checkpoint仍为`None`且不追溯改判。
- 程序继续到连续60帧严格恢复后保存了明确`diagnostic_only`的拒绝点；从该点全新恢复的120帧严格审计通过。该证据只新增本地铲斗接入吞吐测试资格，不能用于APPROACH、人工演示、采集、训练或替代A30正式输入；不得凭末端恢复稳定而抹去前序预算失败。

### 台式机只读审核清单

- 审核范围已在`/home/qunwang/wajueji`：`src/v4/run_layered_mound_fortyseven_rock_four_layer.py`、`tests/v4/test_layered_mound_checkpoint_continuation.py`、`docs/CURRENT_STATUS.md`、本文和`docs/acceptance.md`；本地运行包装、配置、冻结33石输入、最终报告和recovery证据位于`/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/local-47-rebuild-v1/`。
- 核对兼容逻辑只适配报告布局（历史`restore.solver_continuation`与当前`solver_restore`），没有放宽物理、检查点schema或正式A30合同；运行`PYTHONPATH=src/v4 /home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python -m unittest -v tests.v4.test_layered_mound_checkpoint_continuation`、`/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python -m py_compile src/v4/run_layered_mound_fortyseven_rock_four_layer.py`和`python3 scripts/check_docs.py`。
- 从最终报告复核：状态必须为`fortyseven_rock_four_layer_failed`、21项检查唯一失败项为`additional_uncontrolled_hold_all_frames_stable`、正式`checkpoint`为`None`、源33石SHA256仍为`d9567c8d6b8036466cc9ec34574802b562193b0355f667413629b47e229ed38a`。
- 审核只读，不重跑、不删除recovery、不改物理参数或保持门；结论必须明确recovery仅为失败诊断，不能晋级为正式47石输入。

## 资格层级

1. 几何/资源门：资产、SAT间隙、粒子划分、路径和SHA绑定。
2. 构造稳定门：释放窗、额外保持、有限值、边界、接触和动量账本。
3. 检查点门：完整schema-v2写出及来源身份。
4. 独立恢复门：另一张GPU只读恢复、一步连续性和120帧严格保持。
5. 完整场景预接触门：加入开口铲斗代理后验证所有权、净空和零命令稳定。
6. 挖掘门：由S03和S04分别验证APPROACH与真实接触。

不能把前一级结果写成后一级已经通过。

## 验收标准

- 源检查点前后SHA、大小和身份不变。
- 粒子字段、刚体状态、固定网格和warm-start按合同恢复。
- 第一步位置连续且稳定，短窗全部有限。
- 无禁止接触、隐式外部刚性支撑、未映射collider或动量重复记账。
- 新场景不能只凭“最后重新稳定”掩盖首步冷启动或恢复不连续。
- 检查点、报告、配置和运行器SHA相互绑定。

## 不做什么

- 不为视觉密度擅自降低稳定门或移动正式石块。
- 不把单粒子稀疏瞬态规则推广到独立审计的严格保持门。
- 不在A30平台直接热改正式源码；修复回到本地形成新不可变包。

## 支撑材料

- [混合场景专项历史设计](../v4/mixed-material-scene-plan.md)
- [连续三铲详细实施来源](../v4/mixed-material-three-cycle-integration-next-section.md)
- [当前状态](../CURRENT_STATUS.md)
- `docs/decisions.md`中的D-082至D-097

## D-139 连续采集恢复（2026-09-24）

新增cycle全状态合同，包含原生时间/warmstart/加速度/控制输出体速度、新助力及防漏待反力。60连续与30+保存重开+30全24数组、控制、防漏全等；第二铲最终预备点通过独立30帧零接触保持，真实手操3帧与独立同3帧全数组/SHA相同。旧诊断历史缺失边界不消除。故障快照禁止续接，预算/探针不标正常人工终点，审计绑定NPZ和元数据身份及关键物理源码。见[证据](../../work/three-scoop-continuation-20260924-v1/README.md)。


## D-142：零硬化历史溢出与续跑策略（2026-09-24）

本地固定hardening=0的Jp不参与屈服缩放，长时累乘却溢出。新增显式`zero_hardening_frozen_jp_v1`只保留有限正源Jp，不改粒子、应力、弹性、接触或材料强度。该标量不再记录后续塑性历史；非零/混合硬化、坏源或未知策略拒绝。原源点与原失败不改写。

源保持命令显式启用后，cycle元数据记录策略与实现SHA；runtime/人工续铲继承并每步检查适用配置。旧无策略点保留旧行为，不据此宣称旧全链资格。原故障窗993帧通过；同前缀其他23数组全等，部署后6与3+重开+3全24数组一致。完整卸料与A30仍需各自证据。见[验证报告](../../work/passive-jp-overflow-20260924-v1/README.md)。


## D-143 快速接触采集分支（2026-09-24）

训练初态增加显式training_reset标记和库存稳定ID账本；默认连续加载器拒绝。余料位置/速度/材料逐项保留，机构控制及改变几何后的求解缓存重建。短保持与新进程恢复分开验收，不把初始化写成精确原状态续跑。

实现与验收进度以 `docs/CURRENT_STATUS.md` 为准。
