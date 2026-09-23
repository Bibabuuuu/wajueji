# 队友GPT下一任务：短举升结果独立验收审计器

更新：2026-09-20。状态：`已授权实施`。本任务只做离线、只读的验收审计，不运行仿真，不修改轨迹、控制、场景、碰撞代理或物理参数。

## 背景与目标

主线两场受控进斗已经产生短举升结果，但首版判定暴露了身份连续性漏洞：进斗末端原本完整入斗的石块没有随斗抬升，举升过程中后来进入斗包络的另一块石头却被误计为成功携带。修正版按“同一颗石头”收紧后，两场均为`short_lift_hold_needs_review`。

本任务建立独立审计器，验证报告内部的身份集合、阈值、哈希和范围声明，防止执行器再次把“后来进入包络”或“只在地面上被斗包住”写成有效携带。审计器验证的是报告契约，不替代逐步轨迹数据，也不授予物理、满斗率或训练资格。

## 冻结输入（只读）

输入位于主工作区`/home/qunwang/wajueji`：

- 配置：`configs/v4/mixed-material-r2-intake-lift-v1.json`，SHA-256 `4093765f2d01ca3578376481659e38e245f308ec7e53485b538230ca6858bef3`
- 旧版反例：`reports/v4/mixed-material-r2-intake-lift-2026-09-20-v1/report.json`，SHA-256 `74b7a791348d1cbbe6dc847bb8de08ef3653de69e6ef25218aad103b5ff7c07e`
- 修正版正式结果：`reports/v4/mixed-material-r2-intake-lift-2026-09-20-v2/report.json`，SHA-256 `2fff012018fe13e962aec37a01b35e65f2a7d63350a35f4d71856bd4d113af92`
- 修正版执行器：`src/v4/run_mixed_material_r2_intake_lift.py`，SHA-256 `81851a575a7551e59a5dbf6fdb284eb2eb1cba4da5aa2decf4910586d5b8afe8`

旧版只作为必须被识别的反例，不得把它重新标成通过。正式审计对象是v2根报告及其两个场景报告。

## 允许修改

只允许新增或修改以下文件：

- `configs/v4/mixed-material-r2-intake-lift-audit-v1.json`
- `src/v4/audit_mixed_material_r2_intake_lift.py`
- `tests/v4/test_mixed_material_r2_intake_lift_audit.py`
- `reports/v4/mixed-material-r2-intake-lift-audit-2026-09-20-v1/`
- 本文末尾的实施状态

不得修改主线输入、现有进斗/举升执行器、R0/R1/R2桥、MJCF、稳定化模型、场景清单、轨迹、阈值或物理参数。不得运行昂贵仿真。

## 必须审计的契约

对根报告和每个场景报告执行以下检查：

1. 根报告绑定的配置、受控进斗配置/报告、桥源和执行器哈希存在且与声明一致；两个场景报告的实际哈希与根报告逐场声明一致。
2. 场景集合恰好是普通小石主导`26091912`与稀疏大石`26091925`，不得静默增删或替换。
3. `lifted_carried_from_intake_end_full_rock_ids`必须同时是`intake_end_full_rock_ids`与`final_retained_full_rock_ids`的子集，不能消费举升途中才进入包络的ID。
4. 每个宣称成功携带的ID，其`vertical_lift_from_intake_end_m_by_rock`必须达到配置阈值`0.03m`；成功数量必须达到配置最小值。
5. 完整日程、20个身份、有限状态、禁止接触、闭合量、末速度和最终连续留斗检查必须全部显式存在；缺字段不得按默认通过。力/力矩只输出诊断值，不设容量门槛。
6. 根状态必须由逐场结果推导。当前v2两场的预期审计结论是“报告内部一致，但携带门失败/需要复核”，不是审计器自身失败。
7. `physics_qualified`、10秒保持、六场资格、卸料、MPM、满斗率和训练必须全部保持false；审计器不得扩大结论。

审计报告必须区分：

- `contract_valid`：文件、哈希、字段和集合逻辑一致；
- `acceptance_passed`：物理判据是否通过；
- 根`status`：当前正式v2应为`intake_lift_contract_valid_acceptance_failed`。

不要把“审计器正确识别失败”写成“进斗/携带通过”。

## 必测攻击用例

至少覆盖：

1. 把举升途中才进入的ID放入成功集合，必须拒绝。
2. 成功ID不在进斗末端集合、最终留斗集合或抬升字典中，必须拒绝。
3. 抬升量低于`0.03m`、NaN/Infinity、数字字符串或布尔值，必须拒绝。
4. 根/场景状态与实际检查不一致，必须拒绝。
5. 篡改场景报告而不更新根哈希，必须拒绝。
6. 删除任一必需scope false声明或将其改为true，必须拒绝。
7. 旧版v1反例不能得到`acceptance_passed=true`；若旧schema缺少可靠的同一石头集合，结果应是`legacy_contract_insufficient`，不能猜测补全。
8. 正式v2应得到`contract_valid=true`、`acceptance_passed=false`，且失败原因精确指向两场的`same_intake_end_rock_retained_and_lifted=false`。

## 输出与验证

正式输出至少包括根`report.json`、两场审计结果和`README.md`，记录输入绝对路径与SHA-256、逐项检查、失败原因、边界说明和复现命令。相同输入连续生成两次必须逐字节一致。

运行专项测试，并把测试数量、两次逐字节比较和`git diff --check`结果写入本文。目标文件未跟踪时，还要执行等价的行尾/尾随空白检查。

## 完成标准

只有在审计器能拒绝旧版身份洗牌、正确确认v2契约有效但携带门失败、所有攻击测试通过且输出可复现后，本节才可标记`已验证`。本任务完成后，主线再根据失败原因设计下一条低成本轨迹；队友不得自行改轨迹。

## 实施状态

状态：`待实施`。
