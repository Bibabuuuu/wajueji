# 队友GPT下一任务：官方数据来源门禁加固

更新：2026-09-20。状态：`已授权返工`。本任务只修复官方可见数据预检查中的来源门禁和比例类型校验，不修改任何场景、R2、高度场、桥或物理。

## 复核结论

上一节的空模板正式输出、可复现性和现有测试均通过，但主线复核发现两项阻断验收的问题：

1. `synthetic_assumption`来源的非空测量目前会触发`official_visible_measurements_candidate`，并进入`official_visible_*`区间。已复现纯合成输入输出`official_visible_diameter_interval_d_over_bucket=[0.14, 0.34]`。这会把合成假设误标成官方候选。
2. `angularity_class_proportions`在转为`float`前没有拒绝字符串和布尔值；`"0.1"/"0.3"/"0.6"`以及`true/0/0`可绕过数值类型契约。

上一节暂不标记为最终验收通过，先完成本任务。

## 允许修改

- `configs/v4/official-visible-material-summary-template-v1.json`（仅在契约说明确有必要时）
- `src/v4/build_official_scene_calibration_preflight.py`
- `tests/v4/test_official_scene_calibration_preflight.py`
- `reports/v4/official-scene-calibration-preflight-2026-09-20-v1/`（修复后重生成）
- `docs/v4/official-data-calibration-preflight-next-section.md`
- 本文末尾的实施状态

其余文件只读。尤其不得修改`scene-reconstruction*.json`、R0/R1/R2配置、MJCF、桥、物理参数或现有重建/稳定化报告。

## 必须实现的来源门禁

可信官方可见来源集合固定为：

- `observed`
- `human_reviewed_estimate`

以下来源只允许保留在台账或差距说明中，绝不能进入任何`official_visible_*`候选字段：

- `synthetic_assumption`
- `unknown`

候选字段的单项测量必须同时满足：

1. 对应`source_catalog`条目的`review_status`属于可信集合；
2. 该测量自己的`source_status`属于可信集合；
3. 值完整、非空并通过现有数值、单位和不确定度校验。

`absolute_scale`还必须让自身`source_status`满足同一可信门禁；不能因为同一 observation 中另一个字段可信，就把合成尺度用于米制换算。

根状态必须由“至少存在一项通过上述门禁的已知测量”决定：有则为`official_visible_measurements_candidate`，否则为`official_visible_measurements_pending`。不得再以`observations`非空作为候选状态依据。混合来源 observation 可以保留，但每个输出字段只消费自身符合条件的测量。

输出应为每个非空候选区间保留最少但充分的 provenance，至少能追溯`source_id`、`summary_id`和该字段的`source_status`；被忽略的合成/未知值应有可审计计数或说明，但不得复制成官方候选。

## 类型校验

`angularity_class_proportions.values`的三个值在任何`float(...)`转换前都必须逐项验证：

- 只接受`int`或`float`；
- 明确拒绝`bool`；
- 拒绝数字字符串和其他类型；
- 随后继续执行有限值、`[0,1]`范围和总和闭合检查。

## 必测用例

至少新增以下回归：

1. 纯`synthetic_assumption` source + 纯合成测量：输入合法，但根状态保持`pending`，所有`official_visible_*`为`null`，不产生伪候选。
2. `unknown` source + 非空测量：不得进入官方候选；若与现有schema冲突，应给出明确验证错误或保持`pending`，但不能洗白。
3. 混合可信/合成字段：只有可信字段进入对应候选；合成粒径不能因可信覆盖率存在而进入粒径候选，反之亦然。
4. 测量标`observed`但其`source_catalog.review_status`为`synthetic_assumption`：不得进入候选。
5. 可信相对粒径配合合成`absolute_scale`：允许无量纲候选，米制候选必须为`null`。
6. 棱角比例三个数字字符串被拒绝。
7. 棱角比例包含布尔值被拒绝。
8. 原空模板、可信 populated fixture、哈希绑定、逐字节可复现、单位/分位数/不确定度和逐颗缩放拒绝测试继续通过。

## 验收命令与证据

- 运行官方数据专项测试，并记录通过数量。
- 运行现有R0、场景重建、装配和官方数据联合回归，确认无回归。
- 用修复后的生成器重生成正式输出；连续两次临时生成必须与正式4个文件逐字节一致。
- 运行`git diff --check`；由于目标文件可能尚未跟踪，还要对目标文件执行等价的空白检查。
- 在上一节文档中把旧的`已验证`更正为`复核未通过，已完成门禁返工`或在全部验收通过后写成`已验证（来源门禁加固后）`，并更新正式证据哈希。

## 完成标准

只有在纯合成输入无法产生任何`official_visible_*`候选、混合来源按字段隔离、比例类型绕过被阻断且全部回归通过后，本节才可标记`已验证`。本任务仍然只产生校准前检查与建议，不授权修改场景参数，也不代表获得了可用于训练的官方真值。

## 实施状态

状态：`待实施`。
