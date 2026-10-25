# 键盘/USB手柄统一四轴输入实施计划

- 状态：`已实现；本地无物理验证；USB硬件待验`
- 负责人：另一账号的AI（独立小任务）
- 权威范围：人工采集前端、设备映射、输入日志和定向单元测试
- 不属于本任务：47/65石物理、检查点、APPROACH、A30运行包、BC/RL训练、实地接触验收
- 最后复核：2026-09-22（Asia/Shanghai）

## 1. 给执行AI的开始指令

只在`/home/qunwang/wajueji`工作，不另行复制仓库。工作区当前很脏，大量未跟踪文件是用户已有成果；禁止`git reset`、`git clean`、`git checkout --`或覆盖无关修改。

开始前必须完整阅读：

1. `AGENTS.md`
2. `docs/PROJECT_PLAN.md`
3. `docs/CURRENT_STATUS.md`
4. `docs/plans/INDEX.md`及其中S01–S07七份`active`方案
5. 本文档
6. `src/v4/mpm_manual_demo.py`
7. `tests/v4/test_mpm_manual_demo_handshake.py`
8. `src/v4/contracts.py`中`ACTION_SCHEMA`和`AXES`

若发现本计划与最新`CURRENT_STATUS.md`或用户新指令冲突，停止扩大修改并回报，不自行改物理或训练路线。

## 2. 目标

在不改变现有四轴动作语义的前提下，让人工采集入口支持：

- 现有键盘：`J/L`回转、`I/K`动臂、`U/O`斗杆、`N/M`铲斗；
- 本地Linux USB手柄，通过现有Pyglet 2.1.16 controller API读取；
- 两种设备最终均输出`[swing, boom, arm, bucket]`的`excavation_four_axis_action_v1`/`normalized_velocity`；
- 记录设备原始值、死区/缩放后动作、现有端点投影后动作、设备身份、墙钟时间和仿真步；
- 无手柄的A30环境仍能使用原键盘路径，默认行为不变。

## 3. 已冻结的接口

### 3.1 动作语义

不得改变`src/v4/contracts.py`的：

```text
schema = excavation_four_axis_action_v1
axis_order = [swing, boom, arm, bucket]
normalized_velocity ∈ [-1, 1]^4
```

游戏手柄默认映射候选：

| 手柄控件 | 四轴 | 默认方向 |
| --- | --- | --- |
| 左摇杆X | `swing` | 右为`+` |
| 左摇杆Y | `boom` | 上为`+` |
| 右摇杆Y | `arm` | 上为`+` |
| 右摇杆X | `bucket` | 右为`+`（卷斗） |

轴方向必须由版本化JSON配置的`sign`显式表达，不得把某一款手柄的正负号隐藏在事件回调中。首次真实手柄连接时只需校核并修订配置，不需改动作处理代码。

### 3.2 死区

每轴使用对称径向标量死区，第一版配置为`0.12`。纯函数定义为：

```text
|x| <= d  -> 0
|x| > d   -> sign(x) * (|x| - d) / (1 - d)
最后clamp到[-1, 1]
```

输入NaN/无穷、死区不在`[0, 0.5)`、缩放或方向非有限值都必须明确拒绝，不能静默改为0。

### 3.3 安全输入

- 键盘`X`继续是“保存并结束”。
- 手柄`back`作为紧急停止：在下一次物理推进前将动作置零，锁存停止，用`manual_emergency_stop`结束并进入现有安全保存路径。
- 手柄断开、窗口失焦或无法读取时立即输出全零；运行中断开必须写事件，不得保持上一非零动作。
- 第一版不用手柄按钮切换阶段、启动训练或改安全阈值；阶段键仍保持1–6，开始/暂停仍保持Space。

## 4. 限定修改范围

执行AI只修改/新增以下文件：

1. 新增`src/v4/manual_action_input.py`
2. 新增`configs/v4/manual-input-v1.json`
3. 新增`tests/v4/test_manual_action_input.py`
4. 最小集成修改`src/v4/mpm_manual_demo.py`
5. 必要时最小修改`tests/v4/test_mpm_manual_demo_handshake.py`
6. 完成后更新本文档的状态/验证结果，并在`docs/acceptance.md`追加一条仅限输入前端的验收记录

禁止修改：

- `run_layered_47_excavation_preflight.py`
- 任何分层石堆、检查点或A30打包脚本
- MPM/MuJoCo桥、物理参数、观测合同、奖励、BC/RL模型
- 旧episode原始数据或已封存holdout

如果任务必须超出上述文件才能继续，先回报原因，不自行扩大范围。

## 5. 详细实现

### 5.1 纯Python输入核心

`manual_action_input.py`不在import时创建窗口、打开设备或import Newton。建议提供：

- `apply_symmetric_deadzone(value, deadzone) -> float`
- `load_manual_input_config(path) -> dict`，严格验证schema、键集、轴顺序、sign、deadzone和emergency button
- `KeyboardFourAxisState`，保存当前按下的八个机构键并生成旧行为完全一致的四轴动作
- `GamepadFourAxisState`，只接收已标准化的stick事件，保存raw axes，按配置转换四轴动作，处理disconnect和紧急停止锁存
- `ManualInputSample`或等价严格dict构造器，输出第5.4节日志schema

核心模块必须可用伪事件单元测试，不得要求测试机器实际插入USB手柄。

### 5.2 配置

`manual-input-v1.json`至少包含：

```json
{
  "schema": "excavation_manual_input_config_v1",
  "status": "keyboard_and_local_gamepad_candidate",
  "axis_order": ["swing", "boom", "arm", "bucket"],
  "gamepad": {
    "deadzone": 0.12,
    "axes": {
      "swing": {"stick": "leftstick", "component": "x", "sign": 1.0},
      "boom": {"stick": "leftstick", "component": "y", "sign": 1.0},
      "arm": {"stick": "rightstick", "component": "y", "sign": 1.0},
      "bucket": {"stick": "rightstick", "component": "x", "sign": 1.0}
    },
    "emergency_stop_button": "back"
  }
}
```

配置是候选映射而不是“真实手柄已验证”。硬件插入后必须通过无物理干跑核对四轴方向，再把状态升级。

### 5.3 `mpm_manual_demo.py`集成

新增CLI：

```text
--manual-input keyboard|gamepad|auto   # 默认keyboard
--manual-input-config <path>           # gamepad/auto必须绑定
--gamepad-index <int>                  # 默认0
--acquisition-domain <string>          # 例如local-rtx4070-reconstructed-47
```

约束：

- 默认`keyboard`不枚举手柄，不增加硬件依赖，保持A30回退可用。
- `gamepad`模式要求指定index存在并能`open()`，否则在任何物理步之前失败。
- `auto`只在恰好找到一只手柄时选手柄；为0时回退键盘，多于1时拒绝歧义，不默认猜第一只。
- Pyglet controller枚举和`open()`只在窗口/display已建立后执行；使用当前Pyglet 2.1.16的`pyglet.input.get_controllers()`或`ControllerManager`，不增加`pygame`、`evdev`等新依赖。
- 键盘focus逻辑保留；手柄的激活不应被窗口键盘focus错误清零，但窗口失焦时必须因安全要求输出零。
- DAgger人工接管手握必须同时支持“第一个非零键盘动作”和“第一个非零手柄动作”，仍只缓冲一次、仍不推进物理。
- `_select_control_action`不再把所有非DAgger人工动作硬编码为`human_keyboard`；由输入源显式传入`human_keyboard`或`human_gamepad`。保持旧测试可明确传入键盘来源。
- 源哈希清单增加`manual_action_input.py`和使用的配置SHA；不要为此改旧物理源码。

### 5.4 日志

新增`manual-input.ndjson`，每个实际决策步一行，至少包含：

```json
{
  "schema": "excavation_manual_input_sample_v1",
  "episode_id": "...",
  "decision_step": 0,
  "sim_time_s": 0.0,
  "sample_wall_monotonic_s": 0.0,
  "acquisition_domain": "local-rtx4070-reconstructed-47",
  "input_device": "keyboard|gamepad",
  "device_name": null,
  "device_index": null,
  "config_sha256": null,
  "window_focused": true,
  "connected": true,
  "raw_controls": {},
  "normalized_before_projection": [0.0, 0.0, 0.0, 0.0],
  "emergency_stop_latched": false
}
```

`policy-steps.ndjson`现有`raw_action`和`action`保持，并保证`raw_action.normalized_velocity == manual-input.normalized_before_projection`（actor prefix除外）。现有端点投影仍是唯一的应用动作修正点。

顶层`report.json`新增`manual_input`摘要：请求模式、实际模式、设备名/索引、配置SHA、数据域、disconnect数和紧急停止是否触发。不得把“找到手柄”写成“手柄接触演示已验证”。

## 6. 测试顺序

先执行无GPU、无GUI的单元测试：

1. 死区中心、边界、边界外重标定和±1clamp。
2. 四轴顺序、符号、非有限值拒绝。
3. 键盘八键旧映射完全不变。
4. 手柄disconnect立即归零，不保留上一动作。
5. `back`紧急停止锁存，后续轴事件不能恢复非零动作。
6. 配置缺字段、多字段、错axis order、错deadzone和错button都拒绝。
7. DAgger第一非零手柄动作只缓冲一次；普通手柄动作来源为`human_gamepad`。
8. 默认键盘模式不调用controller枚举。

建议命令：

```bash
cd /home/qunwang/wajueji
PYTHONPATH=src/v4 /home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python \
  -m unittest -v \
  tests.v4.test_manual_action_input \
  tests.v4.test_mpm_manual_demo_handshake

/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python \
  -m py_compile src/v4/manual_action_input.py src/v4/mpm_manual_demo.py

python3 scripts/check_docs.py
git diff --check
```

如果已插入手柄，再进行一次“零物理轴监视”人工核对：只显示设备名、raw stick和映射后四轴，不构建MPM、不推进MuJoCo、不写训练episode。未插手柄时不得伪造该验证，只报告单元测试通过和硬件待验。

## 7. 完成标准

以下项全部满足才可把本任务写成`已实现/本地无物理验证`：

- 默认键盘路径语义不变，旧handshake测试通过。
- 手柄映射核心、死区、disconnect、紧急停止和日志schema有单元测试。
- 无新第三方依赖；A30无手柄环境不会因默认keyboard模式失败。
- 手柄和键盘都产生同一`ACTION_SCHEMA`/`AXES`，设备来源与原始控件另行记录。
- 不改物理、检查点、APPROACH、训练或旧episode。
- 所有定向测试、语法、文档检查和diff whitespace检查通过。

此时仍不能宣称“USB手柄已在真实47石接触场景验收”。只有实际手柄无接触干跑、本地47石交互性能门和至少一条试采episode都通过后，才可升级真实硬件资格。

## 8. 交付回报模板

完成后回报：

1. 修改了哪些文件。
2. 最终键盘和手柄映射。
3. 日志新增了哪些字段。
4. 执行的测试命令及结果。
5. 是否实际检测到USB手柄；若没有，明确写“硬件待验”。
6. 任何仍未验证的边界，尤其是真实47石交互和A30键盘回退。

## 9. 实施记录（2026-09-22）

已按本计划白名单完成输入前端实现，未修改47/65石物理、检查点、APPROACH、A30运行包、训练、观测合同或旧episode。新增`src/v4/manual_action_input.py`，其中键盘状态、手柄状态、对称死区、严格JSON配置和单步输入日志构造器均不导入Pyglet、Newton或窗口；新增`configs/v4/manual-input-v1.json`冻结四轴映射和`0.12`死区；新增`tests/v4/test_manual_action_input.py`。

`mpm_manual_demo.py`的默认路径仍为`--manual-input keyboard`，不导入/枚举手柄。`gamepad`和`auto`必须绑定配置，并只在ViewerGL建立display后通过Pyglet 2.1.16 `ControllerManager`枚举和打开设备：`gamepad`要求指定index存在，`auto`仅在恰有一只手柄时启用、零只回退键盘、多只拒绝歧义。运行时将原始控件、死区映射后的投影前动作、设备/配置/数据域、连接和窗口焦点写入`manual-input.ndjson`；断开立即归零并写`manual-input-events.ndjson`，`back`锁存`manual_emergency_stop`并走既有保存路径。普通手柄动作来源为`human_gamepad`；DAgger首个非零动作仍仅缓冲一次。

验证结果：`PYTHONPATH=src/v4 ... -m unittest -v tests.v4.test_manual_action_input tests.v4.test_mpm_manual_demo_handshake`共`13/13`通过；`py_compile`、`python3 scripts/check_docs.py`和`git diff --check`通过。未连接真实USB手柄，未创建窗口、未推进MuJoCo/MPM物理步，故此记录仅授予“输入实现与无硬件单元验证”状态。真实手柄零物理轴监视、本地47石交互吞吐、实际接触episode与A30键盘回退仍待后续独立验证。
