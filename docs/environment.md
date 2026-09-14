## 2026-09-14：连续作业候选与恢复环境

活跃Windows工作树为`C:/Users/14737/.codex/worktrees/55e9/wajueji`。远端通过`desktop-tailscale`连接，候选目录`/home/qunwang/codex-work/autonomous-multiscoop-2026-09-14-v1`；Python固定为`/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python`，不升级运行中依赖。原单铲`autonomous-delivery-2026-09-13`和原compact目录保留。

实测主机为RTX 4070 12GB、i7-13700KF、32GB RAM、CVN Z790M FROZEN D5。当前高保真连续作业使用390400材料点、120Hz机构步、250次MPM迭代，绑定`cuda:0`，显存约0.9—1.0GB；吞吐和升级判据归[硬件决策](hardware-throughput-decision-2026-09-14.md)。运行目录内的`report.json`、`tracking.jsonl`、逐铲观测与路径是恢复状态的依据，旧PID和昨日关机回执不能用于本轮操作。

`geometry-v5`与`empty-2cycle-v4`是当前源码配对；真实接触运行必须携带相同铲次数的已通过空载报告及独立审计，并复核源码哈希。连续序列共用一个土体；独立运行完整新建初态，retention粒子位置/速度快照不含全部材料及控制历史，不能用作物理续跑点。当前进程和唯一下一项由[任务](../.planning/task.md)维护，运行时不改候选源码、不关机、不操作原GUI。

## 2026-09-13 12:04:53 CST交接现查

/proc仅发现连续播放器play_continuous.py PID23032，暂停在41.65833秒，418帧末帧；录制计算已完成，无MPM/bridge/guard/旧静态查看器。用户窗口保持，不自动操作，后续须现查。[现场快照](../evidence/curl-retention-2026-09-13/handoff-runtime-final.json)。新会话必须使用活跃Windows工作树C:/Users/14737/.codex/worktrees/55e9/wajueji，而非注册项目旧目录或新自动cwd。下一项转回成功手操受力跟踪基线。

## 2026-09-13：连续播放器接替静态查看器

远端contact04-continuous保存418个0.1秒实际帧及41.65833秒末帧；补录4999步已结束，4检查点位置/速度逐值一致。contact04-player的play_continuous.py，PID23032，DISPLAY=:1；状态和截图验证播放推进，物理0。旧静态查看器22227已退出。用户可Space暂停、R重播、左右跳2秒、End末帧、C近景/O全景、X退出；2小时上限。不要按旧PID/静态状态操作。

## 2026-09-13：用户请求的失败现场查看器

现查DISPLAY=:1、XAUTHORITY=/run/user/1000/gdm/Xauthority有效。独立autonomous-cycle-2026-09-12/view_saved_failure.py已开启，PID22227，输出contact04-saved-view；原保存帧只读，无Soil/MPM构造、推进0。当前第4帧41.65833秒；1–4切帧、C近景/O全景/X退出，最长2小时。状态与截图见evidence/curl-retention-2026-09-13/view-*。这是用户授权的当前窗口，不自动关停。

## 2026-09-13：contact04失败已停止

独立候选位于autonomous-cycle-2026-09-12/contact04；几何geometry09、空载dynamic04完成。SIGINT在Warp Graph析构被忽略后，对核实本轮runner及直接mpm_bridge子进程SIGTERM，未生成完整final-state；原report的running字段过期，以adjudication.json/final-runtime.json为准。[最终状态](../evidence/curl-retention-2026-09-13/final-runtime.json)确认相关进程为空、基线源码一致。无GUI/正式入口改动。后续需步间协作停止，不能假设SIGINT一定退出。

## 2026-09-13：较浅路径独立试验结束

远端autonomous-cycle-2026-09-12/contact02-diagnosis保存机构精确重放，geometry08/dynamic03/contact03保存较浅路径分层试验。几何、空载通过；contact03收斗失败，未部署正式入口。原环境/源码不变，最终相关进程匹配为空，见[最终状态](../evidence/dynamic-tracking-2026-09-13/shallow-final-runtime.json)。

## 2026-09-13：独立动态试验已结束

远端autonomous-cycle-2026-09-12下dynamic01/02、contact01/02保存分层运行结果，均使用原环境；dynamic02和contact02/sources保存运行时候选源码。contact02最终失败，不能作为正式启动候选。最终/proc检查本轮运行器及桥接进程匹配为空，见[状态](../evidence/dynamic-tracking-2026-09-13/final-runtime.json)。无安装、GUI或正式入口改动，原源码/模型哈希前后一致。

## 2026-09-13：独立几何检查结束

远端autonomous-cycle-2026-09-12/geometry07为本轮最终几何证据；使用原Python、现有pose_guard和原模型，只进行独立姿态查询，没有MPM推进或GUI。最终针对guard/bridge/verify进程检查为空；原compact五个关键源码哈希前后相同。候选planning/cycle_geometry.py、cycle.py与本地工作树哈希一致，正式入口未部署。[证据](../evidence/cycle-geometry-2026-09-13/README.md)。旧进程状态不自动代表后续时刻，执行前仍只读确认。

## 2026-09-12：反力候选试验已结束，原入口保持

新独立远端目录 /home/qunwang/codex-work/soil-reaction-2026-09-12；使用原Newton Python和现有bundle/lib/libmujoco.so.2.3.8、匹配头文件，以g++构建无CA插件机构适配器。无安装升级，无GUI。coupled01全程3544步、half-step01前17秒4080步均完成；最终/proc核对无coupled/mpm_bridge/manual_ready/replay窗口进程，nvidia-smi无计算进程，见evidence/soil-reaction-2026-09-12/final-runtime.json。原模型和compact源码哈希前后一致；当前正式启动入口仍为原compact，不自动替换。原账本冻结。

## 2026-09-12：录屏窗口已退出，账本仅离线

录屏PID41127已由用户正常退出（closed/user_finished，failure空，29.366666667秒）；本轮查询无manual_ready/replay_window/reproduce进程。下方23:05的暂停窗口记录为历史。质量账本仅在Windows当前工作树离线执行NumPy分析与unittest，没有安装依赖、修改原Python/启动器或打开GUI。后续窗口启动前仍须现查，不能自动恢复。
## 2026-09-11 23:05 CST：专用录屏回放

用户授权启动的独立回放目录为/home/qunwang/codex-work/manual-recording-replay-2026-09-11，入口launch_replay.py，原Python、DISPLAY=:1及原XAUTHORITY已成功验证。PID41127/recording-20260911-150442初始paused且0秒；原compact入口未改。窗口实时操作及退出边界见[录屏说明](../evidence/manual-recording-replay-2026-09-11/README.md)。不要自动重开或推进用户录屏窗口。
# 环境记录

**2026-09-11 20:23当前进程状态：** 原compact回退窗口manual-20260911-112551已经closed/window_closed，模拟仍0.066667秒；本轮未操作该窗口。原手操完整复现及10秒保持、分阶段诊断和离屏渲染均已退出，无本轮遗留仿真进程。恢复源码/入口与原Python不变，11项原启动源码哈希再次核对一致。新证据独立保存在/home/qunwang/codex-work/original-manual-audit-2026-09-11。无安装升级/关机/重启；勿按下方历史PID自动开关窗口。见[最终实查](../evidence/original-manual-audit-2026-09-11/final-runtime.json)。

当前恢复版本是 `/home/qunwang/codex-work/compact-bank-2026-09-11`，入口launch_ready.py，使用原MPM Python `/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python`。GUI环境已在本轮验证：`DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority`；跨重启须复核。当前窗口及操作边界以[交接](handoff.md)与[当前任务](../.planning/task.md)为准，勿重开已经运行的窗口。

失败候选 `/home/qunwang/codex-work/soil-separation-2026-09-11` 不作为恢复入口。其Python3.12独立venv通过.pth只读引用原Newton1.5.1/Warp1.17.0/NumPy2.5.3，额外SciPy1.18.1仅装在候选venv，原包未升级。保留用于离线证据/体积分析，不能绕过已撤下的启动资格。

当前无待执行重启、关机或试验；下方历史关机请求已过期，不能据此在新会话关闭任一机器。

<a id="remote-tailscale"></a>
## 当前远程连接：Tailscale + OpenSSH

2026-09-11 已完成安装、同账号登录以及两台均连接 `CMCC-EDU` 校园网 WiFi 时的 SSH 验证。后续 AI 默认从 Windows 笔记本通过此入口进入 Ubuntu 台式机。

| 项目 | 已确认配置 |
| --- | --- |
| 笔记本 | Windows `GGG_bond`，Tailscale IP `100.67.199.121` |
| 台式机 | Ubuntu `qunwang`，Tailscale IP `100.104.193.16`，SSH 用户 `qunwang` |
| Tailscale | 两端版本 `1.102.4`，同一 `Bibabuuuu` GitHub 账号；服务已设为开机启动 |
| Windows 安装目录 | `D:/tailscale`；命令行 `D:/tailscale/tailscale.exe` |
| SSH 快捷入口 | `ssh desktop-tailscale` |
| 本地 SSH 配置 | `C:/Users/14737/.ssh/config` |
| 现有身份文件 | `C:/Users/14737/.ssh/codex_4060`；只引用路径，不复制或输出私钥内容 |

在 Windows PowerShell 中检查并连接：

```powershell
& 'D:\tailscale\tailscale.exe' status
& 'D:\tailscale\tailscale.exe' ping --c=3 --until-direct=false --timeout=5s 100.104.193.16
ssh -o BatchMode=yes -o ConnectTimeout=12 desktop-tailscale 'hostname; whoami'
```

最后一条应返回两行 `qunwang`。交互终端使用 `ssh desktop-tailscale`；传文件可使用 `scp <本地文件> desktop-tailscale:/home/qunwang/<目标路径>`。这些命令仍使用原 OpenSSH 和密钥认证，没有启用 Tailscale 自带的 SSH 服务。

已安装的 SSH 别名配置：

```sshconfig
Host desktop-tailscale
  HostName 100.104.193.16
  User qunwang
  IdentityFile C:/Users/14737/.ssh/codex_4060
  IdentitiesOnly yes
  HostKeyAlias 10.182.237.236
  StrictHostKeyChecking yes
  ServerAliveInterval 30
  ServerAliveCountMax 3
```

`HostName` 才是连接目标；`HostKeyAlias 10.182.237.236` 只沿用旧地址对应的可信主机记录，不会连接该热点 IP。台式机 ED25519 指纹为 `SHA256:cxd/pzF3sRnpmkrsExw8cARbAF1HkEXf8Y3QmZka2lU`，已通过原可信 SSH 通道读取并在 Tailscale 通道核对一致。遇到主机密钥变化先核实原因，不关闭主机密钥校验。

验证证据：两端网络名均为 `CMCC-EDU`；Tailscale ping 三次成功，延迟 `7/56/23 ms`；SSH 返回 `hostname=qunwang`、`whoami=qunwang`，`SSH_CONNECTION` 两端为 `100.67.199.121` 和 `100.104.193.16`。切网初期曾显示东京中继，随后建立直连；直连/中继可随网络变化，不把一次延迟当作长期保证。已验证热点和校园网两种场景，尚未单独验证异网长期稳定性。

排障顺序：确认台式机开机、未休眠且校园网认证完成 → 检查 Windows `Get-Service Tailscale` 和两端 `tailscale status` → 检查设备在线及 IP → Tailscale ping → SSH。能 ping 但 SSH 失败时检查台式机原 OpenSSH 服务、认证与防火墙；不要仅因旧热点 IP 不通就判定台式机离线。台式机本地可运行 `systemctl status tailscaled`、`tailscale status`、`systemctl status ssh`。若 `NeedsLogin`，需完成设备授权；只有实际再次出现 `nodekey already exists` 才考虑显式 logout 后重新登录，不对正常在线设备例行退出或重装。

两端使用 `--accept-dns=false` 保留系统 DNS，没有启用出口节点或子网路由；Ubuntu 用户 `qunwang` 已设为该机 Tailscale operator。安装日志和独立说明在 `D:/tailscale/installer/README.md`、同目录 `install-windows.log` 及台式机 `/home/qunwang/tailscale-setup/install.log`。

下方旧阶段环境快照里的热点地址属于历史；当前远程连接以本节为准。仿真目录、Python 和 GUI 操作仍按最新交接确认，不因连通网络而自动关闭、重置或重启用户窗口。

## 2026-09-11 15:09 保形土台候选

入口 `/home/qunwang/codex-work/soil-shape-2026-09-11/launch_ready.py`，窗口 STABLE SOIL BANK，原Python/显示环境不变；无依赖升级。抗剪8000Pa、原抗拉.0025、全土动态、附加运动扫掠关闭；连续表面可按T切点。新人工轮manual-20260911-070934/PID12775，15:11实查运行8.8167秒/failure空；不关闭或重置用户窗口。原整堆窗口及用户暂停的smoke-20260911-065850未强行关闭；后者仅短测预算6秒，不当作正式手操入口。原v20未改，无待执行关机。完整说明见[记录](../evidence/soil-shape-2026-09-11/README.md)。

## 2026-09-11 14:21 当前整堆手操

新入口 `/home/qunwang/codex-work/bank-full-manual-2026-09-11/launch_manual.py`，沿用原MPM Python、DISPLAY=:1和原XAUTHORITY。新人工轮manual-20260911-062129/PID6226默认暂停，后续须现查。旧局部人工轮manual-20260911-014434已到一小时墙钟预算正常退出并下载冻结，未强制中断；无关机操作。原v20入口未变。详见[操作/证据](../evidence/bank-full-manual-2026-09-11/README.md)。

## 2026-09-11 09:44 当前状态

用户已亲自正常重启Ubuntu并登录。09:42 SSH核实GPU为RTX4070，NVIDIA驱动580.173.02，原驱动/库错配和CUDA804问题已恢复；DISPLAY=:1及原XAUTHORITY路径有效。不存在仍等待执行的旧关机任务。

新独立手操目录 `/home/qunwang/codex-work/bank-manual-2026-09-11`，入口 `launch_manual.py`，沿用原MPM Python；原v20发布不变。人工轮 `manual-20260911-014434` / PID4070，启动默认暂停，须以实时status核对。按X正常保存退出。相关代码、烟测与操作见[记录](../evidence/bank-manual-2026-09-11/README.md)。

## 当前恢复入口与关机状态（2026-09-11）

- Windows工作树：`C:/Users/14737/.codex/worktrees/55e9/wajueji`；PowerShell，本地分析Python为`D:/Python3.11/python.exe`。
- Ubuntu工作站：当前使用 `ssh desktop-tailscale`（`qunwang@100.104.193.16`），配置见[远程连接](#remote-tailscale)；`10.182.237.236` 为历史热点地址。RTX4070 12GB、i7-13700KF，Newton1.5.1/Warp1.17.0。
- 原MPM Python：`/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python`。原运行时依赖在`small-soil-20260910-v17-performance`，不要改这些来源文件来跑候选。
- 最新独立证据根：`/home/qunwang/codex-work/soil-locality-20260910`（10组支承/强度小试验，91文件清单）；`/home/qunwang/codex-work/bucket-bank-20260910`（真实铲斗失败诊断，148文件清单）。两批归档均已下载Windows对应evidence目录并校验，包含source-verification.json及archive-manifest.json。
- 00:05左右本轮查验：没有本批probe、matrix、plan或pose_guard进程，也没有manual_soil_viewer；不要按历史PID操作。原v20发布源码和桌面入口哈希核验保持，未开新GUI，无新增依赖。
- **历史关机请求，已失效；当前无待执行关机。** 机器继续用于用户手操，不因本旧记录关闭任何机器。

恢复时先开目标工作站，验证SSH、GPU和上述Python，再读docs/handoff.md与.planning/task.md。启动原桌面v20只按原启动器，不自动恢复失败候选：`/home/qunwang/codex-work/small-soil-20260910-v20-perception/launch_ready.py`。GUI原环境DISPLAY=:1、XAUTHORITY=/run/user/1000/gdm/Xauthority需重启后重新确认，不保证跨重启仍相同。新试验必须用新输出目录；terrain/moving_sweep.py为失败实验，不能默认开启。

## 当前土方原型目录（2026-09-10）

`/home/qunwang/codex-work/soil-bank-prototype-20260910`，原MPM Python，Newton1.5.1/Warp1.17.0；无新依赖，独立推板/几何试验已退出，无新增GUI。安装源码、原静态碰撞与材料来源哈希见[核验](../evidence/soil-bank-prototype-2026-09-10/source-verification.json)。没有发布混合土体后端，桌面仍v20；本轮进程核查未发现原inspection窗口。复现必须新目录，保留full-bank零步失败用于审计。

## 当前用户GUI检查（2026-09-10 22:10启动）

在下方auto-approach目录启动inspect_viewer.py --run inspection-20260910-221000，PID44893；同原DISPLAY=:1和XAUTHORITY环境。启动初始暂停/感知有效，3600墙钟秒预算，X/窗口关闭正常退出。状态文件是该run/status.json；这是启动快照，不保证未来仍运行。操作与视觉验收边界见[检查说明](../evidence/auto-approach-inspection-2026-09-10/README.md)。

## 当前独立自动接近试验目录（2026-09-10）

`/home/qunwang/codex-work/auto-approach-20260910-v1`保存执行适配器、55项测试、nominal/dropout两轮MPM与来源哈希；使用原MPM Python，运行时复用v17及其原依赖，无新增安装。两轮与native guard已退出，桌面仍v20，原人工输出保留。每次case必须使用新目录，不能覆盖本批原始证据。复现和当前边界见[执行记录](../evidence/auto-approach-2026-09-10/README.md)。

## 当前准备路径/门控验证目录（2026-09-10）

`/home/qunwang/codex-work/approach-gate-20260910`保存最新planning模块、路径和47项测试；沿用原MPM Python，无新依赖，测试已结束。v20源码与桌面不变，人工轮manual-20260910-122337已到120模拟秒正常退出，当前没有该人工窗口。核实原GUI预算为120模拟秒/3600墙钟秒先到者，原先仅写一小时不完整。未重启或重置。接口和下一步见[接近控制](approach-control.md)。下方运行状态属于此前阶段。

## 当前连续验证/可达性运行目录（2026-09-10）

`/home/qunwang/codex-work/phase-dynamic-reach-20260910`保存最新planning模块、连续数据、逆解和35项测试；沿用MPM Python，无新增依赖。测试进程已结束，v20桌面与发布源码保持不变，原人工窗口未重置。旧`phase-observer-20260910-v1`为前一批历史版本。运行范围见[连续验证](../evidence/phase-dynamic-reach-2026-09-10/README.md)。

## 只读阶段判断旁路（2026-09-10）

`/home/qunwang/codex-work/phase-observer-20260910-v1`，使用既有MPM Python与NumPy，无新增依赖。`python -m planning.watch`读取同机v20的latest.json，不导入/启动仿真器。20秒验证进程已结束，退出时输出无效状态；桌面仍v20，原人工窗口保留。启动命令与边界见[阶段判断](phase-observer-v1.md)。

## 当前在线感知v20（2026-09-10）

- 当前桌面“挖掘机土壤测试 v20”指向`/home/qunwang/codex-work/small-soil-20260910-v20-perception/launch_ready.py`。沿用原MPM Python/NumPy/Warp1.17.0，无新增安装，物理/材料/几何哈希与v17一致。
- v20新增`perception/`和GUI只读观察器，默认两视角、墙钟0.2秒采样、0.3秒开发有效期，错过采样不追赶。启动器核验`release-v20.json`源码哈希。
- 新轮`manual-20260910-122337`、PID40129，启动核验时暂停/模拟0秒/感知有效；一小时预算，实时状态以目标目录`status.json`为准。
- v17保留，v20内`desktop-before-v20.desktop`为桌面回退备份。土壤v19对照继续停止。实现、接口与验证边界见[在线感知](perception-online-v2.md)。

## 感知离线第一版（历史，2026-09-10）

- 目标目录`/home/qunwang/codex-work/perception-20260910-v1`，本地源码`perception/`、测试`tests/`；沿用下方MPM虚拟环境的NumPy/Warp1.17.0，无新增安装。
- 回放适配器只读`small-soil-20260910-v17-performance`的CAD、运动学、cohesive-fine01静置状态及replay-full1快照，车厢模型仍来自`manual-comparison-20260908-v1`。不构造Soil或推进物理，输出必须新目录。
- 入口：`python -m perception.replay_newton`，完整命令和接口见[感知说明](perception-v1.md)。当前没有感知后台服务或新桌面窗口，桌面仍v17；v19土壤对照按用户要求停止。

## Newton/Warp MPM 高仿真环境（v17/v18历史实查，当前入口以上方v20为准）

- RTX4070（12282MiB显存）/i7-13700KF；当前GS主求解已使用CUDA Graph。v18候选`/home/qunwang/codex-work/small-soil-20260910-v18-pile`动态验收失败，禁止作为已验证入口；其launcher会按候选判定拒绝启动。新`bench-settled`仅用于诊断，桌面保持v17。人工轮manual-20260910-070118已关闭并标弃用，当前无遗留测试窗口。见[v18记录](../evidence/high-fidelity-v18-pile-profile-2026-09-10/README.md)。

- 当前桌面已切性能版`/home/qunwang/codex-work/small-soil-20260910-v17-performance/launch_ready.py`，图标名“挖掘机土壤测试 v17”。保持120Hz/250迭代/原材料/连续碰撞，仅优化GPU统计及日志；场景、guard-runtime、静置材料继续复用原依赖。v16及独立对照目录`small-soil-20260910-v17-baseline`保留。桌面回退备份为v17目录`desktop-before-v17.desktop`。详见 [v17记录](../evidence/high-fidelity-v17-performance-2026-09-10/README.md)，实时轮次见当前任务。

- v16回退目录为 `/home/qunwang/codex-work/small-soil-20260910-v16-collision`；其人工轮`manual-20260910-053413`已按原3600秒预算正常关闭于59.5333秒。使用cohesive-fine01、60Hz操作帧×2物理步、250迭代、静态原网格连续碰撞。强黏聚fine02已拒绝，不作入口。

- 台式机 `qunwang@10.182.237.236`；本地 SSH 密钥路径 `C:/Users/14737/.ssh/codex_4060`。
- Python：`/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python`；GPU `cuda:0`。
- 保留v15-monitor、v14-ground、v13-boundary、v12-fast60和v11-candidate05；场景仍依赖v13的candidate06，静置材料和guard-runtime仍依赖v7，不删除旧目录。
- 已安装版本为 Newton 1.5.1 / Warp 1.17.0；源码和运行文件哈希见 v14 的 `release-manifest.json`。
- GUI 启动环境：`DISPLAY=:1`，`XAUTHORITY=/run/user/1000/gdm/Xauthority`。入口为当前目录的 `launch_ready.py`，桌面图标已更新。
- 原生 guard 依赖 v13 的 `guard-runtime/lib`；其 MuJoCo 库仅用于零步机构检查，不修改 CA 独立工程。编译诊断头文件位于 `/home/qunwang/codex-work/excavator-workspace-20260907/bundle/mujoco-custom/include`。
- 台式机 MPM 虚拟环境缺少 SciPy/Matplotlib，网格分析与绘图在本地 `D:/Python3.11/python.exe` 完成；未为此改动台式机依赖。

下方为继承的模块 1 虚拟机环境。

## 已验证运行环境

- 主开发环境：Ubuntu 24.04，VMware 虚拟机。
- 编译工具：Clang 18.1.3、CMake 3.28、Python 3.12.3。
- 系统依赖：`build-essential`、`cmake`、`clang`、`git`、`python3`、`python3-jinja2`、`libglfw3-dev`、`libgl1-mesa-dev`、`libxinerama-dev`、`libxcursor-dev`、`libxi-dev`、`libgoogle-glog-dev`、`libgtest-dev`、`libbenchmark-dev`、`unzip`。

## 源码与依赖版本

- 上游仿真器：`KennyVilella/excavator_simulator_mujoco`，基线提交 `4251c86`；项目修改位于独立分支 `wajueji/module1-hold-control`，当前提交 `7ee616b`。
- `soil_dynamics_cpp`：`v1.1.0`。
- 定制 MuJoCo：`KennyVilella/mujoco` 提交 `6463ff7932ff5dbb81de8fba843b65a97c58e2fe`。
- 本地导入的 MuJoCo 核心依赖：lodepng `b4ed2cd7ecf61d29076169b49199371456d4f90b`、MarchingCubeCpp `5b79e5d6bded086a0abe276a4b5a69fc17ae9bf1`、qhull `0c8fc90d2037588024d9964515c1e684f6007ecc`、tinyxml2 `9a89766acc42ddfa9e7133c7d81a5bda108a0ade`、tinyobjloader `1421a10d6ed9742f5b2c1766d22faa6cfbc56248`、libccd `7931e764a19ef6b21b443376c699bbc9c6d4fba8`。

## 可复现构建约束

虚拟机到 GitHub 的 HTTPS 连接超时。构建前须通过 VMware 共享目录导入 `mujoco-custom` 和 `mujoco-core-deps` 源码，再向 CMake 提供对应的 `FETCHCONTENT_SOURCE_DIR_*` 路径。禁止将构建目录、压缩包或二进制提交到项目 Git。

## 当前启动入口

```bash
./build-offline-mujoco/bin/excavator_simulator ./model/excavator/excavator.xml
```

在模块 1 分支已构建的环境中，改动 C++ 源码后使用：

```bash
cmake --build build-offline-mujoco --target excavator_simulator -j2
cp build-offline-mujoco/excavator_simulator/excavator_simulator build-offline-mujoco/bin/excavator_simulator
```

`logs/module1_telemetry.csv` 是运行产物；每次 Reset 会覆盖它，保留试验前先复制或改名。
