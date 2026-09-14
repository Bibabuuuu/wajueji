# 完整自主单铲装车返回证据

2026-09-13T13:11:15.336367+00:00。原物性、真实机构控制与原生MPM反力下，完整11段动作和保持流程通过：16190步/134.9167秒，墙钟2088.99秒。最终车厢连续低速留存净增193.2242kg，其中44.3594kg追溯到运输前内腔群体；斗内残余9.3155kg，单独记录。

## 当前资格与用户口径

用户明确允许少量斗内残余。重新读取命题原图和公开赛事概要，未见必须卸净、残余质量上限或独立扣分条款，废止原型8kg返回门槛。不能把未提供完整评分细则理解为任何残余均不影响效率，也不把残余成因未经试验归结为姿态或土—钢黏附。

当前开发检查采用20kg获取、稳定净接收至少10kg且至少10kg来自运输前内腔，加实际到位/低速确认与安全返回；这些数值不是官方规定土方量。残余不再阻断返回，不删土、不算成车载；下一步连续铲次观察容量与效率影响。额外DUMP_TILT并非当前默认动作。

稳定接收定义为固定ID连续5秒留在实际CAD核心区、速度≤0.05m/s、观测间隔≤0.250001秒；缺测、离开或速度超界重新计时。斗内归属优先排除，边界带和上沿上方未确认土不计入。净增扣除初始库存未留存部分，保留负值。卸料后观测10.0000秒，稳定净增191.0845kg；返回起姿后再观测10.0000秒。这是采样低速留存证据，不是已标定支撑、体积或车容。

## 各版本与证据

- geometry05 / empty04 / contact03 是最终同源组合。empty04共16190步/134.9167秒，17项独立审计通过。
- geometry01几何通过后补齐4Hz采样与携带身份检查；原运行模块在geometry01/source-runtime。
- contact01在70.7秒/8484步因两点越过旧y=4m采集区诊断停止。点速度约0.238/0.249m/s，位于车厢上方和当前斗体范围；真实车体y上界6.3825m。version01和stop-diagnosis.json保留；8项停止证据审计通过，但未完成循环。
- 完整工作域改为原采集XY区与真实车体XY包围盒的并集，保留原-.05/5m高度界；原Soil诊断仍记录，不改接触、物性、机构或账本。最终完整工作域最大越界0；旧采集域最大计数570，两种语义明确分开。14项门控/留存/工作域测试通过，车体三角面与分类源逐点误差0。
- geometry03 / empty02采用完整域；empty01/02全部16190步机构状态、参考和命令逐值相同。contact02卸后观察12.8583秒，稳定接收191.0845kg、残余13.9341kg，因当时的8kg内部门槛主动停止于86.8833秒；version02和6项部分结果审计保留。现在不把这份残余算比赛失败，但未执行返回，仍不是完整循环。
- geometry04 / empty03增加-0.85rad翻斗，几何和空载完成；version03保留。用户澄清后取消强制清残土，没有启动该版本MPM。更深姿态未通过原限位/可达检查的候选全部保留。
- contact03/report.json为实际运行记录；tracking.jsonl逐物理步，soil.jsonl与scene-envelope.jsonl每12步诊断；retention.jsonl和receipt.jsonl是只读固定身份观测。
- analysis.json独立重算机构/时序/限位/阶段/计数；receipt-analysis.json重算连续留存、净增、标记快照CAD分类和最终入车粒子在运输前的位置。总装入包括内腔之外携带的物料，不能用内腔质量直接作总装载分母。
- 复用Soil生成的soil-config.json中coupling说明仍是旧构造器描述；本次实际反力接线以run.py、逐步tracking.jsonl和analysis.json的力系延迟/驱动端方程审计为准，不能据该旧字符串断言没有力反馈。
- retention-*.npz保存完整390400粒子的q/v、质量、斗姿和内腔ID，含initial、hold-start/end、dump-arrival、unloaded、return-arrival、returned、final。这些不是完整MPM可重启状态。
- acquisition-prefix-comparison.json比较前6145步：参考样本一致，真实机构/反力存在小差异，不宣称确定性重复。source-verification.json、final-runtime.json、archive-manifest.json保存源核对、进程退出与归档。

## 复现与停止

SSH desktop-tailscale；原根/home/qunwang/codex-work及原Python环境，依赖compact、v17、autonomous-cycle、制动和土体修复目录，不升级依赖。输出目录必须使用全新名字。

```sh
cd /home/qunwang/codex-work/autonomous-delivery-2026-09-13
PY=/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python
$PY preflight.py
$PY geometry.py --run geometry-new
$PY run.py --run empty-new --geometry geometry-new --mode empty --max-wall 600
$PY audit.py empty-new
$PY run.py --run contact-new --geometry geometry-new --mode contact --empty-report empty-new/report.json --max-wall 3600
$PY audit.py contact-new
$PY tools/audit_delivery_receipt.py "$PWD" "$PWD/contact-new"
```

用`touch contact-new/STOP`逐步协作退出，不能覆盖qpos、删点或重置材料续跑。仿真240秒/带土墙钟3600秒预算；机构、MPM和日志分别计数，异常步保留独立故障记录。

## 未覆盖边界

原GUI/启动源、安装Newton库、材料与账本未改。独立有限旋转修复与mechanism-auto8复用。单次开发循环不证明95%成功率/满斗、100ms端到端、人工90%效率、实土/液压标定、连续三铲或RL就绪。额外投影冲量反馈、连续碰撞证明、途中重规划和堆高支撑仍未验收。下一项为同一土体连续三铲与逐铲重感知。

[徐工赛事概要](https://www.prnasia.com/story/544317-1.shtml)只公开命题概述和任务成功率；详细技术指标以本归档requirements-original.png所示的用户原图为依据。
