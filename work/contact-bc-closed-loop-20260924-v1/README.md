# 原生60Hz BC受保护推理与120帧起步短测

用户已授权D-145推理接线及短段闭环。本轮直接恢复已有第三铲完整预备点a6fed857…，没有从起点运动，也没有修改人工默认入口、物理参数或控制助力。试训检查点4bd50c9f…保持不变。使用第三铲点是因为它已有CycleRuntime完整恢复合同；没有把旧第一铲approach-handoff强改为cycle格式。

## 实现与核验

`contact_bc_inference.py`复用同一特征编码/归一化，在独立CPU进程运行既有训练环境的PyTorch，不修改物理venv。包含检查点SHA、特征schema、有限值、输出形状/范围、请求ID和超时检查。在线历史只用当时状态、上一实际动作和上一帧负载；地形每3帧更新且严格早于动作。初始完整地形现场测量，没有拿未来地形填历史。`run_contact_bc_probe.py`显式候选入口限制0–120帧/180秒，沿用命令投影和全部原硬保护。

15项相关测试通过；验证三条4476帧的流式特征与离线批处理逐值一致，21次独立推理与直接CPU模型输出一致。零步24数组完全一致，控制器/原生状态误差全0，输出预测且0物理推进。

## 实际短测

- 第三铲预备点自主运行120帧＝2秒仿真；全部硬保护、材料和移动铲斗防漏检查通过。
- 铲尖下降54.71cm，X/Y分别移动4.01/4.88cm；同初态人工示范前120帧下降45.68cm、Y移动−2.25cm。两者路径不同，不能据起步通过认定后续方向正确。
- 第106帧开始少量接触，7帧共10个土壤接触采样，0石斗接触子步；尚未进入完整收斗/带载抬升。
- 120帧都有非零策略动作，0帧被命令投影修改。CPU推理往返中位1.268ms、P95 1.403ms；只是推理耗时，物理逐帧日志合计15.47秒，整体非60Hz实时。
- 最终物理快照5333f2a4…保存；原预备点、策略、来源链及快速重置库存保留。该快照不包含在线策略历史，不能直接当作合格策略续跑点。

只授予接口一致性和这一次短窗自主起步通过；不授予完整挖掘、收斗/离土、95%满斗、G7或跨场景泛化。

## 失败与修复边界

保留所有原运行目录：第一次零步因原始状态/compact字段名差异失败，0步；第一次物理探针1帧安全通过后，因续跑日志不含负载字段失败，改从同一底层状态读取；修正后120帧与末态保存完成，但报告中的NumPy整数无法JSON序列化，进程非零退出。已修复布尔计数并加入回归测试。

没有为了报告重跑120帧；[独立审查](probe120-review.json)从原日志/快照恢复结论，21项核对通过。报告修复后的代码只做单测，不冒充同版本再次物理通过。只读审查初稿对NPZ内字符串元数据误用isfinite，修正为仅检查数值数组后重跑通过，没有物理推进。

## 复现与下一步

从仓库根目录运行，输出目录须不存在；共享本地物理锁。`--frames 0`零步，`--frames 120`为本轮短窗上限。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
flock -n /tmp/wajueji-local-47-sparse-manual-contact.lock \
/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python src/v4/run_contact_bc_probe.py \
 --source runs/fast-contact-collection/20260924-115633-1457317/prepare/prepoint.npz \
 --policy runs/contact-bc-native60-pilot-20260924-v1/pilot.pt \
 --policy-sha256 4bd50c9f2bd9d5d31b0adee2335f2320b19056b518a097a23a11d2373bcf371c \
 --training-python /home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python \
 --training-reset-checkpoint --output runs/NEW_BC_PROBE --frames 0
```

下一门：有界延长至实际接触/收斗阶段，跟踪横向偏移和动作衔接；从原预备点重建完整历史或先明确保存策略历史的续跑合同，不能只载本轮物理末态。带载离土停止规则和满斗率仍未验证，不启动长训练或A30。

证据：[接口一致性](parity.json)、[零步与失败记录](verification.json)、[120帧只读审查](probe120-review.json)。大快照和逐帧原始日志留在runs，不进入Git。
