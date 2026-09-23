# 887石结构容量上界时间零预检

状态：`capacity_upper_bound_time_zero_passed`。

D-063已取消887石正式数量硬门；本报告只保留旧精确配方数量作为协议与模型结构的咨询上界。合成887石模型重复生成逐字节一致，同一动态N石桥完成编译、READY和时间零STATE：`nq=6228`、`nv=5340`，模型约`786045`字节，READY约`12692`字节，STATE约`294771`字节；加载至READY约`1.275s`，`init_default`约`0.745s`。

全部9项检查通过：连续稳定ID、`N+1`命名力输入容量、时间零逐石状态与零外力、期望维度、确定性和零物理步。本轮执行`0`次`mj_step`、`0`次MPM/GPU步；不证明887石是正式数量，也不证明真实网格场景、接触、沉降、吞吐或铲运。

复现命令：

```bash
PYTHONPATH=src/v4 python3 src/v4/run_mixed_material_capacity_upper_bound.py \
  --config configs/v4/mixed-material-capacity-upper-bound-v1.json \
  --repository-root . \
  --output reports/v4/mixed-material-capacity-upper-bound-2026-09-20-v1/report.json \
  --mujoco-include /home/qunwang/codex-work/excavator-workspace-20260907/bundle/mujoco-custom/include \
  --mujoco-library-dir /home/qunwang/codex-work/excavator-workspace-20260907/bundle/lib \
  --mechanism-header-dir evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8 \
  --controller-header-dir evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator
```

报告SHA256：`c3f86cba99c9981993066bc1563e34d90b4a518acb5f9b09d451771857607a2f`。
