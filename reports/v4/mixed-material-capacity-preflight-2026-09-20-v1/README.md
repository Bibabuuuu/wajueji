# 多石容量桥时间零预检

状态：`capacity_time_zero_preflight_passed`。

本轮把R3-C固定两石桥隔离派生为按加载模型实际石块数动态分配的容量桥，并完成`64/256/512`石模型生成、C++桥编译、READY和时间零STATE往返。逐石名称到索引使用启动时冻结的哈希表，避免正式场景每拍按石块数做二次扫描。运行总墙钟约`4.85s`，执行`0`次`mj_step`、`0`次MPM/GPU步。

| 石块数 | nq / nv | 模型字节 | 加载至READY | init_default | STATE字节 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 467 / 402 | 367841 | 0.923s | 0.012s | 22941 |
| 256 | 1811 / 1554 | 466001 | 0.955s | 0.107s | 87549 |
| 512 | 3603 / 3090 | 594737 | 1.037s | 0.314s | 169405 |

三档均通过：模型确定性、连续稳定ID、自由关节地址、动态逐石状态数组、`N+1`命名世界力输入容量、时间零逐石载荷全零和消息解析。桥拒绝重复、未知及模型中不存在的石块ID；运行中石块数固定。

复现命令：

```bash
PYTHONPATH=src/v4 python3 src/v4/run_mixed_material_capacity_preflight.py \
  --config configs/v4/mixed-material-capacity-preflight-v1.json \
  --repository-root . \
  --output reports/v4/mixed-material-capacity-preflight-2026-09-20-v1/report.json \
  --mujoco-include /home/qunwang/codex-work/excavator-workspace-20260907/bundle/mujoco-custom/include \
  --mujoco-library-dir /home/qunwang/codex-work/excavator-workspace-20260907/bundle/lib \
  --mechanism-header-dir evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8 \
  --controller-header-dir evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator
```

正式报告SHA256：`103431398eb10dcc2a8e01539d666f6a6c956f05a202a04beae69a32f436a278`。

边界：合成模型只用于结构容量，不是正式`6.1m³`场景；本轮未运行接触、沉降、MPM、逐石真实力路由、单铲或三铲。完整正式N仍须绑定队友生成的冻结场景清单后重新预检。
