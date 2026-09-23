# D-068缩比三层混合场景短动力学烟测

本报告逐哈希绑定`layered-mixed-scene-local-preflight-2026-09-20-v1`的22石清单和72415粒子资产，临时生成实际N MuJoCo模型并复用动态容量桥。Newton MPM按稳定运行时ID汇总每块石的接触力和力矩，并在下一外步回传MuJoCo。

本轮把最终组装态的全部石块和粒子从第0帧同时激活，只运行`12 × 1/120s = 0.1s`。结果为：

- 石—MPM非零接触样本`40250`；
- MPM粒子状态有限，最大粒子速度`0.105477m/s`；
- 最大石线速度`0.393416m/s`、角速度`0.461026rad/s`、位移`0.023249m`；
- 禁止接触为0，机构闭链最大`0.0000468584m`；
- 8项短烟测检查全部通过。

这只证明最终组装态没有在0.1秒内立即非有限或接触爆炸。石块仍在明显沉降，因此`settling_qualified=false`、`wall_free_hold_qualified=false`。下一步必须按foundation/lower/middle/top阶段逐层激活，构造期重力渐增，最后恢复名义参数并做至少1秒撤墙静置。

复现命令：

```bash
PYTHONPATH=src/v4 /home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python \
  src/v4/run_layered_mixed_scene_local_dynamics_smoke.py \
  --config configs/v4/layered-mixed-scene-local-dynamics-smoke-v1.json \
  --repository-root . \
  --output reports/v4/layered-mixed-scene-local-dynamics-smoke-2026-09-20-v1/report.json \
  --mujoco-include /home/qunwang/codex-work/excavator-workspace-20260907/bundle/mujoco-custom/include \
  --mujoco-library-dir /home/qunwang/codex-work/excavator-workspace-20260907/bundle/lib \
  --mechanism-header-dir evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8 \
  --controller-header-dir evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator
```
