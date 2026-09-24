# 新组第三铲快速入口

从第二铲真实末态73aa8f1a…生成，新增9426土粒/2石取走库存，前次14529土粒/5石保留，所有余料初始化数组完全相同。沿用D-144全宽评分和D-143显式快速重置，cycle_index=3、root659f1360…、continuous_physics=false；不运行运送卸料。

新切入点(5.648,0.592,1.031)m，预备尖(6.145,0.644,1.731)m。预备点SHA a6fed857a335c95b6fb6222530b33fe73267841d77e527a712731d5edd0d896b。30帧保持、独立恢复/手操各3帧均零土石斗接触，24数组一致，18准备检查通过。本轮未修改运行源码。只授予本地快速第三铲入口，不授予满斗改善/G4/G5/训练。

```bash
bash scripts/run_47_cycle_manual_local.sh runs/fast-contact-collection/20260924-115633-1457317/prepare/prepoint.npz runs/fast-contact-collection/20260924-115633-1457317/restore-audit/audit.json --training-reset-checkpoint
```

[核验报告](report.json)、[核验脚本](review.py)。
