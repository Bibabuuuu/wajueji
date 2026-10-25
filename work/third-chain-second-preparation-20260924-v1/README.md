# 新组第二铲快速采集准备

来源为第一铲完整末态659f1360…（1411帧、102.38%）。观测载料分离后另记14529土粒/5石库存，未取走余料初始化前后全数组一致，全部ID保留。新点cycle_index=2，保留本组来源链，显式training_reset、continuous_physics=false。

旧预备点上偏0.6m在第6帧有一个约5.23e-8 N·s的网格冲量，严格零接触门拒绝。先调整候选顺序，但较高候选IK不可达，又落回原点，第二次同样被拒；两次记录保留。最终只选上偏0.7m的候选，不能达则重选切入/预备点对，原IK、机构范围、净空和零接触门不变。

最终切入目标(5.548,-0.581,1.204)m，预备铲尖(6.046,-0.633,1.904)m，D-144全宽评分保留一个后续条带。30帧保持、独立恢复3帧及手操3帧均零斗接触，全24数组相同；18项只读准备检查、49项相关单测通过。预备点SHA a88efb6fce3ac9e4cba54593c41f09c99eb1061aa3d16730ea1ba4ba2cc2ec46。

入口直接暂停恢复，无起点接近、运输卸料；只授予本地快速第二铲采集入口，未验证该切入点实际满斗改善，不授予G2全路径/G4/G5或训练资格。绿色是斗宽中心目标，青框是0.60m口宽参考。

```bash
bash scripts/run_47_cycle_manual_local.sh runs/fast-contact-collection/20260924-114750-1451324/prepare/prepoint.npz runs/fast-contact-collection/20260924-114750-1451324/restore-audit/audit.json --training-reset-checkpoint
```

[核验](report.json)、[脚本](review.py)、[首次拒绝](rejected-lower-prepoint.json)、[回落拒绝](rejected-fallback-prepoint.json)。
