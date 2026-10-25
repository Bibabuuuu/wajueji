# 第二铲结束后的统一检查（2026-09-25）

状态：已验证。本次只读审核没有新增物理；接触2725帧、正常补收439帧、固定运输卸料3398帧，共6562帧。旧v4交接后观察预算失败保留，只审核其规划前正常保存的439帧。

- contact-audit.json：59项通过，独立控制复算误差0，行为15项通过。
- carry-audit.json：30项通过，独立控制复算误差0，交接/保存控制历史一致。
- audit-review.json：38项通过，独立控制复算误差0，开发接收22233/22234、原载荷斗内残留0。
- replay-review.json：三个缓存哈希和两处完整边界一致；second-replay.json为6562帧回放入口，播放不推进物理。

实际最终快照：runs/contact-second-fixed-route-20260925-v1/physical/final.npz，SHA256 620b50d6d69fa2d6c036c8f500284265ffab1978cc4506df655fe7c69179f7a9。

后续CPU工作装载评分：CLEAR_PILE末五帧中位数30.21405%，有效开发估计，包含土石和堆高、占据阈值0.2。证据在runs/night-parameter-search-20260925-v2/baseline-score.json，不是稳定混合卸车净体积或95%资格。

本次统一检查授予记录/执行/恢复链可信性。第二铲卸料完成；没有回位、第三铲或跨地形泛化资格。夜间短检和搜索状态另见docs/CURRENT_STATUS.md。
