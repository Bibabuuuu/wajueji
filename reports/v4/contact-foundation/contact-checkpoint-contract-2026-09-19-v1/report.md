# 接触策略 checkpoint 隔离契约审计

更新：2026-09-19。状态：契约已冻结并通过零权重审计；人工连续画面复核仍受证据缺失阻塞，未训练、未加载checkpoint、未运行物理。

## 结论

新接触策略只允许三个动作头：`penetrate_deepen`、`curl_load`、`lift_clear`。完整标签仍保留`approach`、`off_soil_confirm`和`uncertain/recovery`，但前两者属于几何控制，后者只属于安全监督器。旧checkpoint的四个动作头为`approach, penetrate_deepen, curl_load, lift_carry`，不能通过新schema。

新checkpoint必须同时绑定：2412维观察契约、四轴速度顺序、已完成人工复核的标签清单哈希、互斥开发划分、仅训练划分归一化、零holdout读取，以及独立的运行时交接审计回执。训练产物不能自带“运行准入”。

当前34段候选全部仍是`visual_manual_review_pending`，训练资格行数为0；因此审计预期拒绝，拒绝码为`contact_label_schema_not_reviewed`。

## 为什么现在不能补签人工画面复核

允许的数据目录中只找到7张`ready-with-ui.ppm`启动画面；连续视频0个，其他过程图片0张。这些启动画面不能证明接触、卷斗、抬升、离土和接管时刻的视觉意图，因此本次没有把复核清单打勾，也没有把候选标签改成reviewed。

## 隔离结果

- 旧四头checkpoint被新接触契约拒绝：`contact_checkpoint_schema_mismatch`。
- 新接触checkpoint被旧运行加载器拒绝：`v4_actor_checkpoint_contract_mismatch`。
- 新checkpoint即使未来完成训练，也必须另有与其SHA256绑定的`external_contact_handoff_router_v1`运行准入回执。
- 当前候选、旧checkpoint和任何缺少连续视觉复核回执的标签都不能启动接触BC训练。

## 下一步

先决定如何取得连续视觉证据：重新录制带同步视频/关键帧的新示范，或明确接受“旧六条仅作数值先验、不进入新BC训练”。之后才能生成reviewed标签、人工复核回执和M2开发划分。
