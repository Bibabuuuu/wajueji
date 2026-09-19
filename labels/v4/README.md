# v4标签与索引

- `v4-actor-data-index.ndjson`：六条训练示范形成的行为克隆数据索引。
- `subphase-labels.ndjson`：当前阶段/子阶段标签。
- `contact-segments.ndjson`：工作包C生成的34段接触阶段候选，覆盖六条开发训练轨迹。
- `contact-events.ndjson`：接触、卷斗、抬升、最终离土和模糊恢复事件候选。
- `contact-labels-manifest.sha256`：两份新候选标签的完整性清单。

新候选全部为`training_eligible_now=false`，在六条连续画面和精确接管交界完成人工复核前不得替换现有演员索引、重训权重或解释为在线阶段门槛。候选交接点用未来接触事件回看裁出，只用于保留近土训练上下文；部署时必须由独立几何交接契约确定。

未来只有`excavation_contact_segment_reviewed_v1`/`contact_segments_v1_reviewed`才能通过接触checkpoint契约；当前仓库尚不存在该文件。不要通过改名把candidate升级为reviewed，必须同时提交连续视觉复核回执和新的完整性清单。

封存盲测轨迹不在这里，也不得加入训练索引；其原始数据位于本地忽略区 `local-data/archive/v4-full-2026-09-19/03-data/holdout/`。
