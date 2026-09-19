# v4配置

`actor-normalization.json` 是当前行为克隆策略的归一化参数。历史启动命令和D071导出记录位于 `reports/v4/`，因为其中的绝对路径只描述旧运行，不能直接当作新环境配置。

`contact-checkpoint-contract-v1.json`冻结下一版接触策略的元数据契约：完整标签与三个动作头分离，旧四头权重、未reviewed标签和缺少独立运行准入回执的checkpoint都会被拒绝。其状态`frozen_review_blocked`表示契约已确定但当前没有训练资格，不是可训练或可部署配置。

未来的本机路径覆盖文件使用 `local-paths.env`，该文件已被Git忽略；在完成路径可移植性候选前不要伪造此文件已被现有启动器读取。
