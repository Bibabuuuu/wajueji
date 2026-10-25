# D-127 v1：零帧预检失败，保留记录

独立候选GPU基准13项及观测核预检通过。正式场景重放在第0帧被额外快照逐字段检查拒绝，未执行物理。22项原始/续步字段完全一致，但快照内两项reported世界速度与新进程mj_forward重算值不同，差分别5.401408e-6m/s和1.503305e-4rad/s。

只读核验：新进程reported线速逐值等于恢复的刚体qvel前三分量；reported角速与恢复姿态旋转qvel角速度的误差≤7.807e-18，表明这两项是派生读数。原桥init在mj_forward后emit；常规emit只调用mj_kinematics/mj_comPos，缓存时序与恢复重算不同。源实际qpos/qvel、MPM全部状态和warm-start未改变。v2改为22字段逐值一致＋原试采首帧before_state精确匹配＋派生速度按恢复qpos/qvel重算核验，保留并报告两项历史读数差，不夹状态、不改物理门。

第0帧failure诊断完整保留。此目录不算物理失败或候选效果证据，实际短重放由v2承担。
