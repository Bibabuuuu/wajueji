# D-143 快速第二铲采集入口

状态：已实现、本机短保持及独立恢复验证通过。用户明确选择快速采集分支。

源为新第一铲（96.54%）的真实接触退出点1bf4e9ca…；270帧实际上抬34.18cm后，观测载料已分离。用户随后取消完整运输准备，改为直接构造接触训练初态。

取走12024土粒和6石（索引8/13/16/28/40/41），保持稳定ID并整体平移到工作区外隔离地面库存；全场仍为256441土粒、47石，不假装交付车厢。未取走土石的位置、速度和材料字段在重置边界逐数组完全一致，新的机构/全部粒子和石块状态读回也逐值一致。控制器、网格/接触缓存和待结算斗反力在显式训练初态边界重建。库存依然参与物理，本次优化减少铲间准备时间，不承诺提高单帧仿真速度。

第二铲切入点(5.548,-0.581,1.145)m；直接预备铲尖(6.046,-0.633,1.745)m。条带评分考虑剩余两铲，第三铲候选条带(5.25,0.65)m只是空间预留，第二铲后必须重新观察。预备点SHA b324d7bd3738fd7e6cabd70ce491ab3a82b773ff339d0bf5a99f7a1aeeae366b，cycle_index=2，root=3a259cfa…。原第一铲及离土点未改。

验证：46项相关单测通过；构造＋30帧空斗保持约8.19秒，严格零土/石斗接触。独立恢复读回精确，另3帧保持通过；实际手操入口3帧10项运行检查通过，末态全部24数组与独立审计逐值一致。第二次构造额外核验完整零步读回，其预备点NPZ SHA亦相同。首个更近候选第3帧1个残余网格接触采样，约8e-9Ns；外移/抬高各10cm后通过，失败记录保留。全剩余材料用于斗避让，地面连通表面用于条带评分/机构地形检查；空中材料高度柱用于机构检查会产生保守零净空，未据其改变地形或删除散土。

入口：

```bash
bash scripts/run_47_cycle_manual_local.sh runs/fast-contact-collection-20260924-v1/prepare-clear/prepoint.npz runs/fast-contact-collection-20260924-v1/restore-audit/audit.json --training-reset-checkpoint
```

后续已分离的用户完整末态可用`bash scripts/prepare_47_fast_collection_local.sh /path/final-xxxxxx.cycle.npz`生成下一点。未离土时该脚本拒绝，不裁掉连着地面的材料；需先有界实际上抬并复核。第3铲后拒绝生成第4铲。

模式`training_reset=fast_contact_collection_v1`、`continuous_physics=false`、来源链及库存随保存继承，默认连续加载器拒绝；旧diagnostic导入也不能抹去标记。手操UI显示FAST COLLECTION RESET。每铲接触动作仍连续；此分支不代表真实运输卸料、正式G4/G5或训练准入，训练/验证须按原lineage整组划分。

机器可核验摘要：[report.json](report.json)；只读复核脚本：[review.py](review.py)。完整原始状态与逐帧日志留在被Git忽略的runs/目录。

当前已启动人工窗口`manual-contact-20260924-110051-1421079`，ready为暂停0帧、手柄已识别；[启动记录](live-ready.json)。窗口启动不代表本铲采集完成。
