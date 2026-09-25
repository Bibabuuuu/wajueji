# 跨会话交接：D-165保载修复与残余聚团

先完整阅读AGENTS、总体规划、当前状态、要求及七个活动小节。核对实际Git分支/HEAD和进程；唯一仓库`/home/qunwang/wajueji`，分支`codex/contact-bc-collection-20260924`，本轮起始HEAD `38bc708ef24ffbfe3427c5e84ae76af04a2da181`，不得当作下一次当前HEAD。

## 结论

D-165本轮已结束，当前无新增物理或训练任务。主要接口修正已通过45项相关测试、固定轨迹及用户首铲1–1800帧短段验证；新第一铲1080/1680帧满斗148.4953%/111.5174%，原同帧103.16%/29.91%，但2cm占据格9381→5859、峰值17→131仍提示聚团，未根治。新进程3步续算的23个非派生时钟字段、原生控制及扫掠状态逐位一致，派生仿真时钟仅7.105e-15秒浮点差，独立列出容差；没有重复物理前缀。5cm、APIC、法向位置修正及速度外延均未改善，不部署。原示范、保护与旧默认路径保持；新接口需显式选项，不授予稳定保载、完整卸料、三铲或重新录最终视频资格。

## 已实现

- `src/v4/bucket_velocity_origin_candidate.py`：显式修正MuJoCo COM速度→普通body原点速度，输出COM与惯性偏移诊断；40姿态Jacobian核验误差1.11e-16m/s。
- `src/v4/bucket_motion_consistency.py`：D133开始→预测末端→实际末端一致性；旧保存COM速度仅转换一次，不重置物理数组。
- cycle runtime和人工入口支持显式`bucket_motion_consistency`/`--bucket-motion-consistency`；人工模式还要求cycle和原扫掠。
- cycle checkpoint记录policy和两个源码绑定，旧入口拒绝误读修正后的检查点；旧无policy状态与所有旧默认运行不变。

## 数据与验证

权威实验入口`work/bucket-projection-isolation-20260925-v1/README.md`，含全部对照及边界。固定D129满斗源，旧600帧52.3482%，速度原点修正84.2810%，加时序85.2640%；HOLD600帧102.4988%，因此残余不是普通静置沉降。原887帧状态运输后段已张斗，比较闭斗严格取600帧，不能把887末态下降都归为漏土。

原手动8122帧与完整缓存仍在`runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275`。新的首铲输入重算保存于work目录：`manual-prefix-1800/display`实际0–180，`manual-prefix-continuation-1620/display`实际180–1800；两段边界物理同一状态，不是两次挖掘。第180帧只是round(None)报告错误，全硬门通过。`recording-fault-recovery-audit.json`逐项核验来源、全部180帧保护和末态后生成诊断hold副本；原failure不改，通用failure门不放宽，后续从180续算。

新1680/1800完整点在`manual-prefix-continuation-1620/state-001680.npz`与`state-001800.npz`，完整policy与元数据同目录；不是最终保载合格或人工prepoint。123帧被原斗缸端限位预览重新裁剪，记录了原因，不能声称新物理路径与旧路径完全相同。独立3步恢复见`independent-continuation-3/independent-array-audit.json`；首版报告把字符串rigid_ids转float失败，直接审计已保存末态，没有重跑物理。

45项相关测试、暂存版本的CLI help、文档检查和diff检查通过。较细网格/250迭代/APIC/速度外延/法向位置修正均未消除残余，不作为正式修复部署；各失败/无收益报告保留。

## 下一道门

先只读审核接触位置修正与材料体积响应的一致性，带着已保存的前后状态找因果；不要再次泛化扫参或要求用户重录。当前仍存在粒子聚团，即使满斗率超过95%也不能授予根治。重大接触求解/材料路线变更按AGENTS先讨论；已有明确修正不重复索取许可。修复完成后优先重算保存的手柄输入生成新物理缓存，再让用户录屏；全三铲新版本仍未验证。

## 保留边界

不启动RL，不修改原示范、原录像、冻结结果和保护。工作树既存的`audit_layered_47_cycle.py`、`soil_surface.py`、`view_contact_bc_probe.py`及人工采集器原241行workflow差异保留未提交；本轮人工入口只提交motion-consistency增量，切勿整文件覆盖或误收既存变化。大型NPZ/粒子缓存/trace只留本地，不入Git。
