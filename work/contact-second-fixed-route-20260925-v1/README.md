# 第二铲固定路线续接（D-162）

状态：已授权实施，4项CPU静态检查、24数组零步恢复及3398条全路线名义指令检查通过；物理续段准备启动。原生控制状态恢复误差0，最大归一化指令变化0.03333333333333337，预检165.11秒、物理0步。仓库 `/home/qunwang/wajueji`，分支 `codex/contact-bc-collection-20260924`，起始HEAD `a5caf0f`。

用户明确：不追查飞散；零散土粒不应作为运输障碍；沿用纯土固定车位流程。新候选仅取消运输阶段对全场土粒地图的依赖，保留固定抬升/回转/卸料目标、静态车体/地面、限位、闭链、变化率、跟踪、实际保斗和全部原物理保护。没有删土、冻结物料或屏蔽物理碰撞。`bucket_terrain_m=null`明确表示不测量，不能伪造正净空。下一铲选点观察不变。本候选不运行回位、第三铲或训练。

唯一源为 `runs/contact-second-finish-unload-20260925-v4/physical/handoff.npz`，SHA `f750e55864d6189ff6cd149cc1574894c236a334e6817a4b88f66493fd33c84d`，第二接触累计3164帧。是规划失败前保存的正常hold源，不接受recovery或任意失败点。保留完整物理/native/材料/防漏状态，从源内carry.previous_command接运输，不重挖、不重做补收。旧v4及此前输入代码全部只读且SHA绑定。

剩余预算≤5561帧/1830墙秒，与旧439帧/161.044秒合计不超过已授权6000帧/2000秒尾段预算。每120帧、每阶段末及退出保存，保护/超时即止，不自动重跑。输出 `runs/contact-second-fixed-route-20260925-v1/physical/`，物理日志 `runs/contact-second-fixed-route-20260925-v1-physical.log`。中断发送SIGINT给本候选Python，保留recovery和日志；不能假设强杀后具有完整恢复状态。

预检：`test_static.py`验证地形未测量标记、车体/机构/地面拒绝、原两档闭链容差及语法；`prepare_contract.py`绑定本包/源/旧运行及几何输入；`preflight.py`禁止step，验证24数组/native恢复、保存不变、整条固定路线几何和每条名义指令的连续性。预检输出在 `runs/contact-second-fixed-route-20260925-v1/preflight/`、`plan/`及本目录`preflight-review.json`。名义指令通过不等于真实闭环通过。

本机RTX4070、CUDA0，物理Python `/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python`；在仓库根设 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。先完成预检，再执行本包`run.py`一次。全过程保持策略/MPM60Hz、刚体4800Hz，无新物理参数。

结束后统一审核实际指令、半步时钟/力分派、身份/守恒、保存状态及开发卸料接收；生成第二铲完整连续缓存回放。不把开发土粒接收数量当稳定混合体积、95%满斗或三铲成功。夜间搜索仍待新基线串联及短检查通过，不能直接运行绑定旧失败v4的包。
