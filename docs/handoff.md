# 跨会话交接：带载运输抖动修复已验证（2026-10-15）

## 2026-10-15 路演答辩精修与减振同步视频：已实现、已验证（所列范围）

最新展示交付为Windows deliverables/roadshow-defense-ready-20261015/，25页（17主讲＋8隐藏备用），已删除页脚来源/证据及内部备用页提示；第6页加强真实终态、60Hz轨迹和三次保载量测，第8页嵌入12.733秒上下减振对照，第13页为61.5秒两次实际挖掘精剪。新增767物理帧已独立核验，完整修复后两铲链路未重跑，S05/实机/泛化资格不升级；详情统一见[S07](plans/S07-truck-reproduction-submission.md)顶部。

## 2026-10-15 D-200：带载举升/回转抖动修复，单来源完整运输验证通过

状态：已实现、已验证（单来源运输范围）；所有本轮物理预算已关闭。正式opt-in模块为src/v4/layered_47_loaded_transport_support.py及loaded_transport_support.h；默认CycleRuntime保持原控制。以原生native_time_s启止，现有积分支撑按实测投影外载平滑跟踪，配合反射惯量推导的耗散阻尼；原模型、零主动斗缸命令、4cm/2cm、闭链、材料、扫掠、速度/slew、25mm跟踪和15mm端点保护保留。快照记录策略/时钟/窗口/绑定，活动快照使用普通运行器会拒绝。

同命令v3：887帧、29项独立审计通过。最终真实原TransportController闭环v5：同一实例1634帧（原前缀3+举升734参考/30保持+完整回转687参考/60保持+反馈停用后120帧）。举升峰值4.000785°→1.913036°（下降52.18%），RMS1.518199°→0.572984°（下降62.26%）；前120帧后最大0.558376°。完整回转747帧含保持，峰值0.632674°；旧D196仅120帧入口，不把两者当整段成对因果比较。停用反馈后2秒峰值0.524106°，变化范围0.036122°，没有重新出现大幅往复振荡。

40项独立审计通过、522880条有限子步记录；逐帧原LoadAwareReference请求与偏置复算误差均0，参考逐项一致，保护/前3帧状态粒子严格连续；附加阻尼力最大11.224kN，支撑变化率≤30kN/s，原Kp/Kv不变。8个原快照测试+6个新增策略测试通过。另用两个独立原生实例各3帧：活动策略快照由LoadedTransportRuntime恢复、停用快照由原CycleRuntime恢复，零步24数组及6帧全部状态/粒子严格相同、保护通过；普通运行器错误恢复活动策略被CPU拒绝。恢复验证新增6物理帧，不训练、不推理。

前期未达门槛的v1/v2、v4时钟误接与中断退出错误、审计序列化错误均原样保留；只授予v3及v5实际完成范围的资格。初始斗腔队列12056、最终仍在腔内12024（99.7346%）仅为队列诊断，不是标定满斗率或无撒料证明。未新跑完整挖掘—卸料两铲/第三铲、泛化、视频或PPT；S05总体仍进行中，不能把本问题完成升级为项目完成。

证据：runs/loaded-closedloop-20261015-v5/{report.json,independent-audit.json,states.json,substeps.csv}；runs/loaded-restore-20261015-v1/report.json；work/motion-review-20261015-v1/loaded-closedloop-20261015-v5/{protocol.json,closure.json,*tests.log}。Windows交付deliverables/loaded-transport-fix-20261015/。远程HEAD 8ea6cd8f55c3c35c4be73054ce28ed1f815f21a0，当前修复/测试/记录尚未提交；历史4个脏源码保持原样。


以下为本次修复前交接历史，关于默认接手范围和版本的旧条目以本节及权威总体计划为准。

状态：源码及记录整理；物理预算均已关闭，当前没有授权中的仿真或训练。总体状态仍由 [PROJECT_PLAN.md](PROJECT_PLAN.md)、[CURRENT_STATUS.md](CURRENT_STATUS.md) 和小节方案统领。本文件给接手者提供可直接核验的入口，不新增平行规划。

## 1. 环境、版本与接手范围

- 唯一开发仓库：台式机 Linux `/home/qunwang/wajueji`，SSH 别名 `desktop-tailscale`。本次没有新建 worktree；不使用 A30。
- GitHub：`git@github.com:Bibabuuuu/wajueji.git`；当前分支 `codex/roadshow-metrics-20261011`。整理前 HEAD 为 `867bb5460e94dd4fe0566cc53e83b7a6eecfd5c7`；接手时必须查询实际 HEAD 和远端分支，不能把该基线当成上传后的提交。
- Git推送环境：SSH默认PATH不含已有的`/home/qunwang/.local/bin/git-lfs`（3.4.1）。获准推送时使用`PATH=/home/qunwang/.local/bin:$PATH git push`，正常保留LFS钩子；无需安装或删除钩子。
- CPU Python：`/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python`；SB3 隔离依赖：仓库内 `local-data/runtime/sb3-2.9.0-py312`。不联网重装或更换物理依赖。
- Windows 媒体交付根目录：`C:\Users\14737\Desktop\wajueji\deliverables`。源模型、原生检查点、视频与粒子缓存没有纳入 Git；克隆仓库不能独立重跑物理。
- 源码提交：`8305e63b323ae15d5dd064b532a11a87ce7c2769`；交接文档属于其后的记录提交，实际最终HEAD请用下文命令查询。
- 本次提交范围：人工残差/SAC 接口及测试、路演采集/对照源码、近期连续两铲/返回候选的实验脚本、现有权威文档与精选报告。`work/` 脚本仍是实验入口，未据此晋升为正式通用控制器。

首次操作先回报 `pwd`、仓库根目录、分支/HEAD、修改白名单及第一条命令；先完整阅读根 AGENTS 和规定文档链。接手默认只读分析或 CPU 检查；新物理、训练、改变保护/动作映射需要另获授权。

## 2. 已验证结果及结论边界

1. **D192 连续两铲流程阶段成功**：首铲最低短保载显示 `149.957190%`，贴边第二铲 `109.941335%`；均完成 1170 帧人工参考及 0.5 秒保载。首铲→真实运卸返回→原生恢复续接第二铲，土堆未重置；原生恢复与首 3 帧严格对照、47 项独立审计通过。不能表述为从初态单实例完全不中断，也没有第三铲、次铲运输或跨场景成功率证据。显示值不是已标定净装车体积。终点允许少量黏附，不设净空门。
2. **D193 图表及连续视频已交付**：6 组图、6 张关键帧、原始图表数据、1080p 连续主视频 317.97 秒及明确标注的 2 倍速版。Windows `deliverables/roadshow-two-scoop-media-20261015/index.html`；台式机 `runs/roadshow-two-scoop-media-20261015-v1`。快照续接处已核验并标注。等待占比约 64%～68% 指参考暂停推进、允许受限纠偏，不能解释为空转停机。
3. **D194/D196 合并返回短检有效**：同一卸料快照到原预备点，旧流程 `147.0833` 仿真秒，新流程 `45.8667` 仿真秒，本例动作时间减少 `68.8159%`。卸料后重新观测选点，退出车厢后同步返回/调整，保留合格外伸高位点；并非取消所有安全中间点。尚未把这条新路线串接后续完整挖掘。
4. **带载减振仍未解决**：斗缸主动速度保持 0，其余轴采用平滑轨迹和控制交接。举升姿态相对参考的峰值误差仅 `4.11005°→4.00079°`。回转只验证入口 120 帧，不能称全程转运减振成功。斗腔粒子数只是诊断，不证明正式保料率。
5. D196 两组独立原生实例共 `3639` 帧（带载 887、返回 2752）；各自零步 24 数组和原连续首 3 帧严格对照，27 项独立审计通过；0 推理、0 训练。不是一次新完整连续循环。预算于 2026-10-15 13:26:48（Asia/Shanghai）关闭，剩余约 455 秒封存。D195 旧预算在执行前到期，0 物理，不得混为一次运行。

## 3. 源码与证据入口

- 通用人工残差：`src/v4/manual_trajectory_residual.py`、`manual_trajectory_alignment.py`、`manual_residual_sac_env.py`、`manual_anchored_sac.py`；测试见 `tests/v4/test_manual_*.py`。
- 最新实际关节等待纠偏：`work/second-scoop-wait-qerror-20261014-v1/` 及后续 second-scoop 运行脚本；0%～100% 力度、原 10% 轴速限幅、20Hz 决策/60Hz 物理、跨模式旧动作作废。不能把旧设定值误差版本与此版本混成同一接口。
- 两铲：`work/edge-first-two-scoop-20261015-v1/`、`work/edge-adjacent-59um-20261015-v1/`；59 微米候选 `work/second-scoop-rise-tolerance-59um-20261015-v1/`。用户确认的上抬最低点**逐帧**下降容差 `5.9e-5 m`，其余限位/碰撞/土堆间距等保护未放宽；不能当作累计下降许可。
- 图表/视频生成：`work/two-scoop-roadshow-media-20261015-v1/`，原始运行数据仍在 `runs/`。
- 合并返回 CPU 候选：`work/motion-review-20261015-v1/motion_candidate.py`、`return_candidate_v3.py`、`control_handoff_cpu.py`；D196 物理入口 `work/motion-physical-short-20261015-v2/{prepare,run,audit,finalize}.py`。旧 `run.py` 只绑定已关闭预算，**禁止直接重跑**。
- 最新精选证据已复制到 `reports/github-handoff-20261015/evidence/`；`source-manifest.json` 记录源码及证据原路径、SHA256，复制未修改原报告。`staging-allowlist.json` 是本次整理白名单。归档源码通过逐文件 `.gitattributes` 保留原始CRLF与SHA，未重写冻结脚本。Git 中报告副本便于审阅；独立复算仍需原日志、缓存和冻结模型。
- 路演取数合同：[PPT_DATA_REQUEST.md](roadshow/PPT_DATA_REQUEST.md)。当前演示视频使用原合格路线；不能把 D196 的短段结果冒充已更新的完整两铲视频。

## 4. 必须保留的输入与本地差异

关键原生文件（均留台式机，不入 Git）：

| 输入 | 路径（仓库根目录下） | SHA256 |
| --- | --- | --- |
| D196 带载源 | `runs/edge-first-two-scoop-20261015-continuation/policy-1/episode-0001/endpoint.npz` | `46d3f04fa2dfc18082c17f7496e4c146f2491fdb8e08ccd0339cc6169203947a` |
| D196 卸料末态源 | `runs/edge-first-two-scoop-20261015-continuation/dump.npz` | `26d3928d2e41fc1f5bce5020160be6faddcdca7a936dcaddd7eb65a49e1f94b3` |
| 冻结 SAC | `work/manual-strength-overnight-20261013-v1/anchor-grid-v1/seed44-weight100.zip` | `84aade9a3d8819368afb72223b8452ebfbc5e9ac9ad190818207d1142692aab1` |

四个早先未提交文件保留原样：`src/v4/audit_layered_47_cycle.py`、`run_layered_47_sparse_manual_contact_episode.py`、`soil_surface.py`、`view_contact_bc_probe.py`。本次不整文件收进正式源码。精确差异保存为 `reports/github-handoff-20261015/preexisting-uncommitted.patch`，实际工作文件 SHA 在 manifest 中。这些本地版本参与过历史运行；新克隆默认不是完整物理环境，不可直接声称按旧 SHA 复现。若后续确需恢复，在独立工作树先检查补丁再决定，不能覆盖现有脏改动。

禁止覆盖旧失败记录、重置土堆冒充连续、改写原预算或沿用封存剩余时间；不要删除 `runs/`、`local-data/`、模型及原生快照，不要对工作树执行 reset/checkout/stash 来强行清理。

## 5. 可执行的低成本核验

以下命令在台式机仓库运行，不启动物理：

```bash
cd /home/qunwang/wajueji
pwd
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short --untracked-files=no
git ls-remote --heads origin codex/roadshow-metrics-20261011
git diff --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 PYTHONPATH=local-data/runtime/sb3-2.9.0-py312:src/v4 \
 /home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python \
 -m unittest discover -s tests/v4 -p 'test_manual_*.py'
```

本次整理已按原隔离环境通过 54 项人工残差 CPU 回归。首次漏加隔离路径导致 Gymnasium 导入失败，随后使用上列路径通过，无依赖安装。排除上述4个脏文件后的候选树另通过10项路演CPU测试（合计64项）；全部候选Python语法、源码/证据SHA及Git校验见 `reports/github-handoff-20261015/validation.json`；只按记录的实际覆盖范围解释，不能代替物理验证。

## 6. 接手后的下一步与暂停点

该只读诊断现已完成，12项CPU检查通过、0新物理/训练；主要斗姿误差对应斗缸偏移及闭链角度放大，持续受力首因仍未确定。结论、原始SHA、CSV及下一门统一见[S07](plans/S07-truck-reproduction-submission.md)与[诊断报告](../work/motion-review-20261015-v1/loaded-diagnosis-v1/README.md)。建议新增10分钟、最多120帧同轨迹子步受力观察；这是待确认方案，尚未登记新预算，不直接改积分、保持刚度、斗缸动作或冻结D196入口。

若做后续实现，文件白名单先限定 `work/motion-review-20261015-v1/` 的新版本候选/离线检查，复用通用控制器；不得直接改冻结的 D196 绑定文件或正式 `src/v4/` 公共接口。便宜验证通过后回报候选差异和证据，再讨论新的有界物理预算。放开带载斗缸、改变保护、训练或扩大完整流程，均属新的路线/成本决策。

交付要求：精确差异、测试命令/结果、源码与输入 SHA、失败日志/恢复状态位置、已实现/已验证/未验证边界；新增失败必须保留原日志和中间状态。当前不需要 A30 重跑。现有两铲图表和连续视频可用于 PPT，合并返回/减振结果要按本文件边界描述。

---

以下为 D165 历史交接，保留追溯；其中分支、下一步、无 RL 等现状描述已被上文及权威规划替代。


# 跨会话交接：D-165保载修复与残余聚团

> **2026-10-12当前协作入口**：[路演PPT数据请求](roadshow/PPT_DATA_REQUEST.md)。先看CURRENT_STATUS和当前Git；下方D165为历史记录。用户已指定台式GPT推进技术，Windows会话负责PPT，按阶段交付取数。


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

保存轨迹只读复核：原首铲斗腔12086粒子在第1500/1560/1620/1680帧的2cm占据格8569/7148/6054/5859，约1510帧开始加速聚团，至1680帧原粒子仍全部保留。第1080与1680帧该集合的弹性应变行列式中位数0.9959/0.9974、`Jp`中位数均为1.0；与占据格收缩不同量级，但不能单凭此判断成因。接下来应定位每子步位移/材料/接触账本，不再重复跑前缀或盲扫参数。

原手动8122帧与完整缓存仍在`runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275`。新的首铲输入重算保存于work目录：`manual-prefix-1800/display`实际0–180，`manual-prefix-continuation-1620/display`实际180–1800；两段边界物理同一状态，不是两次挖掘。第180帧只是round(None)报告错误，全硬门通过。`recording-fault-recovery-audit.json`逐项核验来源、全部180帧保护和末态后生成诊断hold副本；原failure不改，通用failure门不放宽，后续从180续算。

新1680/1800完整点在`manual-prefix-continuation-1620/state-001680.npz`与`state-001800.npz`，完整policy与元数据同目录；不是最终保载合格或人工prepoint。123帧被原斗缸端限位预览重新裁剪，记录了原因，不能声称新物理路径与旧路径完全相同。独立3步恢复见`independent-continuation-3/independent-array-audit.json`；首版报告把字符串rigid_ids转float失败，直接审计已保存末态，没有重跑物理。

45项相关测试、暂存版本的CLI help、文档检查和diff检查通过。较细网格/250迭代/APIC/速度外延/法向位置修正均未消除残余，不作为正式修复部署；各失败/无收益报告保留。

## 下一道门

先只读审核接触位置修正与材料体积响应的一致性，带着已保存的前后状态找因果；不要再次泛化扫参或要求用户重录。当前仍存在粒子聚团，即使满斗率超过95%也不能授予根治。重大接触求解/材料路线变更按AGENTS先讨论；已有明确修正不重复索取许可。修复完成后优先重算保存的手柄输入生成新物理缓存，再让用户录屏；全三铲新版本仍未验证。

## 保留边界

不启动RL，不修改原示范、原录像、冻结结果和保护。工作树既存的`audit_layered_47_cycle.py`、`soil_surface.py`、`view_contact_bc_probe.py`及人工采集器原241行workflow差异保留未提交；本轮人工入口只提交motion-consistency增量，切勿整文件覆盖或误收既存变化。大型NPZ/粒子缓存/trace只留本地，不入Git。
