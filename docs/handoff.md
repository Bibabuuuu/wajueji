# 跨会话交接：D-153续段预检通过，场景尚未运行

更新：2026-09-24。唯一当前入口为[CURRENT_STATUS.md](CURRENT_STATUS.md)，先读AGENTS要求的总体规划、七个活动小节，再核对实际分支/HEAD/进程。本页下方保留此前冻结资产，不按历史“下一步”擅自重跑。

D-153已实现并通过离线/零步门：历史2400帧命令/阶段精确重建，末三帧BC及序列化后末帧BC一致；24数组及控制器零步恢复精确。1124/1200/1376的小命令预览连续，原速度/关节/禁止下压/地面保护保留。只授予D152特定末态的候选续段入口，普通加载器仍拒绝候选身份。已冻结单次1200帧/360秒协议，准备运行；未推进新MPM，未授予离土、满斗或RL资格。

D-152已完成：完整6×6响应及原生副本预测的4测试/零步门通过；唯一60Hz候选2400帧/655.14秒无硬失败，120/1200/全程行为及独立命令/矩阵/4800半帧/源/身份审核通过。全程桶缸峰速0.05973m/s、半帧最大变化0.02286m/s，旧异常明显改善；成本同前120帧比原60Hz+34.1%。未切默认，未跑首/次铲或RL。

末态`runs/contact-full-response-20260924-v1/candidate60/final.npz`，SHA `37cb63c32f9070bcf119fac3a729be64ad6cde31593fa9f5ac8a17a6b5c6a0f2`；独立候选schema拒绝普通恢复，尚无已验策略续跑资格。末态curl_load/约99.85%收斗，最后5帧离土距离未过3mm，因此不算有效满斗或单铲成功。1124–1549连续426帧零动作的只读原因：非零预览会消除实际软闭链残差，造成0.1–0.5mm与命令大小无关的高度偏移。

见[D-152证据与下一阶段草案](../work/contact-full-response-20260924-v1/README.md)。D-153该范围已授权，仍不能自动追加第二轮或只改schema。冻结协议复制了D150旧描述字段，实际runner及D152_scope均为单次60Hz/2400/总900，`protocol-review.json`保留该元数据问题；后续新协议须清理，不覆盖旧协议。全部物理和只读审核已结束。

最新D-151已完成：用户授权的两异常状态机构响应CPU核验共78个副本探针/84.06秒，无MPM推进。原生恢复/重复/保护、独立来源/中心差分检查与6项拟合测试通过，但现有正质量/惯量接口整体约36%、最大方向约100%误差，未过20%/30%接口门。按事先确认的暂停点未启动完整土壤候选。下一待讨论为完整闭链6×6响应接口的最小扩展，该新接口已获D-152原型授权，尚未验证。不要填任意质量、取消kinematic、继续提高频率或重跑整个前缀。

[D-151工作包](../work/contact-effective-response-20260924-v1/README.md)包含输入/协议/代码/误差/SHA及诊断限制。使用原60Hz第972帧、120Hz第1160帧失败快照仅作独立机构诊断，不能改schema继续正式挖掘。1160末帧完整MPM外载未保存，测量明确是零外载上下文；972另有保持上一MPM外载对照。原示范、源、生产物理、保护与依赖未改。

D-150已证实全程120Hz完整循环比60Hz慢54.9%，后段半帧振荡仍在；原中间360秒预算导致1160帧停止，未达到1200帧门，不为补40帧自动重跑。详细记录继续保留。当前无物理/诊断/训练任务。

用户整体目标仍是今晚连续三铲、本机睡觉期间RL最多8小时、权重本地保存。当前无合格三铲基线或RL训练入口，先解决物理响应并过基线/环境短检，不能根据时钟直接启动训练。

## 1. 位置、版本与第一步

唯一开发仓库`/home/qunwang/wajueji`，分支`codex/contact-bc-collection-20260924`；不使用worktree。本轮接手`1a3f3f8`，诊断/候选实现提交`a69d506`，预算门及阶段证据提交`cfec8b8`，后续收口提交不改变生产源码。当前HEAD应现场核对，不能按旧聊天猜。

```bash
cd /home/qunwang/wajueji
pwd
git rev-parse --show-toplevel
git branch --show-current
git log -3 --oneline
git status --short --untracked-files=no
ps -eo pid,etime,args | rg 'run_contact_geometric_probe|run_layered_47|view_contact_bc_probe|train_contact|mixed-coupled-bridge'
```

先报告目录、分支/HEAD、跟踪差异与实际进程，再按AGENTS完整阅读总体规划、状态、要求、注册表和七个active小节及相关验收/决策。最后核对本页。收口时本机无项目物理、训练、回放进程；双A30未查询或部署。大量既存未跟踪产物保留，不能`git add .`、clean、reset、stash或自动切分支。

当前允许范围包含D-152隔离原型及有条件场景（见顶部授权），以及只读核验与证据整理：`docs/CURRENT_STATUS.md`、本页、相关验收/决策/S03/S04/S05及下列工作包。D-152范围内的物理候选已有授权；超出该范围或切生产默认时再讨论明确方案、预算与暂停点。原始示范、源快照、冻结结果/协议、模型、材料及全部保护不可覆盖或改判。

## 2. 已完成及失败边界

- v7的1500帧只读审核确认423/711/824/981指令跳变，原投影全程未改写。980实际尖峰先于981保护，不能只归因于阶段或BC。
- v8只在原安全求解内加入显式2/s命令变化率；安全优先，变化率不可行时记录例外。默认不启用该候选。900帧切深超调50.92mm，不晋级。
- v9仅将平移上限0.22降为0.15m/s；900帧早段速度改善、无超深，但尚未收斗，原行为门false。没有将它改判通过。
- 同控制v9-handoff显式1200中间窗：返回972帧、201.46秒，435确认触土，959确认收斗，960起BC；第973决策原地面预览拒绝，未发令/未新增物理步。最低CAD29.58mm低于30mm预览余量，切深超调55.07mm超过30mm行为门。返回帧原硬门/材料/扫掠/实际5mm地面门通过；972控制重算与同控制120前缀精确。审核通过不代表行为通过。
- 954–959已出现实际交替，先于BC；965帧铲斗缸命令27.3mm/s而末态速度413.7mm/s。接触/载荷相关尚未做4800Hz子步因果隔离；主动助力归零后仍持续。禁止据此直接归罪/关闭某一组件。
- 第一/第二铲各自零步全24数组和120帧审核通过，仅入口/早段资格，无完整收斗、离土、有效满斗或余料保护结论。
- v7末态独立4cm六邻域连通及实体距离复核仍约0.45μm，未达3mm×5帧。离土、≥95%有效满斗和自主连续三铲均未验证。

证据入口：

| 内容 | 本地工作包（相对仓库） |
| --- | --- |
| v7命令/实际响应及离土只读复核、RL预算 | `work/contact-speed-audit-20260924-v1/` |
| v8及v9原900窗 | `work/contact-geometric-feedback-20260924-v8/`、`work/contact-geometric-feedback-20260924-v9/` |
| 972帧失败审核与950–972诊断 | `work/contact-geometric-feedback-20260924-v9-handoff/` |
| 第一铲初始化来源、原失败及新保存点 | `work/contact-multisource-validation-20260924-v1/` |
| 第一/第二铲零步和120帧 | `work/contact-first-source-20260924-v1/`、`work/contact-second-source-20260924-v1/` |

## 3. 输入、SHA及恢复边界

| 用途 | 仓库相对路径 | SHA256 |
| --- | --- | --- |
| 第三铲开发源 | `runs/fast-contact-collection/20260924-115633-1457317/prepare/prepoint.npz` | `a6fed857a335c95b6fb6222530b33fe73267841d77e527a712731d5edd0d896b` |
| 第二铲真实挖后源 | `runs/fast-contact-collection/20260924-114750-1451324/prepare/prepoint.npz` | `a88efb6fce3ac9e4cba54593c41f09c99eb1061aa3d16730ea1ba4ba2cc2ec46` |
| 第一铲原APPROACH源 | `runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089/approach-handoff.npz` | `6344e74f809c8b5323942b211f5d62cccb6356da87bf8c85698910e1f6dfb3f4` |
| 第一铲新完整保存点 | `runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz` | `700f76be970c5754bde905837faf642cca0317fef097e31d17cf48f008ba0545` |
| 阶段BC | `runs/contact-phase-bc-pilot-v1/pilot.pt` | `b4bbc98cf472bcad23f89e014a025cfc933c36091ada91ced4c4cc9680d94095` |
| v9-handoff故障物理态 | `runs/contact-geometric-probe-20260924-v9-handoff/contact1200/physics/failure.npz` | `54ef3a70042909f81357565c33591e642583e222ed036a82362457d5a6f92ce1` |

第一铲新点不是把旧schema改名：按原手操冷初始化后只复现原第1帧零命令，机构/控制6字段和47石姿态/速度与原记录误差0，再保存完整运行态。22个微量MPM接触样本与原示范一致（约7e−8N），`strict_zero_sample_qualified=false`、`full_G2_qualified=false`；历史native状态缺失标记保留，原零步时钟失败、严格零样本失败不改判。新点独立恢复的24数组精确，不等于原APPROACH有完整native历史。

第二/三铲保留`training_reset`和库存/lineage，必须显式选择该恢复域，不冒充同实例连续三铲。未来有新授权时仍直接恢复这些保存点，不从起点慢移。故障物理态虽然对应最后有效帧，但缺策略短历史、阶段器、反馈锁存，不能精确策略续跑；v7正常final也有该限制。

## 4. 环境、验证与后续暂停点

物理Python为`/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python`；推理Python为`/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python`。外部几何依赖在`/home/qunwang/codex-work/small-soil-20260910-v17-performance`，斗腔标定在`/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json`。`runs/`、模型和大依赖只在本机，Git仅代码/协议/精选报告。异机先核对资产，不从文件名猜测或重建示范。

当前生产实现63项定向测试通过，覆盖几何反馈、安全优先变化率、阶段/BC、地面保护和900/1200预算门。可做的便宜核验：

```bash
cd /home/qunwang/wajueji
python3 scripts/check_docs.py
git diff --check
```

任何工作包`run.sh`、`prepare_first*.py`都会生成物理或产物，**暂停期间不得执行**；旧协议绑定旧源码，不得直接重跑覆盖。只读审核也需确认脚本不会推进物理，并保留原报告；qpos轴地址`[7,9,12,15]`，qvel为`[6,8,11,14]`。

后续待讨论方案为有界执行器/接触响应诊断，未授权实施。应先明确触发窗口、子步日志成本、最小可隔离对照、源SHA、输出、失败保留与暂停点，再决定是否改底层。原地面/超深/限位/材料/扫掠门不放宽；门未过不晋级完整窗。满斗必须先离土，再按原0.2占据阈值含土石/堆高口径测量，不按视觉放行。

RL仍为后续候选：v7同负载0.2306秒/步粗估10万步6.4小时、100万步64小时，未含额外重置/训练/评价，A30未实测，无双卡线性加速假设。长训练、远端与批量试验须先讨论预算，当前没有后台任务等待接续。
