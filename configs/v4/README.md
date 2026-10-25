# v4配置

`layered-mound-thirtythree-rock-three-layer-v1.json`当前由新`a30-layered-checkpoint-v3`包用于重放33石并生成schema v2精确续跑检查点。旧schema v1检查点`d30db1bd...f3f545`已经通过文件与公开数组完整性核验，但首步连续性失败，只保留为物理快照。`layered-mound-checkpoint-hold-audit-v1.json`与`layered-mound-fortyseven-rock-four-layer-v1.json`中的旧输入绑定在新A30检查点返回前视为待替换，不能直接运行；届时必须同时绑定新NPZ SHA、源配置/执行器SHA和Newton/Warp版本。

`mpm-rigid-free-velocity-probe.json`绑定土先沉降单石v2与三石覆土v1，只修改自由石块传给MPM的速度预测，保持可挖MPM材料及所有稳定阈值。命令行`--predictor off/on`用于A/B，`--verification-hold-frames 240`追加同状态无控制2秒核验。当前一/三石稳定通过，未授予65石全场或挖掘资格。

`layered-mound-nine-rock-lower-layer-v1.json`冻结完整底层9石的`7–16mm`不等短压入、128881个可变形MPM粒子、连续120帧释放稳定门和随后240帧同状态无控制保持。它不改变三石已验证材料、时间步、阻尼或稳定阈值；当前运行通过，只授予完整底层静载资格，不授予第二层、65石全场或挖取资格。

`layered-mound-twenty-rock-two-layer-v1.json`冻结前两层20石、三段共180554个MPM粒子和第二层为消除真实穿插所需的`0–82mm`竖向避让。两段cover分别沉降后才释放全部刚体，稳定门仍为连续120帧加240帧额外保持；当前运行通过，只授予前两层静载资格。

`layered-mound-thirtythree-rock-three-layer-v1.json`冻结三层33石、四段共221882个MPM粒子及第三层3石的最小竖向避让；当前运行通过连续120帧稳定及240帧保持，只授予三层静载资格。

`mixed-material-capacity-preflight-v1.json`冻结正式场景到来前的`64/256/512`时间零容量阶梯；三档均已通过模型生成、动态桥编译、READY/STATE。`mixed-material-capacity-upper-bound-v1.json`把旧887石数量明确降为D-063咨询上界，并已通过同样的时间零结构容量门；两者均未执行物理。`fullscale-graded-gravel-slope-v1.json`保留D-060精确配方及首版局部SAT装填失败回归，不再代表D-063正式数量或体积硬门。

`formal-scene-model-adapter-v1.json`冻结D-063视觉稳定清单到实际N运行时模型的最低接口：稳定性/外观六项真值、三档粒径、四资产哈希、质量惯量复算、最多1024石，以及源ID到连续`r1-rock-*`运行时ID映射。它只授权清单验证与时间零模型生成，不授权物理步、沉降或正式场景资格。

`formal-scene-mpm-fill-v1.json`冻结同一视觉稳定清单到MPM填隙资产的离线几何边界：绑定原390400点银行及清单适配器，只删除真实凸石内部的粒子中心，禁止为追参考体积额外删点；`6.1m³`和`5.2–7.0m³`仅记录参考/宽护栏。该配置不授权运行MPM、沉降或混合物理。

`formal-scene-geometry-preparation-v1.json`把上述实际N模型适配器和MPM填隙配置逐哈希绑定为一个原子离线交付包；两阶段必须消费同一稳定清单并对齐石块数，任一阶段失败都不发布半成品。该编排仍不授权时间零、物理步、MPM运行或正式资格。

`formal-scene-time-zero-v1.json`冻结原子几何包的实际N加载入口：复核清单、几何包、模型清单和全部包内文件哈希，复用动态N桥完成MuJoCo READY与时间零STATE，并要求实际N、有限状态、零外力和零禁止接触采样。该门固定执行0次`mj_step`、0次MPM/GPU步，不做沉降，也不授权全尺寸场景或P1a。

`layered-mixed-scene-local-preflight-v1.json`冻结D-068三层联合沉积的本地缩比几何门：下层2大+2中石按真实网格高度嵌入45–60%，中层6块小石嵌入35–50%，顶层12块小石嵌入20–35%；原土粒按foundation/lower/middle/top四阶段激活，并且只carve真实石体内部中心。该配置明确执行零物理步，不把几何嵌入写成稳定性通过。

`layered-mound-viewer-preview-v1.json`冻结用户通过的五层65石视觉基线；`layered-mound-physical-preflight-v1.json`逐哈希消费该视觉配置，把每层簇状目标映射到对应高度的土堆横截面，并用真实凸网格拒绝相交候选。它要求至少1mm SAT间隙、只carve真实石内粒子中心，执行零物理步；视觉基线原坐标的61对相交由独立审计诚实保留，只有重排后的清单获准进入时间零。`layered-mound-time-zero-v1.json`绑定实际65石/263673粒子的零步装载门；`layered-mound-single-rock-press-smoke-v1.json`再绑定时间零回执，但从未carve的源土体选择85620个foundation粒子，只授权`lower-000`的24帧预沉降＋36帧低速2cm压入，不授权多石、全深度或稳定性。

`layered-mound-three-rock-adaptive-press-v1.json`按用户修正后的准入语义冻结三块底层石的`12/8/16mm`不等短压入，并允许释放后侧移、滚动和石—石接触；它只要求有限、范围内和连续0.5秒低速稳定窗，不要求回到预览姿态。首轮裸露foundation释放2秒未达到稳定窗，配置与失败报告保留为“覆土前释放支撑不足”证据，不授权扩到9石整层。

`layered-mound-three-rock-cover-release-v1.json`保持三块石受控期间激活并沉降覆土；`...-slow-release-v1.json`只把完全释放改为60帧重力渐变；`...-arrest-release-v1.json`再只允许渐变结束后的一个外步线性制动，随后要求至少120帧完全无控制稳定。覆土阶段通过，但三种释放都保留约`g/120`的共同向下速度，最终候选诚实失败。该链只用于锁定显式耦合静载交接问题，不授权更多石块，也不允许继续用场景参数或重复反馈掩盖失败。

`layered-mound-one-rock-free-deposition-v1.json`首次验证土先沉降、石头随后从首个动力学帧自由受力，但5mm间隙碰撞体在预沉降期已有531个接触样本，不能单独回答用户假设。`...-v2.json`逐哈希绑定该失败并只把碰撞体在60帧土壤预沉降时停放到远处，得到严格0预接触；随后单石自然接触3秒仍以`0.0818016m/s`下移，故诚实失败且禁止扩到三石。

`layered-mixed-scene-local-dynamics-smoke-v1.json`逐哈希绑定上述22石清单、72415粒子资产和动态N桥，只执行最终组装态0.1秒双向接触烟测。它故意关闭逐层激活、重力渐增和构造墙，因此只能发现立即非有限/爆炸或路由错误；即使通过也保持`settling_qualified=false`和`wall_free_hold_qualified=false`。

`layered-mixed-scene-staged-settling-audit-v1.json`是主线独立准入合同：在执行器首轮运行前冻结同一几何输入哈希、四阶段累计粒子/石块数量、逐层6秒收敛窗和最终1秒名义保持阈值，并要求执行配置/源码/石映射哈希、七类MPM激活历史、未激活MPM三路零计数、构造控制和top-N异常证据。它只审核队友候选报告并保留诚实失败；即使全部通过也只授权把同一构造方法放大全尺寸，不直接授权P1a单铲，也不授予全尺寸或三铲资格。

`fullscale-layered-mixed-scene-qualification-audit-v1.json`冻结P1a前的独立全尺寸资格门。候选执行器必须分别提交稳定清单、原子几何包、实际N时间零和全尺寸分层沉降报告；审计器从这些逐哈希绑定的原始报告重算八项资格，拒绝执行器自报布尔值与证据不一致、路径换绑或把失败写成通过。诚实失败保留；只有独立状态`fullscale_scene_admitted_for_p1a`可以向下游授权单铲。

`mixed-material-p1a-handoff-v1.json`冻结全尺寸场景到既有连续三铲模块的离线接入链。缩比审计只作为方法门；它现在只接受上述独立全尺寸审计回执，而不直接信任执行器候选报告。八项资格全部通过后，场景才可绑定到canonical state、融合观测、既有选点器、dispatch几何复核和同实例三铲循环。配置本身不运行物理/GPU，也不授予三铲资格。

`actor-normalization.json` 是当前行为克隆策略的归一化参数。历史启动命令和D071导出记录位于 `reports/v4/`，因为其中的绝对路径只描述旧运行，不能直接当作新环境配置。

`contact-checkpoint-contract-v1.json`冻结下一版接触策略的元数据契约：完整标签与三个动作头分离，旧四头权重、未reviewed标签和缺少独立运行准入回执的checkpoint都会被拒绝。其状态`frozen_review_blocked`表示契约已确定但当前没有训练资格，不是可训练或可部署配置。

`mixed-material-r0.json`冻结可铲运碎石专项R0的铲斗源网格哈希、坐标修正、内腔开口、石块密度假设、通规尺寸和三类确定性场景种子。密度是显式仿真假设，不是官方石料真值；R0只生成几何与资产契约，不启动物理或训练。

`mixed-material-r1.json`定义开口分段凸碰撞代理、短时探针、大石稀疏策略，以及一石`26/24/12`或两石`33/30/12`候选的源哈希、接触位和冻结单元场景。除加载、自由落体、斜面、互撞和推石外，分阶段命令的真实机构单石入口和举升后10秒保持也已通过。常规场景至少80%石块不超过`0.16m`，`0.22–0.27m`每场最多1块且不超过刚体石料等效实体积的20%，`0.285m`以上仅用于卡滞边界。当前不授予整石包络、多石携带、卸料或MPM混合物理资格。

`scene-reconstruction-v1.json`定义纯几何重建区域、合成表面、粒径/原型分布、聚集、净空和3类×3固定种子。输出已确定性生成并通过SAT无穿插及质量惯量复算；所有来源仍标记为合成假设，不授予物理或训练资格。

`scene-assembly-preflight-v1.json`把9个场景逐石绑定到R0资产、稳定名称、质量/惯量和未来自由关节地址，并生成现有一石/两石回归批次。批次不能替代全场同时物理。

`mixed-material-r2.json`冻结一个最大容量20的R2候选口径：模型加载时实际石块数固定且不超过20，不允许占位石块或运行时增删。首轮只准入6个20石普通/稀疏场景，3个13石卡滞场景延期；配置还把已审核场景清单的解析曲面确定性采样为`65×65`浮点高度场，并绑定源清单、资产哈希和`0.15mm`最大插值误差门槛。当前授权MJCF生成、`mj_loadXML`结构检查、从冻结R1源码确定性派生R2桥、时间0初始化，以及1秒零命令沉降烟测；不授权修改R1桥、受控挖掘、多石物理资格或训练。

`mixed-material-r2-stabilization.json`冻结预沉降回烘门槛：最长6秒、线速度`0.002m/s`、角速度`0.03rad/s`连续0.5秒；回烘模型另做1秒复验，最大位移门槛`5mm`。原清单/装配计划、石块数量和尺寸不得修改，原`0.08m`边界安全距必须保留。该配置不授权受控挖掘或训练。

`sparse-large-stable-admission-v1.json`冻结稀疏大石替换候选的种子顺序和“几何→预沉降→回烘→边界”准入链。失败候选必须保留，达到两个替代场景后停止，不得放宽`0.08m`门槛或手工移石。

`scene-reconstruction-v2.json`、`scene-assembly-preflight-v2.json`、`mixed-material-r2-v2.json`和`mixed-material-r2-stabilization-v2.json`构成最终六场哈希链：普通`26091911/12/13`、稀疏`26091922/25/26`。v1文件继续保留失败历史；v2配置不授权受控进斗、MPM或训练。

`mixed-material-r2-controlled-intake-v1.json`冻结普通`26091912`和稀疏`26091925`的低速接近—卷斗轨迹；`mixed-material-r2-intake-lift-v1.json`追加1.5秒举升和2秒保持并保留旧严格包络门反例。`mixed-material-r2-effective-load-v1.json`采用用户确认的“离土—随斗”口径，首版满斗率只计算稳定携带石头实体体积除以固定`0.255164729950989m³`斗容；不补偿空隙，不授权卸料、车厢、六场、MPM或训练。

`mixed-material-r2-single-scene-load-development-v1.json`只在普通`26091912`加入1.5秒高位负Y预回转，结果仍为2块并保留失败状态；`...-v2.json`哈希绑定v1失败证据，只把相同低速举升从1.5秒延长到2.0秒，禁止改有效载荷门槛、做参数网格或扩场景。

`mixed-material-r2-manual-session-v1.json`冻结普通`26091912`人工控制入口：J/L回转、I/K动臂、U/O斗杆、M/N铲斗，释放即零命令保持；F3标记举升开始，F4标记最终静止评估开始。候选至少须携带3块且实体体积满斗率严格高于冻结脚本`1.2045%`，并通过最终1秒零命令、闭链和禁止接触门。配置只授权人工可行性与同步记录，不授权训练、扩场景、卸料或MPM。

`mixed-material-r3-a.json`冻结单石—小型MPM基质分体反力闭环；`mixed-material-r3-b.json`在不改R3-A的前提下冻结铲斗固定外步力矩参考、Newton `backward`碰撞速度、四工况接触所有权和`1/120→1/240s`步长对照。R3-B只授权单石小夹具的MPM/刚石/合计载荷记账，不授权多石、官方包络、材料标定或训练。

`mixed-material-r3-c.json`冻结R3-C第一道结构门：恰好两块`0.12m angular_octa`、稳定ID顺序、`33/30/12`模型和铲斗＋逐石三个命名外力系。当前只授权确定性生成、C++编译和时间零协议；`gpu_mpm_step=false`、`physics_qualified=false`，不授权双石动力学、20石、官方包络、材料标定或训练。

`mixed-material-r3-c-routing-v1.json`独立冻结两条逐ID路由小门：每轮只为一个目标石块开启局部MPM碰撞，验证稳定ID→Newton body→collider及命名载荷回显。该门明确关闭铲斗MPM、四工况、反馈、重复和步长资格；通过后仍保持整体`r3_c_physics_qualified=false`。

`mixed-material-r3-c-shared-four-round-v1.json`冻结两石共享小基质的clear/MPM-only/rigid-only/combined四轮门、逐石/铲斗参考点与分项残差阈值。该配置只授权一次`1/120s`四轮接触分区，不授权重复、步长对照、完整进程内reset、整体R3-C、官方包络或训练。

`mixed-material-r3-c-convergence-v1.json`哈希绑定shared门，冻结粗/细外步为`1/120s`和`1/240s`、内部MuJoCo步长不变、各两次重建、无反馈对照及重复/步长阈值。正式结果未通过细步重复和粗细步长门，不能放宽配置换取资格。

`mixed-material-three-cycle-v1.json`是连续三铲P0契约候选：哈希绑定R3-C共享门及收敛失败、现有逐铲重感知三铲循环、选点器和装载车接收器。它只授权无GPU合成契约测试；代表场景仍为pending，`p1_execution_authorized=false`，不能据此宣称物理单铲或三铲通过。

未来的本机路径覆盖文件使用 `local-paths.env`，该文件已被Git忽略；在完成路径可移植性候选前不要伪造此文件已被现有启动器读取。
