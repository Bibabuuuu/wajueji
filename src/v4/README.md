# v4运行源码

`layered_mound_checkpoint.py`提供schema v2分层检查点的严格只读加载、完整字段/元数据/数量/有限值核验、恢复读回和原子写新检查点；除公开粒子/MPM/刚体状态外，它还保存固定网格构造基准、网格身份、Newton网格热启动场和刚体历史，并绑定Newton/Warp与求解模式。源文件在加载前后都以大小、mtime和SHA256确认未变化，旧schema v1会被明确拒绝。`layered_mound_checkpoint_runtime.py`负责用原始网格基准创建fixture、显式`qpos/qvel`初始化MuJoCo、恢复Newton MPM及solver续步状态并核算一步/短窗连续性。

`run_layered_mound_fortyseven_rock_four_layer.py`只接受未来A30生成并逐哈希绑定的schema v2三层检查点：先精确恢复33石并通过连续性，再通过Newton非同构网格场接口把热启动转入47体fixture，按恢复位姿为upper 14石重算SAT和最小竖向避让，逐石短压入、激活upper cover、释放47石并额外保持2秒；全部通过后才另存47石检查点。`run_layered_mound_checkpoint_hold_audit.py`是GPU1只读旁路，不增加实体、不写检查点。旧`d30db1bd...f3f545`只保留为schema v1物理快照；新schema v2 A30检查点返回前，两条下游配置尚未完成最终输入绑定，不得运行。

`mpm_rigid_free_velocity.py`提供自由石块的本步重力预测：只给MPM代理使用`v_n+g·dt`，MuJoCo实际状态不修改。`run_mpm_rigid_free_velocity_probe.py`绑定冻结的一/三石夹具，提供开/关对照、逐步反力与原始刚体线动量核算及额外同状态2秒稳定复验。一石A/B与三石覆土均证明该修正消除此前`g·dt`持续下沉；保持/规定运动阶段不应用自由速度预测，关节机构和密集刚体接触尚不在本次资格内。

`run_layered_mound_nine_rock_lower_layer.py`复用冻结的覆土施工顺序，把自由速度预测扩大到五层清单的完整底层9石。它逐石短压入、恢复休眠cover历史、沉降覆土、九石同时释放，并在连续1秒稳定后继续同一状态2秒；逐步核对原始MuJoCo动量、MPM反力和刚性接触支撑。当前128881粒子GPU运行全部检查通过，但不授权第二层、65石全场或挖取。

`run_layered_mound_twenty_rock_two_layer.py`把同一构造扩到`lower+lower_middle`前两层20石：后层碰撞体先停放到无土位置，两段cover依次恢复历史和沉降，最终20石同时自由释放并额外保持2秒。它在运行前重算全部190对SAT，并用全局动量核对容纳允许的石—石内力；当前180554粒子运行通过，仍不授权第三层或完整场景。

`run_layered_mound_thirtythree_rock_three_layer.py`继续增加`middle`13石和第三段cover，总计33石/221882粒子；当前三层联合沉降、连续1秒释放稳定和额外2秒保持均通过，但不授权第四层或65石全场。

`run_layered_mound_three_layer_checkpoint.py`只在上述三层源报告真实通过后保存schema v2精确续跑状态；源报告先完成、桥正常关闭后才原子写NPZ，避免失败运行留下看似可续跑的半检查点。临时MuJoCo模型可通过`EXCAVATOR_MUJOCO_MESH_DIR`与`EXCAVATOR_MUJOCO_TEXTURE_DIR`重绑离线网格/纹理；生成器同时把冻结XML中的仓库资产绝对路径改到当前仓库，并在加载前验证每个外部文件。A30包必须先做粒子构造、真实模型生成、桥编译和READY门，再进入昂贵MPM重放。

`build_mixed_material_capacity_preflight.py`与`run_mixed_material_capacity_preflight.py`派生动态N石桥并完成`64/256/512`时间零容量预检；逐石名称使用启动时冻结的哈希索引。`run_mixed_material_capacity_upper_bound.py`复用同一桥，把旧887石只作为D-063咨询上界完成时间零结构预检，不恢复正式数量声明。`build_fullscale_graded_gravel_slope.py`保留D-060精确配方与“重叠格点＋局部SAT松弛”失败回归，失败时不carve、不输出半成品；下一实现使用独立的无穿插稀松初态＋自适应分批沉降。

`build_formal_scene_model_adapter.py`消费队友的视觉稳定场景清单，逐石复算四种冻结资产的体积、质量和机体系惯量，拒绝重复源ID、错误资产、非单位姿态、非零冻结速度或缺失稳定检查，再映射为动态桥要求的连续运行时ID并生成实际N模型。合成三类石清单已完成模型生成、桥编译、MuJoCo加载和时间零STATE；尚未消费真实正式清单或执行物理步。

`build_formal_scene_mpm_fill.py`消费同一已验证清单，从冻结原土体只carve真实凸石内部的粒子中心，确定性写出保留`q/material_id/original_index`和`removed_index`，同时报告实际MPM/刚石/合计体积、质量和逐石删除点数。它不为命中参考比例额外删点，也不推进Newton/Warp；合成清单双跑已逐字节一致并通过零保留中心落石检查。

`prepare_formal_scene_geometry.py`是两者的薄编排层：一次校验同一清单，生成实际N模型、模型清单、MPM填隙资产和交叉哈希报告，全部成功后才原子发布目标目录。它不运行动态桥、MuJoCo步进或Newton/Warp，物理资格仍由后续独立门给出。

`run_formal_scene_time_zero.py`消费上述原子几何包，逐文件复核哈希和清单/实际N交叉绑定，确定性生成并编译动态N桥，只执行READY与`init_default`时间零STATE。报告直接符合全尺寸独立审计器的时间零输入schema；它核对有限值、ID顺序、维度、零外力和零禁止接触采样，明确记录`mj_step_calls=0`、`mpm_or_gpu_steps=0`，不授予物理、全尺寸场景或P1a资格。

`build_layered_mixed_scene_local_preflight.py`实现D-068缩比三层联合沉积的确定性几何入口。它用每块真实变换凸网格的竖向跨度反算中心高度和嵌入比例，检查石—石SAT与局部边界，只删除石内MPM粒子中心，并为保留粒子写入四阶段激活标签。输出22石清单、局部MPM资产、指标和诚实范围报告；没有Newton/Warp/MuJoCo导入或物理资格声明。

`audit_layered_mound_preview_geometry.py`用真实凸网格审计用户通过的五层65石视觉坐标，保留相交失败而不发布物理清单。`build_layered_mound_physical_preflight.py`保持ID、五层数量、尺寸、形状、朝向和材质，按各层土堆横截面确定性重排XY位置，通过SAT、边界、精确carve和六阶段激活门后原子发布时间零输入；两者均执行零物理步。

`run_layered_mound_time_zero.py`把上述重排清单实际装入动态N MuJoCo桥和Newton MPM，核对65个稳定ID/碰撞体、263673粒子、有限零速状态与零外力，但执行0次求解步。`run_layered_mound_single_rock_press_smoke.py`只验证下一项施工原语：它重新从冻结源bank选择未预挖孔的foundation土体，短暂预沉降后，以公开`forward`碰撞速度把`lower-000`锁定姿态低速压入2cm并采集接触反力。该脚本不会把最终组装空腔bank冒充挤土过程，也不授予全深度、多石、释放稳定或完整场景资格。

`run_layered_mound_three_rock_adaptive_press.py`把单石门扩到三块不等短行程，压入时逐块保持，之后用动态N MuJoCo桥同时解除三块约束并把Newton MPM逐石反力回传。报告允许并量化释放后的平移与旋转，只以连续低速窗判稳定。首轮裸露foundation候选全程有限但持续竖向下沉，诚实输出失败；该脚本不自动追加覆土、延长到触底或改变稳定阈值。

`run_layered_mound_three_rock_cover_release.py`验证“保持石块—激活覆土—覆土稳定—释放”的原始施工顺序；`run_layered_mound_three_rock_slow_release.py`在不改材料、阻尼和阈值的前提下加入60帧重力渐变；`run_layered_mound_three_rock_arrest_release.py`只在渐变结束时施加一次按实测线速度计算的施工制动力，之后不再控制。三者共同确认覆土可稳定，但当前外步耦合会留下约一个`g·dt`的刚体速度偏置；最终脚本失败后按预注册止损停止场景级补偿。

`run_layered_mound_one_rock_free_deposition.py`让基础土先沉降、再从5mm表面间隙加入首帧自由石头；其首版报告暴露预沉降期仍有接触核样本。`..._v2.py`不复制动力学实现，只在夹具初始化和60个土壤预沉降步把石碰撞体停放到远处，随后恢复原零速自由加入流程。v2严格证明土先独立沉降仍不能消除跨求解器`g·dt`残余。

`run_layered_mixed_scene_local_dynamics_smoke.py`从同一22石清单派生临时实际N MuJoCo模型，复用动态容量桥并在Newton MPM中按稳定ID回传逐石力矩。首轮只把最终72415粒子和22石同时激活0.1秒，验证有限值、非零接触、身份路由、闭链和禁止接触；它不实现或冒充D-068要求的逐层沉降与1秒撤墙名义保持。

`audit_layered_mixed_scene_staged_settling.py`是与队友执行器分离的纯报告审计器。它从冻结输入和预注册阈值重新推导四阶段及名义保持资格，拒绝数量/哈希漂移、未激活侧接触、阈值破门后仍声明通过或把缩比结果上升为全尺寸/三铲资格；诚实失败会被保留而不是误判为格式错误。

`audit_fullscale_layered_mixed_scene_qualification.py`是P1a前的第二层独立审计器。它读取候选绑定的稳定清单、原子几何包、实际N时间零和全尺寸沉降报告，自己核对文件哈希、实际N、状态/检查一致性，并重新形成稳定清单、几何交叉绑定、时间零、分层沉降、名义保持、画面、MPM连通性和身份质量八项结果。执行器伪报通过会被拒绝，诚实物理失败会保留；审计器本身不执行物理或GPU。

`prepare_mixed_material_p1a_handoff.py`是缩比沉降之后的离线交接适配器。它拒绝把缩比通过直接当作单铲授权，也不再直接信任全尺寸执行器候选；只有上述独立审计回执为`fullscale_scene_admitted_for_p1a`才生成P1a就绪收据。收据复用`mixed_three_cycle_state.py`、`mixed_terrain_observation.py`、既有选点和dispatch复核，不执行物理或GPU，也不声明三铲通过。

这里是2026-09-19整理出的四轴模仿学习唯一受版本管理源码。Python文件来自最新D071观察运行时；测试放在 `tests/v4/`，历史报告放在 `reports/v4/`，原始轨迹放在Git忽略的 `local-data/`。

当前源码仍保留历史绝对依赖路径，以维持原始审计哈希。路径迁移必须作为新的候选提交并重新预检，不能静默修改后继续引用旧D071资格。

`mixed_material_r0.py`是碎石专项的零物理几何构建器：核验实际铲斗与内腔契约哈希，生成闭合凸石块资产、24姿态斗口通规和确定性无初始穿插场景清单。它不导入MuJoCo、Newton或Warp，也不具备推进物理或启动训练的入口。

`mixed_material_r1.py`从冻结内腔截面生成11段斗壳、2块侧板和5个斗齿的开口凸碰撞代理，并写出独立MuJoCo探针夹具。`mixed_material_r1_probe.cc`只测试石块能否从斗口外进入并接触斗壳；`run_mixed_material_r1_probes.py`把探针临时编译到系统临时目录并记录结果。

`mixed_material_r1_mechanism.py`在临时目录从冻结机构XML生成一石或两石候选，禁用原整斗网格的碰撞但保留视觉，在同一bucket body挂入18个代理，并显式写入R0质量和完整惯量。`mixed_material_r1_model_inspect.cc`执行只加载契约检查；`mixed_rigid_bridge.cc`是隔离的新`26/24/12`或`33/30/12`桥，输出原机构状态、逐石稳定ID/状态、分类接触及石块—铲斗力系字段，旧`19/18/12`桥不修改。`run_mixed_rigid_bridge_smoke.py`支持配置冻结的恒定或分阶段命令，现已通过自由落体、斜面、互撞、推石以及实际凸石块全部顶点的真实机构入口、举升后10秒保持、回转和固定接收箱卸料；`compare_mixed_rigid_bucket_push.py`比较接触/无接触报告。

`build_scene_reconstruction.py`把显式合成参数编译为3类×3固定种子的稳定清单、实测指标和SVG预览；放置采用实际旋转网格表面支撑与凸体SAT窄相检查。输出是纯几何候选，不导入或运行物理。

`build_scene_assembly_preflight.py`离线复核9个场景的逐石资产、质量、惯量、稳定命名与地址，并输出全场装配计划。`mixed_material_r2_mechanism.py`消费首批6个20石计划，复用R1的18个开口代理生成隔离MJCF，同时从已审核场景清单生成`65×65`浮点高度场并验证解析曲面插值误差；`mixed_material_r2_model_inspect.cc`只调用`mj_loadXML`检查`159/138/12`结构、名称、地址、碰撞位、质量和高度场；`run_mixed_material_r2_load.py`编译一次检查器并完成六场景回归。`build_mixed_rigid_bridge_r2.py`核对R1桥哈希后确定性生成支持实际`1–20`石的R2源码，不修改R1文件，并把`r2_surface_`接触归为地面而非坡道；`run_mixed_material_r2_bridge_init.py`用同一生成源码逐场完成时间0初始化和20条状态输出；`run_mixed_material_r2_settling.py`执行六场各1秒零命令烟测并汇总身份、闭链、接触分类以及水平/垂直位移。

`run_mixed_material_r2_stabilization.py`在不覆盖源清单的前提下执行预沉降、保存逐石稳定位置/四元数、以零速度回烘新模型，并做1秒独立复验及边界/解析表面审计。`select_sparse_large_stable_candidates.py`按冻结种子序列调用同一几何和稳定化门，保留失败证据并只选足两个替代场景。最终v2六场全部保留`0.08m`边界，回烘后1秒最大位移`0.152–0.216mm`；该状态只允许下一步受控多石进斗分项，不代表满斗率或混合物理通过。

`run_mixed_material_r2_controlled_intake.py`和`run_mixed_material_r2_intake_lift.py`保留两场低速进斗及旧严格包络短举升证据。`instrument_r2_bridge_contacts.py`从冻结桥确定性派生只增加逐石地面/斗接触计数的观测版，不改动力学；`run_mixed_material_r2_effective_load.py`用持续离地、放宽斗邻域、相对运动和末速度判定稳定携带，并以逐石清单实体体积除以固定斗容输出首版满斗率。该结果仍不是空隙补偿、视觉堆高、10秒保持或卸料资格。

`run_mixed_material_r2_single_scene_load_development.py`保留单普通场负Y预回转的v1失败证据；`..._v2.py`继承并哈希核对v1输入，只延长0.5秒低速举升，输出3块有效携带候选。两个执行器都明确关闭参数网格、第二场、六场、卸料、MPM和训练资格。

`mixed_material_r2_manual_viewer.cc`是纯显示/键盘输入镜像，不计算物理；`run_mixed_material_r2_manual_session.py`把它与冻结R2桥配对，只在普通`26091912`执行人工四轴操作。入口以120Hz保存物理测量、20Hz保存完整状态—动作对，并用F3/F4分别绑定举升起点和最终静止评估窗。原始样本固定为`training_eligible=false`；只有人工会话复核通过后，才可另行进入行为克隆数据筛选。smoke只验证窗口、桥、协议和记录链，不宣称人工装载成功。

`build_mixed_coupled_bridge_r3_a.py`和`run_mixed_material_r3_a_probe.py`从冻结R1逻辑派生单石—小型MPM基质分体反力闭环。`build_mixed_coupled_bridge_r3_b.py`再从冻结R3-A输出派生铲斗双向入口，把刚石冲量矩和外部MPM力矩统一到每个外步开始时的固定铲斗原点；`run_mixed_material_r3_b_probe.py`用Newton `backward`碰撞体速度、互斥collider所有权以及clear/MPM-only/rigid-only/combined四工况记录本步施加和本步实测的MPM/刚石/合计载荷。R3-B只通过单石小夹具记账与合计响应，不授权多石、官方包络或训练。

`build_mixed_multirock_candidate_r3_c.py`和`build_mixed_coupled_bridge_r3_c.py`从冻结R1/R3-B输入派生恰好两石的`33/30/12`小夹具与三路命名外力系桥；`run_mixed_material_r3_c_bridge_preflight.py`只重复生成、编译并读取时间零协议。R3-C当前16/16项结构检查通过，但执行零物理步；下一步才实现稳定ID到Newton collider的动态映射、两条单石路由隔离和双石四工况，不授权20石、官方包络或训练。

`mixed_material_r3_c_routing_contract.py`冻结下一小门的范围；`run_mixed_material_r3_c_routing_probe.py`分别只为一块目标石头开启Newton粒子碰撞，从稳定ID映射body index并反查collider集合，再经R3-C命名力系桥回传。两条路由13/13项通过但铲斗MPM关闭、基质彼此隔离；整体R3-C仍未通过，下一步是共享基质的双石四工况。

`build_mixed_coupled_bridge_r3_c_dynamics.py`从冻结R3-C基础桥派生观测协议：逐石MPM载荷增加显式参考点回显，刚体石—铲斗冲量按稳定石块ID分拆，同时继续保留聚合量。`mixed_material_r3_c_shared_contract.py`和`run_mixed_material_r3_c_shared_probe.py`在同一共享基质运行clear/MPM-only/rigid-only/combined；正式15/15项通过并授予`shared_four_round_qualified=true`。当前仅单次`1/120s`，仍不授权整体R3-C、20石、官方包络或训练。

`mixed_material_r3_c_convergence_contract.py`和`run_mixed_material_r3_c_convergence_probe.py`在shared门之后运行粗步两次、无反馈对照和细步两次。粗步重复、反馈因果和全部健康检查通过；细步重复及粗细步长门失败，报告保持`r3_c_physics_qualified=false`。该失败证据用于下一步定位接触时序/外步耦合分叉，不授权放宽阈值。

`mixed_three_cycle_state.py`、`mixed_terrain_observation.py`和`mixed_material_three_cycle_contract.py`构成连续三铲P0离线契约：状态适配器统一19/18维机构、MPM历史、逐石稳定ID、斗/车库存和四区账本；融合观测从保存的MPM/刚石表面采样重建网格并绑定当前状态；三铲契约重放冻结选点和dispatch复核，检查同实例边界、逐铲新地形、全序列质量/ID、阶段状态链以及获取—斗内—车内转移。当前只允许`synthetic_contract_fixture`且`physics_executed=false`，不代表代表场景或正式三铲物理通过。

本地快速测试：

```bash
PYTHONPATH=src/v4 python3 -m unittest discover -s tests/v4 -p "test_*.py" -v
```
