# 当前状态

- 状态：`第二铲卸料与统一检查通过；第二铲120帧短验证通过；夜间搜索未启动；第一／第二铲双起点准备待接线`
- 最后复核：2026-09-25（Asia/Shanghai）
- 唯一仓库：`/home/qunwang/wajueji`；分支 `codex/contact-bc-collection-20260924`
- 权威范围：此刻进度、运行任务和下一道门；历史看验收、决策及Git。

## 当前主线实证

第一铲D159运输、卸料及返回4120帧/1273.792秒，37项独立审核通过，开发接收原斗腔土粒25371/25372、残留0。工作满斗45.53%；不是正式稳定混合体积或95%资格。全局8101帧返回点SHA `be05b2ef64e1f2fd877368883866b93d639ef4b5ecef745974019b1fc8dd31f0`。

第二铲真实预备点2619帧/816.881秒。严格零接触资格false保留：尾30帧MPM接触样本最多11、石接触0，按授权继续开发流程，不是正式G2。接触2725帧/907.961秒，补收439帧/161.04354秒，≥99%保持3秒且持续上升ready，正常handoff SHA `f750e55864d6189ff6cd149cc1574894c236a334e6817a4b88f66493fd33c84d`。随后旧v4规划报cycle_observation_grid_budget；失败原件保留，正常前缀独立审核通过，不重挖/重做补收。

D-162从正常handoff直接恢复，固定抬升→回转→卸料3398帧/1104.554693秒正常完成，未回位/第三铲。开发接收22233/22234原载荷土粒、斗内残留0；最终SHA `620b50d6d69fa2d6c036c8f500284265ffab1978cc4506df655fe7c69179f7a9`。独立三段审核59＋30＋38＝127项全部通过，控制复算误差均0；接触行为15项通过；6562帧回放SHA及两处状态/粒子边界精确一致。[统一检查](../work/contact-second-fixed-route-review-20260925-v1/README.md)。CLEAR_PILE末五帧工作装载中位数30.21405%，有效开发估计，未达95%。

当前无主线物理/夜间批次/神经训练；检查自动化automation-2已暂停，防止重复执行。用户要求今晚只完成第二铲卸料，三铲仍未验证。

## 120帧短验证结果

用户要求先结束审核及短检，因此新增`batch.py --check-only`：只评分和验证，不启动候选。12项CPU测试通过。v2最终三段证据已冻结；第二铲全1倍率短检120帧/43.213983秒，零步24数组全等、原生控制误差0、全部执行硬门通过；末态24数组与原gate-120全部精确一致，NPZ SHA同为 `97e4212b7eec410211383a577b2e3221242bfd0695c5315699fb6737c859cbdd`。

评分＋短检合计66.631579秒。`runs/night-parameter-search-20260925-v2/precheck.json`为passed；不存在batch.json，未试候选。后续同版本--run须验证manifest及检查证据SHA才能复用，扣除上述已用预算，不重新跑120帧；失败不自动重跑，已开始检查禁止prepare重绑。[短检证据](../work/night-parameter-search-20260925-v2/short-check-review.json)。

## 夜间8h：用户新增双起点要求

用户明确希望今晚同时使用第一、第二铲固定预备点。两源均有保存点和120帧参考，不需要重新接近。第一源 `runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz`，SHA `700f76be970c5754bde905837faf642cca0317fef097e31d17cf48f008ba0545`；第二源 `runs/contact-next-prepoint-20260925-v1/physical/prepoint.npz`，SHA `ab4f529501c3991022f6e72a276d4d7b48def9a888fcd85db3148356c73e420e`。

状态区分：双起点方向已获用户要求，具体运行包尚未实现/验证。当前v2绑定第二源，第一源为原cycle schema、cycle1、truck_enabled=false，现第二源专用加载器/尾段会拒绝，不能只替换路径。下一步独立接入第一源、保持来源/原生控制/碰撞身份，再做零步和120帧核验；不得改已冻结v2或将第二源短检冒充双源通过。

建议每组参数在两源成对比较，要求两源有效且不以明显牺牲一铲换另一铲改善；8h内减少参数组合数量，为最佳双源复验预留预算。具体候选表/双源基线对齐尚待收口，未启动新第一源物理。固定两源也不证明第三铲或任意地形泛化；同源复验只证明复现性。

仍为本机RTX4070有限参数搜索，只有近土预备点→挖掘→带载抬离参与优化；运输卸料回位固定。不更新BC神经权重，不部署A30。原v2单源方案为最多6组收斗<95%阶段意图倍率，普通试验止CLEAR_PILE、最佳复验接卸料；双源扩展另准备，不直接启动旧六组单源批次。

## 保持的边界

- 10mm累计回落仅FINISH_CURL；抬升/运输/回转/卸料不受该新增限制。
- 固定运输不依赖全场土粒地形图；远粒不追查、不删除，静态车体/地面/限位/保斗和实际物理保护保留；下一铲选点仍观察余土。
- 三个取料/晃动/疑似视觉穿模问题交专项任务`01a0d459-acca-70d3-b21f-a69d646ddece`充分讨论后处理；原图在evidence/user-first-scoop-review-20260925-v1。
- 原示范、BC权重、冻结失败和保护均保留。重复起点试验不冒充连续三铲；正式95%满斗、稳定混合卸车体积、连续第三铲和神经RL未验证。

## 权威入口

[总体规划](PROJECT_PLAN.md)、[要求](requirements.md)、[七个活动小节](plans/INDEX.md)、[验收](acceptance.md)、[决策](decisions.md)、[交接](handoff.md)。

- [S01-scene-physics-checkpoints.md](plans/S01-scene-physics-checkpoints.md)
- [S02-observation-target-selection.md](plans/S02-observation-target-selection.md)
- [S03-approach-control-handoff.md](plans/S03-approach-control-handoff.md)
- [S04-contact-excavation-demonstrations.md](plans/S04-contact-excavation-demonstrations.md)
- [S05-imitation-and-rl.md](plans/S05-imitation-and-rl.md)
- [S06-three-scoop-objective-evaluation.md](plans/S06-three-scoop-objective-evaluation.md)
- [S07-truck-reproduction-submission.md](plans/S07-truck-reproduction-submission.md)
