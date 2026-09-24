# 当前状态

- 状态：`第二铲卸料与审核通过；D-163双源夜间参数优化已启动；先跑完整基线；尚无候选改善结论`
- 最后复核：2026-09-25（Asia/Shanghai）
- 唯一仓库：`/home/qunwang/wajueji`；分支 `codex/contact-bc-collection-20260924`
- 权威范围：此刻进度、运行任务和下一道门；历史看验收、决策及Git。

## 当前主线实证

第一铲D159运输、卸料及返回4120帧/1273.792秒，37项独立审核通过，开发接收原斗腔土粒25371/25372、残留0。工作满斗45.53%；不是正式稳定混合体积或95%资格。全局8101帧返回点SHA `be05b2ef64e1f2fd877368883866b93d639ef4b5ecef745974019b1fc8dd31f0`。

第二铲真实预备点2619帧/816.881秒。严格零接触资格false保留：尾30帧MPM接触样本最多11、石接触0，按授权继续开发流程，不是正式G2。接触2725帧/907.961秒，补收439帧/161.04354秒，≥99%保持3秒且持续上升ready，正常handoff SHA `f750e55864d6189ff6cd149cc1574894c236a334e6817a4b88f66493fd33c84d`。随后旧v4规划报cycle_observation_grid_budget；失败原件保留，正常前缀独立审核通过，不重挖/重做补收。

D-162从正常handoff直接恢复，固定抬升→回转→卸料3398帧/1104.554693秒正常完成，未回位/第三铲。开发接收22233/22234原载荷土粒、斗内残留0；最终SHA `620b50d6d69fa2d6c036c8f500284265ffab1978cc4506df655fe7c69179f7a9`。独立三段审核59＋30＋38＝127项全部通过，控制复算误差均0；接触行为15项通过；6562帧回放SHA及两处状态/粒子边界精确一致。[统一检查](../work/contact-second-fixed-route-review-20260925-v1/README.md)。CLEAR_PILE末五帧工作装载中位数30.21405%，有效开发估计，未达95%。

连续三铲主线已停止；D-163独立双固定起点夜间优化正在运行，非神经训练。检查自动化automation-2已暂停，防止重复执行。用户要求今晚只完成第二铲卸料，三铲仍未验证。

## 120帧短验证结果

用户要求先结束审核及短检，因此新增`batch.py --check-only`：只评分和验证，不启动候选。12项CPU测试通过。v2最终三段证据已冻结；第二铲全1倍率短检120帧/43.213983秒，零步24数组全等、原生控制误差0、全部执行硬门通过；末态24数组与原gate-120全部精确一致，NPZ SHA同为 `97e4212b7eec410211383a577b2e3221242bfd0695c5315699fb6737c859cbdd`。

评分＋短检合计66.631579秒。`runs/night-parameter-search-20260925-v2/precheck.json`为passed；不存在batch.json，未试候选。后续同版本--run须验证manifest及检查证据SHA才能复用，扣除上述已用预算，不重新跑120帧；失败不自动重跑，已开始检查禁止prepare重绑。[短检证据](../work/night-parameter-search-20260925-v2/short-check-review.json)。

## 夜间8h：用户新增双起点要求

用户明确希望今晚同时使用第一、第二铲固定预备点。两源均有保存点和120帧参考，不需要重新接近。第一源 `runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz`，SHA `700f76be970c5754bde905837faf642cca0317fef097e31d17cf48f008ba0545`；第二源 `runs/contact-next-prepoint-20260925-v1/physical/prepoint.npz`，SHA `ab4f529501c3991022f6e72a276d4d7b48def9a888fcd85db3148356c73e420e`。

[双源入口准备](../work/night-dual-start-preparation-20260925-v2/README.md)已验证：第一源独立零步24数组/原生控制恢复全等，120帧37.547386秒，末态24数组及NPZ SHA与原第一源gate-120全等，SHA `b05c0ecd5fb38ee3e65e154057d093568874ee02dedc54fa80af71e5b042ed09`，执行硬门及120帧行为门通过。第二源明确复用未改动的夜间v2执行链，全部manifest/precheck绑定SHA复核通过，没有重跑第二源物理。10项CPU检查及成对评分规则通过。

首次双源v1的零步已通过，随后旧D154入口只允许0/3000，120请求在创建物理目录前被拒（0物理）；日志/失败冻结。v2仅在独立first_loop.py把允许帧数改0/120、墙时收紧180秒，逐字确认其余不变；复用已过零步，执行剩余120预算，没有重跑挖掘前缀。两个入口仍保持自己的原始车体/控制/材料历史，不把第一源改标签冒充第二源。

D-163：用户明确选择当前D152物理，不合入多接触点候选；再次确认“优先满斗率改善：继续挖掘参数优化，失败候选淘汰后继续，8小时内保留最佳并复验”。新[双源运行包](../work/night-dual-search-20260925-v1/README.md)及第一源完整尾段已接线，16项CPU测试通过；两源/尾段加载检查通过（0物理）。新批次已于2026-09-25 03:58:18（Asia/Shanghai）由systemd启动，PID 2132580，启动确认阶段baseline/first；截止2026-09-25 11:56:16。完整效果未验证。

两个全1基线首先跑到CLEAR_PILE，前120帧在同次运行内检查与旧参考数组精确一致。最多六组既有倍率作为候选池，每组两源成对评价；预算不足则少跑，始终优先留7200秒最佳双源固定卸料复验。总28800秒扣除既有检查121.096636秒。只改curl_load且收斗<95%的BC意图倍率，不改神经权重。

已知动作保护失败只淘汰该候选，保留失败状态/日志，下一组恢复原预备点；同组第一源失败无需浪费第二源。数值/材料/闭链异常、恢复/来源不一致、未知错误、保存失败或基线无法复现才停整批；绝不从失败状态自动续挖、不放宽保护、不重跑失败候选。单源或未复验改善不选为最佳，两源退步>2pp拒绝；复验还要求开发接收比例≥98%。没有改善就保留基线。

运行目录`runs/night-dual-search-20260925-v1/`，后台unit `wajueji-dual-search-20260925-v1.service`；实际状态以batch.json/launch.json为准。自动保存，完成/预算耗尽自动退出；用户要求不盯跑，不安排Codex监控，automation-2保持暂停。双固定源不证明连续第三铲/任意地形；正式95%资格和神经RL均未授予。

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
