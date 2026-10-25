# D-137 遗留推石悬板清理（2026-09-24）

状态：已实现，定点碰撞回归及30帧恢复保持已验证；未重新采集完整单铲。

用户授权清理后，仅对新生成的人工接触/运输MuJoCo模型删除具名静态碰撞体 `r1_ground_push_shelf`。验证其worldbody归属、位置、尺寸、姿态和碰撞掩码；若身份变化/重名/被引用则拒绝修改。冻结资产、原预备点与历史模型不写入，真正地面/车体/铲斗/石块几何与碰撞保护不变。清理模块SHA及模型清理前后SHA随新运行保存。D-136主动收斗候选仍显式选择，未改助力参数。

## 验证

- `tests.log`：3项清理单测及23项现有手操测试，共26项通过。
- `model-check.json`：9项检查通过。上次1498帧故障姿态仅做原生MuJoCo `mj_forward`，不积分物理。228→227个geom，8→7个接触点，仅移除悬板与rock020接触，其余接触对全等，其他XML结构/属性全等，源XML和故障快照未变。
- `manual-model-check.json`：将资源路径绑定为人工启动脚本的离线bundle，清理前后模型SHA与实际30帧入口报告精确一致；5项检查通过，同样只消除悬板接触。第一次模型比较使用了独立回放的资源路径，两者XML SHA不同，已核对并补做实际人工模型版本，未跳过身份检查。
- `restore-check.json`：从D-115原完整预备点直接恢复，ready暂停0帧，随后30帧零指令/0.5秒仿真；10项运行检查全过、每帧硬门全过，前3帧after_state与清理前D-136探针全等。物理计算3.984秒。预备点SHA仍为`6344e74f809c8b5323942b211f5d62cccb6356da87bf8c85698910e1f6dfb3f4`。

实际探针：`runs/layered-47-sparse-manual-contact-v1/manual-contact-20260924-005017-1076447`。入口：`bash scripts/run_47_contact_bc_active_curl_local.sh --manual-input gamepad --zero-probe-frames 30`。未重放1498帧、未推进原故障现场、未启动新人工采集窗口。下一步用清理后的D-136候选从原预备点采集新的完整接触单铲；高满斗切削、稳定卸车及训练准入仍须各自证据。

## 复核与保留

`check_model.py`编译并运行只读碰撞探针，随后`check_manual_model.py`核对实际人工模型，`check_restore.py`汇总已保存的30帧日志。使用项目现有venv并设置`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:.`。冻结模型、NPZ、完整逐帧日志、生成XML和探针二进制保留本地，不纳入源码提交。

初次链接因bundle仅含版本化MuJoCo库失败，改用`-l:libmujoco.so.2.3.8`后通过；首次30帧命令被旧参数choices拒绝，增加显式30帧选项后重新成功。两次失败均未推进物理，原日志保留本地。
