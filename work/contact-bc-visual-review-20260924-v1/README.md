# BC起步回放与摇杆输入核验

用户要求检查可能的摇杆误触并查看自主动作。原示范、模型、死区和策略不变。

对应第三铲人工示范115812-1458627的前120帧，104帧出现小回转且伴随超过30%的动臂命令。原始左横轴常为−0.13725，超过现配置0.12死区后约−0.01961（−1.96%）归一化回转命令。可疑于纵向操作带来的附带横移，但日志不能判定人的意图。模型120帧平均回转+3.02%，峰值+47.98%，主横偏与这段示范不同，不能把模型偏移直接归因误触；该条还是开发验证集，不是直接训练样本。十条输入描述性统计与各源SHA见[输入核验](joystick-review.json)，没有自动清零“疑似误触”标签。

## 画面与一致性

`src/v4/view_contact_bc_probe.py`先核对既有审查输入/输出SHA，独立恢复原第三铲预备点，执行原120帧已记录命令；每帧前后机构/石块状态必须与原记录一致，最终24项物理数组全部一致后才开放播放。准备阶段确实重建了120步物理（21.87秒），并缓存0–120帧全部256441粒子坐标；观看阶段无物理、无实时策略推理或手柄输入。不能把此重放当成第二条独立自主试验。

窗口已在DISPLAY=:1启动，最初PID1493335，0帧暂停；截图核验完整场景/连续土表面/目标标记。窗口关闭状态以进程为准。操作：Space播放/暂停；左右键逐帧；R退回起点并暂停；Home/End到首尾；X/Esc关闭。半速播放，到末帧自动暂停，窗口不会因播完自动关闭。沿用原场景相机和鼠标视角操作。

缓存占约372MB，保存在Git忽略的`runs/contact-bc-visual-review-20260924-v1`，不写回源文件。查看时持有本地GPU锁；先关闭窗口再开始新物理任务。

```bash
DISPLAY=:1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/v4:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
flock -n /tmp/wajueji-local-47-sparse-manual-contact.lock \
/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python \
 src/v4/view_contact_bc_probe.py --view-only --output runs/contact-bc-visual-review-20260924-v1
```

首次重建用`--review work/contact-bc-closed-loop-20260924-v1/probe120-review.json --output runs/NEW_REPLAY --prepare-only`，输出目录必须不存在，然后用`--view-only`打开。没有扩展到120帧以后的接触/收斗阶段。

[重建证明](reconstruction.json)、[窗口启动](viewer-ready.json)。图像为runs/contact-bc-visual-review-20260924-v1/viewer-ready.png。
