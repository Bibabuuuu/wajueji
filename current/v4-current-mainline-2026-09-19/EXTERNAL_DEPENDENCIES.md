# 外部运行依赖

为避免再次复制超过 1 GB 的仿真后端和多个虚拟环境，本工作包没有复制以下大型或共享依赖。当前 `01-runtime/` 中的启动脚本仍使用这些绝对路径。

## 必需路径

- Python/MPM 环境：`/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python`
- 快速档仿真后端：`/home/qunwang/codex-work/autonomous-multiscoop-speed-opt-2026-09-14-v1`
- 满斗率模块：`/home/qunwang/codex-work/fill-volume-2026-09-15-v1`
- 保留体积几何：`/home/qunwang/codex-work/autonomous-multiscoop-speed-opt-2026-09-14-v1/retention-front-rim.npz`
- 土体分离依赖：`/home/qunwang/codex-work/soil-separation-2026-09-11/.venv/lib/python3.12/site-packages`
- 策略依赖：`/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/lib/python3.12/site-packages`

## 启动前检查

```bash
test -x /home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python
test -d /home/qunwang/codex-work/autonomous-multiscoop-speed-opt-2026-09-14-v1
test -d /home/qunwang/codex-work/fill-volume-2026-09-15-v1
nvidia-smi
ps -eo pid,etime,cmd | rg -i 'mpm|warp|excavat|v4'
```

不要把缺失依赖静默替换为其他版本。若未来需要真正独立搬迁，应按已有报告里的 SHA256 清单复制并重新做预检。
