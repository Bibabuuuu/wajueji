# Submission GLB build

This work package exports static, self-contained GLB models for the 2026-09-25
submission material set.  It reads frozen source geometry and does not advance
the simulator.  The current exporter is the enhanced presentation revision:
PBR material roles are separated and the visual soil surface uses a 25 mm grid,
three earth tones, and a closed side wall.

Source bindings and per-file validation are recorded in the generated
`GLB模型校验报告.json`.  Generated binaries live outside Git at:

`/home/qunwang/Documents/Codex/2026-09-25/复杂介质工况自主挖掘-三维模型材料-升级版`

Build command used:

```bash
PYTHONPATH=/tmp/wajueji-glb-tools:/home/qunwang/codex-work/small-soil-20260910-v17-performance \
python3 work/submission-glb-20260925-v1/build_models.py \
  --output /home/qunwang/Documents/Codex/2026-09-25/复杂介质工况自主挖掘-三维模型材料-升级版
```

The temporary tool directory contained `trimesh==4.8.3` and `scipy==1.16.2`;
they are export/validation tools, not project runtime dependencies.  The final
files were also checked with Khronos `gltf-validator@2.0.0-dev.3.10`: all three
reported zero errors and zero warnings.

The previous output folder is retained only as a rollback baseline.  The
enhanced folder is the recommended submission set and contains an interactive
`model-viewer` page plus a WebGL screenshot rendered from the actual GLB.
