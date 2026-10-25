# Scene assembly preflight v1

Status: `scene_assembly_preflight_candidate`.

The nine admitted geometry candidates were validated and converted into
deterministic offline assembly plans. No MJCF was generated and no physics
engine or training process was run.

## Result

- Scene plans: 9
- Rocks mapped and independently recomputed: 159
- One/two-rock unit regression batches: 81
- Observed full-scene capacities: {'13': 3, '20': 6}
- Confirmed maximum capacity: 20
- First-round admission scenes: 6
- Deferred boundary diagnostics: 3

The maximum capacity is confirmed for the next bridge design but is not
implemented. It is an upper bound, not a padded 20-slot runtime contract.
Each run will use the fixed rock count of its loaded model. Unit batches only
reuse the current one/two-rock contract and cannot establish full-scene physics.

## Reproduce

```bash
PYTHONPATH=src/v4 python3 src/v4/build_scene_assembly_preflight.py \
  --config configs/v4/scene-assembly-preflight-v1.json \
  --output-dir reports/v4/scene-assembly-preflight-2026-09-19-v1
```

The output directory must not already exist.
