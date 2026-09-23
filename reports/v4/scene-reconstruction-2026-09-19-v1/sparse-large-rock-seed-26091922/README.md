# sparse-large-rock-seed-26091922

Status: `geometry_reconstruction_candidate`.

This directory is a deterministic, geometry-only candidate. It is not an
official-scene reconstruction, physics qualification, or training input.

## Reproduce

```bash
PYTHONPATH=src/v4 python3 src/v4/build_scene_reconstruction.py \
  --config configs/v4/scene-reconstruction-v1.json \
  --output-dir reports/v4/scene-reconstruction-2026-09-19-v1
```

## Admission

- `rock_count_preserved`: passed
- `minimum_small_count_fraction`: passed
- `maximum_regular_large_count`: passed
- `maximum_regular_large_volume_fraction`: passed
- `jam_isolation`: passed
- `region_and_boundary_clearance`: passed
- `surface_support`: passed
- `convex_narrow_phase_no_intersection`: passed
- `minimum_rock_gap`: passed
- `finite_values`: passed

The next interface is the R0 asset ID, body-frame inertia, world pose and
coordinate-frame mapping recorded in `scene-manifest.json`. The R1/R2 mainline
must repeat collision and support checks in its actual physics representation.
