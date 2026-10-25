# Scene reconstruction v1

Status: `geometry_reconstruction_candidate`.

Nine deterministic geometry candidates were compiled from explicitly marked
synthetic parameters and the frozen R0 convex assets. They are not an official
scene reconstruction and have no physics or training qualification.

## Reproduce

```bash
PYTHONPATH=src/v4 python3 src/v4/build_scene_reconstruction.py \
  --config configs/v4/scene-reconstruction-v1.json \
  --output-dir reports/v4/scene-reconstruction-2026-09-19-v1
```

The output directory must not already exist. Generate into two fresh temporary
directories and compare files to verify byte-for-byte determinism.

## Scenes

- `ordinary-small-rock-dominant-seed-26091911`: geometry_reconstruction_candidate
- `ordinary-small-rock-dominant-seed-26091912`: geometry_reconstruction_candidate
- `ordinary-small-rock-dominant-seed-26091913`: geometry_reconstruction_candidate
- `sparse-large-rock-seed-26091922`: geometry_reconstruction_candidate
- `sparse-large-rock-seed-26091925`: geometry_reconstruction_candidate
- `sparse-large-rock-seed-26091926`: geometry_reconstruction_candidate
- `jam-boundary-seed-26091931`: geometry_reconstruction_candidate
- `jam-boundary-seed-26091932`: geometry_reconstruction_candidate
- `jam-boundary-seed-26091933`: geometry_reconstruction_candidate

## Boundary

The compiler only establishes explicit provenance, stable IDs, R0-compatible
mass/inertia, surface support, boundary clearance and convex SAT non-intersection.
R1/R2 must independently qualify the actual collision proxies, settling and
rock/bucket dynamics before any scene may be called physics-qualified.
