# Final project archive 2026-10-23

## Current entry points: 2026-10-25

- The default `main` branch now contains the final source and recovery tools, preserving both the previous `main` and `codex/portable-restore-20261024` histories.
- The [homepage](../README.md) links to the real two-scoop preview, current 149.96% / 109.94% short-hold display results, recovery instructions and final materials.
- Published Release assets and existing tags remain unchanged. The portable archive continues to pin its original frozen source; merging documentation into `main` does not rebuild or replace it.
- A fresh checkout provides source and small records. Runtime assets are in the [portable restoration Release](https://github.com/Bibabuuuu/wajueji/releases/tag/portable-restore-20261024); final PPT and media are in the [materials Release](https://github.com/Bibabuuuu/wajueji/releases/tag/final-archive-20261023).
- The repository was observed as public on 2026-10-25. This integration does not change visibility. References to a private repository below describe the original archival date.
- This is source and documentation integration only. No new physics, training, full-cycle validation or deletion of original workspaces is performed.

The user has ended the competition work and authorized archiving the final PPT,
code, images and videos to the existing private GitHub repository, followed by
cleanup of obsolete logs and temporary artifacts.

- Code snapshot branch: `archive/final-20261023`.
- Final materials: GitHub Release `final-archive-20261023`.
- Starting desktop HEAD: `8ea6cd8f55c3c35c4be73054ce28ed1f815f21a0`.
- Source freeze: `reports/final-archive-20261023/source-freeze.json`.
- Cleanup allowlist: `reports/final-archive-20261023/cleanup-plan.json`.
- Small reports, including failures: `run-summaries-before-cleanup.tar.gz` and manifest.

This is a project archival milestone. Existing technical validation boundaries
remain in the original reports; no new physical test or training is implied.
Final media and relevant inputs are archived separately from Git source files.
The declared obsolete `display/particles.npy` caches may be removed after upload
verification. All other files in those runs, especially inputs, endpoints,
states, commands, models and reports, remain available on the desktop.
The retained run roots are listed explicitly in the cleanup plan.

## Completed cleanup

Release: https://github.com/Bibabuuuu/wajueji/releases/tag/final-archive-20261023

{
  "deleted_files": 50,
  "deleted_logical_GiB": 355.04,
  "reclaimed_GiB": 216.26,
  "free_before_GiB": 29.42,
  "free_after_GiB": 245.68,
  "used_percent_after": 60.7,
  "retained_final_cache_files_verified": 10,
  "source_commit": "de813c431a42e4977c9238327decec3ee842a168",
  "retained_model_and_input_files_sha256_verified": 76,
  "preserved": [
    "All sources and environments",
    "All model weights",
    "All input and endpoint NPZ files",
    "All states, command traces, reports and failed-run summaries",
    "All declared final replay roots"
  ],
  "scope": "Only the 50 explicitly inventoried superseded runs/**/display/particles.npy files."
}

## Portable restoration supplement 2026-10-24

The user authorized completing non-environment runtime dependencies and dropping historical PPTs as a retention requirement. See [recovery commands](../deployment/README.zh-CN.md) and [exact verification](../deployment/restore-verification-20261024.json).

The standalone archive contains 7941 manifest-verified files. Extraction to a new directory, frozen-policy inference, exact 24-array zero-step restoration, native controller error 0, 3 physical frames, visual-model loading and soil-surface generation passed. All 7 original project resource roots were actively blocked; a full-process syscall file trace recorded 0 accesses to them. The Python/CUDA and package environment remains an external prerequisite; no full excavation rerun or training was performed.

Published release: https://github.com/Bibabuuuu/wajueji/releases/tag/portable-restore-20261024

The original source snapshot and original final-media release remain unchanged. All 4 assets match server SHA256 digests. The complete 300189971-byte package was independently downloaded from GitHub and matches SHA256 400a129ecb4ba06660455b1e33586e198640b65e736303d69fb1552fe4b54564. See [cloud receipt](../deployment/github-verification-20261024.json). Five extra final/transport checkpoints passed schema and binding validation; an attempted old-project read was rejected. See [state validation](../deployment/extra-state-validation.json).
