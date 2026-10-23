"""Read-only audit of the rejected second manual showcase attempt."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/qunwang/wajueji")
RUN = ROOT / "runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-135414-2443995"
OUT = ROOT / "work/manual-three-scoop-showcase-20260925-v1/attempt-02-read-only-audit.json"
KIN = Path("/home/qunwang/codex-work/small-soil-20260910-v17-performance")
MEASURE = Path("/home/qunwang/codex-work/fill-volume-2026-09-15-v1/fill_volume.py")
CONTRACT = Path("/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json")
FRONT = CONTRACT.parent / "front-rim.npz"


def main() -> None:
    sys.path.insert(0, str(KIN))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src/v4"))
    from layered_excavation_approach_planner import _load_kinematics

    spec = importlib.util.spec_from_file_location("_attempt02_measure", MEASURE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    contract = json.loads(CONTRACT.read_text())["bucket"]
    with np.load(FRONT) as data:
        front = data["front"].copy()
    assembly, _ = _load_kinematics(
        KIN, KIN / "scene/input/model/A.xml", KIN / "scene/input/mesh")

    display = RUN / "display"
    states = json.loads((display / "states.json").read_text())
    particles = np.load(display / "particles.npy", mmap_mode="r")
    with np.load(RUN / "failure-002220.rejected-raw.npz") as failure:
        jp = failure["particle_Jp"]
        bad = np.flatnonzero(~np.isfinite(jp))
        bad_positions = failure["particle_q"][bad]

    physics = [json.loads(line) for line in (RUN / "physics.ndjson").open()]
    sample_frames = sorted(set(range(0, len(states), 30)) | {len(states) - 1})
    samples = []
    masks = {}
    for frame in sample_frames:
        state = states[frame]
        q = np.asarray(state["mechanism_qpos"])
        pose = assembly.assemble(q, include_meshes=False)[1]["bucket"]
        local = (particles[frame] - pose[:3, 3]) @ pose[:3, :3]
        mask = module.cavity_mask(
            local, np.asarray(contract["polygon_xz_m"]),
            np.asarray(contract["y_limits_m"]), front)
        masks[frame] = mask
        samples.append({
            "frame": frame,
            "sim_time_s": float(state["time_s"]),
            "soil_cavity_particle_count": int(mask.sum()),
            "bucket_origin_world_m": np.asarray(pose[:3, 3]).tolist(),
            "swing_q": float(q[7]),
            "boom_q": float(q[9]),
            "arm_q": float(q[12]),
            "bucket_q": float(q[15]),
        })

    counts = np.array([row["soil_cavity_particle_count"] for row in samples])
    peak_i = int(np.argmax(counts))
    peak_frame = samples[peak_i]["frame"]
    peak_mask = masks[peak_frame]
    for row in samples[peak_i:]:
        frame = row["frame"]
        mask = masks[frame]
        row["peak_cohort_retained"] = int((mask & peak_mask).sum())
        row["peak_cohort_lost"] = int((~mask & peak_mask).sum())
        row["new_since_peak"] = int((mask & ~peak_mask).sum())

    cohort_anchors = [600, 840, 1080, 1200, 1800]
    cohort_retention = []
    for anchor in cohort_anchors:
        cohort = masks[anchor]
        history = []
        for frame in sample_frames:
            if frame < anchor:
                continue
            current = masks[frame]
            history.append({
                "frame": frame,
                "retained": int((current & cohort).sum()),
                "lost": int((~current & cohort).sum()),
            })
        cohort_retention.append({
            "anchor_frame": anchor,
            "anchor_count": int(cohort.sum()),
            "retained_at_frame_1800": next(
                (r["retained"] for r in history if r["frame"] == 1800), None),
            "retained_at_frame_2160": next(
                (r["retained"] for r in history if r["frame"] == 2160), None),
            "retained_at_frame_2220": history[-1]["retained"],
            "history": history,
        })

    # Material-guard onset and the location of the exact rejected particles.
    finite_jp = []
    for row in physics:
        guard = row.get("hencky_material_guard") or {}
        finite_jp.append({
            "frame": int(row["frame"]),
            "maximum_Jp": guard.get("maximum_Jp"),
            "nonfinite_particle_Jp": int(
                (guard.get("nonfinite_values") or {}).get("particle_Jp", 0)),
            "truck_mpm_contact_samples": int(row.get("truck_mpm_contact_samples", 0)),
            "bucket_mpm_contact_samples": int(row.get("bucket_mpm_contact_samples", 0)),
        })
    growth = [r for r in finite_jp if r["maximum_Jp"] is not None and r["maximum_Jp"] >= 1e20]
    result = {
        "schema": "manual_showcase_attempt_read_only_audit_v1",
        "status": "hard_guard_cause_confirmed_and_bucket_membership_sampled",
        "physics_steps_added": 0,
        "source_run": str(RUN.relative_to(ROOT)),
        "cache_frames": len(states),
        "sample_stride_frames": 30,
        "peak_cavity": samples[peak_i],
        "last_sample": samples[-1],
        "sampled_cavity_history": samples,
        "cohort_retention": cohort_retention,
        "nonfinite_particle_Jp_count": int(len(bad)),
        "nonfinite_particle_indices": bad.tolist(),
        "nonfinite_particle_world_bounds_m": {
            "minimum": bad_positions.min(axis=0).tolist(),
            "maximum": bad_positions.max(axis=0).tolist(),
            "mean": bad_positions.mean(axis=0).tolist(),
        },
        "material_growth_first_over_1e20": growth[0] if growth else None,
        "material_growth_tail": finite_jp[-30:],
        "limitations": [
            "Cavity membership uses particle centers and is not a formal fill-volume estimate.",
            "Thirty-frame sampling can miss exit and re-entry between samples.",
            "This audit performs no simulator construction or physical stepping.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "peak": result["peak_cavity"],
        "last": result["last_sample"],
        "bad_bounds": result["nonfinite_particle_world_bounds_m"],
        "growth_start": result["material_growth_first_over_1e20"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
