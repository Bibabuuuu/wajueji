"""D-128 read-only comparison with the frozen matched original replay."""
import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASE = OUT.parent / "manual-soil-loss-replay-20260923-v1"
START = ROOT / "runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089/approach-handoff.npz"
ORIGINAL_TRACE = ROOT / "runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-182307-822279/physics.ndjson"
COMMON_IDS = OUT.parent / "soil-local-onset-audit-20260923-v1/common-hole-initial-particle-ids.npy"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


a0 = np.load(START)
q0 = a0["particle_q"].astype(float)
flags0 = a0["particle_flags"]
bottom = q0[:, 2] < 0.2
xylo, xyhi = q0[:, :2].min(axis=0), q0[:, :2].max(axis=0)
v0, n0 = np.unique(np.floor(q0[bottom] / 0.1).astype(int), axis=0, return_counts=True)
common_ids = np.load(COMMON_IDS)
common_cells = np.unique(np.floor(q0[common_ids] / 0.1).astype(int), axis=0)
assert len(common_cells) == 70


def snapshot(path):
    a = np.load(path)
    q = a["particle_q"].astype(float)
    qd = a["particle_qd"].astype(float)
    F = a["particle_elastic_strain"].astype(float)
    Jp = a["particle_Jp"]
    displacement = np.linalg.norm(q - q0, axis=1)
    outside = ((q[:, :2] < xylo - 0.1) | (q[:, :2] > xyhi + 0.1)).any(axis=1)
    vf, nf = np.unique(np.floor(q[q[:, 2] < 0.2] / 0.1).astype(int), axis=0, return_counts=True)
    counts = {tuple(v): int(n) for v, n in zip(vf, nf)}
    depleted = []
    for v, n in zip(v0, n0):
        now = counts.get(tuple(v), 0)
        if n >= 32 and now < 0.25 * n:
            depleted.append((tuple(v), int(n), now))
    common_occupancy = [counts.get(tuple(v), 0) for v in common_cells]
    common_initial = [int(n0[np.flatnonzero(np.all(v0 == v, axis=1))[0]]) for v in common_cells]
    sv = np.linalg.svd(F, compute_uv=False)
    speed = np.linalg.norm(qd, axis=1)
    finite_Jp = Jp[np.isfinite(Jp)]
    nonfinite = {k: int((~np.isfinite(a[k])).sum()) for k in a.files
                 if a[k].dtype.kind in "fc" and not np.isfinite(a[k]).all()}
    return {
        "file": path.name,
        "sha256": sha(path),
        "particle_count": len(q),
        "flags_unchanged": bool(np.array_equal(flags0, a["particle_flags"])),
        "nonfinite_fields": nonfinite,
        "Jp_nonfinite": int((~np.isfinite(Jp)).sum()),
        "Jp_min_p50_p99_max_finite": np.quantile(finite_Jp, [0, 0.5, 0.99, 1]).tolist(),
        "F_singular_min": float(sv.min()),
        "F_max_singular_p50_p99_max": np.quantile(sv[:, 0], [0.5, 0.99, 1]).tolist(),
        "F_frobenius_over1000": int((np.linalg.norm(F, axis=(1, 2)) > 1000).sum()),
        "bottom_displacement_median_p99_max_m": np.quantile(displacement[bottom], [0.5, 0.99, 1]).tolist(),
        "bottom_moved_over20cm": int((bottom & (displacement > 0.2)).sum()),
        "bottom_outside_footprint": int((bottom & outside).sum()),
        "all_outside_footprint": int(outside.sum()),
        "bottom_depleted_cells": len(depleted),
        "common70_current_particles": int(sum(common_occupancy)),
        "common70_initial_particles": int(sum(common_initial)),
        "common70_cells_below25pct_initial": int(sum(now < 0.25 * initial for now, initial in zip(common_occupancy, common_initial))),
        "minimum_z_m": float(q[:, 2].min()),
        "particles_below_minus5mm": int((q[:, 2] < -0.005).sum()),
        "particle_speed_p99_max_m_s": np.quantile(speed, [0.99, 1]).tolist(),
    }


frames = [600, 800, 1000, 1193, 1329]
snapshots = {}
for frame in frames:
    baseline_path = BASE / f"diagnostic-{frame:06d}.npz"
    candidate_path = OUT / f"candidate-{frame:06d}.npz"
    snapshots[str(frame)] = {"baseline": snapshot(baseline_path), "candidate": snapshot(candidate_path)}

q1193_base = np.load(BASE / "diagnostic-001193.npz")["particle_q"].astype(float)
q1329_base = np.load(BASE / "diagnostic-001329.npz")["particle_q"].astype(float)
q1193_new = np.load(OUT / "candidate-001193.npz")["particle_q"].astype(float)
q1329_new = np.load(OUT / "candidate-001329.npz")["particle_q"].astype(float)


def late(a, b):
    d = np.linalg.norm(b - a, axis=1)
    moved = d > 0.2
    return {
        "all_moved_over20cm": int(moved.sum()),
        "initial_bottom_moved_over20cm": int((bottom & moved).sum()),
        "moved_over20cm_initial_z_min_p50_max_m": None if not moved.any() else np.quantile(q0[moved, 2], [0, 0.5, 1]).tolist(),
        "initial_bottom_displacement_p50_p99_max_m": np.quantile(d[bottom], [0.5, 0.99, 1]).tolist(),
    }


def load_trace(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def projection(rows, start_frame=1194):
    selected = [r["postsolve_projection"] for r in rows if r["frame"] >= start_frame]
    max_position_index = int(np.argmax([p["maximum_position_delta_m"] for p in selected]))
    return {
        "frames": len(selected),
        "sum_corrected_particle_count": int(sum(p["corrected_particle_count"] for p in selected)),
        "max_corrected_particle_count": int(max(p["corrected_particle_count"] for p in selected)),
        "max_position_delta_m": float(max(p["maximum_position_delta_m"] for p in selected)),
        "max_position_delta_frame": int(rows[start_frame - 1 + max_position_index]["frame"]),
        "frames_position_delta_over2cm": int(sum(p["maximum_position_delta_m"] > 0.02 for p in selected)),
        "max_velocity_delta_m_s": float(max(p["maximum_velocity_delta_m_s"] for p in selected)),
        "max_pre_projection_speed_m_s": float(max(p["pre_projection_max_speed_m_s"] for p in selected)),
    }


baseline_trace = load_trace(ORIGINAL_TRACE)
candidate_trace = load_trace(OUT / "trace.ndjson")
protocol = json.loads((OUT / "protocol.json").read_text())
report = json.loads((OUT / "report.json").read_text())
final_base = snapshots["1329"]["baseline"]
final_new = snapshots["1329"]["candidate"]
result = {
    "definitions": {
        "bottom_cohort": "same particle IDs whose initial z < 0.2m",
        "outside_footprint": "outside initial particle XY AABB expanded by 0.1m",
        "depleted_cells": "fixed 0.1m cells below z=0.2m, initial >=32 particles, current <25% initial",
        "common70": "70 retrospective bottom cells depleted in both original and no-late-projection D123/D124 finals; diagnosis only",
    },
    "initial_particle_count": len(q0),
    "initial_bottom_count": int(bottom.sum()),
    "snapshots": snapshots,
    "last136": {
        "baseline": late(q1193_base, q1329_base),
        "candidate": late(q1193_new, q1329_new),
    },
    "last136_projection": {
        "baseline": projection(baseline_trace),
        "candidate": projection(candidate_trace),
    },
    "final_candidate_over_baseline_ratios": {
        "bottom_moved_over20cm": final_new["bottom_moved_over20cm"] / final_base["bottom_moved_over20cm"],
        "bottom_outside_footprint": final_new["bottom_outside_footprint"] / final_base["bottom_outside_footprint"],
        "bottom_depleted_cells": final_new["bottom_depleted_cells"] / final_base["bottom_depleted_cells"],
        "common70_current_particles": final_new["common70_current_particles"] / final_base["common70_current_particles"],
    },
    "checks": {
        "1329_frames_completed": report["status"] == "candidate_full_replay_passed" and report["frames"] == 1329,
        "all_candidate_frame_guards_passed": all(all(r["guards"].values()) for r in candidate_trace),
        "all_bindings_unchanged": all(sha(path) == expected for path, expected in protocol["bindings"].items()),
        "same_original_applied_commands": len(candidate_trace) == 1329 and all(
            a["frame"] == b["frame"] and a["applied_command_si"] == b["applied_command_si"]
            for a, b in zip(candidate_trace, baseline_trace)
        ),
        "final_material_finite": not final_new["nonfinite_fields"],
        "final_particle_flags_unchanged": final_new["flags_unchanged"],
        "bottom_migration_materially_reduced": final_new["bottom_moved_over20cm"] <= 0.1 * final_base["bottom_moved_over20cm"],
        "bottom_depleted_cells_materially_reduced": final_new["bottom_depleted_cells"] <= 0.25 * final_base["bottom_depleted_cells"],
        "no_particles_below_minus5mm": final_new["particles_below_minus5mm"] == 0,
        "full_human_scoop_qualified": False,
        "training_eligible": False
    },
    "limits": [
        "The candidate changes the coupled trajectory from frame 1; equal commands do not mean equal bucket/rock state after that point.",
        "The common70 set is retrospective diagnostic evidence, not a pre-registered generalization metric.",
        "Particle counts and fixed-cell occupancy are not calibrated bucket volume or mass loss.",
        "Passing the recorded trajectory does not replace a new human full-scoop test with the repaired soft-limit command chain.",
        "Jp remains finite but continues to grow; with hardening=0 it does not change yield strength in this configuration.",
    ],
}
(OUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
