"""CPU-only sampled soil-ID audit; no simulator construction or physical stepping."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "cached-cavity-review.json"
KIN = Path("/home/qunwang/codex-work/small-soil-20260910-v17-performance")
MEASURE = Path("/home/qunwang/codex-work/fill-volume-2026-09-15-v1/fill_volume.py")
CONTRACT = Path("/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json")
FRONT = CONTRACT.parent / "front-rim.npz"
SOURCES = (
    ("D152", "contact-full-response-20260924-v1/candidate60", 2400, 0,
     [0, 600, 900, 1123, 1200, 1500, 1550, 1800, 2100, 2400],
     "37cb63c32f9070bcf119fac3a729be64ad6cde31593fa9f5ac8a17a6b5c6a0f2"),
    ("D153", "contact-preview-lift-20260924-v1/continuation", 424, 2400,
     [0, 60, 120, 240, 300, 360, 420, 424],
     "5cb69a1fd46cc50d0f7fa7f7990026f37436b4a876967583a619a15d0a074ffe"),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    started = time.perf_counter()
    sys.path.insert(0, str(ROOT / "src/v4"))
    from layered_excavation_approach_planner import _load_kinematics

    spec = importlib.util.spec_from_file_location("_fill_loss_measure", MEASURE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    contract = json.loads(CONTRACT.read_text())["bucket"]
    with np.load(FRONT) as data:
        front = data["front"].copy()
    assembly, _ = _load_kinematics(KIN, KIN / "scene/input/model/A.xml", KIN / "scene/input/mesh")

    inputs = {}
    for path in (Path(__file__), MEASURE, CONTRACT, FRONT,
                 ROOT / "src/v4/layered_excavation_approach_planner.py",
                 KIN / "scene_assembly.py", KIN / "fast_kinematics.py",
                 KIN / "closed_chain.py", KIN / "scene/input/model/A.xml"):
        inputs[str(path)] = sha256(path)
    rows, source_reviews = [], []
    previous_mask = cohort = boundary_particles = boundary_q = None
    for label, relative, final_frame, offset, frames, expected_final_sha in SOURCES:
        run = ROOT / "runs" / relative
        display = run / "display"
        manifest = json.loads((display / "reconstruction.json").read_text())
        inputs[str(display / "reconstruction.json")] = sha256(display / "reconstruction.json")
        for name in ("particles.npy", "states.json", "scene.json"):
            path = display / name
            actual = sha256(path)
            if actual != manifest["cache_sha256"][name]:
                raise ValueError(f"cache_sha_mismatch:{label}:{name}")
            inputs[str(path)] = actual
        source_sha = sha256(run / "final.npz")
        if source_sha != expected_final_sha:
            raise ValueError(f"final_source_sha_mismatch:{label}")
        inputs[str(run / "final.npz")] = source_sha
        states = json.loads((display / "states.json").read_text())
        particles = np.load(display / "particles.npy", mmap_mode="r")
        if len(states) != final_frame + 1 or particles.shape[0] < len(states):
            raise ValueError("cache_valid_frame_range_mismatch")
        with np.load(run / "final.npz") as final:
            error = float(np.max(np.abs(final["particle_q"] - particles[final_frame])))
            q_error = float(np.max(np.abs(final["mechanism_qpos"] - states[-1]["mechanism_qpos"])))
            if error != 0 or q_error != 0 or not np.all(final["particle_flags"] == 1):
                raise ValueError("final_cache_or_active_identity_mismatch")
        if boundary_particles is not None:
            if not np.array_equal(boundary_particles, particles[0]):
                raise ValueError("continuation_particle_boundary_mismatch")
            if not np.array_equal(boundary_q, states[0]["mechanism_qpos"]):
                raise ValueError("continuation_mechanism_boundary_mismatch")
        source_reviews.append(dict(source=label, valid_local_frame_range=[0, final_frame],
                                   cache_allocated_frames=int(particles.shape[0]),
                                   particle_count=int(particles.shape[1]),
                                   final_particle_max_abs_error=error,
                                   final_mechanism_max_abs_error=q_error,
                                   all_final_particles_active=True,
                                   full_cache_sha_matches_frozen_manifest=True))
        for frame in frames:
            q = np.asarray(states[frame]["mechanism_qpos"])
            pose = assembly.assemble(q, include_meshes=False)[1]["bucket"]
            local = (particles[frame] - pose[:3, 3]) @ pose[:3, :3]
            mask = module.cavity_mask(local, np.asarray(contract["polygon_xz_m"]),
                                      np.asarray(contract["y_limits_m"]), front)
            if label == "D152" and frame == 1123:
                cohort = mask.copy()
            row = dict(source=label, local_frame=frame, global_frame=offset + frame,
                       soil_cavity_count=int(mask.sum()),
                       effective_curl_fraction=float((q[15] - .370) / (.852 - .370)),
                       bucket_origin_z_m=float(pose[2, 3]))
            if cohort is not None:
                row.update(cohort1123_retained=int((mask & cohort).sum()),
                           cohort1123_out=int((~mask & cohort).sum()),
                           new_since1123=int((mask & ~cohort).sum()))
            if previous_mask is not None:
                row.update(entered_since_sample=int((mask & ~previous_mask).sum()),
                           left_since_sample=int((~mask & previous_mask).sum()))
            previous_mask = mask.copy()
            rows.append(row)
        boundary_particles = np.array(particles[final_frame])
        boundary_q = np.asarray(states[final_frame]["mechanism_qpos"])

    # These are the exact findings obtained in the initial read-only investigation.
    expected_counts = [0, 674, 6296, 8031, 8064, 8540, 8616, 15099, 21978,
                       24094, 24094, 24132, 24166, 24238, 24256, 24282, 24332, 24335]
    if [row["soil_cavity_count"] for row in rows] != expected_counts:
        raise ValueError("initial_investigation_count_regression")
    if any(row["cohort1123_out"] != (18 if row["global_frame"] == 1200 else 0)
           for row in rows if "cohort1123_out" in row):
        raise ValueError("initial_investigation_cohort_regression")
    result = dict(schema="sampled_cached_soil_cavity_id_review_v1",
                  status="read_only_cpu_audit_passed", physics_steps_added=0,
                  gpu_or_training_started=False, source_reviews=source_reviews,
                  continuation_boundary_particles_and_mechanism_exact=True,
                  cohort_definition="D152 global frame 1123: soil array indices whose centers lie in fixed CAD cavity",
                  cohort_count=int(cohort.sum()), input_sha256=inputs, samples=rows,
                  limitations=[
                      "Soil particle center counts are not occupied volume or fill percentage.",
                      "No heap-above-mouth or rock inventory tracking is included.",
                      "Sampled endpoint membership does not exclude exit/re-entry between samples.",
                      "Cavity membership while embedded in soil does not prove detached carried load.",
                      "No inference that 99% cylinder closure implies 95% payload fill.",
                      "This is the separate third-scoop development source, not an autonomous continuous three-scoop chain.",
                      "Input hashes bind listed files; they are not a complete environment or mesh dependency manifest."],
                  wall_seconds=time.perf_counter() - started)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(status=result["status"], output=str(OUT), samples=len(rows),
                          wall_seconds=result["wall_seconds"])))


if __name__ == "__main__":
    main()
