#!/usr/bin/env python3
"""Run R3-C reconstructed repeats, dt refinement, and feedback causality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import tempfile
import time

import newton
import numpy as np
import warp as wp

from build_mixed_coupled_bridge_r3_c_dynamics import write_bridge
from build_mixed_multirock_candidate_r3_c import build_two_rock_candidate, sha256
from mixed_material_r3_c_convergence_contract import validate_config
from run_mixed_material_r3_a_probe import _compile_bridge, _write_json
from run_mixed_material_r3_c_shared_probe import _episode


REPORT_SCHEMA = "excavation_mixed_material_r3_c_convergence_report_v1"
ROCK_IDS = ("r1-rock-000", "r1-rock-001")


def _state_vector(episode: dict) -> np.ndarray:
    values = list(episode["final_mechanism_qpos"])
    for rock_id in ROCK_IDS:
        values.extend(episode["final_rock_qpos"][rock_id])
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (33,) or not np.isfinite(vector).all():
        raise ValueError("r3_c_convergence_final_state_contract")
    return vector


def _impulse(episode: dict, key: str) -> float:
    value = episode["accumulated_impulse_magnitudes"]
    for part in key.split("."):
        value = value[part]
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("r3_c_convergence_impulse_contract")
    return result


def _relative_difference(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), 1.0e-30)


def _episode_health(episode: dict, shared_probe: dict) -> bool:
    ownership = episode["collider_ownership"]
    return bool(
        episode["matrix"]["finite"] and
        abs(episode["matrix"]["target_mass_error_kg"]) <=
        float(shared_probe["maximum_mass_error_kg"]) and
        episode["maximum_forbidden_contact_count"] == 0 and
        episode["maximum_mechanism_closure_m"] <=
        float(shared_probe["maximum_closure_m"]) and
        episode["maximum_per_rock_rigid_force_partition_residual_n"] <=
        float(shared_probe["maximum_force_partition_residual_n"]) and
        episode["maximum_per_rock_rigid_torque_partition_residual_nm"] <=
        float(shared_probe["maximum_torque_partition_residual_nm"]) and
        episode["maximum_total_force_partition_residual_n"] <=
        float(shared_probe["maximum_force_partition_residual_n"]) and
        episode["maximum_total_torque_partition_residual_nm"] <=
        float(shared_probe["maximum_torque_partition_residual_nm"]) and
        episode["maximum_applied_wrench_echo_residual"] <=
        float(shared_probe["maximum_force_partition_residual_n"]) and
        episode["maximum_reference_point_residual_m"] <=
        float(shared_probe["maximum_reference_point_residual_m"]) and
        ownership["mapping_disjoint"] and
        ownership["unmapped_active_sample_count"] == 0 and
        ownership["nonfinite_sample_count"] == 0 and
        ownership["rock_id_to_newton_body_index"] ==
        {"r1-rock-000": 0, "r1-rock-001": 1} and
        ownership["bucket_newton_body_index"] == 2 and
        episode["mpm_history_reset"]["passed"])


def run(args) -> dict:
    repository = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    inventory = Path(args.inventory).resolve()
    output = Path(args.output).resolve()
    artifact_dir = (Path(args.artifact_dir).resolve()
                    if args.artifact_dir is not None else None)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, repository)
    frozen = config["frozen_inputs"]
    shared_config_path = repository / frozen["shared_config_path"]
    shared_config = json.loads(shared_config_path.read_text(encoding="utf-8"))
    r3_c_config_path = (
        repository / shared_config["frozen_inputs"]["r3_c_config_path"])
    r3_c_config = json.loads(r3_c_config_path.read_text(encoding="utf-8"))
    r3_b_config = json.loads((
        repository / shared_config["frozen_inputs"]["r3_b_config_path"]
    ).read_text(encoding="utf-8"))
    probe = config["probe"]
    duration = float(probe["duration_s"])
    coarse_frames = int(round(duration / float(probe["coarse_output_dt_s"])))
    fine_frames = int(round(duration / float(probe["refined_output_dt_s"])))
    if args.frames is not None:
        coarse_frames = int(args.frames)
        if coarse_frames < 1 or coarse_frames > 120:
            raise ValueError("frames_must_be_within_1_and_120")
        fine_frames = coarse_frames * 2
    coarse_config = json.loads(json.dumps(shared_config))
    coarse_config["probe"]["output_dt_s"] = float(
        probe["coarse_output_dt_s"])
    coarse_config["probe"]["mujoco_internal_substeps"] = int(
        probe["coarse_mujoco_internal_substeps"])
    fine_config = json.loads(json.dumps(shared_config))
    fine_config["probe"]["output_dt_s"] = float(probe["refined_output_dt_s"])
    fine_config["probe"]["mujoco_internal_substeps"] = int(
        probe["refined_mujoco_internal_substeps"])

    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
            prefix="mixed-material-r3-c-convergence-") as temporary:
        work = Path(temporary)
        model = work / "model.xml"
        model_manifest = build_two_rock_candidate(
            r3_c_config_path, repository, inventory, model)
        source = work / "bridge.cc"
        source_manifest = write_bridge(r3_c_config_path, repository, source)
        binary = work / "bridge"
        compile_command = _compile_bridge(
            args.cxx, source, binary,
            [args.mechanism_header_dir, args.controller_header_dir,
             args.mujoco_include], Path(args.mujoco_library_dir))
        episodes = {
            "coarse_first": _episode(
                coarse_config, r3_c_config, r3_b_config, repository, inventory,
                binary, model, args.device, "combined", coarse_frames, True),
            "coarse_repeat": _episode(
                coarse_config, r3_c_config, r3_b_config, repository, inventory,
                binary, model, args.device, "combined", coarse_frames, True),
            "coarse_no_feedback": _episode(
                coarse_config, r3_c_config, r3_b_config, repository, inventory,
                binary, model, args.device, "combined", coarse_frames, False),
            "refined_first": _episode(
                fine_config, r3_c_config, r3_b_config, repository, inventory,
                binary, model, args.device, "combined", fine_frames, True),
            "refined_repeat": _episode(
                fine_config, r3_c_config, r3_b_config, repository, inventory,
                binary, model, args.device, "combined", fine_frames, True),
        }
        model_sha256 = sha256(model)
        source_sha256 = sha256(source)
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model, artifact_dir / "generated-model.xml")
            shutil.copy2(source, artifact_dir / "generated-bridge.cc")
            _write_json(
                artifact_dir / "generated-model-manifest.json", model_manifest)
            _write_json(
                artifact_dir / "generated-bridge-manifest.json", source_manifest)

    coarse_first = episodes["coarse_first"]
    coarse_repeat = episodes["coarse_repeat"]
    no_feedback = episodes["coarse_no_feedback"]
    refined_first = episodes["refined_first"]
    refined_repeat = episodes["refined_repeat"]
    coarse_state_delta = float(np.max(np.abs(
        _state_vector(coarse_first) - _state_vector(coarse_repeat))))
    refined_state_delta = float(np.max(np.abs(
        _state_vector(refined_first) - _state_vector(refined_repeat))))
    feedback_mechanism_delta = float(np.max(np.abs(
        np.asarray(coarse_first["final_mechanism_qpos"], dtype=np.float64) -
        np.asarray(no_feedback["final_mechanism_qpos"], dtype=np.float64))))
    coarse_mean_state = 0.5 * (
        _state_vector(coarse_first) + _state_vector(coarse_repeat))
    refined_mean_state = 0.5 * (
        _state_vector(refined_first) + _state_vector(refined_repeat))
    dt_state_delta = float(np.max(np.abs(coarse_mean_state - refined_mean_state)))

    repeat_impulse_keys = [
        "bucket_mpm_ns", "bucket_rigid_ns", "bucket_total_ns",
        "rock_mpm_ns.r1-rock-000", "rock_mpm_ns.r1-rock-001",
        "bucket_rigid_by_rock_ns.r1-rock-000",
        "bucket_rigid_by_rock_ns.r1-rock-001",
    ]
    coarse_repeat_impulse = {
        key: _relative_difference(
            _impulse(coarse_first, key), _impulse(coarse_repeat, key))
        for key in repeat_impulse_keys
    }
    refined_repeat_impulse = {
        key: _relative_difference(
            _impulse(refined_first, key), _impulse(refined_repeat, key))
        for key in repeat_impulse_keys
    }
    dt_impulse = {
        key: _relative_difference(
            0.5 * (_impulse(coarse_first, key) +
                   _impulse(coarse_repeat, key)),
            0.5 * (_impulse(refined_first, key) +
                   _impulse(refined_repeat, key)))
        for key in repeat_impulse_keys
    }
    all_episodes = list(episodes.values())
    checks = {
        "coarse_reconstructed_repeat_bounded": (
            coarse_state_delta <=
            float(probe["maximum_repeat_state_difference"]) and
            max(coarse_repeat_impulse.values()) <=
            float(probe["maximum_repeat_impulse_relative_difference"])),
        "refined_reconstructed_repeat_bounded": (
            refined_state_delta <=
            float(probe["maximum_repeat_state_difference"]) and
            max(refined_repeat_impulse.values()) <=
            float(probe["maximum_repeat_impulse_relative_difference"])),
        "dt_refinement_bounded": (
            dt_state_delta <= float(probe["maximum_dt_state_difference"]) and
            max(dt_impulse[key] for key in probe["dt_acceptance_impulses"]) <=
            float(probe["maximum_dt_impulse_relative_difference"])),
        "feedback_changes_mechanism": (
            float(probe["minimum_feedback_mechanism_qpos_difference_m"]) <=
            feedback_mechanism_delta <=
            float(probe["maximum_feedback_mechanism_qpos_difference_m"])),
        "all_episode_health_checks_pass": all(
            _episode_health(episode, shared_config["probe"])
            for episode in all_episodes),
        "coarse_repeat_is_byte_exact": (
            coarse_first["trace_sha256"] == coarse_repeat["trace_sha256"] and
            coarse_first["matrix"]["particle_q_sha256"] ==
            coarse_repeat["matrix"]["particle_q_sha256"] and
            coarse_first["matrix"]["particle_qd_sha256"] ==
            coarse_repeat["matrix"]["particle_qd_sha256"]),
        "refined_repeat_is_byte_exact": (
            refined_first["trace_sha256"] == refined_repeat["trace_sha256"] and
            refined_first["matrix"]["particle_q_sha256"] ==
            refined_repeat["matrix"]["particle_q_sha256"] and
            refined_first["matrix"]["particle_qd_sha256"] ==
            refined_repeat["matrix"]["particle_qd_sha256"]),
        "scope_not_overclaimed": (
            config["scope"]["reconstructed_repeat_qualification"] and
            config["scope"]["dt_refinement_qualification"] and
            config["scope"]["feedback_causality_qualification"] and
            not config["scope"]["complete_in_process_reset"] and
            not config["scope"]["official_envelope"] and
            not config["scope"]["training"]),
    }
    qualified = bool(all(checks.values()))
    report = {
        "schema": REPORT_SCHEMA,
        "status": ("r3_c_convergence_probe_passed" if qualified else
                   "r3_c_convergence_probe_failed"),
        "convergence_qualified": qualified,
        "r3_c_physics_qualified": qualified,
        "training_qualified": False,
        "checks": {key: bool(value) for key, value in checks.items()},
        "comparison": {
            "feedback": {
                "maximum_final_mechanism_qpos_difference_m":
                    feedback_mechanism_delta,
            },
            "coarse_repeat": {
                "maximum_final_state_difference": coarse_state_delta,
                "impulse_relative_differences": coarse_repeat_impulse,
            },
            "refined_repeat": {
                "maximum_final_state_difference": refined_state_delta,
                "impulse_relative_differences": refined_repeat_impulse,
            },
            "dt_refinement": {
                "coarse_output_dt_s": float(probe["coarse_output_dt_s"]),
                "refined_output_dt_s": float(probe["refined_output_dt_s"]),
                "mujoco_internal_dt_preserved": True,
                "comparison_basis": "mean_of_two_reconstructed_episodes_per_dt",
                "maximum_final_state_difference": dt_state_delta,
                "impulse_relative_differences": dt_impulse,
                "acceptance_impulses": probe["dt_acceptance_impulses"],
                "rigid_components_are_diagnostic": True,
            },
        },
        "episodes": episodes,
        "inputs": {
            "config_sha256": sha256(config_path),
            "shared_config_sha256": sha256(shared_config_path),
            "shared_report_sha256": sha256(
                repository / frozen["shared_report_path"]),
            "derived_model_sha256": model_sha256,
            "model_manifest": model_manifest,
            "bridge_source_sha256": source_sha256,
            "bridge_source_manifest": source_manifest,
            "persisted_generated_artifacts": (
                ["generated-model.xml", "generated-model-manifest.json",
                 "generated-bridge.cc", "generated-bridge-manifest.json"]
                if artifact_dir is not None else []),
        },
        "implementation": {
            "runner_sha256": sha256(Path(__file__)),
            "compile_command": compile_command,
            "total_wall_seconds": time.perf_counter() - started,
        },
        "backend": {
            "python": platform.python_version(), "newton": newton.__version__,
            "warp": wp.__version__, "numpy": np.__version__,
            "device_requested": args.device,
            "device_resolved": str(wp.get_device()),
        },
        "reset_claims": {
            "reconstructed_episode_reset": True,
            "mpm_history_reset_audited_after_each_episode": True,
            "complete_in_process_mixed_reset": False,
        },
        "scope_limit": (
            "exactly two rocks in one shared small MPM matrix; reconstructed "
            "repeat, 1/120-to-1/240 outer-step refinement, and feedback "
            "causality only; no complete in-process mixed reset, official "
            "envelope, calibrated material, complete excavation, or training"),
    }
    _write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    report = run(parser.parse_args())
    print(json.dumps({
        "status": report["status"],
        "convergence_qualified": report["convergence_qualified"],
        "r3_c_physics_qualified": report["r3_c_physics_qualified"],
        "checks": report["checks"],
        "comparison": report["comparison"],
    }, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
