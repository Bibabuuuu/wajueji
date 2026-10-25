#!/usr/bin/env python3
"""Retry the covered three-rock gate with one change: slow gravity release."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile

import numpy as np

import run_layered_mound_three_rock_cover_release as base
from build_mixed_material_capacity_preflight import sha256
from run_layered_mound_single_rock_press_smoke import _resolve
from run_mixed_material_r3_a_probe import _write_json


CONFIG_SCHEMA = "excavation_layered_mound_three_rock_slow_release_config_v1"
REPORT_SCHEMA = "excavation_layered_mound_three_rock_slow_release_report_v1"


def load_config(path: str | Path, repository_root: str | Path) -> tuple[dict, dict]:
    repository = Path(repository_root).resolve()
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (value.get("schema") != CONFIG_SCHEMA or value.get("status") !=
            "single_factor_slow_gravity_release_after_cover"):
        raise ValueError("slow_release_config_schema_or_status")
    binding = value.get("base_candidate", {})
    for field in ("config", "runner", "report"):
        target = _resolve(repository, binding.get(field + "_path", ""))
        if (not target.is_file() or sha256(target) !=
                binding.get(field + "_sha256")):
            raise ValueError(f"slow_release_base_{field}_binding")
    base_config = base.load_config(_resolve(
        repository, binding["config_path"]), repository)
    report = json.loads(_resolve(
        repository, binding["report_path"]).read_text(encoding="utf-8"))
    checks = report.get("checks", {})
    if (report.get("status") != "three_rocks_cover_then_free_release_failed"
            or checks.get("cover_continuous_stable_window_reached") is not True
            or checks.get("release_continuous_stable_window_reached") is not False
            or not all(result for name, result in checks.items()
                       if name != "release_continuous_stable_window_reached")):
        raise ValueError("slow_release_base_failure_contract")
    if value.get("correction") != {
            "reason":
            "avoid_one_external_step_full_gravity_impulse_when_releasing_held_rocks",
            "change_scope": [
                "release.external_upward_support_force_linear_ramp",
                "release.maximum_frames",
                "release.required_continuous_window_including_post_ramp_window"],
            "gravity_release_frames": 60,
            "gravity_release_duration_s": 0.5,
            "support_force_formula":
            "Fz=(1-alpha)*mass*9.81; alpha=(frame+1)/60",
            "maximum_release_frames": 300,
            "required_total_continuous_stable_frames": 120,
            "required_fully_released_stable_frames": 60,
            "target_runtime_rock_ids": [
                "r1-rock-000", "r1-rock-001", "r1-rock-002"],
            "mpm_material_changed": False,
            "rigid_damping_changed": False,
            "stability_thresholds_changed": False,
            "press_schedule_changed": False,
            "cover_schedule_changed": False}:
        raise ValueError("slow_release_correction_contract")
    if value.get("acceptance") != {
            "rock_translation_during_release_allowed": True,
            "rock_rotation_during_release_allowed": True,
            "final_manifest_pose_required": False,
            "fully_released_post_ramp_window_required": True}:
        raise ValueError("slow_release_acceptance_contract")
    if value.get("scope") != {
            "active_rock_count": 3, "other_62_rocks_active": False,
            "full_lower_layer": False, "full_scene": False,
            "stable_scene_claimed": False, "fullscale_qualified": False,
            "training": False}:
        raise ValueError("slow_release_scope_contract")
    return value, base_config


def run(args) -> dict:
    repository = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("slow_release_output_exists")
    config, base_config = load_config(config_path, repository)
    correction = config["correction"]
    manifest = json.loads(_resolve(
        repository, base_config["physical_geometry"]["manifest_path"]
    ).read_text(encoding="utf-8"))
    by_id = {rock["rock_id"]: rock for rock in manifest["rocks"]}
    masses = [float(by_id[target["manifest_rock_id"]]["mass_kg"])
              for target in base_config["targets"]]
    runtime_ids = list(correction["target_runtime_rock_ids"])
    ramp_frames = int(correction["gravity_release_frames"])
    ramp_state = {"calls": 0, "samples": []}
    original_step = base.RuntimeBridge.step

    def step_with_support(self, dt: float, substeps: int,
                          wrenches: dict[str, dict]) -> dict:
        frame = ramp_state["calls"]
        alpha = min(1.0, (frame + 1) / ramp_frames)
        adjusted = {}
        support_by_id = {}
        for body_name, wrench in wrenches.items():
            adjusted[body_name] = {
                "force": np.asarray(wrench["force"], dtype=np.float64).copy(),
                "torque": np.asarray(wrench["torque"], dtype=np.float64).copy(),
                "point": np.asarray(wrench["point"], dtype=np.float64).copy(),
            }
        for rock_id, mass in zip(runtime_ids, masses, strict=True):
            support = (1.0 - alpha) * mass * 9.81
            adjusted[rock_id]["force"][2] += support
            support_by_id[rock_id] = support
        if frame in {0, ramp_frames - 1}:
            ramp_state["samples"].append({
                "frame": frame, "gravity_fraction": alpha,
                "upward_support_force_n_by_runtime_id": support_by_id})
        ramp_state["calls"] += 1
        return original_step(self, dt, substeps, adjusted)

    legacy = copy.deepcopy(base_config)
    legacy["probe"]["maximum_release_frames"] = int(
        correction["maximum_release_frames"])
    legacy["probe"]["release_stable_window_frames"] = int(
        correction["required_total_continuous_stable_frames"])
    base.RuntimeBridge.step = step_with_support
    try:
        with tempfile.TemporaryDirectory(prefix="three-rock-slow-release-config-") as raw:
            legacy_path = Path(raw) / "legacy-compatible.json"
            legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
            legacy_args = argparse.Namespace(**vars(args))
            legacy_args.config = str(legacy_path)
            report = base.run(legacy_args)
    finally:
        base.RuntimeBridge.step = original_step

    release_steps = int(report["release"]["steps"])
    required_total = int(correction["required_total_continuous_stable_frames"])
    required_post = int(correction["required_fully_released_stable_frames"])
    post_ramp_window = (
        report["checks"]["release_continuous_stable_window_reached"]
        and release_steps >= ramp_frames + required_post
        and int(report["release"]["stable_window_frames"]) >= required_total)
    checks = dict(report["checks"])
    checks.update({
        "gravity_release_ramp_completed": ramp_state["calls"] >= ramp_frames,
        "fully_released_post_ramp_stable_window_reached": bool(post_ramp_window),
        "single_factor_scope_preserved": (
            not correction["mpm_material_changed"]
            and not correction["rigid_damping_changed"]
            and not correction["stability_thresholds_changed"]
            and not correction["press_schedule_changed"]
            and not correction["cover_schedule_changed"]),
    })
    report["schema"] = REPORT_SCHEMA
    report["status"] = ("three_rocks_slow_release_post_ramp_stable_passed"
                        if all(checks.values()) else
                        "three_rocks_slow_release_post_ramp_stable_failed")
    report["checks"] = checks
    report["gravity_release"] = {
        "method": "external_upward_support_force_linear_ramp_to_zero",
        "frames": ramp_frames,
        "duration_s": correction["gravity_release_duration_s"],
        "bridge_step_calls": ramp_state["calls"],
        "samples": ramp_state["samples"],
        "fully_released_post_ramp_required_frames": required_post,
        "fully_released_post_ramp_window_reached": bool(post_ramp_window),
    }
    report["inputs"].update({
        "config_sha256": sha256(config_path),
        "wrapper_runner_sha256": sha256(Path(__file__).resolve()),
        "base_config_sha256": config["base_candidate"]["config_sha256"],
        "base_runner_sha256": config["base_candidate"]["runner_sha256"],
        "base_failed_report_sha256": config["base_candidate"]["report_sha256"],
    })
    report["acceptance"] = config["acceptance"]
    report["scope"] = config["scope"]
    report["next_gate"] = (
        "review_three_rock_slow_release_before_any_full_lower_layer")
    _write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "status": report["status"], "checks": report["checks"],
        "gravity_release": report["gravity_release"],
        "release": {key: value for key, value in report["release"].items()
                    if key != "trace"},
        "execution": report["execution"],
    }, sort_keys=True, allow_nan=False))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
