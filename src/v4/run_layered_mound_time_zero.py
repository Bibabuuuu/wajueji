#!/usr/bin/env python3
"""Load the admitted 65-rock mound and carved MPM bank at time zero only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import tempfile
import time

import newton
import numpy as np
import warp as wp

from build_mixed_material_capacity_preflight import render_capacity_bridge, sha256
from run_layered_mixed_scene_local_dynamics_smoke import Fixture, _build_provisional_model
from run_mixed_material_capacity_preflight import _time_zero
from run_mixed_material_r3_a_probe import _compile_bridge, _write_json


CONFIG_SCHEMA = "excavation_layered_mound_time_zero_config_v1"
REPORT_SCHEMA = "excavation_layered_mound_time_zero_report_v1"


def _finite_tree(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    return False


def load_config(path: str | Path, repository_root: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (value.get("schema") != CONFIG_SCHEMA or value.get("status") !=
            "actual_65_rock_and_mpm_time_zero_only"):
        raise ValueError("layered_mound_time_zero_schema_or_status")
    for group, fields in (("geometry", ("config", "report", "manifest", "bank")),
                          ("capacity_bridge", ("config",))):
        binding = value.get(group, {})
        for field in fields:
            target = repository / binding.get(field + "_path", "")
            if (not target.is_file() or sha256(target) !=
                    binding.get(field + "_sha256")):
                raise ValueError(f"layered_mound_time_zero_{group}_{field}_binding")
    report = json.loads((repository / value["geometry"]["report_path"])
                        .read_text(encoding="utf-8"))
    manifest = json.loads((repository / value["geometry"]["manifest_path"])
                          .read_text(encoding="utf-8"))
    if (report.get("status") != "nonintersecting_geometry_preflight_passed"
            or not all(report.get("checks", {}).values())
            or report.get("stable_scene_claimed") is not False
            or manifest.get("status") != "nonintersecting_geometry_admitted_for_time_zero"
            or manifest.get("rock_count") != 65
            or not all(manifest.get("checks", {}).values())
            or manifest.get("stable_scene_claimed") is not False):
        raise ValueError("layered_mound_time_zero_geometry_scope")
    with np.load(repository / value["geometry"]["bank_path"], allow_pickle=False) as bank:
        if (set(bank.files) != {"q", "material_id", "activation_stage",
                               "original_bank_index", "removed_original_bank_index"}
                or bank["q"].shape != (263673, 3)
                or set(bank["activation_stage"].tolist()) != set(range(6))
                or not np.isfinite(bank["q"]).all()):
            raise ValueError("layered_mound_time_zero_bank_contract")
    if value.get("expected") != {
            "rock_count": 65, "particle_count": 263673,
            "activation_stage_count": 6, "nq": 474, "nv": 408, "nu": 12}:
        raise ValueError("layered_mound_time_zero_expected_contract")
    if value.get("scope") != {
            "mujoco_ready_and_init_default": True,
            "newton_model_and_collider_setup": True,
            "mpm_initial_state_load": True,
            "mj_step_calls": 0, "newton_solver_step_calls": 0,
            "settling": False, "stable_scene_claimed": False,
            "fullscale_qualified": False, "training": False}:
        raise ValueError("layered_mound_time_zero_scope")
    return value


def run(args) -> dict:
    repository = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("layered_mound_time_zero_output_exists")
    config = load_config(config_path, repository)
    manifest_path = repository / config["geometry"]["manifest_path"]
    bank_path = repository / config["geometry"]["bank_path"]
    capacity_path = repository / config["capacity_bridge"]["config_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = config["expected"]
    started = time.perf_counter()
    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    with tempfile.TemporaryDirectory(prefix="layered-mound-time-zero-") as raw:
        work = Path(raw)
        model = work / "actual-65.xml"
        model_manifest = _build_provisional_model(
            capacity_path, repository, manifest, model)
        source_path = work / "bridge.cc"
        source, bridge_manifest = render_capacity_bridge(capacity_path, repository)
        source_path.write_text(source, encoding="utf-8")
        binary = work / "bridge"
        compile_command = _compile_bridge(
            args.cxx, source_path, binary,
            [args.mechanism_header_dir, args.controller_header_dir, args.mujoco_include],
            Path(args.mujoco_library_dir))
        ready, state, timings = _time_zero(binary, model)
        fixture = Fixture(config, repository, manifest, bank_path, state, args.device)
        particle = fixture.metrics()
        body_q = fixture.state.body_q.numpy()
        body_qd = fixture.state.body_qd.numpy()

    ids = [f"r1-rock-{index:03d}" for index in range(expected["rock_count"])]
    rocks = state.get("rocks", [])
    applied = state.get("applied_rocks_mpm", [])
    checks = {
        "mujoco_ready_protocol": (
            ready.get("rock_count") == expected["rock_count"]
            and ready.get("maximum_external_wrench_count") == expected["rock_count"] + 1
            and ready.get("allowed_external_wrench_bodies") == ["bucket", *ids]),
        "mujoco_time_zero_state": state.get("time_s") == 0,
        "mujoco_actual_n_and_dimensions": (
            [item.get("rock_id") for item in rocks] == ids
            and len(state.get("mechanism_qpos", [])) == 19
            and len(state.get("mechanism_qvel", [])) == 18
            and model_manifest["rock_count"] == expected["rock_count"]
            and model_manifest["base_capacity_model_manifest"]["expected_nq"] == expected["nq"]
            and model_manifest["base_capacity_model_manifest"]["expected_nv"] == expected["nv"]),
        "mujoco_state_finite": _finite_tree(state),
        "mujoco_zero_external_wrenches": (
            [item.get("rock_id") for item in applied] == ids
            and all(item.get("force_world_n") == [0, 0, 0]
                    and item.get("torque_world_nm") == [0, 0, 0] for item in applied)),
        "mujoco_zero_forbidden_contacts": state.get("forbidden_contact_count") == 0,
        "newton_particle_count": particle["particle_count"] == expected["particle_count"],
        "newton_particle_state_finite_and_zero_speed": (
            particle["finite"] and particle["maximum_particle_speed_m_s"] == 0.0),
        "newton_rock_collider_identity": (
            len(fixture.collider_ids) == expected["rock_count"]
            and set(fixture.collider_ids) == set(ids)
            and all(len(value) == 1 for value in fixture.collider_ids.values())),
        "newton_body_state_finite": (
            np.isfinite(body_q).all() and np.isfinite(body_qd).all()
            and np.allclose(body_qd, 0.0, rtol=0.0, atol=0.0)),
        "zero_physics_steps": True,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": ("actual_65_rock_and_mpm_time_zero_passed"
                   if all(checks.values()) else "actual_65_rock_and_mpm_time_zero_failed"),
        "checks": checks,
        "metrics": {
            "rock_count": len(rocks), "particle": particle,
            "newton_collider_count": len(fixture.collider_ids),
            "mujoco": timings,
            "model_nq": expected["nq"], "model_nv": expected["nv"],
        },
        "inputs": {
            "config_sha256": sha256(config_path),
            "geometry_manifest_sha256": sha256(manifest_path),
            "geometry_bank_sha256": sha256(bank_path),
            "model_manifest": model_manifest,
            "bridge_manifest": bridge_manifest,
        },
        "environment": {
            "python": platform.python_version(), "newton": newton.__version__,
            "warp": wp.__version__, "numpy": np.__version__,
            "device": str(wp.get_device()),
        },
        "execution": {
            "compile_command": compile_command,
            "mj_step_calls": 0, "newton_solver_step_calls": 0,
            "wall_seconds": time.perf_counter() - started,
        },
        "stable_scene_claimed": False,
        "next_gate": "one_lower_rock_controlled_press_short_smoke",
    }
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
    print(json.dumps({"status": report["status"], "checks": report["checks"],
                      "metrics": report["metrics"]}, sort_keys=True, allow_nan=False))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
