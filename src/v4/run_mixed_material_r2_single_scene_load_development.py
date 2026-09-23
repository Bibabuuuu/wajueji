"""Run one geometry-selected R2 trajectory adjustment before cross-scene confirmation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile

from instrument_r2_bridge_contacts import instrument_per_rock_contacts
from mixed_material_r2_mechanism import sha256, write_json
from run_mixed_material_r2_bridge_init import _compile
from run_mixed_material_r2_controlled_intake import (
    COMMAND_LIMITS,
    _exact_keys,
    _number,
    _validate_config as _validate_intake_config,
)
from run_mixed_material_r2_effective_load import (
    _run_scene,
    _validate_config as _validate_effective_load_config,
)
from run_mixed_material_r2_intake_lift import (
    _validate_config as _validate_lift_config,
)


SCHEMA = "excavation_mixed_material_r2_single_scene_load_development_config_v1"


def _validate_config(config: dict) -> None:
    _exact_keys(config, {
        "schema", "source_effective_load_config", "source_effective_load_report",
        "selected_scene", "baseline", "trajectory_change", "acceptance", "scope",
    }, "single_scene_load_development")
    if config["schema"] != SCHEMA:
        raise ValueError("invalid_single_scene_load_development_schema")
    _exact_keys(config["source_effective_load_config"], {"path", "sha256"},
                "source_effective_load_config")
    _exact_keys(config["source_effective_load_report"],
                {"path", "sha256", "required_status"},
                "source_effective_load_report")
    if (config["source_effective_load_report"]["required_status"] !=
            "two_scene_effective_carried_load_candidate"
            or config["selected_scene"] !=
            "ordinary-small-rock-dominant-seed-26091912"):
        raise ValueError("invalid_single_scene_source_scope")
    _exact_keys(config["baseline"],
                {"carried_rock_count", "preliminary_solid_volume_fill_rate"},
                "baseline")
    if (not isinstance(config["baseline"]["carried_rock_count"], int)
            or isinstance(config["baseline"]["carried_rock_count"], bool)
            or config["baseline"]["carried_rock_count"] != 2):
        raise ValueError("invalid_baseline_count")
    _number(config["baseline"]["preliminary_solid_volume_fill_rate"],
            "baseline_fill_rate", positive=True)
    change = config["trajectory_change"]
    _exact_keys(change, {"insert_after_phase", "phase", "rationale"},
                "trajectory_change")
    if change["insert_after_phase"] != "settle" or not change["rationale"]:
        raise ValueError("invalid_trajectory_change_anchor")
    phase = change["phase"]
    _exact_keys(phase, {"name", "duration_s", "command"}, "trajectory_phase")
    if (phase["name"] != "pre_align_rotation"
            or phase["duration_s"] != 1.5
            or phase["command"] != [-0.06, 0.0, 0.0, 0.0]):
        raise ValueError("trajectory_change_not_frozen")
    for axis, value in enumerate(phase["command"]):
        number = _number(value, "trajectory_command")
        if abs(number) > COMMAND_LIMITS[axis] + 1e-12:
            raise ValueError("trajectory_command_above_bridge_limit")
    acceptance = config["acceptance"]
    _exact_keys(acceptance, {
        "minimum_carried_rock_count", "require_fill_rate_above_baseline",
        "required_forbidden_contact_count", "maximum_closure_m",
    }, "development_acceptance")
    if (acceptance["minimum_carried_rock_count"] != 3
            or acceptance["require_fill_rate_above_baseline"] is not True
            or acceptance["required_forbidden_contact_count"] != 0
            or acceptance["maximum_closure_m"] != 0.001):
        raise ValueError("invalid_development_acceptance")
    scope = config["scope"]
    _exact_keys(scope, {
        "one_scene_trajectory_development", "trajectory_grid_search",
        "second_scene_confirmation", "all_six_scene_qualification", "dump",
        "truck_cad", "mpm_coupling", "training", "physics_qualified",
    }, "development_scope")
    if (scope["one_scene_trajectory_development"] is not True
            or any(scope[key] is not False for key in (
                "trajectory_grid_search", "second_scene_confirmation",
                "all_six_scene_qualification", "dump", "truck_cad",
                "mpm_coupling", "training", "physics_qualified"))):
        raise ValueError("expanded_single_scene_development_scope")


def _insert_phase(base_config: dict, config: dict) -> dict:
    candidate = copy.deepcopy(base_config)
    phases = candidate["phases"]
    anchor = config["trajectory_change"]["insert_after_phase"]
    indices = [index for index, phase in enumerate(phases)
               if phase["name"] == anchor]
    if indices != [0]:
        raise ValueError("single_scene_phase_anchor_changed")
    phases.insert(1, copy.deepcopy(config["trajectory_change"]["phase"]))
    candidate["selected_scenes"] = [config["selected_scene"]]
    return candidate


def run(config_path: str | Path, output_root: str | Path,
        repository_root: str | Path, cxx: str, mujoco_include: str | Path,
        mujoco_library_dir: str | Path, mechanism_header_dir: str | Path,
        controller_header_dir: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise ValueError("single_scene_load_output_already_exists")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    effective_path = repository / config["source_effective_load_config"]["path"]
    source_report_path = repository / config["source_effective_load_report"]["path"]
    if sha256(effective_path) != config["source_effective_load_config"]["sha256"]:
        raise ValueError("single_scene_effective_config_sha256_mismatch")
    if sha256(source_report_path) != config["source_effective_load_report"]["sha256"]:
        raise ValueError("single_scene_effective_report_sha256_mismatch")
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if source_report["status"] != config[
            "source_effective_load_report"]["required_status"]:
        raise ValueError("single_scene_effective_report_status")
    baseline = next(scene for scene in source_report["scenes"]
                    if scene["scene_id"] == config["selected_scene"])
    if (baseline["carried_rock_count"] != config["baseline"]["carried_rock_count"]
            or abs(baseline["preliminary_solid_volume_fill_rate"]
                   - config["baseline"]["preliminary_solid_volume_fill_rate"])
            > 1e-15):
        raise ValueError("single_scene_baseline_mismatch")
    effective_config = json.loads(effective_path.read_text(encoding="utf-8"))
    _validate_effective_load_config(effective_config)
    intake_path = repository / effective_config["source_intake_config"]["path"]
    lift_path = repository / effective_config["source_lift_config"]["path"]
    base_config = json.loads(intake_path.read_text(encoding="utf-8"))
    lift_config = json.loads(lift_path.read_text(encoding="utf-8"))
    _validate_intake_config(base_config)
    _validate_lift_config(lift_config)
    development_base = _insert_phase(base_config, config)

    stable_root = repository / base_config["stable_model_root"]["path"]
    stable_report = json.loads((stable_root / "report.json").read_text(encoding="utf-8"))
    stable_index = {item["scene_id"]: item for item in stable_report["scenes"]}
    frozen_bridge = stable_root / "generated-bridge-r2.cc"
    r2_config = json.loads((
        repository / base_config["source_r2_config"]["path"]
    ).read_text(encoding="utf-8"))
    plan_path = (repository / r2_config["assembly_preflight"]["root"] /
                 config["selected_scene"] / "assembly-plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    geometry = base_config["bucket_geometry"]
    inventory = json.loads(Path(geometry["inventory_path"]).read_text(encoding="utf-8"))
    proxy = json.loads((repository / geometry["proxy_manifest_path"]).read_text(
        encoding="utf-8"))
    model_path = stable_root / config["selected_scene"] / "model.xml"
    if sha256(model_path) != stable_index[config["selected_scene"]]["model_sha256"]:
        raise ValueError("single_scene_model_sha256_mismatch")

    output.mkdir(parents=True)
    instrumented = output / "generated-bridge-r2-per-rock-contacts.cc"
    instrumented.write_text(instrument_per_rock_contacts(
        frozen_bridge.read_text(encoding="utf-8")), encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="mixed-r2-single-scene-load-") as temporary:
        binary = Path(temporary) / "mixed-rigid-bridge-r2"
        compile_command = _compile(
            cxx, instrumented, binary,
            [mechanism_header_dir, controller_header_dir, mujoco_include],
            Path(mujoco_library_dir))
        compile_record = [
            "<temporary>/mixed-rigid-bridge-r2"
            if value == str(binary.resolve()) else value
            for value in compile_command]
        metrics = _run_scene(
            binary, model_path, plan, inventory, proxy,
            development_base, lift_config, effective_config)
    checks = {
        "effective_load_contract_passed": (
            metrics["status"] == "effective_carried_load_scene_candidate"),
        "carried_count_improved_to_three": metrics["carried_rock_count"] >= int(
            config["acceptance"]["minimum_carried_rock_count"]),
        "fill_rate_above_baseline": metrics[
            "preliminary_solid_volume_fill_rate"] > config["baseline"][
                "preliminary_solid_volume_fill_rate"],
        "no_forbidden_contacts": metrics[
            "maximum_forbidden_contact_count"] == 0,
        "closure_within_limit": metrics["maximum_closure_m"] <= float(
            config["acceptance"]["maximum_closure_m"]),
    }
    status = ("single_scene_load_development_candidate"
              if all(checks.values()) else "single_scene_load_development_needs_review")
    scene_report = {
        "schema": "excavation_mixed_material_r2_single_scene_load_development_scene_report_v1",
        "scene_id": config["selected_scene"],
        "scene_class": plan["scene_class"],
        "status": status,
        "development_checks": checks,
        "baseline": config["baseline"],
        "trajectory_change": config["trajectory_change"],
        "metrics": metrics,
        "physics_qualified": False,
    }
    scene_dir = output / config["selected_scene"]
    scene_dir.mkdir(parents=True)
    scene_path = scene_dir / "report.json"
    write_json(scene_path, scene_report)
    report = {
        "schema": "excavation_mixed_material_r2_single_scene_load_development_report_v1",
        "status": status,
        "scene_id": config["selected_scene"],
        "development_checks": checks,
        "baseline_carried_rock_count": config["baseline"]["carried_rock_count"],
        "candidate_carried_rock_count": metrics["carried_rock_count"],
        "baseline_preliminary_solid_volume_fill_rate": config["baseline"][
            "preliminary_solid_volume_fill_rate"],
        "candidate_preliminary_solid_volume_fill_rate": metrics[
            "preliminary_solid_volume_fill_rate"],
        "candidate_carried_rock_ids": metrics["carried_rock_ids"],
        "candidate_carried_mass_kg": metrics["carried_mass_kg"],
        "candidate_carried_solid_volume_m3": metrics["carried_solid_volume_m3"],
        "trajectory_grid_search": False,
        "second_scene_confirmation": False,
        "all_six_scene_qualification": False,
        "dump_qualified": False,
        "truck_cad_qualified": False,
        "mpm_coupling_qualified": False,
        "training_qualified": False,
        "physics_qualified": False,
        "inputs": {
            "config_sha256": sha256(config_path),
            "source_effective_load_config_sha256": sha256(effective_path),
            "source_effective_load_report_sha256": sha256(source_report_path),
            "model_sha256": sha256(model_path),
            "assembly_plan_sha256": sha256(plan_path),
            "frozen_bridge_sha256": sha256(frozen_bridge),
            "instrumented_bridge_sha256": sha256(instrumented),
        },
        "implementation": {
            "runner_sha256": sha256(
                repository /
                "src/v4/run_mixed_material_r2_single_scene_load_development.py"),
            "compile_command": compile_record,
        },
        "scene_report_sha256": sha256(scene_path),
        "scope_limit": (
            "one ordinary scene and one geometry-selected pre-alignment change; no "
            "grid search, second-scene confirmation, six-scene qualification, dump, "
            "truck, MPM, hydraulic capacity or training qualification"),
    }
    write_json(output / "report.json", report)
    (output / "README.md").write_text(
        "# R2单场装入量轨迹开发\n\n"
        f"状态：`{status}`。只在普通`26091912`加入高位负Y预回转；其余下降、"
        "卷斗、举升、保持及有效载荷门不变。该结果不代表第二场、六场、卸料、"
        "车厢、MPM或训练资格。\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    args = parser.parse_args()
    report = run(
        args.config, args.output_root, args.repository_root, args.cxx,
        args.mujoco_include, args.mujoco_library_dir,
        args.mechanism_header_dir, args.controller_header_dir)
    print(json.dumps({
        "status": report["status"],
        "candidate_carried_rock_count": report["candidate_carried_rock_count"],
        "candidate_preliminary_solid_volume_fill_rate": report[
            "candidate_preliminary_solid_volume_fill_rate"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
