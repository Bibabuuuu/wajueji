"""Re-run the frozen v1 pre-alignment with one bounded lift-duration extension."""

from __future__ import annotations

import argparse
import copy
import json
import math
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
from run_mixed_material_r2_single_scene_load_development import (
    _insert_phase,
    _validate_config as _validate_v1_config,
)


SCHEMA = "excavation_mixed_material_r2_single_scene_load_development_config_v2"
RUNNER_PATH = "src/v4/run_mixed_material_r2_single_scene_load_development_v2.py"


def _validate_config(config: dict) -> None:
    _exact_keys(config, {
        "schema", "source_v1_config", "source_v1_report",
        "lift_phase_extension", "acceptance", "scope",
    }, "single_scene_load_development_v2")
    if config["schema"] != SCHEMA:
        raise ValueError("invalid_single_scene_load_development_v2_schema")
    _exact_keys(config["source_v1_config"], {"path", "sha256"},
                "source_v1_config")
    _exact_keys(config["source_v1_report"],
                {"path", "sha256", "required_status"},
                "source_v1_report")
    if config["source_v1_report"]["required_status"] != (
            "single_scene_load_development_needs_review"):
        raise ValueError("invalid_v1_development_status")
    change = config["lift_phase_extension"]
    _exact_keys(change, {
        "phase_name", "source_duration_s", "candidate_duration_s",
        "command", "rationale",
    }, "lift_phase_extension")
    if (change["phase_name"] != "short_lift"
            or change["source_duration_s"] != 1.5
            or change["candidate_duration_s"] != 2.0
            or change["command"] != [0.0, 0.02, 0.0, 0.0]
            or not isinstance(change["rationale"], str)
            or not change["rationale"].strip()):
        raise ValueError("lift_phase_extension_not_frozen")
    for name in ("source_duration_s", "candidate_duration_s"):
        value = _number(change[name], name, positive=True)
        if not math.isfinite(value):
            raise ValueError("nonfinite_lift_duration")
    for axis, value in enumerate(change["command"]):
        number = _number(value, "lift_phase_command")
        if not math.isfinite(number) or abs(number) > COMMAND_LIMITS[axis] + 1e-12:
            raise ValueError("lift_phase_command_above_bridge_limit")
    acceptance = config["acceptance"]
    _exact_keys(acceptance, {
        "minimum_carried_rock_count", "require_fill_rate_above_baseline",
        "required_forbidden_contact_count", "maximum_closure_m",
    }, "development_v2_acceptance")
    if (acceptance["minimum_carried_rock_count"] != 3
            or acceptance["require_fill_rate_above_baseline"] is not True
            or acceptance["required_forbidden_contact_count"] != 0
            or acceptance["maximum_closure_m"] != 0.001):
        raise ValueError("invalid_development_v2_acceptance")
    scope = config["scope"]
    _exact_keys(scope, {
        "one_scene_trajectory_development", "trajectory_grid_search",
        "second_scene_confirmation", "all_six_scene_qualification", "dump",
        "truck_cad", "mpm_coupling", "training", "physics_qualified",
    }, "development_v2_scope")
    if (scope["one_scene_trajectory_development"] is not True
            or any(scope[key] is not False for key in (
                "trajectory_grid_search", "second_scene_confirmation",
                "all_six_scene_qualification", "dump", "truck_cad",
                "mpm_coupling", "training", "physics_qualified"))):
        raise ValueError("expanded_single_scene_development_v2_scope")


def _extend_lift(lift_config: dict, config: dict) -> dict:
    candidate = copy.deepcopy(lift_config)
    change = config["lift_phase_extension"]
    matches = [phase for phase in candidate["additional_phases"]
               if phase["name"] == change["phase_name"]]
    if len(matches) != 1:
        raise ValueError("single_short_lift_phase_required")
    phase = matches[0]
    if (phase["duration_s"] != change["source_duration_s"]
            or phase["command"] != change["command"]):
        raise ValueError("source_short_lift_phase_changed")
    phase["duration_s"] = change["candidate_duration_s"]
    return candidate


def run(config_path: str | Path, output_root: str | Path,
        repository_root: str | Path, cxx: str, mujoco_include: str | Path,
        mujoco_library_dir: str | Path, mechanism_header_dir: str | Path,
        controller_header_dir: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise ValueError("single_scene_load_v2_output_already_exists")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)

    v1_config_path = repository / config["source_v1_config"]["path"]
    v1_report_path = repository / config["source_v1_report"]["path"]
    if sha256(v1_config_path) != config["source_v1_config"]["sha256"]:
        raise ValueError("source_v1_config_sha256_mismatch")
    if sha256(v1_report_path) != config["source_v1_report"]["sha256"]:
        raise ValueError("source_v1_report_sha256_mismatch")
    v1_config = json.loads(v1_config_path.read_text(encoding="utf-8"))
    v1_report = json.loads(v1_report_path.read_text(encoding="utf-8"))
    _validate_v1_config(v1_config)
    if v1_report["status"] != config["source_v1_report"]["required_status"]:
        raise ValueError("source_v1_report_status_mismatch")

    effective_path = repository / v1_config["source_effective_load_config"]["path"]
    effective_report_path = repository / v1_config["source_effective_load_report"]["path"]
    if sha256(effective_path) != v1_config["source_effective_load_config"]["sha256"]:
        raise ValueError("v2_effective_config_sha256_mismatch")
    if sha256(effective_report_path) != v1_config["source_effective_load_report"]["sha256"]:
        raise ValueError("v2_effective_report_sha256_mismatch")
    effective_config = json.loads(effective_path.read_text(encoding="utf-8"))
    effective_report = json.loads(effective_report_path.read_text(encoding="utf-8"))
    _validate_effective_load_config(effective_config)
    if effective_report["status"] != v1_config[
            "source_effective_load_report"]["required_status"]:
        raise ValueError("v2_effective_report_status_mismatch")
    baseline = next(scene for scene in effective_report["scenes"]
                    if scene["scene_id"] == v1_config["selected_scene"])
    if (baseline["carried_rock_count"] != v1_config["baseline"]["carried_rock_count"]
            or abs(baseline["preliminary_solid_volume_fill_rate"]
                   - v1_config["baseline"]["preliminary_solid_volume_fill_rate"])
            > 1e-15):
        raise ValueError("v2_baseline_mismatch")

    intake_path = repository / effective_config["source_intake_config"]["path"]
    lift_path = repository / effective_config["source_lift_config"]["path"]
    base_config = json.loads(intake_path.read_text(encoding="utf-8"))
    lift_config = json.loads(lift_path.read_text(encoding="utf-8"))
    _validate_intake_config(base_config)
    _validate_lift_config(lift_config)
    development_base = _insert_phase(base_config, v1_config)
    development_lift = _extend_lift(lift_config, config)

    stable_root = repository / base_config["stable_model_root"]["path"]
    stable_report = json.loads((stable_root / "report.json").read_text(encoding="utf-8"))
    stable_index = {item["scene_id"]: item for item in stable_report["scenes"]}
    frozen_bridge = stable_root / "generated-bridge-r2.cc"
    r2_config = json.loads((
        repository / base_config["source_r2_config"]["path"]
    ).read_text(encoding="utf-8"))
    scene_id = v1_config["selected_scene"]
    plan_path = repository / r2_config["assembly_preflight"]["root"] / scene_id / "assembly-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    geometry = base_config["bucket_geometry"]
    inventory = json.loads(Path(geometry["inventory_path"]).read_text(encoding="utf-8"))
    proxy_path = repository / geometry["proxy_manifest_path"]
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    model_path = stable_root / scene_id / "model.xml"
    if sha256(model_path) != stable_index[scene_id]["model_sha256"]:
        raise ValueError("single_scene_v2_model_sha256_mismatch")

    output.mkdir(parents=True)
    instrumented = output / "generated-bridge-r2-per-rock-contacts.cc"
    instrumented.write_text(instrument_per_rock_contacts(
        frozen_bridge.read_text(encoding="utf-8")), encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="mixed-r2-single-scene-load-v2-") as temporary:
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
            development_base, development_lift, effective_config)

    checks = {
        "effective_load_contract_passed": (
            metrics["status"] == "effective_carried_load_scene_candidate"),
        "carried_count_improved_to_three": metrics["carried_rock_count"] >= int(
            config["acceptance"]["minimum_carried_rock_count"]),
        "fill_rate_above_baseline": metrics[
            "preliminary_solid_volume_fill_rate"] > v1_config["baseline"][
                "preliminary_solid_volume_fill_rate"],
        "no_forbidden_contacts": metrics["maximum_forbidden_contact_count"] == 0,
        "closure_within_limit": metrics["maximum_closure_m"] <= float(
            config["acceptance"]["maximum_closure_m"]),
    }
    status = ("single_scene_load_development_candidate"
              if all(checks.values()) else "single_scene_load_development_needs_review")
    scene_report = {
        "schema": "excavation_mixed_material_r2_single_scene_load_development_scene_report_v2",
        "scene_id": scene_id,
        "scene_class": plan["scene_class"],
        "status": status,
        "development_checks": checks,
        "baseline": v1_config["baseline"],
        "inherited_pre_alignment": v1_config["trajectory_change"],
        "lift_phase_extension": config["lift_phase_extension"],
        "metrics": metrics,
        "physics_qualified": False,
    }
    scene_dir = output / scene_id
    scene_dir.mkdir(parents=True)
    scene_path = scene_dir / "report.json"
    write_json(scene_path, scene_report)
    report = {
        "schema": "excavation_mixed_material_r2_single_scene_load_development_report_v2",
        "status": status,
        "scene_id": scene_id,
        "development_checks": checks,
        "baseline_carried_rock_count": v1_config["baseline"]["carried_rock_count"],
        "v1_candidate_carried_rock_count": v1_report["candidate_carried_rock_count"],
        "candidate_carried_rock_count": metrics["carried_rock_count"],
        "baseline_preliminary_solid_volume_fill_rate": v1_config["baseline"][
            "preliminary_solid_volume_fill_rate"],
        "v1_candidate_preliminary_solid_volume_fill_rate": v1_report[
            "candidate_preliminary_solid_volume_fill_rate"],
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
            "source_v1_config_sha256": sha256(v1_config_path),
            "source_v1_report_sha256": sha256(v1_report_path),
            "source_effective_load_config_sha256": sha256(effective_path),
            "source_effective_load_report_sha256": sha256(effective_report_path),
            "model_sha256": sha256(model_path),
            "assembly_plan_sha256": sha256(plan_path),
            "frozen_bridge_sha256": sha256(frozen_bridge),
            "instrumented_bridge_sha256": sha256(instrumented),
        },
        "implementation": {
            "runner_sha256": sha256(repository / RUNNER_PATH),
            "compile_command": compile_record,
        },
        "scene_report_sha256": sha256(scene_path),
        "scope_limit": (
            "one ordinary scene, the frozen v1 pre-alignment and one 0.5 s lift "
            "extension; no threshold relaxation, grid search, second-scene "
            "confirmation, six-scene qualification, dump, truck, MPM, hydraulic "
            "capacity or training qualification"),
    }
    write_json(output / "report.json", report)
    (output / "README.md").write_text(
        "# R2单场装入量轨迹开发 v2\n\n"
        f"状态：`{status}`。保留v1高位负Y预回转，只把低速短举升从1.5秒延长"
        "到2.0秒；有效载荷判定门槛不变。该结果不代表第二场、六场、卸料、"
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
