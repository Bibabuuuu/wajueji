"""Generate and load-inspect the six first-admission R2 scene models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from mixed_material_r2_mechanism import (
    _validate_config,
    build_candidate,
    sha256,
    write_json,
)


def _compile(cxx: str, source: Path, output: Path, include_dir: Path,
             library_dir: Path) -> list[str]:
    command = [
        cxx, "-O2", "-std=c++17", "-Wall", "-Wextra",
        "-I" + str(include_dir.resolve()), str(source.resolve()),
        "-L" + str(library_dir.resolve()),
        "-Wl,-rpath," + str(library_dir.resolve()),
        "-l:libmujoco.so.2.3.8", "-o", str(output.resolve()),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError("r2_inspector_compile_failed:" + completed.stderr)
    return command


def build_all(config_path: str | Path, output_root: str | Path,
              repository_root: str | Path, cxx: str,
              mujoco_include: str | Path,
              mujoco_library_dir: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    output = Path(output_root).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    preflight_root = repository / config["assembly_preflight"]["root"]
    preflight_report_path = preflight_root / "report.json"
    if sha256(preflight_report_path) != config["assembly_preflight"]["root_report_sha256"]:
        raise ValueError("assembly_preflight_root_report_sha256_mismatch")
    preflight = json.loads(preflight_report_path.read_text(encoding="utf-8"))
    indexed = {record["scene_id"]: record for record in preflight["scenes"]}
    source_model = repository / config["source_model"]["path"]
    source_hash_before = sha256(source_model)
    inspector_source = repository / "src/v4/mixed_material_r2_model_inspect.cc"
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mixed-material-r2-load-") as temporary:
        inspector_binary = Path(temporary) / "mixed-material-r2-model-inspect"
        compile_command = _compile(
            cxx, inspector_source, inspector_binary,
            Path(mujoco_include), Path(mujoco_library_dir))
        compile_command_record = [
            "<temporary>/mixed-material-r2-model-inspect"
            if value == str(inspector_binary.resolve()) else value
            for value in compile_command
        ]
        scene_reports = []
        for scene_id in config["first_admission"]["scene_ids"]:
            if scene_id not in indexed:
                raise ValueError("first_admission_scene_missing_from_preflight")
            indexed_record = indexed[scene_id]
            if indexed_record["rock_count"] != 20:
                raise ValueError("first_admission_index_not_twenty_rocks")
            plan_path = preflight_root / scene_id / "assembly-plan.json"
            if sha256(plan_path) != indexed_record["assembly_plan_sha256"]:
                raise ValueError("assembly_plan_index_sha256_mismatch:" + scene_id)
            scene_output = output / scene_id
            scene_output.mkdir(parents=True, exist_ok=True)
            model_path = scene_output / "model.xml"
            manifest = build_candidate(
                config_path, plan_path, model_path, repository)
            manifest_path = scene_output / "model-manifest.json"
            write_json(manifest_path, manifest)
            completed = subprocess.run(
                [str(inspector_binary), str(model_path), "20"],
                text=True, capture_output=True, check=False)
            if completed.returncode:
                raise RuntimeError(
                    "r2_model_load_failed:" + scene_id + ":" + completed.stderr)
            inspection = json.loads(completed.stdout)
            expected = manifest["expected_candidate_contract"]
            checks = {
                "load_inspection_passed": inspection["status"] == "passed",
                "load_only": inspection["load_only"] is True,
                "no_forward_dynamics": inspection["forward_dynamics_called"] is False,
                "no_time_step": inspection["time_step_called"] is False,
                "not_physics_qualified": inspection["physics_qualified"] is False,
                "exact_twenty_rocks": inspection["rock_count"] == 20,
                "no_placeholder_rocks": inspection["placeholder_rock_count"] == 0,
                "single_capacity_twenty": inspection["maximum_rock_capacity"] == 20,
                "dimension_contract": all(
                    inspection[key] == expected[key]
                    for key in ("nq", "nv", "nu", "nplugin", "rock_count")
                ),
                "bucket_proxy_contract": (
                    inspection["bucket_proxy_geom_count"] ==
                    manifest["bucket_proxy_geom_count"] == 18
                ),
                "static_heightfield_contract": (
                    inspection["heightfield_count"] == 1 and
                    inspection["heightfield_nrow"] ==
                    manifest["surface_support"]["nrow"] == 65 and
                    inspection["heightfield_ncol"] ==
                    manifest["surface_support"]["ncol"] == 65 and
                    inspection["source_ground_plane_rock_collision_enabled"]
                    is False and
                    manifest["surface_support"]["maximum_interpolation_error_m"] <=
                    config["surface_support"]["maximum_interpolation_error_m"]
                ),
                "mass_contract": abs(
                    inspection["total_rock_mass_kg"] -
                    manifest["total_rock_mass_kg"]
                ) <= 1e-8,
            }
            if not all(checks.values()):
                failed = [key for key, value in checks.items() if not value]
                raise RuntimeError("r2_load_contract_failed:" + scene_id + ":" + ",".join(failed))
            scene_report = {
                "schema": "excavation_mixed_material_r2_scene_load_report_v1",
                "status": "r2_load_only_passed",
                "scene_id": scene_id,
                "scene_class": manifest["scene_class"],
                "rock_count": 20,
                "maximum_rock_capacity": 20,
                "placeholder_rock_count": 0,
                "assembly_plan_sha256": sha256(plan_path),
                "model_sha256": sha256(model_path),
                "model_manifest_sha256": sha256(manifest_path),
                "heightfield_asset_sha256": manifest["surface_support"]["asset_sha256"],
                "heightfield_maximum_interpolation_error_m": (
                    manifest["surface_support"]["maximum_interpolation_error_m"]),
                "inspection": inspection,
                "checks": checks,
                "physics_qualified": False,
                "scope_limit": (
                    "mj_loadXML structural inspection only; no forward dynamics, "
                    "settling, excavation, retention, dump or fill-rate qualification"
                ),
            }
            scene_report_path = scene_output / "report.json"
            write_json(scene_report_path, scene_report)
            scene_reports.append({
                "scene_id": scene_id,
                "scene_class": manifest["scene_class"],
                "rock_count": 20,
                "model_sha256": scene_report["model_sha256"],
                "model_manifest_sha256": scene_report["model_manifest_sha256"],
                "heightfield_asset_sha256": scene_report["heightfield_asset_sha256"],
                "heightfield_maximum_interpolation_error_m": scene_report[
                    "heightfield_maximum_interpolation_error_m"],
                "report_sha256": sha256(scene_report_path),
            })

    source_unchanged = sha256(source_model) == source_hash_before
    checks = {
        "six_first_admission_scenes": len(scene_reports) == 6,
        "all_scenes_have_twenty_real_rocks": all(
            record["rock_count"] == 20 for record in scene_reports),
        "single_maximum_capacity_twenty": config["capacity"]["maximum_rock_count"] == 20,
        "no_placeholder_rocks": config["capacity"]["placeholder_rocks_allowed"] is False,
        "runtime_count_fixed": config["capacity"]["runtime_add_remove_allowed"] is False,
        "source_model_unchanged": source_unchanged,
        "jam_boundary_deferred": (
            config["first_admission"]["deferred_scene_class"] == "jam-boundary" and
            config["first_admission"]["deferred_scene_count"] == 3
        ),
        "load_only_not_physics": config["scope"]["physics_qualification"] is False,
        "static_heightfield_in_every_scene": all(
            record["heightfield_asset_sha256"] for record in scene_reports),
    }
    if not all(checks.values()):
        raise RuntimeError("r2_root_load_contract_failed")
    report = {
        "schema": "excavation_mixed_material_r2_load_report_v1",
        "status": "r2_six_scene_load_only_passed",
        "scene_count": len(scene_reports),
        "rock_count_per_scene": 20,
        "total_loaded_rock_records": sum(record["rock_count"] for record in scene_reports),
        "maximum_rock_capacity": 20,
        "bridge_implementation_count_planned": 1,
        "placeholder_rocks_used": False,
        "runtime_add_remove_used": False,
        "deferred_jam_boundary_scene_count": 3,
        "physics_qualified": False,
        "checks": checks,
        "scenes": scene_reports,
        "inputs": {
            "config_sha256": sha256(config_path),
            "assembly_preflight_report_sha256": sha256(preflight_report_path),
            "source_model_sha256": sha256(source_model),
        },
        "implementation": {
            "assembler_sha256": sha256(
                repository / "src/v4/mixed_material_r2_mechanism.py"),
            "inspector_sha256": sha256(inspector_source),
            "compile_command": compile_command_record,
        },
        "scope_limit": (
            "R2 candidate generation plus load-only structural inspection; the bridge "
            "is not modified and no multi-rock physics or bucket fill-rate result is claimed"
        ),
    }
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    args = parser.parse_args()
    report = build_all(
        args.config, args.output_root, args.repository_root, args.cxx,
        args.mujoco_include, args.mujoco_library_dir)
    print(json.dumps({
        "status": report["status"],
        "scene_count": report["scene_count"],
        "rock_count_per_scene": report["rock_count_per_scene"],
        "physics_qualified": report["physics_qualified"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
