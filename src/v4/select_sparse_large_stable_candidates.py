"""Select replacement sparse-large scenes by fixed-sequence post-settling admission."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

from build_mixed_rigid_bridge_r2 import write_bridge
from build_scene_assembly_preflight import (
    _build_plan,
    _load_prototypes as load_preflight_prototypes,
    _validate_rock,
    load_config as load_preflight_config,
    write_json as write_preflight_json,
)
from build_scene_reconstruction import (
    _build_scene,
    _load_prototypes as load_reconstruction_prototypes,
    load_config as load_reconstruction_config,
)
from mixed_material_r2_mechanism import build_candidate, sha256, write_json
from run_mixed_material_r2_bridge_init import _compile, _finite
from run_mixed_material_r2_stabilization import (
    _pre_settle,
    _spatial_review,
    _stable_state,
    _validate_config as validate_stabilization_config,
    _verify_rebaked,
)


CONFIG_SCHEMA = "excavation_sparse_large_stable_admission_config_v1"


def _validate_config(config: dict) -> None:
    if config.get("schema") != CONFIG_SCHEMA or not _finite(config):
        raise ValueError("invalid_sparse_large_admission_config")
    selection = config["selection"]
    seeds = selection["fixed_seed_sequence"]
    if (selection["scene_class"] != "sparse-large-rock" or
            not isinstance(seeds, list) or len(seeds) < 2 or
            len(seeds) != len(set(seeds)) or
            any(not isinstance(seed, int) or seed <= 0 for seed in seeds) or
            int(selection["required_selected_count"]) != 2 or
            selection["stop_after_required_count"] is not True or
            selection["preserve_rejected_candidates"] is not True or
            selection["existing_retained_scene_id"] !=
            "sparse-large-rock-seed-26091922"):
        raise ValueError("invalid_sparse_large_selection_contract")
    admission = config["admission"]
    if (int(admission["required_rock_count"]) != 20 or
            float(admission["minimum_boundary_clearance_m_after_settling"]) != 0.08 or
            float(admission["maximum_rebaked_displacement_m"]) != 0.005 or
            not all(admission[key] is True for key in (
                "require_all_vertices_inside_source_region",
                "require_rock_sizes_and_masses_unchanged",
                "require_pre_settling_stability",
                "require_rebaked_stability"))):
        raise ValueError("invalid_sparse_large_admission_thresholds")
    scope = config["scope"]
    if (scope["final_scene_count_changed"] or scope["official_data_calibration"] or
            scope["source_configs_modified"] or scope["controlled_excavation"] or
            scope["training"]):
        raise ValueError("sparse_large_selection_scope_expanded")


def _source_paths(config: dict, repository: Path) -> dict[str, Path]:
    result = {}
    for key, record in config["source_configs"].items():
        path = repository / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError("source_config_sha256_mismatch:" + key)
        result[key] = path
    return result


def _candidate_plan(preflight_config: dict, reconstruction_config: dict,
                    manifest: dict, manifest_path: Path,
                    metrics_path: Path, repository: Path) -> dict:
    prototypes = load_preflight_prototypes(preflight_config, repository)
    seen_ids = set()
    rocks = [
        _validate_rock(
            rock, prototypes, reconstruction_config["sources"], seen_ids)
        for rock in manifest["rocks"]
    ]
    scene = {
        "scene_id": manifest["scene_id"],
        "scene_class": manifest["scene_class"],
        "seed": manifest["seed"],
        "rock_count": len(rocks),
        "rocks": rocks,
        "input_hashes": {
            "scene_report_sha256": sha256(metrics_path),
            "scene_manifest_sha256": sha256(manifest_path),
            "scene_metrics_sha256": sha256(metrics_path),
            "preview_sha256": sha256(manifest_path),
        },
    }
    return _build_plan(preflight_config, scene)


def run(config_path: str | Path, output_root: str | Path,
        repository_root: str | Path, cxx: str, mujoco_include: str | Path,
        mujoco_library_dir: str | Path, mechanism_header_dir: str | Path,
        controller_header_dir: str | Path) -> dict:
    repository = Path(repository_root).resolve()
    config_path = Path(config_path).resolve()
    output = Path(output_root).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    sources = _source_paths(config, repository)
    reconstruction_config = load_reconstruction_config(
        sources["scene_reconstruction"])
    preflight_config = load_preflight_config(sources["scene_assembly_preflight"])
    r2_config = json.loads(
        sources["mixed_material_r2"].read_text(encoding="utf-8"))
    stabilization_config = json.loads(
        sources["stabilization"].read_text(encoding="utf-8"))
    validate_stabilization_config(stabilization_config)
    scene_class = next(
        item for item in reconstruction_config["scene_classes"]
        if item["scene_class"] == config["selection"]["scene_class"])
    reconstruction_prototypes = load_reconstruction_prototypes(
        reconstruction_config, repository)
    output.mkdir(parents=True, exist_ok=True)
    generated_source = output / "generated-bridge-r2.cc"
    source_manifest = write_bridge(
        sources["mixed_material_r2"], repository, generated_source)
    write_json(output / "generated-bridge-r2-manifest.json", source_manifest)

    with tempfile.TemporaryDirectory(prefix="sparse-large-selection-") as temporary:
        binary = Path(temporary) / "mixed-rigid-bridge-r2"
        compile_command = _compile(
            cxx, generated_source, binary,
            [mechanism_header_dir, controller_header_dir, mujoco_include],
            Path(mujoco_library_dir))
        compile_command_record = [
            "<temporary>/mixed-rigid-bridge-r2"
            if value == str(binary.resolve()) else value
            for value in compile_command
        ]
        selected = []
        candidates = []
        required = int(config["selection"]["required_selected_count"])
        for sequence_index, seed in enumerate(
                config["selection"]["fixed_seed_sequence"]):
            manifest, metrics = _build_scene(
                reconstruction_config, scene_class, seed,
                reconstruction_prototypes)
            scene_id = manifest["scene_id"]
            scene_output = output / scene_id
            scene_output.mkdir(parents=True, exist_ok=True)
            manifest_path = scene_output / "scene-manifest.json"
            metrics_path = scene_output / "scene-metrics.json"
            write_json(manifest_path, manifest)
            write_json(metrics_path, metrics)
            plan = _candidate_plan(
                preflight_config, reconstruction_config, manifest,
                manifest_path, metrics_path, repository)
            plan_path = scene_output / "assembly-plan.json"
            write_preflight_json(plan_path, plan)

            baseline_model = scene_output / "baseline-model.xml"
            baseline_manifest = build_candidate(
                sources["mixed_material_r2"], plan_path, baseline_model,
                repository, surface_scene_manifest_path=manifest_path)
            write_json(scene_output / "baseline-model-manifest.json", baseline_manifest)
            final_state, pre_metrics = _pre_settle(
                binary, baseline_model, stabilization_config["pre_settling"])
            stable = _stable_state(
                scene_id, plan, final_state, pre_metrics, baseline_model)
            stable["source_assembly_plan_sha256"] = sha256(plan_path)
            stable_path = scene_output / "stable-initial-state.json"
            write_json(stable_path, stable)
            spatial = _spatial_review(
                plan, stable, manifest, repository, stabilization_config)
            rebaked_model = scene_output / "model.xml"
            rebaked_manifest = build_candidate(
                sources["mixed_material_r2"], plan_path, rebaked_model,
                repository, stable_path, manifest_path)
            rebaked_manifest_path = scene_output / "model-manifest.json"
            write_json(rebaked_manifest_path, rebaked_manifest)
            verification = _verify_rebaked(
                binary, rebaked_model,
                stabilization_config["rebaked_verification"])
            checks = {
                "geometry_admission_passed": metrics["all_admission_checks_passed"],
                "required_rock_count": len(plan["rocks"]) ==
                int(config["admission"]["required_rock_count"]),
                "pre_settling_stable": all(pre_metrics["checks"].values()),
                "all_vertices_inside_source_region": spatial["checks"][
                    "all_vertices_inside_source_region"],
                "minimum_boundary_clearance_retained": spatial[
                    "minimum_boundary_clearance_m"] >= float(
                        config["admission"][
                            "minimum_boundary_clearance_m_after_settling"]),
                "rock_sizes_and_masses_unchanged": spatial["checks"][
                    "rock_sizes_and_masses_unchanged"],
                "surface_penetration_within_limit": spatial["checks"][
                    "surface_penetration_within_limit"],
                "rebaked_stability_passed": all(verification["checks"].values()),
                "rebaked_displacement_within_limit": verification[
                    "maximum_rock_displacement_m"] <= float(
                        config["admission"]["maximum_rebaked_displacement_m"]),
            }
            admitted = all(checks.values())
            if admitted:
                selected.append(scene_id)
            candidate_report = {
                "schema": "excavation_sparse_large_stable_candidate_report_v1",
                "status": "selected" if admitted else "rejected",
                "sequence_index": sequence_index,
                "seed": seed,
                "scene_id": scene_id,
                "checks": checks,
                "pre_settling": pre_metrics,
                "spatial_review": spatial,
                "rebaked_verification": verification,
                "scene_manifest_sha256": sha256(manifest_path),
                "assembly_plan_sha256": sha256(plan_path),
                "stable_initial_state_sha256": sha256(stable_path),
                "model_sha256": sha256(rebaked_model),
                "physics_qualified": False,
            }
            candidate_report_path = scene_output / "report.json"
            write_json(candidate_report_path, candidate_report)
            candidates.append({
                "sequence_index": sequence_index,
                "seed": seed,
                "scene_id": scene_id,
                "status": candidate_report["status"],
                "minimum_boundary_clearance_m": spatial[
                    "minimum_boundary_clearance_m"],
                "pre_settling_elapsed_s": pre_metrics["elapsed_s"],
                "rebaked_maximum_displacement_m": verification[
                    "maximum_rock_displacement_m"],
                "report_sha256": sha256(candidate_report_path),
            })
            if (len(selected) >= required and
                    config["selection"]["stop_after_required_count"]):
                break
    sufficient = len(selected) == int(config["selection"]["required_selected_count"])
    report = {
        "schema": "excavation_sparse_large_stable_admission_report_v1",
        "status": ("sparse_large_replacements_selected" if sufficient
                   else "sparse_large_replacements_insufficient"),
        "fixed_seed_sequence": config["selection"]["fixed_seed_sequence"],
        "evaluated_candidate_count": len(candidates),
        "required_selected_count": config["selection"]["required_selected_count"],
        "selected_scene_ids": selected,
        "existing_retained_scene_id": config["selection"][
            "existing_retained_scene_id"],
        "final_sparse_large_scene_count": 1 + len(selected),
        "final_total_first_round_scene_count": 3 + 1 + len(selected),
        "rejected_candidates_preserved": True,
        "candidates": candidates,
        "source_configs_modified": False,
        "controlled_excavation": False,
        "physics_qualified": False,
        "inputs": {
            key + "_sha256": sha256(path) for key, path in sources.items()
        },
        "implementation": {
            "selector_sha256": sha256(
                repository / "src/v4/select_sparse_large_stable_candidates.py"),
            "generated_bridge_source_sha256": sha256(generated_source),
            "compile_command": compile_command_record,
        },
        "scope_limit": (
            "fixed-sequence candidate selection through geometry, settling, rebake and "
            "boundary checks; selected scenes still require final v2 reconstruction and "
            "assembly hash-chain regeneration before R2 admission"),
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
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    args = parser.parse_args()
    report = run(
        args.config, args.output_root, args.repository_root, args.cxx,
        args.mujoco_include, args.mujoco_library_dir,
        args.mechanism_header_dir, args.controller_header_dir)
    print(json.dumps({
        "status": report["status"],
        "evaluated_candidate_count": report["evaluated_candidate_count"],
        "selected_scene_ids": report["selected_scene_ids"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
