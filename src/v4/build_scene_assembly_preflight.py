"""Build deterministic offline assembly plans for scene-reconstruction candidates.

This module validates the complete scene hash chain and plans fixed-capacity
addresses and two-rock regression batches.  It never creates MJCF, imports a
physics engine, advances simulation, or modifies the R1 bridge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Iterable

import numpy as np

from mixed_material_r0 import load_obj, mass_properties, quaternion_matrix_wxyz


CONFIG_SCHEMA = "excavation_scene_assembly_preflight_config_v1"
PLAN_SCHEMA = "excavation_scene_assembly_plan_v1"
COMPATIBILITY_SCHEMA = "excavation_scene_assembly_compatibility_v1"
REPORT_SCHEMA = "excavation_scene_assembly_preflight_report_v1"
ALLOWED_SOURCE_STATUSES = {
    "measured", "human_reviewed_estimate", "synthetic_assumption", "unknown"
}
SAFE_SCENE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stable_json_value(value):
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non_finite_json_value")
        if number == 0.0:
            return 0.0
        return float(f"{number:.10g}")
    if isinstance(value, (list, tuple)):
        return [_stable_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _stable_json_value(item) for key, item in value.items()}
    raise ValueError(f"unsupported_json_type:{type(value).__name__}")


def _json_bytes(value) -> bytes:
    return (json.dumps(_stable_json_value(value), indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def write_json(path: str | Path, value) -> None:
    Path(path).write_bytes(_json_bytes(value))


def _load_json(path: str | Path) -> dict:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate_json_key:{key}")
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return value


def _assert_finite(value, path: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"non_finite_value:{path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}/{key}")
        return
    raise ValueError(f"unsupported_value_type:{path}")


def _validate_sha(value: str, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"invalid_sha256:{label}")


def _resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _tree_snapshot(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise ValueError(f"missing_input_root:{root}")
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _tree_hash(snapshot: dict[str, str]) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_config(config: dict) -> None:
    _assert_finite(config)
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("invalid_assembly_preflight_config_schema")
    for section in ("input", "r0_assets", "r1_evidence", "current_bridge",
                    "future_candidate", "scope"):
        if section not in config:
            raise ValueError(f"missing_config_section:{section}")
    for key, value in config["input"].items():
        if key.endswith("sha256"):
            _validate_sha(value, f"input/{key}")
    for key, value in config["r1_evidence"].items():
        if key.endswith("sha256"):
            _validate_sha(value, f"r1_evidence/{key}")
    for key, value in config["r0_assets"]["assets"].items():
        _validate_sha(value, f"r0_assets/{key}")
    bridge = config["current_bridge"]
    if int(bridge["maximum_rock_count"]) != 2:
        raise ValueError("current_bridge_capacity_must_match_two_rock_evidence")
    if (int(bridge["mechanism_nq"]), int(bridge["mechanism_nv"]),
            int(bridge["actuator_count"]), int(bridge["plugin_count"])) != (19, 18, 12, 0):
        raise ValueError("unexpected_mechanism_base_contract")
    if (int(bridge["qpos_per_rock"]), int(bridge["qvel_per_rock"])) != (7, 6):
        raise ValueError("unexpected_freejoint_address_width")
    if int(config["future_candidate"]["unit_regression_batch_capacity"]) != 2:
        raise ValueError("unit_regression_capacity_must_match_current_bridge")
    if config["future_candidate"]["capacity_policy"] != "maximum_observed_scene_rock_count":
        raise ValueError("unsupported_capacity_policy")
    future = config["future_candidate"]
    if int(future.get("maximum_capacity", 0)) != 20:
        raise ValueError("maximum_capacity_must_be_20")
    if (future.get("capacity_decision_status")
            != "user_confirmed_unique_maximum_capacity"):
        raise ValueError("maximum_capacity_decision_not_confirmed")
    if future.get("runtime_count_policy") != "loaded_model_rock_count_at_startup":
        raise ValueError("invalid_runtime_count_policy")
    first_round = future.get("first_round_admission_scene_classes")
    deferred = future.get("deferred_boundary_scene_classes")
    if first_round != ["ordinary-small-rock-dominant", "sparse-large-rock"]:
        raise ValueError("invalid_first_round_admission_classes")
    if deferred != ["jam-boundary"]:
        raise ValueError("invalid_deferred_boundary_classes")
    if set(first_round) & set(deferred):
        raise ValueError("overlapping_scene_admission_classes")
    if not future.get("preserve_input_order"):
        raise ValueError("input_order_must_be_preserved")
    scope = config["scope"]
    if scope.get("status") != "scene_assembly_preflight_candidate":
        raise ValueError("invalid_output_status")
    for key in ("physics_run", "mjcf_generation", "bridge_modification", "training"):
        if scope.get(key) is not False:
            raise ValueError(f"scope_must_remain_false:{key}")


def load_config(path: str | Path) -> dict:
    value = _load_json(path)
    _validate_config(value)
    return value


def _load_prototypes(config: dict, repository_root: Path) -> dict[str, dict]:
    asset_root = _resolve(repository_root, config["r0_assets"]["asset_root"])
    result = {}
    for prototype_id, expected_hash in sorted(config["r0_assets"]["assets"].items()):
        path = asset_root / f"{prototype_id}.obj"
        if not path.is_file():
            raise ValueError(f"missing_r0_asset:{prototype_id}")
        if sha256(path) != expected_hash:
            raise ValueError(f"r0_asset_hash_mismatch:{prototype_id}")
        vertices, faces = load_obj(path)
        properties = mass_properties(vertices, faces)
        diameter = (6.0 * float(properties["volume"]) / math.pi) ** (1.0 / 3.0)
        if not math.isclose(diameter, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"r0_asset_not_unit_diameter:{prototype_id}")
        if float(np.linalg.norm(properties["center"])) > 1e-12:
            raise ValueError(f"r0_asset_not_com_centered:{prototype_id}")
        result[prototype_id] = {
            "path": path,
            "path_repository": path.relative_to(repository_root).as_posix(),
            "sha256": expected_hash,
            "unit_volume_m3": float(properties["volume"]),
            "unit_inertia_m5": properties["inertia_unit_density"],
        }
    return result


def _validate_r1_evidence(config: dict, repository_root: Path) -> dict:
    evidence = config["r1_evidence"]
    loaded = {}
    for prefix in ("config", "assembler", "bridge", "two_rock_report"):
        path = _resolve(repository_root, evidence[f"{prefix}_path"])
        if not path.is_file():
            raise ValueError(f"missing_r1_evidence:{prefix}")
        actual = sha256(path)
        if actual != evidence[f"{prefix}_sha256"]:
            raise ValueError(f"r1_evidence_hash_mismatch:{prefix}")
        loaded[f"{prefix}_path"] = path
        loaded[f"{prefix}_sha256"] = actual
    r1_config = _load_json(loaded["config_path"])
    report = _load_json(loaded["two_rock_report_path"])
    if report.get("status") != evidence["required_two_rock_status"]:
        raise ValueError("two_rock_evidence_not_passed")
    bridge = config["current_bridge"]
    mechanism = r1_config["mechanism_candidate"]
    source_contract = mechanism["source_model_contract"]
    expected_source = {
        "mechanism_nq": bridge["mechanism_nq"],
        "mechanism_nv": bridge["mechanism_nv"],
        "actuator_count": bridge["actuator_count"],
        "plugin_count": bridge["plugin_count"],
    }
    if source_contract != expected_source:
        raise ValueError("r1_source_contract_mismatch")
    if mechanism["contact_bits"] != bridge["contact_bits"]:
        raise ValueError("r1_contact_bits_mismatch")
    expected_two = {
        "nq": bridge["mechanism_nq"] + 2 * bridge["qpos_per_rock"],
        "nv": bridge["mechanism_nv"] + 2 * bridge["qvel_per_rock"],
        "nu": bridge["actuator_count"], "nplugin": bridge["plugin_count"],
        "rock_count": 2,
    }
    candidate = report["candidate_manifest"]
    if candidate["expected_candidate_contract"] != expected_two:
        raise ValueError("two_rock_candidate_contract_mismatch")
    if report["bridge_ready"].get("rock_count") != 2:
        raise ValueError("two_rock_bridge_ready_mismatch")
    if candidate["bucket_proxy_geom_count"] != bridge["bucket_proxy_geom_count"]:
        raise ValueError("bucket_proxy_geom_count_mismatch")
    if candidate["contact_bits"] != bridge["contact_bits"]:
        raise ValueError("two_rock_report_contact_bits_mismatch")
    if report["build"].get("bridge_sha256") != evidence["bridge_sha256"]:
        raise ValueError("two_rock_report_bridge_hash_mismatch")
    return {
        "config_sha256": loaded["config_sha256"],
        "assembler_sha256": loaded["assembler_sha256"],
        "bridge_sha256": loaded["bridge_sha256"],
        "two_rock_report_sha256": loaded["two_rock_report_sha256"],
        "two_rock_contract": expected_two,
        "two_rock_status": report["status"],
    }


def _validate_reconstruction_sources(config: dict, repository_root: Path) -> dict:
    input_config = config["input"]
    path = _resolve(repository_root, input_config["reconstruction_config_path"])
    if not path.is_file() or sha256(path) != input_config["reconstruction_config_sha256"]:
        raise ValueError("reconstruction_config_hash_mismatch")
    reconstruction = _load_json(path)
    sources = reconstruction.get("sources", {})
    if not sources:
        raise ValueError("reconstruction_sources_missing")
    for source_id, source in sources.items():
        if source.get("status") not in ALLOWED_SOURCE_STATUSES:
            raise ValueError(f"reconstruction_source_status_invalid:{source_id}")
    return reconstruction


def _allclose(actual, expected, label: str, tolerance: float = 1e-9) -> None:
    if not np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                       rtol=1e-9, atol=tolerance):
        raise ValueError(f"numeric_contract_mismatch:{label}")


def _validate_rock(rock: dict, prototypes: dict[str, dict],
                   sources: dict, seen_ids: set[str]) -> dict:
    _assert_finite(rock, f"rock/{rock.get('rock_id', 'missing')}")
    rock_id = rock.get("rock_id")
    if not isinstance(rock_id, str) or not rock_id:
        raise ValueError("missing_rock_id")
    if rock_id in seen_ids:
        raise ValueError(f"duplicate_rock_id:{rock_id}")
    seen_ids.add(rock_id)
    prototype_id = rock.get("prototype_id")
    if prototype_id not in prototypes:
        raise ValueError(f"unknown_prototype:{prototype_id}")
    prototype = prototypes[prototype_id]
    if rock.get("asset_path") != prototype["path_repository"]:
        raise ValueError(f"asset_path_mismatch:{rock_id}")
    if rock.get("asset_sha256") != prototype["sha256"]:
        raise ValueError(f"asset_hash_mismatch:{rock_id}")
    diameter = float(rock["equivalent_diameter_m"])
    if diameter <= 0.0 or not math.isclose(
            diameter, float(rock["uniform_linear_scale_from_unit_asset"]),
            rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"invalid_rock_scale:{rock_id}")
    density = float(rock["density_kg_m3"])
    if density <= 0.0:
        raise ValueError(f"invalid_rock_density:{rock_id}")
    volume = prototype["unit_volume_m3"] * diameter ** 3
    mass = density * volume
    inertia_body = density * prototype["unit_inertia_m5"] * diameter ** 5
    quaternion = np.asarray(rock["orientation_quaternion_wxyz"], dtype=float)
    if quaternion.shape != (4,) or not math.isclose(
            float(np.linalg.norm(quaternion)), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"invalid_rock_quaternion:{rock_id}")
    rotation = quaternion_matrix_wxyz(quaternion)
    inertia_world = rotation @ inertia_body @ rotation.T
    _allclose(rock["volume_m3"], volume, f"volume:{rock_id}", 1e-10)
    _allclose(rock["mass_kg"], mass, f"mass:{rock_id}", 1e-8)
    _allclose(rock["inertia_about_com_body_kg_m2"], inertia_body,
              f"body_inertia:{rock_id}", 1e-9)
    _allclose(rock["inertia_about_com_world_kg_m2"], inertia_world,
              f"world_inertia:{rock_id}", 1e-9)
    if len(rock["position_world_m"]) != 3:
        raise ValueError(f"invalid_rock_position:{rock_id}")
    parameter_source = rock.get("parameter_source")
    asset_source = rock.get("asset_source")
    if parameter_source not in sources or asset_source not in sources:
        raise ValueError(f"unknown_rock_source:{rock_id}")
    return {
        "rock_id": rock_id,
        "prototype_id": prototype_id,
        "asset_path": prototype["path_repository"],
        "asset_sha256": prototype["sha256"],
        "equivalent_diameter_m": diameter,
        "uniform_linear_scale_from_unit_asset": diameter,
        "density_kg_m3": density,
        "volume_m3": volume,
        "mass_kg": mass,
        "inertia_about_com_body_kg_m2": inertia_body.tolist(),
        "inertia_about_com_world_kg_m2": inertia_world.tolist(),
        "position_world_m": rock["position_world_m"],
        "orientation_quaternion_wxyz": rock["orientation_quaternion_wxyz"],
        "parameter_source": {
            "source_id": parameter_source,
            "status": sources[parameter_source]["status"],
        },
        "asset_source": {
            "source_id": asset_source,
            "status": sources[asset_source]["status"],
        },
    }


def _contract_for_capacity(config: dict, capacity: int) -> dict:
    bridge = config["current_bridge"]
    return {
        "rock_capacity": capacity,
        "nq": int(bridge["mechanism_nq"]) + int(bridge["qpos_per_rock"]) * capacity,
        "nv": int(bridge["mechanism_nv"]) + int(bridge["qvel_per_rock"]) * capacity,
        "nu": int(bridge["actuator_count"]),
        "nplugin": int(bridge["plugin_count"]),
        "state_top_level_fields": int(bridge["state_top_level_fields"]),
        "per_rock_state_named_fields": int(bridge["per_rock_state_named_fields"]),
        "estimated_state_named_fields_including_rock_records": (
            int(bridge["state_top_level_fields"])
            + int(bridge["per_rock_state_named_fields"]) * capacity),
        "state_numeric_values": (
            int(bridge["fixed_state_numeric_values_excluding_rocks"])
            + int(bridge["per_rock_state_numeric_values"]) * capacity),
        "legacy_first_rock_alias_numeric_values": int(
            bridge["legacy_first_rock_alias_numeric_values"]),
        "ready_top_level_fields": int(bridge["ready_top_level_fields"]),
    }


def _contact_upper_bounds(config: dict, count: int) -> dict:
    bridge = config["current_bridge"]
    rock_rock = count * (count - 1) // 2
    ground = count * int(bridge["ground_geom_count"])
    bucket = count * int(bridge["bucket_proxy_geom_count"])
    truck = count * int(bridge["truck_receiver_geom_count"])
    return {
        "scope": "combinatorial candidate geom/body pairs, not simultaneous contacts or manifolds",
        "rock_rock_body_pairs": rock_rock,
        "rock_ground_geom_pairs": ground,
        "rock_bucket_proxy_geom_pairs": bucket,
        "rock_truck_receiver_geom_pairs": truck,
        "total_candidate_geom_pairs": rock_rock + ground + bucket + truck,
        "total_body_or_environment_role_pairs": rock_rock + 3 * count,
    }


def _full_binding(config: dict, slot: int) -> dict:
    bridge = config["current_bridge"]
    future = config["future_candidate"]
    return {
        "slot": slot,
        "body_name": future["body_name_template"].format(slot=slot),
        "mesh_name": future["mesh_name_template"].format(slot=slot),
        "geom_name": future["geom_name_template"].format(slot=slot),
        "freejoint_name": future["freejoint_name_template"].format(slot=slot),
        "qpos_address": int(bridge["mechanism_nq"]) + int(bridge["qpos_per_rock"]) * slot,
        "qpos_width": int(bridge["qpos_per_rock"]),
        "qvel_address": int(bridge["mechanism_nv"]) + int(bridge["qvel_per_rock"]) * slot,
        "qvel_width": int(bridge["qvel_per_rock"]),
        "contact_role": "rock",
        "contact_bits": bridge["contact_bits"],
    }


def _unit_binding(config: dict, slot: int) -> dict:
    bridge = config["current_bridge"]
    return {
        "unit_slot": slot,
        "body_name": bridge["body_name_template"].format(slot=slot),
        "mesh_name": bridge["mesh_name_template"].format(slot=slot),
        "geom_name": bridge["geom_name_template"].format(slot=slot),
        "freejoint_name": bridge["freejoint_name_template"].format(slot=slot),
        "qpos_address": int(bridge["mechanism_nq"]) + int(bridge["qpos_per_rock"]) * slot,
        "qvel_address": int(bridge["mechanism_nv"]) + int(bridge["qvel_per_rock"]) * slot,
    }


def _build_unit_batches(config: dict, scene_id: str, rocks: list[dict]) -> list[dict]:
    capacity = int(config["future_candidate"]["unit_regression_batch_capacity"])
    batches = []
    for batch_index, start in enumerate(range(0, len(rocks), capacity)):
        members = []
        for unit_slot, rock in enumerate(rocks[start:start + capacity]):
            members.append({
                "source_rock_id": rock["rock_id"],
                "full_scene_slot": start + unit_slot,
                "current_bridge_binding": _unit_binding(config, unit_slot),
            })
        batches.append({
            "batch_id": f"{scene_id}-unit-batch-{batch_index:03d}",
            "purpose": "asset_and_one_or_two_rock_regression_only",
            "does_not_qualify_full_scene_physics": True,
            "rock_count": len(members),
            "expected_current_bridge_contract": _contract_for_capacity(
                config, len(members)),
            "members": members,
        })
    flattened = [member["source_rock_id"] for batch in batches for member in batch["members"]]
    expected = [rock["rock_id"] for rock in rocks]
    if flattened != expected or len(flattened) != len(set(flattened)):
        raise ValueError(f"unit_batch_partition_failed:{scene_id}")
    return batches


def _load_scenes(config: dict, repository_root: Path,
                 prototypes: dict[str, dict], reconstruction: dict) -> tuple[list[dict], dict]:
    input_config = config["input"]
    scene_root = _resolve(repository_root, input_config["scene_root"])
    root_report_path = scene_root / "report.json"
    if not root_report_path.is_file() or sha256(root_report_path) != input_config["root_report_sha256"]:
        raise ValueError("scene_root_report_hash_mismatch")
    root_report = _load_json(root_report_path)
    if root_report.get("schema") != input_config["expected_root_schema"]:
        raise ValueError("scene_root_schema_mismatch")
    if root_report.get("status") != input_config["expected_status"]:
        raise ValueError("scene_root_status_mismatch")
    if (root_report.get("scene_count") != input_config["expected_scene_count"]
            or len(root_report.get("scenes", [])) != input_config["expected_scene_count"]
            or root_report.get("all_scenes_passed") is not True):
        raise ValueError("scene_root_count_or_admission_mismatch")
    if root_report.get("config_sha256") != input_config["reconstruction_config_sha256"]:
        raise ValueError("scene_root_config_hash_mismatch")
    if root_report.get("asset_sha256") != config["r0_assets"]["assets"]:
        raise ValueError("scene_root_asset_hashes_mismatch")
    scene_ids = [item.get("scene_id") for item in root_report["scenes"]]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("duplicate_scene_id")
    seen_rock_ids: set[str] = set()
    loaded_scenes = []
    for index_record in root_report["scenes"]:
        scene_id = index_record["scene_id"]
        if not SAFE_SCENE_ID.fullmatch(scene_id):
            raise ValueError(f"unsafe_scene_id:{scene_id}")
        directory = scene_root / scene_id
        report_path = directory / "report.json"
        manifest_path = directory / "scene-manifest.json"
        metrics_path = directory / "scene-metrics.json"
        preview_path = directory / "preview.svg"
        for path in (report_path, manifest_path, metrics_path, preview_path):
            if not path.is_file():
                raise ValueError(f"missing_scene_input:{scene_id}:{path.name}")
        if sha256(report_path) != index_record["report_sha256"]:
            raise ValueError(f"scene_report_hash_mismatch:{scene_id}")
        report = _load_json(report_path)
        manifest = _load_json(manifest_path)
        metrics = _load_json(metrics_path)
        if report.get("schema") != input_config["expected_scene_report_schema"]:
            raise ValueError(f"scene_report_schema_mismatch:{scene_id}")
        if manifest.get("schema") != input_config["expected_manifest_schema"]:
            raise ValueError(f"scene_manifest_schema_mismatch:{scene_id}")
        if metrics.get("schema") != input_config["expected_metrics_schema"]:
            raise ValueError(f"scene_metrics_schema_mismatch:{scene_id}")
        if any(value.get("status") != input_config["expected_status"]
               for value in (index_record, report, manifest)):
            raise ValueError(f"scene_status_mismatch:{scene_id}")
        if (report.get("all_admission_checks_passed") is not True
                or metrics.get("all_admission_checks_passed") is not True):
            raise ValueError(f"scene_not_admitted:{scene_id}")
        for key in ("scene_id", "scene_class", "seed"):
            expected = index_record[key]
            if report.get(key) != expected or manifest.get(key) != expected or metrics.get(key) != expected:
                raise ValueError(f"scene_identity_mismatch:{scene_id}:{key}")
        hash_expectations = {
            "scene_manifest_sha256": manifest_path,
            "scene_metrics_sha256": metrics_path,
            "preview_sha256": preview_path,
        }
        for field, path in hash_expectations.items():
            if report.get(field) != sha256(path):
                raise ValueError(f"scene_payload_hash_mismatch:{scene_id}:{field}")
        for field in ("input_sha256", "config_sha256", "generator_sha256", "asset_sha256"):
            if report.get(field) != root_report.get(field):
                raise ValueError(f"scene_root_binding_mismatch:{scene_id}:{field}")
        if manifest.get("parameter_sources") != reconstruction.get("parameter_sources"):
            raise ValueError(f"scene_parameter_sources_mismatch:{scene_id}")
        rocks = manifest.get("rocks", [])
        if manifest.get("rock_count") != len(rocks) or metrics.get("rock_count") != len(rocks):
            raise ValueError(f"scene_rock_count_mismatch:{scene_id}")
        validated_rocks = [
            _validate_rock(rock, prototypes, reconstruction["sources"], seen_rock_ids)
            for rock in rocks
        ]
        total_mass = sum(rock["mass_kg"] for rock in validated_rocks)
        if not math.isclose(total_mass, float(metrics["total_mass_kg"]),
                            rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError(f"scene_total_mass_mismatch:{scene_id}")
        loaded_scenes.append({
            "scene_id": scene_id,
            "scene_class": manifest["scene_class"],
            "seed": manifest["seed"],
            "rock_count": len(validated_rocks),
            "rocks": validated_rocks,
            "input_hashes": {
                "scene_report_sha256": sha256(report_path),
                "scene_manifest_sha256": sha256(manifest_path),
                "scene_metrics_sha256": sha256(metrics_path),
                "preview_sha256": sha256(preview_path),
            },
        })
    return loaded_scenes, root_report


def _build_plan(config: dict, scene: dict) -> dict:
    future = config["future_candidate"]
    maximum_capacity = int(future["maximum_capacity"])
    if scene["rock_count"] > maximum_capacity:
        raise ValueError(f"scene_exceeds_maximum_capacity:{scene['scene_id']}")
    if scene["scene_class"] in future["first_round_admission_scene_classes"]:
        if scene["rock_count"] != maximum_capacity:
            raise ValueError(f"first_round_scene_not_at_maximum:{scene['scene_id']}")
        qualification = "first_round_admission_candidate"
        first_round_admission = True
    elif scene["scene_class"] in future["deferred_boundary_scene_classes"]:
        qualification = "deferred_boundary_diagnostic"
        first_round_admission = False
    else:
        raise ValueError(f"unclassified_scene_for_admission:{scene['scene_id']}")
    rocks = []
    for slot, rock in enumerate(scene["rocks"]):
        rocks.append({**rock, "full_scene_binding": _full_binding(config, slot)})
    batches = _build_unit_batches(config, scene["scene_id"], rocks)
    return {
        "schema": PLAN_SCHEMA,
        "status": config["scope"]["status"],
        "scope": "offline assembly plan only; no MJCF, physics or bridge modification",
        "scene_id": scene["scene_id"],
        "scene_class": scene["scene_class"],
        "seed": scene["seed"],
        "source_input_hashes": scene["input_hashes"],
        "rock_count": scene["rock_count"],
        "maximum_capacity": maximum_capacity,
        "loaded_model_rock_count": scene["rock_count"],
        "runtime_contract": _contract_for_capacity(config, scene["rock_count"]),
        "runtime_count_policy": future["runtime_count_policy"],
        "padding_forbidden": True,
        "first_round_admission": first_round_admission,
        "qualification": qualification,
        "candidate_contact_pair_upper_bounds": _contact_upper_bounds(
            config, scene["rock_count"]),
        "rocks": rocks,
        "unit_regression_batches": batches,
        "unit_regression_scope_limit": (
            "each batch is independently compatible with the current one/two-rock "
            "contract and cannot qualify simultaneous full-scene physics"),
    }


def _compatibility_report(config: dict, plans: list[dict], r1: dict,
                          input_tree_hash: str) -> dict:
    counts: dict[int, int] = {}
    for plan in plans:
        counts[plan["rock_count"]] = counts.get(plan["rock_count"], 0) + 1
    maximum = max(counts)
    minimum = min(counts)
    maximum_capacity = int(config["future_candidate"]["maximum_capacity"])
    if maximum_capacity != maximum:
        raise ValueError("maximum_capacity_not_maximum_observed")
    maximum_contract = _contract_for_capacity(config, maximum_capacity)
    first_round_plans = [plan for plan in plans if plan["first_round_admission"]]
    deferred_plans = [plan for plan in plans if not plan["first_round_admission"]]
    if len(first_round_plans) != 6 or any(
            plan["rock_count"] != maximum_capacity for plan in first_round_plans):
        raise ValueError("invalid_first_round_admission_set")
    if len(deferred_plans) != 3 or any(
            plan["qualification"] != "deferred_boundary_diagnostic"
            or plan["rock_count"] != 13
            for plan in deferred_plans):
        raise ValueError("invalid_deferred_boundary_set")
    total_batches = sum(len(plan["unit_regression_batches"]) for plan in plans)
    total_members = sum(plan["rock_count"] for plan in plans)
    return {
        "schema": COMPATIBILITY_SCHEMA,
        "status": config["scope"]["status"],
        "input_tree_sha256": input_tree_hash,
        "scene_count": len(plans),
        "observed_scene_rock_counts": {
            str(key): counts[key] for key in sorted(counts)},
        "minimum_observed_rock_count": minimum,
        "maximum_observed_rock_count": maximum,
        "current_bridge": {
            "qualified_capacity": config["current_bridge"]["maximum_rock_count"],
            "contract": _contract_for_capacity(
                config, config["current_bridge"]["maximum_rock_count"]),
            "two_rock_status": r1["two_rock_status"],
            "scope_limit": "one/two-rock regression only; not a full-scene bridge",
        },
        "maximum_capacity_decision": {
            "maximum_capacity": maximum_capacity,
            "maximum_capacity_contract": maximum_contract,
            "selection_rule": config["future_candidate"]["capacity_policy"],
            "evidence": (
                f"maximum actual count across nine admitted manifests is {maximum}; "
                f"successful ordinary/sparse scenes use 20 and jam scenes use 13"),
            "decision_status": "user_confirmed_unique_maximum_not_implemented",
            "bridge_program_rule": (
                "one bridge implementation reads the loaded model rock_count at startup; "
                "each run has a fixed actual count not greater than 20"),
            "padding_forbidden": True,
        },
        "observed_runtime_contracts": {
            str(value): _contract_for_capacity(config, value)
            for value in sorted(counts)
        },
        "first_round_admission": {
            "scene_count": len(first_round_plans),
            "required_rock_count": maximum_capacity,
            "scene_ids": [plan["scene_id"] for plan in first_round_plans],
            "scene_classes": sorted({plan["scene_class"] for plan in first_round_plans}),
            "scope": "only these six 20-rock ordinary/sparse scenes are first-round candidates",
        },
        "deferred_boundary_diagnostics": {
            "scene_count": len(deferred_plans),
            "rock_count": 13,
            "scene_ids": [plan["scene_id"] for plan in deferred_plans],
            "scene_classes": sorted({plan["scene_class"] for plan in deferred_plans}),
            "runtime_contract": _contract_for_capacity(config, 13),
            "scope": "retained as diagnostics; not eligible for first-round success",
        },
        "rejected_designs": [
            {
                "option": "separate_bridge_or_message_implementations_for_13_and_20",
                "status": "rejected_by_user_decision",
                "reason": "one bridge program must handle the loaded model count up to 20",
            },
            {
                "option": "pad_13_rock_models_to_20_runtime_slots",
                "status": "rejected_by_user_decision",
                "reason": "13-rock diagnostics must naturally use nq=110 and nv=96",
            },
            {
                "option": "admit_13_rock_jam_scenes_in_first_round",
                "status": "deferred_by_user_decision",
                "reason": "the first round is limited to six 20-rock ordinary/sparse scenes",
            },
        ],
        "state_message_formula": {
            "nq": "19 + 7 * rock_capacity",
            "nv": "18 + 6 * rock_capacity",
            "nu": "12",
            "numeric_values": "91 + 16 * rock_capacity",
            "named_fields_including_rock_records": "19 + 4 * rock_capacity",
            "top_level_fields": config["current_bridge"]["state_top_level_fields"],
            "per_rock_named_fields": config["current_bridge"]["per_rock_state_named_fields"],
            "legacy_first_rock_alias_numeric_values_in_fixed_term": 16,
        },
        "contact_pair_upper_bounds": {
            str(value): _contact_upper_bounds(config, value)
            for value in sorted(set(counts) | {2, maximum})
        },
        "unit_regression": {
            "batch_capacity": 2,
            "batch_count": total_batches,
            "member_count": total_members,
            "all_members_exactly_once": True,
            "scope_limit": "does not replace simultaneous full-scene physics",
        },
        "unresolved_before_r2_bridge": [
            "implement startup validation for the loaded model rock_count in the range 1 through 20",
            "decide whether legacy first-rock aliases remain in the multirock state schema",
            "implement and load a multirock model before claiming address or contact runtime qualification",
            "repeat whole-rock envelope, contact, settling, transport and dump checks in actual physics",
        ],
        "boundaries": [
            "no MJCF was generated",
            "no MuJoCo, Newton or Warp run was performed",
            "no R1 bridge or report was modified",
            "unit batches do not qualify a complete scene",
        ],
    }


def _root_readme(report: dict, compatibility: dict) -> str:
    return f"""# Scene assembly preflight v1

Status: `scene_assembly_preflight_candidate`.

The nine admitted geometry candidates were validated and converted into
deterministic offline assembly plans. No MJCF was generated and no physics
engine or training process was run.

## Result

- Scene plans: {report['scene_count']}
- Rocks mapped and independently recomputed: {report['rock_count']}
- One/two-rock unit regression batches: {report['unit_regression_batch_count']}
- Observed full-scene capacities: {compatibility['observed_scene_rock_counts']}
- Confirmed maximum capacity: {compatibility['maximum_capacity_decision']['maximum_capacity']}
- First-round admission scenes: {compatibility['first_round_admission']['scene_count']}
- Deferred boundary diagnostics: {compatibility['deferred_boundary_diagnostics']['scene_count']}

The maximum capacity is confirmed for the next bridge design but is not
implemented. It is an upper bound, not a padded 20-slot runtime contract.
Each run will use the fixed rock count of its loaded model. Unit batches only
reuse the current one/two-rock contract and cannot establish full-scene physics.

## Reproduce

```bash
PYTHONPATH=src/v4 python3 src/v4/build_scene_assembly_preflight.py \\
  --config configs/v4/scene-assembly-preflight-v1.json \\
  --output-dir reports/v4/scene-assembly-preflight-2026-09-19-v1
```

The output directory must not already exist.
"""


def build(config_path: str | Path, output_dir: str | Path,
          repository_root: str | Path | None = None) -> dict:
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output_directory_exists:{output_dir}")
    root = (Path(repository_root).resolve() if repository_root is not None
            else Path(__file__).resolve().parents[2])
    config = load_config(config_path)
    scene_root = _resolve(root, config["input"]["scene_root"])
    before_snapshot = _tree_snapshot(scene_root)
    before_tree_hash = _tree_hash(before_snapshot)
    prototypes = _load_prototypes(config, root)
    r1 = _validate_r1_evidence(config, root)
    reconstruction = _validate_reconstruction_sources(config, root)
    scenes, input_root_report = _load_scenes(
        config, root, prototypes, reconstruction)
    plans = [_build_plan(config, scene) for scene in scenes]
    compatibility = _compatibility_report(config, plans, r1, before_tree_hash)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        scene_records = []
        for plan in plans:
            directory = temporary / plan["scene_id"]
            directory.mkdir()
            path = directory / "assembly-plan.json"
            write_json(path, plan)
            scene_records.append({
                "scene_id": plan["scene_id"],
                "scene_class": plan["scene_class"],
                "seed": plan["seed"],
                "rock_count": plan["rock_count"],
                "first_round_admission": plan["first_round_admission"],
                "qualification": plan["qualification"],
                "unit_regression_batch_count": len(plan["unit_regression_batches"]),
                "assembly_plan_sha256": sha256(path),
                "status": plan["status"],
            })
        compatibility_path = temporary / "compatibility-report.json"
        write_json(compatibility_path, compatibility)
        after_snapshot = _tree_snapshot(scene_root)
        after_tree_hash = _tree_hash(after_snapshot)
        if before_snapshot != after_snapshot or before_tree_hash != after_tree_hash:
            raise ValueError("input_scene_tree_modified_during_build")
        report = {
            "schema": REPORT_SCHEMA,
            "status": config["scope"]["status"],
            "scope": "offline assembly preflight only; no MJCF, physics, bridge changes or training",
            "config_sha256": sha256(config_path),
            "generator_sha256": sha256(Path(__file__)),
            "input_root_report_sha256": sha256(scene_root / "report.json"),
            "input_tree_sha256_before": before_tree_hash,
            "input_tree_sha256_after": after_tree_hash,
            "input_tree_unchanged": True,
            "input_reconstruction_status": input_root_report["status"],
            "r0_asset_sha256": {
                key: prototypes[key]["sha256"] for key in sorted(prototypes)},
            "r1_evidence": r1,
            "scene_count": len(plans),
            "rock_count": sum(plan["rock_count"] for plan in plans),
            "unit_regression_batch_count": sum(
                len(plan["unit_regression_batches"]) for plan in plans),
            "maximum_capacity": compatibility[
                "maximum_capacity_decision"]["maximum_capacity"],
            "first_round_admission_scene_count": compatibility[
                "first_round_admission"]["scene_count"],
            "deferred_boundary_scene_count": compatibility[
                "deferred_boundary_diagnostics"]["scene_count"],
            "scenes": scene_records,
            "compatibility_report_sha256": sha256(compatibility_path),
            "checks": {
                "nine_input_scenes_validated": len(plans) == 9,
                "all_input_statuses_geometry_candidate": True,
                "all_hash_chains_validated": True,
                "all_assets_mapped": True,
                "all_mass_inertia_pose_source_records_validated": True,
                "all_rocks_partitioned_exactly_once": True,
                "input_tree_unchanged": True,
                "current_two_rock_evidence_validated": True,
                "unique_maximum_capacity_20_confirmed": (
                    compatibility["maximum_capacity_decision"][
                        "maximum_capacity"] == 20),
                "first_round_limited_to_six_20_rock_scenes": (
                    compatibility["first_round_admission"]["scene_count"] == 6),
                "thirteen_rock_scenes_deferred_without_padding": (
                    compatibility["deferred_boundary_diagnostics"]["scene_count"] == 3),
                "no_physics_or_mjcf_generation": True,
            },
            "boundaries": compatibility["boundaries"],
        }
        if not all(report["checks"].values()):
            raise ValueError("assembly_preflight_checks_failed")
        write_json(temporary / "report.json", report)
        (temporary / "README.md").write_text(
            _root_readme(report, compatibility), encoding="utf-8", newline="\n")
        os.replace(temporary, output_dir)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path)
    args = parser.parse_args(argv)
    report = build(args.config, args.output_dir, args.repository_root)
    print(json.dumps({
        "status": report["status"],
        "scene_count": report["scene_count"],
        "rock_count": report["rock_count"],
        "unit_regression_batch_count": report["unit_regression_batch_count"],
        "output_dir": str(args.output_dir),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
