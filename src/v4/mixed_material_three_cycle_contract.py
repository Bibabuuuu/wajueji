"""Frozen P0 contracts for a same-instance mixed-material three-scoop run."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

from mixed_three_cycle_state import (
    SHA256_RE,
    canonical_sha256,
    state_sha256,
    validate_cycle_boundary,
    validate_state,
)
from mixed_terrain_observation import validate_mixed_terrain_observation


CONFIG_SCHEMA = "excavation_mixed_material_three_cycle_config_v1"
REPORT_SCHEMA = "excavation_mixed_material_three_cycle_report_v1"
OBSERVATION_SUMMARY_SCHEMA = (
    "excavation_mixed_terrain_observation_summary_v1")
REQUIRED_STAGES = [
    "APPROACH", "CUT", "DEEPEN", "CURL", "HOLD", "TURN", "DUMP",
    "SETTLE", "RETURN", "FINAL_HOLD",
]
FROZEN_INPUT_KEYS = {
    "r3_c_shared_config", "r3_c_shared_report", "r3_c_convergence_config",
    "r3_c_convergence_report", "existing_three_cycle_runner",
    "existing_runtime_plan", "existing_truck_receiver",
    "existing_truck_contract", "existing_truck_mesh",
    "existing_cycle_selector", "existing_cycle_geometry",
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"mixed_three_cycle_keys:{name}")


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"mixed_three_cycle_number_required:{name}")
    result = float(value)
    if (not math.isfinite(result) or
            (result < 0.0 if allow_zero else result <= 0.0)):
        raise ValueError(f"mixed_three_cycle_positive_required:{name}")
    return result


def validate_config(config: dict, repository_root: str | Path) -> None:
    """Validate P0 scope and every immutable capability binding."""

    repository = Path(repository_root).resolve()
    _exact_keys(
        config,
        {"schema", "status", "frozen_inputs", "scene_binding",
         "representative_scene_target",
         "runtime_contract", "development_acceptance", "scope"},
        "config",
    )
    if (config["schema"] != CONFIG_SCHEMA or
            config["status"] != "p0_contract_candidate"):
        raise ValueError("mixed_three_cycle_config_schema_or_status")
    frozen = config["frozen_inputs"]
    if not isinstance(frozen, dict) or set(frozen) != FROZEN_INPUT_KEYS:
        raise ValueError("mixed_three_cycle_frozen_input_set")
    for name, binding in frozen.items():
        _exact_keys(binding, {"path", "sha256"}, f"frozen_inputs.{name}")
        if (not isinstance(binding["path"], str) or
                not isinstance(binding["sha256"], str) or
                not SHA256_RE.fullmatch(binding["sha256"])):
            raise ValueError(f"mixed_three_cycle_frozen_input_format:{name}")
        path = repository / binding["path"]
        if not path.is_file() or sha256(path) != binding["sha256"]:
            raise ValueError(f"mixed_three_cycle_frozen_input_mismatch:{name}")

    shared_report = json.loads((repository / frozen[
        "r3_c_shared_report"]["path"]).read_text(encoding="utf-8"))
    if (shared_report.get("status") !=
            "r3_c_shared_matrix_four_round_passed" or
            not shared_report.get("shared_four_round_qualified") or
            shared_report.get("r3_c_physics_qualified") is not False):
        raise ValueError("mixed_three_cycle_r3_c_shared_claim_drift")
    convergence = json.loads((repository / frozen[
        "r3_c_convergence_report"]["path"]).read_text(encoding="utf-8"))
    if (convergence.get("status") != "r3_c_convergence_probe_failed" or
            convergence.get("convergence_qualified") is not False or
            convergence.get("r3_c_physics_qualified") is not False):
        raise ValueError("mixed_three_cycle_convergence_failure_not_preserved")

    if config["scene_binding"] != {
        "representative_scene_manifest": None,
        "representative_scene_binding_pending": True,
        "p1_execution_authorized": False,
    }:
        raise ValueError("mixed_three_cycle_p1_prematurely_authorized")
    if config["representative_scene_target"] != {
        "scenario_name":
            "mixed-autonomous-fullscale-graded-gravel-slope-truck-v1",
        "nominal_material_volume_m3": 6.1,
        "volume_fraction_basis": "material_equivalent_solid_volume",
        "volume_fractions": {
            "small_rigid_rock": 0.6,
            "mpm_soil_and_gap_fill": 0.3,
            "medium_rigid_rock": 0.08,
            "large_rigid_rock": 0.02,
        },
        "equivalent_diameter_ranges_m": {
            "small": [0.16, 0.24],
            "medium": [0.24, 0.32],
            "large": [0.32, 0.4],
        },
        "placement": {
            "small_rock_surface_dominant": True,
            "medium_and_large_rock_lower_region": True,
            "soil_fills_inter_rock_space": True,
        },
        "rock_count_policy": "derived_from_target_volume_not_fixed",
        "planning_estimate_only": {
            "small": 840, "medium": 40, "large": 5, "total": 885,
        },
        "capacity_preflight_rock_counts": [64, 256, 512],
        "formal_count_source":
            "frozen_scene_manifest_after_volume_closure",
    }:
        raise ValueError("mixed_three_cycle_scene_target_drift")
    if config["runtime_contract"] != {
        "cycles_requested": 3,
        "output_dt_s": 1.0 / 120.0,
        "mujoco_internal_substeps": 40,
        "one_physics_instance": True,
        "fresh_planning_observation_each_cycle": True,
        "fresh_dispatch_observation_each_cycle": True,
        "next_cycle_uses_previous_cycle_terrain": True,
        "material_reset_between_cycles": False,
        "material_refill_between_cycles": False,
        "material_deletion_between_cycles": False,
        "cycle_index_entry_lookup": False,
    }:
        raise ValueError("mixed_three_cycle_runtime_contract_drift")
    acceptance = config["development_acceptance"]
    _exact_keys(
        acceptance,
        {"minimum_positive_acquisition_kg", "minimum_positive_delivery_kg",
         "submission_thresholds_frozen"},
        "development_acceptance",
    )
    _positive(
        acceptance["minimum_positive_acquisition_kg"],
        "minimum_positive_acquisition_kg",
    )
    _positive(
        acceptance["minimum_positive_delivery_kg"],
        "minimum_positive_delivery_kg",
    )
    if acceptance["submission_thresholds_frozen"] is not False:
        raise ValueError("mixed_three_cycle_submission_threshold_overclaim")
    if config["scope"] != {
        "p0_contract_only": True,
        "gpu_run": False,
        "physics_run": False,
        "mixed_material": True,
        "reuse_existing_truck": True,
        "reuse_existing_three_cycle_loop": True,
        "complete_in_process_reset": False,
        "r3_c_physics_qualified": False,
        "official_envelope": False,
        "training": False,
        "frozen_prior_evidence_modified": False,
    }:
        raise ValueError("mixed_three_cycle_scope_expanded")


def _validate_observation_summary(
        observation: dict, *, cycle_id: int, physics_instance_id: str,
        previous_end_hash: str | None, minimum_step: int,
        minimum_sequence: int,
) -> None:
    _exact_keys(
        observation,
        {"schema", "physics_instance_id", "cycle_id", "observation_sequence",
         "capture_physics_step", "capture_sim_time_s", "source_state_sha256",
         "previous_cycle_end_state_sha256", "observation_sha256",
         "terrain_content_sha256",
         "artifact_path", "artifact_sha256", "source_state", "includes_mpm",
         "includes_rigid_rocks", "visible_rock_ids", "terrain_observation"},
        "observation_summary",
    )
    if observation["schema"] != OBSERVATION_SUMMARY_SCHEMA:
        raise ValueError("mixed_three_cycle_observation_schema")
    if (observation["cycle_id"] != cycle_id or
            observation["physics_instance_id"] != physics_instance_id):
        raise ValueError("mixed_three_cycle_observation_identity")
    for key in ("capture_physics_step", "observation_sequence"):
        value = observation[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("mixed_three_cycle_observation_integer")
    if observation["capture_physics_step"] <= minimum_step:
        raise ValueError("mixed_three_cycle_stale_observation_step")
    if observation["observation_sequence"] <= minimum_sequence:
        raise ValueError("mixed_three_cycle_stale_observation_sequence")
    _positive(
        observation["capture_sim_time_s"], "observation.capture_sim_time_s",
        allow_zero=True,
    )
    for key in ("source_state_sha256", "observation_sha256"):
        if (not isinstance(observation[key], str) or
                not SHA256_RE.fullmatch(observation[key])):
            raise ValueError("mixed_three_cycle_observation_hash")
    if (not isinstance(observation["artifact_path"], str) or
            not observation["artifact_path"] or
            Path(observation["artifact_path"]).is_absolute() or
            ".." in Path(observation["artifact_path"]).parts or
            not isinstance(observation["artifact_sha256"], str) or
            not SHA256_RE.fullmatch(observation["artifact_sha256"])):
        raise ValueError("mixed_three_cycle_observation_artifact_binding")
    source_state = observation["source_state"]
    validate_state(source_state)
    if (source_state["physics_instance_id"] != physics_instance_id or
            source_state["cycle_id"] != cycle_id or
            source_state["physics_step"] !=
            observation["capture_physics_step"] or
            abs(float(source_state["sim_time_s"]) -
                float(observation["capture_sim_time_s"])) > 1e-12 or
            state_sha256(source_state) != observation["source_state_sha256"]):
        raise ValueError("mixed_three_cycle_observation_source_state_mismatch")
    terrain_observation = observation["terrain_observation"]
    validate_mixed_terrain_observation(
        terrain_observation,
        minimum_capture_physics_step=minimum_step,
    )
    if (terrain_observation["physics_instance_id"] != physics_instance_id or
            terrain_observation["cycle_id"] != cycle_id or
            terrain_observation["observation_sequence"] !=
            observation["observation_sequence"] or
            terrain_observation["capture_physics_step"] !=
            observation["capture_physics_step"] or
            terrain_observation["capture_sim_time_s"] !=
            observation["capture_sim_time_s"] or
            terrain_observation["source_state_sha256"] !=
            observation["source_state_sha256"] or
            terrain_observation["previous_cycle_end_state_sha256"] !=
            previous_end_hash or
            terrain_observation["observation_sha256"] !=
            observation["observation_sha256"] or
            terrain_observation["content_sha256"] !=
            observation["terrain_content_sha256"]):
        raise ValueError("mixed_three_cycle_observation_payload_mismatch")
    sampling = terrain_observation["sampling_evidence"]
    if (sampling["source_mpm_state_sha256"] !=
            source_state["mpm"]["state_sha256"] or
            sampling["source_rocks_state_sha256"] !=
            canonical_sha256(source_state["rocks"])):
        raise ValueError("mixed_three_cycle_observation_sampling_state_mismatch")
    if observation["previous_cycle_end_state_sha256"] != previous_end_hash:
        raise ValueError("mixed_three_cycle_previous_end_binding")
    if (observation["includes_mpm"] is not True or
            observation["includes_rigid_rocks"] is not True):
        raise ValueError("mixed_three_cycle_not_mixed_observation")
    ids = observation["visible_rock_ids"]
    if (not isinstance(ids, list) or ids != sorted(ids) or not ids or
            len(ids) != len(set(ids)) or
            any(not isinstance(item, str) or not item for item in ids)):
        raise ValueError("mixed_three_cycle_visible_rock_ids")
    if ids != [rock["id"] for rock in source_state["rocks"]]:
        raise ValueError("mixed_three_cycle_observation_rock_state_mismatch")
    if ids != terrain_observation["rocks"]["visible_ids"]:
        raise ValueError("mixed_three_cycle_observation_visible_ids_mismatch")


def _validate_transfer(value: dict, name: str) -> None:
    _exact_keys(
        value,
        {"total_mass_kg", "mpm_mass_kg", "rock_ids", "rock_mass_kg",
         "stable"},
        name,
    )
    total = _positive(value["total_mass_kg"], f"{name}.total_mass_kg")
    mpm = _positive(
        value["mpm_mass_kg"], f"{name}.mpm_mass_kg", allow_zero=True)
    rock_mass = _positive(
        value["rock_mass_kg"], f"{name}.rock_mass_kg", allow_zero=True)
    if abs(total - mpm - rock_mass) > 1e-8:
        raise ValueError(f"mixed_three_cycle_transfer_mass_mismatch:{name}")
    rock_ids = value["rock_ids"]
    if (not isinstance(rock_ids, list) or rock_ids != sorted(rock_ids) or
            len(rock_ids) != len(set(rock_ids)) or
            any(not isinstance(item, str) or not item for item in rock_ids)):
        raise ValueError(f"mixed_three_cycle_transfer_rock_ids:{name}")
    if (rock_mass > 0.0) != bool(rock_ids):
        raise ValueError(f"mixed_three_cycle_transfer_rock_mass_ids:{name}")
    if value["stable"] is not True:
        raise ValueError(f"mixed_three_cycle_transfer_not_stable:{name}")


def _rock_mass_map(snapshot: dict) -> dict[str, float]:
    return {rock["id"]: float(rock["mass_kg"]) for rock in snapshot["rocks"]}


def _validate_conserved_inventory(snapshot: dict, baseline: dict) -> None:
    if (snapshot["mpm"]["particle_count"] != baseline["particle_count"] or
            abs(float(snapshot["mpm"]["total_mass_kg"]) -
                float(baseline["total_mpm_mass_kg"])) > 1e-8 or
            _rock_mass_map(snapshot) != baseline["rock_mass_kg_by_id"]):
        raise ValueError("mixed_three_cycle_global_material_identity_drift")


def _validate_bound_file(
        repository: Path, binding: dict, name: str, *, nullable: bool = False,
) -> None:
    _exact_keys(binding, {"path", "sha256"}, name)
    path_value = binding["path"]
    hash_value = binding["sha256"]
    if nullable and path_value is None and hash_value is None:
        return
    if (not isinstance(path_value, str) or not path_value or
            Path(path_value).is_absolute() or ".." in Path(path_value).parts or
            not isinstance(hash_value, str) or not SHA256_RE.fullmatch(hash_value)):
        raise ValueError(f"mixed_three_cycle_file_binding_format:{name}")
    path = repository / path_value
    if not path.is_file() or sha256(path) != hash_value:
        raise ValueError(f"mixed_three_cycle_file_binding_mismatch:{name}")


def _load_selector(repository: Path, selector_path: str):
    path = repository / selector_path
    spec = importlib.util.spec_from_file_location(
        "_mixed_three_cycle_frozen_selector", path)
    if spec is None or spec.loader is None:
        raise ValueError("mixed_three_cycle_selector_import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selector_terrain(terrain_observation: dict) -> dict:
    grid = terrain_observation["grid"]
    point_count = [
        [grid["mpm_point_count"][row][column] +
         grid["rock_point_count"][row][column]
         for column in range(grid["shape_yx"][1])]
        for row in range(grid["shape_yx"][0])
    ]
    return {
        "valid": True,
        "height_m": grid["height_m"],
        "point_count": point_count,
        "origin_xy_m": grid["origin_xy_m"],
        "resolution_m": grid["resolution_m"],
    }


def _replay_selector(
        repository: Path, selector_path: str, terrain_observation: dict,
        previous_entries: list[list[float]],
) -> dict:
    module = _load_selector(repository, selector_path)
    return module.select_cut(
        _selector_terrain(terrain_observation),
        excluded_entry_xy=previous_entries)


def _replay_dispatch_validation(
        repository: Path, selector_path: str, terrain_observation: dict,
        selection: dict, previous_entries: list[list[float]],
) -> dict:
    module = _load_selector(repository, selector_path)
    return module.validate_cut(
        _selector_terrain(terrain_observation), selection,
        excluded_entry_xy=previous_entries)


def _terrain_changed_near_previous_entry(
        previous: dict, current: dict, entry_xy_m: list[float],
) -> bool:
    old = previous["grid"]
    new = current["grid"]
    if (old["origin_xy_m"] != new["origin_xy_m"] or
            old["resolution_m"] != new["resolution_m"] or
            old["shape_yx"] != new["shape_yx"]):
        raise ValueError("mixed_three_cycle_terrain_grid_changed")
    resolution = float(old["resolution_m"])
    column = math.floor(
        (float(entry_xy_m[0]) - float(old["origin_xy_m"][0])) / resolution)
    row = math.floor(
        (float(entry_xy_m[1]) - float(old["origin_xy_m"][1])) / resolution)
    half_width = max(1, math.ceil(0.60 / resolution) // 2)
    for y_index in range(max(0, row - half_width),
                         min(old["shape_yx"][0], row + half_width + 1)):
        for x_index in range(max(0, column - 5),
                             min(old["shape_yx"][1], column + 2)):
            old_height = old["height_m"][y_index][x_index]
            new_height = new["height_m"][y_index][x_index]
            if ((old_height is None) != (new_height is None) or
                    (old_height is not None and
                     abs(float(old_height) - float(new_height)) > 1e-6)):
                return True
    return False


def validate_report(
        report: dict, config: dict, repository_root: str | Path,
) -> None:
    """Validate a P0 synthetic fixture without granting physical qualification."""

    repository = Path(repository_root).resolve()
    validate_config(config, repository)

    _exact_keys(
        report,
        {"schema", "status", "inputs", "implementation",
         "physics_instance_id", "cycles_requested", "cycles_completed",
         "fixed_timestep", "scope", "cycles", "global_checks"},
        "report",
    )
    if (report["schema"] != REPORT_SCHEMA or
            report["status"] != "mixed_material_three_cycle_contract_fixture_passed" or
            report["cycles_requested"] != 3 or
            report["cycles_completed"] != 3):
        raise ValueError("mixed_three_cycle_report_completion")
    physics_instance_id = report["physics_instance_id"]
    if not isinstance(physics_instance_id, str) or not physics_instance_id:
        raise ValueError("mixed_three_cycle_physics_instance_id")
    _exact_keys(report["inputs"], {"config", "scene_manifest"}, "report.inputs")
    _exact_keys(
        report["implementation"],
        {"contract", "state_adapter", "observation_adapter"},
        "report.implementation",
    )
    _validate_bound_file(repository, report["inputs"]["config"], "inputs.config")
    _validate_bound_file(
        repository, report["inputs"]["scene_manifest"],
        "inputs.scene_manifest", nullable=True)
    if report["inputs"]["scene_manifest"] != {"path": None, "sha256": None}:
        raise ValueError("mixed_three_cycle_p0_scene_must_be_unbound")
    for name, binding in report["implementation"].items():
        _validate_bound_file(repository, binding, f"implementation.{name}")
    expected_paths = {
        "contract": "src/v4/mixed_material_three_cycle_contract.py",
        "state_adapter": "src/v4/mixed_three_cycle_state.py",
        "observation_adapter": "src/v4/mixed_terrain_observation.py",
    }
    if report["inputs"]["config"]["path"] != (
            "configs/v4/mixed-material-three-cycle-v1.json") or any(
                report["implementation"][name]["path"] != path
                for name, path in expected_paths.items()):
        raise ValueError("mixed_three_cycle_unexpected_file_binding")
    if report["fixed_timestep"] != {
        "output_dt_s": 1.0 / 120.0,
        "mujoco_internal_substeps": 40,
    }:
        raise ValueError("mixed_three_cycle_report_timestep")
    if report["scope"] != {
        "synthetic_contract_fixture": True,
        "physics_executed": False,
        "mixed_material": True,
        "same_physics_instance": True,
        "fresh_observation_each_cycle": True,
        "material_reset_between_cycles": False,
        "cycle_index_entry_lookup": False,
        "complete_in_process_reset": False,
        "r3_c_physics_qualified": False,
        "official_envelope": False,
        "training": False,
    }:
        raise ValueError("mixed_three_cycle_report_scope_overclaim")
    if report["global_checks"] != {
        "finite": True,
        "forbidden_contacts_zero": True,
        "mpm_mass_conserved": True,
        "rock_ids_and_mass_conserved": True,
        "same_instance_continuity": True,
        "fresh_terrain_for_cycles_2_and_3": True,
        "truck_inventory_monotonic": True,
        "r3_c_convergence_failure_preserved": True,
    }:
        raise ValueError("mixed_three_cycle_global_checks")
    cycles = report["cycles"]
    if not isinstance(cycles, list) or len(cycles) != 3:
        raise ValueError("mixed_three_cycle_cycle_count")

    acceptance = config["development_acceptance"]
    prior_end = None
    prior_end_hash = None
    prior_observation_sequence = 0
    baseline = None
    observation_hashes: set[str] = set()
    observation_artifact_hashes: set[str] = set()
    observation_state_hashes: set[str] = set()
    planning_terrain_content_hashes: set[str] = set()
    previous_entries: list[list[float]] = []
    previous_planning_terrain = None
    for expected_cycle_id, cycle in enumerate(cycles, start=1):
        _exact_keys(
            cycle,
            {"cycle_id", "start_state", "planning_observation", "selection",
             "dispatch_observation", "dispatch_validation", "stages",
             "stage_boundary_states", "acquisition", "delivery",
             "acquisition_state", "trajectory_artifact", "end_state", "status"},
            f"cycles[{expected_cycle_id}]",
        )
        if (cycle["cycle_id"] != expected_cycle_id or
                cycle["status"] != "completed"):
            raise ValueError("mixed_three_cycle_cycle_order_or_status")
        start = cycle["start_state"]
        end = cycle["end_state"]
        validate_state(start)
        validate_state(end)
        if baseline is None:
            baseline = {
                "particle_count": start["mpm"]["particle_count"],
                "total_mpm_mass_kg": float(start["mpm"]["total_mass_kg"]),
                "rock_mass_kg_by_id": _rock_mass_map(start),
            }
        _validate_conserved_inventory(start, baseline)
        _validate_conserved_inventory(end, baseline)
        if (start["cycle_id"] != expected_cycle_id or
                end["cycle_id"] != expected_cycle_id or
                start["physics_instance_id"] != physics_instance_id or
                end["physics_instance_id"] != physics_instance_id):
            raise ValueError("mixed_three_cycle_state_identity")
        if (end["physics_step"] <= start["physics_step"] or
                float(end["sim_time_s"]) <= float(start["sim_time_s"]) or
                end["mpm"]["history_step"] <= start["mpm"]["history_step"]):
            raise ValueError("mixed_three_cycle_state_time_not_advanced")
        if prior_end is not None:
            validate_cycle_boundary(prior_end, start)

        planning = cycle["planning_observation"]
        _validate_observation_summary(
            planning,
            cycle_id=expected_cycle_id,
            physics_instance_id=physics_instance_id,
            previous_end_hash=prior_end_hash,
            minimum_step=(prior_end["physics_step"] if prior_end is not None
                          else start["physics_step"]),
            minimum_sequence=prior_observation_sequence,
        )
        _validate_conserved_inventory(planning["source_state"], baseline)
        if planning["terrain_content_sha256"] in planning_terrain_content_hashes:
            raise ValueError("mixed_three_cycle_planning_terrain_content_reused")
        planning_terrain_content_hashes.add(planning["terrain_content_sha256"])
        if (previous_planning_terrain is not None and
                not _terrain_changed_near_previous_entry(
                    previous_planning_terrain,
                    planning["terrain_observation"], previous_entries[-1])):
            raise ValueError("mixed_three_cycle_previous_cut_terrain_unchanged")
        selection = cycle["selection"]
        _exact_keys(
            selection,
            {"source_observation_sha256", "source_state_sha256", "entry_xy_m",
             "selection_mode", "cycle_index_lookup",
             "remaining_material_constraint_applied",
             "historical_entry_constraint_applied", "selector_path",
             "selector_sha256", "selector_output_sha256", "selector_output"},
            "selection",
        )
        if (selection["source_observation_sha256"] !=
                planning["observation_sha256"] or
                selection["source_state_sha256"] !=
                planning["source_state_sha256"]):
            raise ValueError("mixed_three_cycle_selection_stale_observation")
        selector_binding = config["frozen_inputs"]["existing_cycle_selector"]
        if (selection["selector_path"] != selector_binding["path"] or
                selection["selector_sha256"] != selector_binding["sha256"]):
            raise ValueError("mixed_three_cycle_selector_binding")
        entry = selection["entry_xy_m"]
        if (not isinstance(entry, list) or len(entry) != 2 or
                any(isinstance(value, bool) or
                    not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) for value in entry)):
            raise ValueError("mixed_three_cycle_selection_entry")
        if (selection["selection_mode"] != "observation_driven" or
                selection["cycle_index_lookup"] is not False or
                selection["remaining_material_constraint_applied"] is not True or
                selection["historical_entry_constraint_applied"] is not True):
            raise ValueError("mixed_three_cycle_preset_or_unconstrained_selection")
        replayed_selection = _replay_selector(
            repository, selection["selector_path"],
            planning["terrain_observation"], previous_entries)
        if (selection["selector_output"] != replayed_selection or
                selection["selector_output_sha256"] !=
                canonical_sha256(replayed_selection) or
                entry != replayed_selection["entry_xy_m"]):
            raise ValueError("mixed_three_cycle_selection_replay_mismatch")

        dispatch = cycle["dispatch_observation"]
        _validate_observation_summary(
            dispatch,
            cycle_id=expected_cycle_id,
            physics_instance_id=physics_instance_id,
            previous_end_hash=prior_end_hash,
            minimum_step=planning["capture_physics_step"],
            minimum_sequence=planning["observation_sequence"],
        )
        _validate_conserved_inventory(dispatch["source_state"], baseline)
        if dispatch["capture_physics_step"] >= end["physics_step"]:
            raise ValueError("mixed_three_cycle_dispatch_not_before_end")
        replayed_dispatch = _replay_dispatch_validation(
            repository, selection["selector_path"],
            dispatch["terrain_observation"], selection["selector_output"],
            previous_entries)
        dispatch_validation = cycle["dispatch_validation"]
        _exact_keys(
            dispatch_validation, {"receipt", "receipt_sha256"},
            "dispatch_validation")
        if (dispatch_validation["receipt"] != replayed_dispatch or
                dispatch_validation["receipt_sha256"] !=
                canonical_sha256(replayed_dispatch)):
            raise ValueError("mixed_three_cycle_dispatch_replay_mismatch")
        for observation in (planning, dispatch):
            for key, seen, error in (
                ("observation_sha256", observation_hashes,
                 "mixed_three_cycle_observation_reused"),
                ("artifact_sha256", observation_artifact_hashes,
                 "mixed_three_cycle_observation_artifact_reused"),
                ("source_state_sha256", observation_state_hashes,
                 "mixed_three_cycle_observation_state_reused"),
            ):
                value = observation[key]
                if value in seen:
                    raise ValueError(error)
                seen.add(value)

        trajectory = cycle["trajectory_artifact"]
        _exact_keys(trajectory, {"path", "sha256"}, "trajectory_artifact")
        if (not isinstance(trajectory["path"], str) or
                not trajectory["path"] or Path(trajectory["path"]).is_absolute() or
                ".." in Path(trajectory["path"]).parts or
                not isinstance(trajectory["sha256"], str) or
                not SHA256_RE.fullmatch(trajectory["sha256"])):
            raise ValueError("mixed_three_cycle_trajectory_binding")
        stages = cycle["stages"]
        if not isinstance(stages, list) or len(stages) != len(REQUIRED_STAGES):
            raise ValueError("mixed_three_cycle_stage_sequence")
        boundary_states = cycle["stage_boundary_states"]
        if (not isinstance(boundary_states, list) or
                len(boundary_states) != len(REQUIRED_STAGES) + 1):
            raise ValueError("mixed_three_cycle_stage_boundary_count")
        for boundary in boundary_states:
            validate_state(boundary)
            _validate_conserved_inventory(boundary, baseline)
            if (boundary["physics_instance_id"] != physics_instance_id or
                    boundary["cycle_id"] != expected_cycle_id or
                    boundary["mpm"]["history_step"] !=
                    boundary["physics_step"] or
                    boundary["contacts"]["forbidden_contact_count"] != 0):
                raise ValueError("mixed_three_cycle_stage_boundary_state")
        previous_stage = None
        for index, (name, stage) in enumerate(zip(REQUIRED_STAGES, stages)):
            _exact_keys(
                stage,
                {"name", "status", "start_physics_step", "end_physics_step",
                 "start_sim_time_s", "end_sim_time_s", "start_state_sha256",
                 "end_state_sha256", "trajectory_sha256"},
                "stage",
            )
            if (stage["name"] != name or stage["status"] != "completed" or
                    stage["trajectory_sha256"] != trajectory["sha256"] or
                    not isinstance(stage["start_physics_step"], int) or
                    isinstance(stage["start_physics_step"], bool) or
                    not isinstance(stage["end_physics_step"], int) or
                    isinstance(stage["end_physics_step"], bool) or
                    stage["end_physics_step"] <= stage["start_physics_step"] or
                    _positive(stage["end_sim_time_s"], "stage.end_sim_time_s",
                              allow_zero=True) <=
                    _positive(stage["start_sim_time_s"], "stage.start_sim_time_s",
                              allow_zero=True) or
                    not SHA256_RE.fullmatch(str(stage["start_state_sha256"])) or
                    not SHA256_RE.fullmatch(str(stage["end_state_sha256"]))):
                raise ValueError("mixed_three_cycle_stage_evidence")
            start_boundary = boundary_states[index]
            end_boundary = boundary_states[index + 1]
            if (stage["start_physics_step"] != start_boundary["physics_step"] or
                    stage["end_physics_step"] != end_boundary["physics_step"] or
                    float(stage["start_sim_time_s"]) !=
                    float(start_boundary["sim_time_s"]) or
                    float(stage["end_sim_time_s"]) !=
                    float(end_boundary["sim_time_s"]) or
                    stage["start_state_sha256"] != state_sha256(start_boundary) or
                    stage["end_state_sha256"] != state_sha256(end_boundary)):
                raise ValueError("mixed_three_cycle_stage_boundary_binding")
            if previous_stage is None:
                if (stage["start_physics_step"] !=
                        dispatch["capture_physics_step"] or
                        abs(float(stage["start_sim_time_s"]) -
                            float(dispatch["capture_sim_time_s"])) > 1e-12 or
                        stage["start_state_sha256"] !=
                        dispatch["source_state_sha256"]):
                    raise ValueError("mixed_three_cycle_stage_dispatch_binding")
            elif (stage["start_physics_step"] !=
                    previous_stage["end_physics_step"] or
                    abs(float(stage["start_sim_time_s"]) -
                        float(previous_stage["end_sim_time_s"])) > 1e-12 or
                    stage["start_state_sha256"] !=
                    previous_stage["end_state_sha256"]):
                raise ValueError("mixed_three_cycle_stage_timeline_gap")
            previous_stage = stage
        if (previous_stage["end_physics_step"] != end["physics_step"] or
                abs(float(previous_stage["end_sim_time_s"]) -
                    float(end["sim_time_s"])) > 1e-12 or
                previous_stage["end_state_sha256"] != state_sha256(end)):
            raise ValueError("mixed_three_cycle_stage_end_binding")

        acquisition_state = cycle["acquisition_state"]
        validate_state(acquisition_state)
        _validate_conserved_inventory(acquisition_state, baseline)
        hold_stage = stages[REQUIRED_STAGES.index("HOLD")]
        if (acquisition_state["physics_instance_id"] != physics_instance_id or
                acquisition_state["cycle_id"] != expected_cycle_id or
                acquisition_state["physics_step"] !=
                hold_stage["end_physics_step"] or
                state_sha256(acquisition_state) !=
                hold_stage["end_state_sha256"]):
            raise ValueError("mixed_three_cycle_acquisition_state_binding")
        ordered_states = [
            start, planning["source_state"], dispatch["source_state"],
            acquisition_state, end,
        ]
        for state_item in ordered_states:
            if (state_item["mpm"]["history_step"] !=
                    state_item["physics_step"] or
                    state_item["contacts"]["forbidden_contact_count"] != 0):
                raise ValueError("mixed_three_cycle_state_history_or_contact")
        for earlier, later in zip(ordered_states, ordered_states[1:]):
            if (later["physics_step"] <= earlier["physics_step"] or
                    float(later["sim_time_s"]) <= float(earlier["sim_time_s"]) or
                    later["mpm"]["history_step"] <=
                    earlier["mpm"]["history_step"] or
                    later["mpm"]["state_sha256"] ==
                    earlier["mpm"]["state_sha256"]):
                raise ValueError("mixed_three_cycle_state_history_not_monotonic")
        _validate_transfer(cycle["acquisition"], "acquisition")
        _validate_transfer(cycle["delivery"], "delivery")
        acquisition = cycle["acquisition"]
        delivery = cycle["delivery"]
        if (float(acquisition["total_mass_kg"]) <
                float(acceptance["minimum_positive_acquisition_kg"]) or
                float(delivery["total_mass_kg"]) <
                float(acceptance["minimum_positive_delivery_kg"])):
            raise ValueError("mixed_three_cycle_nonpositive_transfer")
        if (float(delivery["mpm_mass_kg"]) >
                float(acquisition["mpm_mass_kg"]) + 1e-8 or
                not set(delivery["rock_ids"]).issubset(acquisition["rock_ids"])):
            raise ValueError("mixed_three_cycle_delivery_not_from_acquisition")
        acquired_mpm = (
            float(acquisition_state["bucket"]["residual_mpm_mass_kg"]) -
            float(start["bucket"]["residual_mpm_mass_kg"]))
        acquired_rock_ids = (
            set(acquisition_state["bucket"]["carried_rock_ids"]) -
            set(start["bucket"]["carried_rock_ids"]))
        if (not set(acquisition["rock_ids"]).issubset(
                    baseline["rock_mass_kg_by_id"]) or
                acquisition_state["truck"] != start["truck"] or
                abs(float(start["ledger"]["source_mpm_mass_kg"]) -
                    float(acquisition_state["ledger"]["source_mpm_mass_kg"]) -
                    float(acquisition["mpm_mass_kg"])) > 1e-8 or
                set(acquisition["rock_ids"]) != (
                    set(start["ledger"]["source_rock_ids"]) -
                    set(acquisition_state["ledger"]["source_rock_ids"])) or
                abs(acquired_mpm - float(acquisition["mpm_mass_kg"])) > 1e-8 or
                acquired_rock_ids != set(acquisition["rock_ids"]) or
                abs(sum(baseline["rock_mass_kg_by_id"][rock_id]
                        for rock_id in acquired_rock_ids) -
                    float(acquisition["rock_mass_kg"])) > 1e-8):
            raise ValueError("mixed_three_cycle_acquisition_ledger_not_closed")
        truck_mass = float(end["truck"]["stable_mpm_mass_kg"])
        truck_ids = set(end["truck"]["stable_rock_ids"])
        delivered_rock_ids = truck_ids - set(start["truck"]["stable_rock_ids"])
        if (not set(delivery["rock_ids"]).issubset(
                    baseline["rock_mass_kg_by_id"]) or
                abs(float(end["ledger"]["source_mpm_mass_kg"]) -
                    float(acquisition_state["ledger"]["source_mpm_mass_kg"])) >
                    1e-8 or
                end["ledger"]["source_rock_ids"] !=
                    acquisition_state["ledger"]["source_rock_ids"] or
                abs(float(acquisition_state["bucket"]["residual_mpm_mass_kg"]) -
                    float(end["bucket"]["residual_mpm_mass_kg"]) -
                    float(delivery["mpm_mass_kg"])) > 1e-8 or
                delivered_rock_ids != (
                    set(acquisition_state["bucket"]["carried_rock_ids"]) -
                    set(end["bucket"]["carried_rock_ids"])) or
                abs(truck_mass - float(start["truck"]["stable_mpm_mass_kg"]) -
               float(delivery["mpm_mass_kg"])) > 1e-8 or
                delivered_rock_ids != set(delivery["rock_ids"]) or
                abs(sum(baseline["rock_mass_kg_by_id"][rock_id]
                        for rock_id in delivered_rock_ids) -
                    float(delivery["rock_mass_kg"])) > 1e-8):
            raise ValueError("mixed_three_cycle_delivery_truck_ledger_not_closed")
        if float(end["contacts"]["forbidden_contact_count"]) != 0:
            raise ValueError("mixed_three_cycle_forbidden_contact")

        prior_end = end
        prior_end_hash = state_sha256(end)
        prior_observation_sequence = dispatch["observation_sequence"]
        previous_entries.append(list(entry))
        previous_planning_terrain = planning["terrain_observation"]
