import copy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "v4"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from mixed_material_three_cycle_contract import (  # noqa: E402
    OBSERVATION_SUMMARY_SCHEMA,
    REPORT_SCHEMA,
    REQUIRED_STAGES,
    sha256 as file_sha256,
    validate_config,
    validate_report,
)
from mixed_terrain_observation import (  # noqa: E402
    build_mixed_terrain_observation,
    validate_mixed_terrain_observation,
)
from mixed_three_cycle_state import (  # noqa: E402
    STATE_SCHEMA,
    build_state_snapshot,
    canonical_sha256,
    state_sha256,
    validate_cycle_boundary,
    validate_state,
)
from planning.cycle import select_cut, validate_cut  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "v4" / "mixed-material-three-cycle-v1.json"
HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


def state(cycle_id, step, truck_mass, bucket_mass=0.0):
    source_mass = 100.0 - truck_mass - bucket_mass
    return {
        "schema": STATE_SCHEMA,
        "physics_instance_id": "physics-instance-001",
        "cycle_id": cycle_id,
        "sim_time_s": step / 120.0,
        "physics_step": step,
        "mechanism": {
            "qpos": [0.01 * step] + [0.0] * 18,
            "qvel": [0.0] * 18,
            "command": [0.0] * 4,
        },
        "mpm": {
            "particle_count": 1000,
            "total_mass_kg": 100.0,
            "state_sha256": f"{step:064x}",
            "history_step": step,
        },
        "rocks": [
            {
                "id": rock_id,
                "mass_kg": 2.0,
                "position_world_m": [float(index), 0.0, 0.2],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "linear_velocity_world_m_s": [0.0, 0.0, 0.0],
                "angular_velocity_world_rad_s": [0.0, 0.0, 0.0],
                "contact_objects": ["mpm"],
            }
            for index, rock_id in enumerate(("r1-rock-000", "r1-rock-001"))
        ],
        "bucket": {
            "state_sha256": HEX_A,
            "residual_mpm_mass_kg": bucket_mass,
            "carried_rock_ids": [],
        },
        "truck": {
            "stable_mpm_mass_kg": truck_mass,
            "stable_rock_ids": [],
        },
        "ledger": {
            "source_mpm_mass_kg": source_mass,
            "bucket_mpm_mass_kg": bucket_mass,
            "truck_mpm_mass_kg": truck_mass,
            "unassigned_mpm_mass_kg": 0.0,
            "source_rock_ids": ["r1-rock-000", "r1-rock-001"],
            "bucket_rock_ids": [],
            "truck_rock_ids": [],
            "unassigned_rock_ids": [],
            "mpm_mass_residual_kg": 0.0,
            "rock_mass_residual_kg": 0.0,
        },
        "contacts": {"forbidden_contact_count": 0},
    }


def observation(cycle_id, sequence, step, previous_end_hash, digest,
                source_state, name):
    origin = [0.0, -1.0]
    resolution = 0.1
    shape = [20, 70]
    base_height = 0.6 + 0.01 * cycle_id + 0.001 * sequence
    points = [
        [origin[0] + resolution * (column + 0.5),
         origin[1] + resolution * (row + 0.5), base_height]
        for row in range(shape[0])
        for column in range(shape[1])
    ]
    terrain = build_mixed_terrain_observation(
        mpm_surface_points_world_m=points,
        rock_surface_points_world_m={
            "r1-rock-000": [[5.25, -0.75, base_height + 0.05]],
            "r1-rock-001": [[5.25, 0.75, base_height + 0.04]],
        },
        origin_xy_m=origin,
        resolution_m=resolution,
        shape_yx=shape,
        physics_instance_id="physics-instance-001",
        cycle_id=cycle_id,
        observation_sequence=sequence,
        capture_physics_step=step,
        capture_sim_time_s=step / 120.0,
        source_state_sha256=state_sha256(source_state),
        source_mpm_state_sha256=source_state["mpm"]["state_sha256"],
        source_rocks_state_sha256=canonical_sha256(source_state["rocks"]),
        previous_cycle_end_state_sha256=previous_end_hash,
        bucket_mouth_width_m=0.5,
    )
    return {
        "schema": OBSERVATION_SUMMARY_SCHEMA,
        "physics_instance_id": "physics-instance-001",
        "cycle_id": cycle_id,
        "observation_sequence": sequence,
        "capture_physics_step": step,
        "capture_sim_time_s": step / 120.0,
        "source_state_sha256": state_sha256(source_state),
        "previous_cycle_end_state_sha256": previous_end_hash,
        "observation_sha256": terrain["observation_sha256"],
        "terrain_content_sha256": terrain["content_sha256"],
        "artifact_path": f"cycle-{cycle_id:02d}/{name}.npz",
        "artifact_sha256": canonical_sha256({"name": name, "terrain": terrain}),
        "source_state": source_state,
        "includes_mpm": True,
        "includes_rigid_rocks": True,
        "visible_rock_ids": ["r1-rock-000", "r1-rock-001"],
        "terrain_observation": terrain,
    }


def valid_report():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cycles = []
    previous_end = None
    previous_entries = []
    sequence = 0
    start = state(1, 0, 0.0)
    for cycle_id in range(1, 4):
        if previous_end is not None:
            start = copy.deepcopy(previous_end)
            start["cycle_id"] = cycle_id
        previous_hash = None if previous_end is None else state_sha256(previous_end)
        planning_step = start["physics_step"] + 1
        dispatch_step = start["physics_step"] + 2
        starting_truck_mass = float(start["truck"]["stable_mpm_mass_kg"])
        planning_state = state(cycle_id, planning_step, starting_truck_mass)
        dispatch_state = state(cycle_id, dispatch_step, starting_truck_mass)
        planning = observation(
            cycle_id, sequence + 1, planning_step, previous_hash,
            f"{cycle_id * 2:064x}", planning_state, "planning-observation",
        )
        dispatch = observation(
            cycle_id, sequence + 2, dispatch_step, previous_hash,
            f"{cycle_id * 2 + 1:064x}", dispatch_state,
            "dispatch-observation",
        )
        boundary_states = [dispatch_state]
        for boundary_index in range(1, len(REQUIRED_STAGES) + 1):
            boundary_step = dispatch_step + 2 * boundary_index
            if 4 <= boundary_index <= 6:
                boundary_states.append(state(
                    cycle_id, boundary_step, starting_truck_mass, 1.0))
            elif boundary_index >= 7:
                boundary_states.append(state(
                    cycle_id, boundary_step, starting_truck_mass + 1.0))
            else:
                boundary_states.append(state(
                    cycle_id, boundary_step, starting_truck_mass))
        end = boundary_states[-1]
        trajectory_sha = f"{200 + cycle_id:064x}"
        grid = planning["terrain_observation"]["grid"]
        selector_terrain = {
            "valid": True,
            "height_m": grid["height_m"],
            "point_count": [
                [grid["mpm_point_count"][row][column] +
                 grid["rock_point_count"][row][column]
                 for column in range(grid["shape_yx"][1])]
                for row in range(grid["shape_yx"][0])
            ],
            "origin_xy_m": grid["origin_xy_m"],
            "resolution_m": grid["resolution_m"],
        }
        selector_output = select_cut(
            selector_terrain, excluded_entry_xy=previous_entries)
        dispatch_grid = dispatch["terrain_observation"]["grid"]
        dispatch_terrain = {
            "valid": True,
            "height_m": dispatch_grid["height_m"],
            "point_count": [
                [dispatch_grid["mpm_point_count"][row][column] +
                 dispatch_grid["rock_point_count"][row][column]
                 for column in range(dispatch_grid["shape_yx"][1])]
                for row in range(dispatch_grid["shape_yx"][0])
            ],
            "origin_xy_m": dispatch_grid["origin_xy_m"],
            "resolution_m": dispatch_grid["resolution_m"],
        }
        dispatch_receipt = validate_cut(
            dispatch_terrain, selector_output,
            excluded_entry_xy=previous_entries)
        cycles.append({
            "cycle_id": cycle_id,
            "start_state": start,
            "planning_observation": planning,
            "selection": {
                "source_observation_sha256": planning["observation_sha256"],
                "source_state_sha256": planning["source_state_sha256"],
                "entry_xy_m": selector_output["entry_xy_m"],
                "selection_mode": "observation_driven",
                "cycle_index_lookup": False,
                "remaining_material_constraint_applied": True,
                "historical_entry_constraint_applied": True,
                "selector_path": config["frozen_inputs"][
                    "existing_cycle_selector"]["path"],
                "selector_sha256": config["frozen_inputs"][
                    "existing_cycle_selector"]["sha256"],
                "selector_output_sha256": canonical_sha256(selector_output),
                "selector_output": selector_output,
            },
            "dispatch_observation": dispatch,
            "dispatch_validation": {
                "receipt": dispatch_receipt,
                "receipt_sha256": canonical_sha256(dispatch_receipt),
            },
            "stages": [
                {
                    "name": name,
                    "status": "completed",
                    "start_physics_step": boundary_states[index]["physics_step"],
                    "end_physics_step": boundary_states[index + 1]["physics_step"],
                    "start_sim_time_s": boundary_states[index]["sim_time_s"],
                    "end_sim_time_s": boundary_states[index + 1]["sim_time_s"],
                    "start_state_sha256": state_sha256(boundary_states[index]),
                    "end_state_sha256": state_sha256(boundary_states[index + 1]),
                    "trajectory_sha256": trajectory_sha,
                }
                for index, name in enumerate(REQUIRED_STAGES)
            ],
            "acquisition": {
                "total_mass_kg": 1.0,
                "mpm_mass_kg": 1.0,
                "rock_ids": [],
                "rock_mass_kg": 0.0,
                "stable": True,
            },
            "delivery": {
                "total_mass_kg": 1.0,
                "mpm_mass_kg": 1.0,
                "rock_ids": [],
                "rock_mass_kg": 0.0,
                "stable": True,
            },
            "acquisition_state": boundary_states[
                REQUIRED_STAGES.index("HOLD") + 1],
            "stage_boundary_states": boundary_states,
            "trajectory_artifact": {
                "path": f"cycle-{cycle_id:02d}/trajectory.jsonl",
                "sha256": trajectory_sha,
            },
            "end_state": end,
            "status": "completed",
        })
        previous_end = end
        previous_entries.append(selector_output["entry_xy_m"])
        sequence += 2
    return {
        "schema": REPORT_SCHEMA,
        "status": "mixed_material_three_cycle_contract_fixture_passed",
        "inputs": {
            "config": {
                "path": "configs/v4/mixed-material-three-cycle-v1.json",
                "sha256": file_sha256(CONFIG_PATH),
            },
            "scene_manifest": {"path": None, "sha256": None},
        },
        "implementation": {
            "contract": {
                "path": "src/v4/mixed_material_three_cycle_contract.py",
                "sha256": file_sha256(
                    SOURCE / "mixed_material_three_cycle_contract.py"),
            },
            "state_adapter": {
                "path": "src/v4/mixed_three_cycle_state.py",
                "sha256": file_sha256(SOURCE / "mixed_three_cycle_state.py"),
            },
            "observation_adapter": {
                "path": "src/v4/mixed_terrain_observation.py",
                "sha256": file_sha256(SOURCE / "mixed_terrain_observation.py"),
            },
        },
        "physics_instance_id": "physics-instance-001",
        "cycles_requested": 3,
        "cycles_completed": 3,
        "fixed_timestep": {
            "output_dt_s": 1.0 / 120.0,
            "mujoco_internal_substeps": 40,
        },
        "scope": {
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
        },
        "cycles": cycles,
        "global_checks": {
            "finite": True,
            "forbidden_contacts_zero": True,
            "mpm_mass_conserved": True,
            "rock_ids_and_mass_conserved": True,
            "same_instance_continuity": True,
            "fresh_terrain_for_cycles_2_and_3": True,
            "truck_inventory_monotonic": True,
            "r3_c_convergence_failure_preserved": True,
        },
    }


class MixedMaterialThreeCycleConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_p0_config_freezes_existing_capabilities_without_authorizing_p1(self):
        validate_config(self.config, ROOT)
        self.assertFalse(self.config["scene_binding"]["p1_execution_authorized"])
        self.assertIsNone(
            self.config["scene_binding"]["representative_scene_manifest"])
        self.assertTrue(self.config["scope"]["reuse_existing_truck"])
        self.assertTrue(self.config["scope"]["reuse_existing_three_cycle_loop"])
        self.assertFalse(self.config["scope"]["r3_c_physics_qualified"])
        target = self.config["representative_scene_target"]
        self.assertEqual(target["nominal_material_volume_m3"], 6.1)
        self.assertEqual(target["volume_fractions"], {
            "small_rigid_rock": 0.6,
            "mpm_soil_and_gap_fill": 0.3,
            "medium_rigid_rock": 0.08,
            "large_rigid_rock": 0.02,
        })
        self.assertEqual(target["capacity_preflight_rock_counts"],
                         [64, 256, 512])
        self.assertEqual(target["rock_count_policy"],
                         "derived_from_target_volume_not_fixed")

    def test_rejects_hash_drift_premature_execution_and_scope_overclaim(self):
        attacks = []
        changed = copy.deepcopy(self.config)
        changed["frozen_inputs"]["existing_truck_mesh"]["sha256"] = "0" * 64
        attacks.append((changed, "frozen_input_mismatch"))
        changed = copy.deepcopy(self.config)
        changed["scene_binding"]["p1_execution_authorized"] = True
        attacks.append((changed, "prematurely_authorized"))
        changed = copy.deepcopy(self.config)
        changed["scope"]["r3_c_physics_qualified"] = True
        attacks.append((changed, "scope_expanded"))
        changed = copy.deepcopy(self.config)
        changed["runtime_contract"]["cycle_index_entry_lookup"] = True
        attacks.append((changed, "runtime_contract_drift"))
        changed = copy.deepcopy(self.config)
        changed["representative_scene_target"]["planning_estimate_only"][
            "total"] = 240
        attacks.append((changed, "scene_target_drift"))
        changed = copy.deepcopy(self.config)
        changed["representative_scene_target"][
            "equivalent_diameter_ranges_m"]["small"] = [0.08, 0.14]
        attacks.append((changed, "scene_target_drift"))
        changed = copy.deepcopy(self.config)
        changed["representative_scene_target"]["volume_fractions"][
            "small_rigid_rock"] = 0.5
        attacks.append((changed, "scene_target_drift"))
        for candidate, message in attacks:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_config(candidate, ROOT)


class MixedThreeCycleStateTest(unittest.TestCase):
    def test_live_bridge_fields_adapt_to_canonical_state(self):
        expected = state(1, 0, 0.0)
        bridge = {
            "time_s": expected["sim_time_s"],
            "mechanism_qpos": expected["mechanism"]["qpos"],
            "mechanism_qvel": expected["mechanism"]["qvel"],
            "rocks": [
                {
                    "rock_id": rock["id"],
                    "qpos": rock["position_world_m"] + rock["quaternion_wxyz"],
                    "position": rock["position_world_m"],
                    "linear_velocity_world_m_s":
                        rock["linear_velocity_world_m_s"],
                    "angular_velocity_world_rad_s":
                        rock["angular_velocity_world_rad_s"],
                }
                for rock in expected["rocks"]
            ],
            "forbidden_contact_count": 0,
        }
        actual = build_state_snapshot(
            physics_instance_id=expected["physics_instance_id"],
            cycle_id=1,
            physics_step=0,
            bridge_state=bridge,
            command=expected["mechanism"]["command"],
            mpm=expected["mpm"],
            rock_mass_kg={rock["id"]: rock["mass_kg"]
                          for rock in expected["rocks"]},
            rock_contact_objects={rock["id"]: rock["contact_objects"]
                                  for rock in expected["rocks"]},
            bucket=expected["bucket"],
            truck=expected["truck"],
            ledger=expected["ledger"],
        )
        self.assertEqual(actual, expected)

    def test_state_and_exact_boundary_pass(self):
        end = state(1, 20, 1.0)
        start = copy.deepcopy(end)
        start["cycle_id"] = 2
        validate_state(end)
        validate_cycle_boundary(end, start)
        self.assertEqual(len(state_sha256(end)), 64)

    def test_state_rejects_mass_id_quaternion_and_boundary_reset_attacks(self):
        attacks = []
        changed = state(1, 0, 0.0)
        changed["ledger"]["source_mpm_mass_kg"] -= 1.0
        attacks.append((changed, "mass_not_conserved"))
        changed = state(1, 0, 0.0)
        changed["rocks"][1]["id"] = "r1-rock-000"
        attacks.append((changed, "rock_ids_not_stable_sorted_unique"))
        changed = state(1, 0, 0.0)
        changed["rocks"][0]["quaternion_wxyz"] = [2.0, 0.0, 0.0, 0.0]
        attacks.append((changed, "quaternion_not_unit"))
        for candidate, message in attacks:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_state(candidate)

        end = state(1, 20, 1.0)
        start = copy.deepcopy(end)
        start["cycle_id"] = 2
        start["mpm"]["particle_count"] -= 1
        with self.assertRaisesRegex(ValueError, "state_reset:mpm"):
            validate_cycle_boundary(end, start)


class MixedTerrainObservationTest(unittest.TestCase):
    def build(self):
        return build_mixed_terrain_observation(
            mpm_surface_points_world_m=[
                [0.1, 0.1, 1.0], [1.1, 0.1, 0.5]],
            rock_surface_points_world_m={
                "r1-rock-000": [[1.1, 0.1, 2.0], [0.1, 1.1, 3.0]],
            },
            origin_xy_m=[0.0, 0.0],
            resolution_m=1.0,
            shape_yx=[2, 2],
            physics_instance_id="physics-instance-001",
            cycle_id=2,
            observation_sequence=3,
            capture_physics_step=21,
            capture_sim_time_s=21.0 / 120.0,
            source_state_sha256=HEX_A,
            source_mpm_state_sha256=HEX_C,
            source_rocks_state_sha256=HEX_B,
            previous_cycle_end_state_sha256=HEX_B,
            bucket_mouth_width_m=0.5,
        )

    def test_fuses_mpm_rigid_mixed_unknown_and_is_deterministic(self):
        first = self.build()
        second = self.build()
        validate_mixed_terrain_observation(first, minimum_capture_physics_step=20)
        self.assertEqual(first, second)
        self.assertEqual(
            first["grid"]["material"],
            [["mpm", "mixed"], ["rigid_rock", "unknown"]],
        )
        self.assertEqual(first["grid"]["height_m"], [[1.0, 2.0], [3.0, None]])
        self.assertFalse(
            first["rocks"]["bucket_mouth_passable"]["r1-rock-000"])

    def test_rejects_stale_or_tampered_observation(self):
        value = self.build()
        with self.assertRaisesRegex(ValueError, "stale_physics_step"):
            validate_mixed_terrain_observation(
                value, minimum_capture_physics_step=21)
        value = self.build()
        value["grid"]["height_m"][0][0] = 999.0
        with self.assertRaisesRegex(
                ValueError, "derived_cell_mismatch|hash_mismatch"):
            validate_mixed_terrain_observation(value)

    def test_rejects_rehashed_origin_count_layer_and_ghost_id_attacks(self):
        def resign(value):
            value["content_sha256"] = canonical_sha256({
                "grid": value["grid"], "rocks": value["rocks"]})
            payload = copy.deepcopy(value)
            payload.pop("observation_sha256", None)
            value["observation_sha256"] = canonical_sha256(payload)

        attacks = []
        value = self.build()
        value["grid"]["origin_xy_m"] = ["not-a-number", 0.0]
        resign(value)
        attacks.append((value, "origin"))
        value = self.build()
        value["grid"]["mpm_point_count"][0][0] = -1
        resign(value)
        attacks.append((value, "point_count"))
        value = self.build()
        value["grid"]["material"][0][0] = "rigid_rock"
        resign(value)
        attacks.append((value, "derived_cell_mismatch"))
        value = self.build()
        value["grid"]["rock_ids_by_cell"][0][1].append("ghost-rock")
        value["grid"]["rock_ids_by_cell"][0][1].sort()
        resign(value)
        attacks.append((value, "visible_rock_cell_mismatch"))
        value = self.build()
        value["rocks"]["visible_ids"].append("ghost-rock")
        value["rocks"]["visible_ids"].sort()
        value["rocks"]["equivalent_diameter_m"]["ghost-rock"] = 0.1
        value["rocks"]["bucket_mouth_passable"]["ghost-rock"] = True
        resign(value)
        attacks.append((value, "visible_rock_cell_mismatch"))
        for candidate, message in attacks:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_mixed_terrain_observation(candidate)


class MixedMaterialThreeCycleReportTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_valid_same_instance_fresh_terrain_three_cycle_report(self):
        validate_report(valid_report(), self.config, ROOT)

    def test_rejects_continuity_stale_observation_preset_and_incomplete_attacks(self):
        attacks = []
        changed = valid_report()
        changed["cycles"][1]["start_state"]["physics_instance_id"] = "new-instance"
        attacks.append((changed, "physics_instance_changed|state_identity"))
        changed = valid_report()
        changed["cycles"][1]["planning_observation"][
            "capture_physics_step"] = changed["cycles"][0]["end_state"][
                "physics_step"]
        attacks.append((changed, "stale_observation_step"))
        changed = valid_report()
        changed["cycles"][2]["selection"]["cycle_index_lookup"] = True
        attacks.append((changed, "preset_or_unconstrained_selection"))
        changed = valid_report()
        changed["cycles"][2]["stages"].pop()
        attacks.append((changed, "stage_sequence"))
        changed = valid_report()
        changed["cycles_completed"] = 2
        attacks.append((changed, "report_completion"))
        changed = valid_report()
        changed["cycles"][1]["planning_observation"][
            "previous_cycle_end_state_sha256"] = HEX_C
        attacks.append((changed, "previous_end_binding"))
        for candidate, message in attacks:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_report(candidate, self.config, ROOT)

    def test_rejects_material_reset_id_drift_zero_delivery_and_overclaim(self):
        attacks = []
        changed = valid_report()
        changed["cycles"][1]["start_state"]["mpm"]["particle_count"] -= 1
        attacks.append((changed, "global_material_identity_drift|state_reset:mpm"))
        changed = valid_report()
        changed["cycles"][2]["delivery"]["total_mass_kg"] = 0.0
        changed["cycles"][2]["delivery"]["mpm_mass_kg"] = 0.0
        attacks.append((changed, "positive_required|nonpositive_transfer"))
        changed = valid_report()
        changed["scope"]["r3_c_physics_qualified"] = True
        attacks.append((changed, "scope_overclaim"))
        changed = valid_report()
        changed["cycles"][1]["end_state"]["truck"][
            "stable_mpm_mass_kg"] = 0.5
        attacks.append((changed, "truck_mpm_ledger_mismatch"))
        for candidate, message in attacks:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_report(candidate, self.config, ROOT)

    def test_rejects_relabelled_terrain_global_drift_fake_selection_and_hashes(self):
        attacks = []

        changed = valid_report()
        old = changed["cycles"][0]["planning_observation"][
            "terrain_observation"]
        planning = changed["cycles"][1]["planning_observation"]
        terrain = planning["terrain_observation"]
        terrain["grid"] = copy.deepcopy(old["grid"])
        terrain["rocks"] = copy.deepcopy(old["rocks"])
        terrain["content_sha256"] = canonical_sha256({
            "grid": terrain["grid"], "rocks": terrain["rocks"]})
        terrain_payload = copy.deepcopy(terrain)
        terrain_payload.pop("observation_sha256")
        terrain["observation_sha256"] = canonical_sha256(terrain_payload)
        planning["terrain_content_sha256"] = terrain["content_sha256"]
        planning["observation_sha256"] = terrain["observation_sha256"]
        planning["artifact_sha256"] = canonical_sha256(terrain)
        changed["cycles"][1]["selection"][
            "source_observation_sha256"] = planning["observation_sha256"]
        attacks.append((changed, "sampling_reconstruction_mismatch|planning_terrain_content_reused"))

        changed = valid_report()
        end = changed["cycles"][0]["end_state"]
        end["mpm"]["total_mass_kg"] = 110.0
        end["ledger"]["source_mpm_mass_kg"] += 10.0
        attacks.append((changed, "global_material_identity_drift"))

        changed = valid_report()
        changed["cycles"][0]["end_state"]["rocks"][0]["mass_kg"] = 20.0
        attacks.append((changed, "global_material_identity_drift"))

        changed = valid_report()
        end = changed["cycles"][0]["end_state"]
        end["rocks"][0]["id"] = "r1-rock-100"
        end["rocks"].sort(key=lambda item: item["id"])
        end["ledger"]["source_rock_ids"] = [
            "r1-rock-001", "r1-rock-100"]
        attacks.append((changed, "global_material_identity_drift"))

        changed = valid_report()
        end = changed["cycles"][1]["end_state"]
        end["truck"]["stable_mpm_mass_kg"] = 50.0
        end["ledger"]["truck_mpm_mass_kg"] = 50.0
        end["ledger"]["source_mpm_mass_kg"] = 50.0
        changed["cycles"][1]["stages"][-1]["end_state_sha256"] = (
            state_sha256(end))
        attacks.append((changed, "delivery_truck_ledger_not_closed"))

        changed = valid_report()
        changed["cycles"][0]["selection"]["entry_xy_m"] = [1.0, 2.0]
        attacks.append((changed, "selection_replay_mismatch"))

        changed = valid_report()
        changed["inputs"]["config"]["sha256"] = "0" * 64
        attacks.append((changed, "file_binding_mismatch"))

        changed = valid_report()
        changed["status"] = "mixed_material_three_cycle_passed"
        attacks.append((changed, "report_completion"))

        for candidate, message in attacks:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_report(candidate, self.config, ROOT)

    def test_rejects_dispatch_rollback_history_nan_and_hidden_contact_attacks(self):
        def rebuild_summary(summary, *, mpm_points=None):
            old = summary["terrain_observation"]
            sampling = old["sampling_evidence"]
            rebuilt = build_mixed_terrain_observation(
                mpm_surface_points_world_m=(
                    sampling["mpm_surface_points_world_m"]
                    if mpm_points is None else mpm_points),
                rock_surface_points_world_m=
                    sampling["rock_surface_points_world_m"],
                origin_xy_m=old["grid"]["origin_xy_m"],
                resolution_m=old["grid"]["resolution_m"],
                shape_yx=old["grid"]["shape_yx"],
                physics_instance_id=summary["physics_instance_id"],
                cycle_id=summary["cycle_id"],
                observation_sequence=summary["observation_sequence"],
                capture_physics_step=summary["capture_physics_step"],
                capture_sim_time_s=summary["capture_sim_time_s"],
                source_state_sha256=summary["source_state_sha256"],
                source_mpm_state_sha256=
                    summary["source_state"]["mpm"]["state_sha256"],
                source_rocks_state_sha256=canonical_sha256(
                    summary["source_state"]["rocks"]),
                previous_cycle_end_state_sha256=
                    summary["previous_cycle_end_state_sha256"],
                bucket_mouth_width_m=sampling["bucket_mouth_width_m"],
            )
            summary["terrain_observation"] = rebuilt
            summary["observation_sha256"] = rebuilt["observation_sha256"]
            summary["terrain_content_sha256"] = rebuilt["content_sha256"]
            summary["artifact_sha256"] = canonical_sha256(rebuilt)

        attacks = []

        changed = valid_report()
        dispatch = changed["cycles"][1]["dispatch_observation"]
        low_points = [
            [point[0], point[1], 0.1]
            for point in dispatch["terrain_observation"]["sampling_evidence"]
            ["mpm_surface_points_world_m"]
        ]
        rebuild_summary(dispatch, mpm_points=low_points)
        attacks.append((changed, "cut_selection_invalid|dispatch_replay_mismatch"))

        changed = valid_report()
        acquisition_state = changed["cycles"][1]["acquisition_state"]
        acquisition_state["truck"]["stable_mpm_mass_kg"] = 0.0
        acquisition_state["ledger"]["truck_mpm_mass_kg"] = 0.0
        acquisition_state["ledger"]["source_mpm_mass_kg"] = 99.0
        acquisition_state["ledger"]["bucket_mpm_mass_kg"] = 1.0
        hold_index = REQUIRED_STAGES.index("HOLD")
        changed["cycles"][1]["stage_boundary_states"][hold_index + 1] = (
            acquisition_state)
        changed["cycles"][1]["stages"][hold_index]["end_state_sha256"] = (
            state_sha256(acquisition_state))
        changed["cycles"][1]["stages"][hold_index + 1][
            "start_state_sha256"] = state_sha256(acquisition_state)
        attacks.append((changed, "acquisition_ledger_not_closed"))

        changed = valid_report()
        planning = changed["cycles"][1]["planning_observation"]
        planning["source_state"]["mpm"]["history_step"] = 0
        planning["source_state_sha256"] = state_sha256(planning["source_state"])
        rebuild_summary(planning)
        changed["cycles"][1]["selection"]["source_state_sha256"] = (
            planning["source_state_sha256"])
        changed["cycles"][1]["selection"]["source_observation_sha256"] = (
            planning["observation_sha256"])
        attacks.append((changed, "state_history_or_contact|state_history_not_monotonic"))

        changed = valid_report()
        for stage in changed["cycles"][0]["stages"]:
            stage["start_sim_time_s"] = "nan"
            stage["end_sim_time_s"] = "nan"
        attacks.append((changed, "number_required:stage|stage_evidence"))

        changed = valid_report()
        acquisition_state = changed["cycles"][0]["acquisition_state"]
        acquisition_state["contacts"]["forbidden_contact_count"] = 1
        hold_index = REQUIRED_STAGES.index("HOLD")
        changed["cycles"][0]["stage_boundary_states"][hold_index + 1] = (
            acquisition_state)
        changed["cycles"][0]["stages"][hold_index]["end_state_sha256"] = (
            state_sha256(acquisition_state))
        changed["cycles"][0]["stages"][hold_index + 1][
            "start_state_sha256"] = state_sha256(acquisition_state)
        attacks.append((changed, "stage_boundary_state|history_or_contact"))

        changed = valid_report()
        prior = changed["cycles"][0]["planning_observation"][
            "terrain_observation"]["sampling_evidence"]
        planning = changed["cycles"][1]["planning_observation"]
        reused = copy.deepcopy(prior["mpm_surface_points_world_m"])
        reused[0][2] += 1e-9
        planning["terrain_observation"]["sampling_evidence"][
            "rock_surface_points_world_m"] = copy.deepcopy(
                prior["rock_surface_points_world_m"])
        rebuild_summary(planning, mpm_points=reused)
        changed["cycles"][1]["selection"]["source_observation_sha256"] = (
            planning["observation_sha256"])
        attacks.append((changed, "previous_cut_terrain_unchanged"))

        changed = valid_report()
        prior = changed["cycles"][0]["planning_observation"][
            "terrain_observation"]["sampling_evidence"]
        planning = changed["cycles"][1]["planning_observation"]
        duplicated = copy.deepcopy(prior["mpm_surface_points_world_m"])
        duplicated.append(copy.deepcopy(duplicated[0]))
        planning["terrain_observation"]["sampling_evidence"][
            "rock_surface_points_world_m"] = copy.deepcopy(
                prior["rock_surface_points_world_m"])
        rebuild_summary(planning, mpm_points=duplicated)
        changed["cycles"][1]["selection"]["source_observation_sha256"] = (
            planning["observation_sha256"])
        attacks.append((changed, "previous_cut_terrain_unchanged"))

        for candidate, message in attacks:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_report(candidate, self.config, ROOT)


if __name__ == "__main__":
    unittest.main()
