import copy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "v4"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from audit_layered_mixed_scene_staged_settling import (  # noqa: E402
    audit_candidate,
    load_config,
)


CONFIG = (
    ROOT / "configs" / "v4"
    / "layered-mixed-scene-staged-settling-audit-v1.json")


def passing_candidate(config):
    stages = []
    for index, expected in enumerate(config["expected"]["stages"]):
        history_hashes = {
            name: {"before": f"{index + 1:064x}",
                   "after": f"{index + 1:064x}"}
            for name in config["required_evidence"][
                "activation_history_components"]
        }
        stages.append({
            **expected,
            "start_sim_time_s": 4.0 * index,
            "end_sim_time_s": 4.0 * (index + 1),
            "duration_s": 4.0,
            "step_count": 480,
            "wall_clock_s": 5.0,
            "consecutive_converged_window_s": 0.5,
            "max_rock_linear_speed_m_s": 0.009,
            "max_rock_angular_speed_rad_s": 0.14,
            "mpm_speed_p99_m_s": 0.009,
            "mpm_speed_max_m_s": 0.14,
            "inactive_mujoco_max_pose_drift_m": 0.0,
            "inactive_mujoco_contact_count": 0,
            "inactive_newton_contact_sample_count": 0,
            "inactive_mpm_transfer_count": 0,
            "inactive_mpm_material_count": 0,
            "inactive_mpm_contact_sample_count": 0,
            "activation_history_preserved": True,
            "activation_history_hashes": history_hashes,
            "mass_identity_closed": True,
            "particle_indices_preserved": True,
            "rock_ids_preserved": True,
            "finite": True,
            "converged": True,
        })
    return {
        "schema": config["candidate_report_schema"],
        "status": "staged_settling_passed",
        "input_bindings": copy.deepcopy(config["frozen_inputs"]),
        "implementation": {
            "execution_config": {"path": "configs/v4/candidate.json",
                                 "sha256": "a" * 64},
            "execution_source": {"path": "src/v4/candidate.py",
                                 "sha256": "b" * 64},
            "stable_rock_mapping_sha256": "c" * 64,
        },
        "construction_controls": {
            "pre_registered": True,
            "execution_config_sha256": "a" * 64,
            "gravity_ramp_events": [
                {"start_step": 0, "end_step": 60, "value": 1.0}],
            "temporary_damping_events": [],
            "unregistered_control_changes": False,
            "nominal_values_restored_before_hold": True,
        },
        "nonfinite_failure": None,
        "identity": {
            "same_solver_object": True,
            "same_state_object": True,
            "reset_count": 1,
            "particle_count": 72415,
            "rock_count": 22,
            "particle_indices_preserved": True,
            "rock_ids_preserved": True,
        },
        "stages": stages,
        "nominal_hold": {
            "duration_s": 1.0,
            "tail_window_s": 0.5,
            "max_rock_displacement_m": 0.004,
            "final_max_rock_linear_speed_m_s": 0.001,
            "final_max_rock_angular_speed_rad_s": 0.02,
            "tail_mpm_speed_p99_m_s": 0.009,
            "tail_mpm_speed_max_m_s": 0.14,
            "max_closure_error_m": 0.0009,
            "min_particle_z_m": -0.001,
            "forbidden_contact_count": 0,
            "finite": True,
            "nominal_parameters_restored": True,
            "construction_velocities_zeroed": True,
            "positions_unchanged_by_velocity_zeroing": True,
            "mass_identity_closed": True,
            "particle_indices_preserved": True,
            "rock_ids_preserved": True,
            "construction_walls_present": False,
            "qualified": True,
        },
        "outliers": {
            "top_mpm_speed_particles": [{
                "original_particle_index": 7,
                "position_world_m": [0.0, 0.0, 0.1],
                "speed_m_s": 0.14,
            }],
            "top_rock_displacements": [{
                "rock_id": "layered-rock-000",
                "max_displacement_m": 0.004,
            }],
        },
        "qualifications": {
            "staged_construction_qualified": True,
            "nominal_hold_qualified": True,
            "fullscale_scene_qualified": False,
            "three_cycle_qualified": False,
        },
    }


class LayeredStagedSettlingAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG, ROOT)

    def test_config_freezes_geometry_inventory_thresholds_and_scope(self):
        self.assertEqual(self.config["expected"]["particle_count"], 72415)
        self.assertEqual(self.config["expected"]["rock_count"], 22)
        self.assertEqual(
            [stage["active_particle_count"]
             for stage in self.config["expected"]["stages"]],
            [17747, 30977, 50148, 72415])
        self.assertFalse(self.config["scope"]["physics_execution"])
        self.assertFalse(self.config["scope"]["three_cycle_qualification"])

    def test_passing_candidate_is_only_admitted_for_fullscale_scaleup(self):
        audited = audit_candidate(
            self.config, passing_candidate(self.config))
        self.assertEqual(audited["status"], "admitted_for_fullscale_scaleup")
        self.assertTrue(
            audited["derived_qualifications"]["fullscale_scaleup_authorized"])
        self.assertFalse(
            audited["derived_qualifications"]["p1a_single_scoop_authorized"])
        self.assertFalse(
            audited["derived_qualifications"]["fullscale_scene_qualified"])
        self.assertFalse(
            audited["derived_qualifications"]["three_cycle_qualified"])

    def test_honest_stage_failure_is_preserved_not_rejected(self):
        candidate = passing_candidate(self.config)
        candidate["stages"][2]["mpm_speed_max_m_s"] = 0.2
        candidate["stages"][2]["converged"] = False
        candidate["qualifications"]["staged_construction_qualified"] = False
        candidate["status"] = "staged_settling_failed"
        audited = audit_candidate(self.config, candidate)
        self.assertEqual(audited["status"], "honest_failure_preserved")
        self.assertFalse(
            audited["derived_qualifications"]["fullscale_scaleup_authorized"])

    def test_nonfinite_stage_with_null_metrics_is_preserved(self):
        candidate = passing_candidate(self.config)
        stage = candidate["stages"][1]
        stage["finite"] = False
        stage["converged"] = False
        for name in (
                "max_rock_linear_speed_m_s",
                "max_rock_angular_speed_rad_s",
                "mpm_speed_p99_m_s", "mpm_speed_max_m_s"):
            stage[name] = None
        candidate["nonfinite_failure"] = {
            "detected": True,
            "stage_id": "lower",
            "global_frame": 60,
            "local_frame": 0,
            "phase": "post_coupled_step_pre_trace",
            "fields": [{
                "path": "particle.qd[31415].x",
                "value_class": "nan",
                "entity_type": "particle",
                "entity_id": 31415,
            }],
            "solver_advance_stopped": True,
        }
        candidate["qualifications"]["staged_construction_qualified"] = False
        candidate["status"] = "staged_settling_failed"
        audited = audit_candidate(self.config, candidate)
        self.assertEqual(audited["status"], "honest_failure_preserved")
        self.assertTrue(
            audited["checks"]["nonfinite_failure_evidence_consistent"])
        self.assertFalse(audited["stage_audits"][1]["finite"])
        self.assertFalse(
            audited["stage_audits"][1]["metrics_within_gate"])

    def test_null_metrics_without_nonfinite_evidence_are_rejected(self):
        candidate = passing_candidate(self.config)
        candidate["stages"][1]["finite"] = False
        candidate["stages"][1]["converged"] = False
        candidate["stages"][1]["mpm_speed_max_m_s"] = None
        candidate["qualifications"]["staged_construction_qualified"] = False
        candidate["status"] = "staged_settling_failed"
        audited = audit_candidate(self.config, candidate)
        self.assertEqual(audited["status"], "candidate_report_rejected")
        self.assertFalse(
            audited["checks"]["nonfinite_failure_evidence_consistent"])

    def test_pre_activation_failure_preserves_previous_inventory_and_zero_steps(self):
        candidate = passing_candidate(self.config)
        stage = candidate["stages"][1]
        stage.update({
            "active_particle_count": 17747,
            "active_rock_count": 0,
            "inactive_rock_count": 22,
            "finite": False,
            "converged": False,
            "duration_s": 0.0,
            "consecutive_converged_window_s": 0.0,
            "step_count": 0,
            "end_sim_time_s": stage["start_sim_time_s"],
            "max_rock_linear_speed_m_s": None,
            "max_rock_angular_speed_rad_s": None,
            "mpm_speed_p99_m_s": None,
            "mpm_speed_max_m_s": None,
        })
        candidate["nonfinite_failure"] = {
            "detected": True,
            "stage_id": "lower",
            "global_frame": 60,
            "local_frame": 0,
            "phase": "pre_activation",
            "fields": [{
                "path": "particle.stress[31415].0.0",
                "value_class": "nan",
                "entity_type": "particle",
                "entity_id": 31415,
            }],
            "solver_advance_stopped": True,
        }
        candidate["qualifications"]["staged_construction_qualified"] = False
        candidate["status"] = "staged_settling_failed"
        audited = audit_candidate(self.config, candidate)
        self.assertEqual(audited["status"], "honest_failure_preserved")
        self.assertTrue(audited["stage_audits"][1]["inventory_exact"])
        self.assertTrue(audited["stage_audits"][1]["timing_consistent"])

    def test_false_stage_pass_is_rejected(self):
        candidate = passing_candidate(self.config)
        candidate["stages"][2]["mpm_speed_max_m_s"] = 0.2
        audited = audit_candidate(self.config, candidate)
        self.assertEqual(audited["status"], "candidate_report_rejected")
        self.assertFalse(
            audited["checks"]["all_stage_declarations_consistent"])

    def test_inventory_or_binding_drift_is_rejected(self):
        candidate = passing_candidate(self.config)
        candidate["stages"][1]["active_particle_count"] -= 1
        audited = audit_candidate(self.config, candidate)
        self.assertEqual(audited["status"], "candidate_report_rejected")
        candidate = passing_candidate(self.config)
        candidate["input_bindings"]["particle_bank"]["sha256"] = "0" * 64
        audited = audit_candidate(self.config, candidate)
        self.assertEqual(audited["status"], "candidate_report_rejected")

    def test_hold_overclaim_and_inactive_contact_are_rejected(self):
        candidate = passing_candidate(self.config)
        candidate["nominal_hold"]["max_rock_displacement_m"] = 0.006
        audited = audit_candidate(self.config, candidate)
        self.assertEqual(audited["status"], "candidate_report_rejected")
        candidate = passing_candidate(self.config)
        candidate["stages"][0]["inactive_mujoco_contact_count"] = 1
        audited = audit_candidate(self.config, candidate)
        self.assertEqual(audited["status"], "candidate_report_rejected")

    def test_config_hash_or_threshold_drift_is_rejected(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["frozen_inputs"]["particle_bank"]["sha256"] = "0" * 64
        temporary = ROOT / "work" / "staged-settling-audit-bad-config.json"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(config), encoding="utf-8")
        try:
            with self.assertRaisesRegex(ValueError, "binding_mismatch"):
                load_config(temporary, ROOT)
        finally:
            temporary.unlink(missing_ok=True)

    def test_missing_plan_required_evidence_is_rejected(self):
        candidate = passing_candidate(self.config)
        candidate.pop("construction_controls")
        audited = audit_candidate(self.config, candidate)
        self.assertEqual(audited["status"], "candidate_report_rejected")
        self.assertFalse(audited["checks"]["construction_controls_complete"])

        candidate = passing_candidate(self.config)
        candidate["stages"][1]["inactive_mpm_transfer_count"] = 1
        candidate["stages"][1]["converged"] = False
        candidate["qualifications"]["staged_construction_qualified"] = False
        candidate["status"] = "staged_settling_failed"
        audited = audit_candidate(self.config, candidate)
        self.assertEqual(audited["status"], "honest_failure_preserved")
        self.assertFalse(audited["stage_audits"][1]["inactive_mpm_zero"])


if __name__ == "__main__":
    unittest.main()
