import json
from pathlib import Path
import tempfile
import unittest

from audit_contact_data_contract import (
    analyze_corrections,
    analyze_kinematic_logs,
    analyze_policy_episode,
    joint_addresses_from_xml,
    load_actor_index,
    require_safe_actor_index,
)
from contracts import ACTION_SCHEMA, AXES, OBSERVATION_SCHEMA, FourAxisAction
from runtime_adapter import plan_step


def action(values):
    return {
        "schema": ACTION_SCHEMA,
        "axis_order": list(AXES),
        "normalized_velocity": list(values),
    }


def observation(step, previous, *, velocity=None, cylinders=None, bucket=None):
    sim_time = step * 0.05
    return {
        "schema": OBSERVATION_SCHEMA,
        "decision_step": step,
        "sim_time_s": sim_time,
        "terrain": {
            "height_m": [[0.5] * 38 for _ in range(30)],
            "known_mask": [[True] * 38 for _ in range(30)],
            "valid": True,
            "capture_sim_s": 0.0,
            "age_s": sim_time,
            "sequence": 0,
        },
        "target_world_m": [5.0, 0.0, 0.5],
        "target_valid": True,
        "proprioception": {
            "swing_sin_cos": [0.0, 1.0],
            "cylinder_position_normalized": cylinders or [0.0, 0.0, 0.0],
            "axis_velocity_normalized": velocity or [0.0, 0.0, 0.0, 0.0],
            "axis_valid_mask": [True, True, True, True],
        },
        "bucket": {
            "position_world_m": bucket or [5.0, 0.0, 1.0],
            "orientation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "relative_to_target_m": [0.0, 0.0, 0.5],
            "mouth_angle_deg": 6.5,
            "valid": True,
        },
        "wrench_history": {
            "values_world_si": [None] * 4,
            "age_s": [0.0, 0.05, 0.1, 0.15],
            "valid_mask": [False] * 4,
        },
        "previous_action": action(previous),
        "elapsed_fraction": sim_time / 60.0,
        "digging_interaction_observed": False,
    }


def policy_row(step, observation_value, next_observation, values,
               *, source="human_keyboard", include_raw=False):
    action_value = action(values)
    row = {
        "decision_step": step,
        "observation": observation_value,
        "action": action_value,
        "action_source": source,
        "phase": "approach",
        "plan": plan_step(
            step, step * 0.05, FourAxisAction.parse(action_value)).as_dict(),
        "next_observation": next_observation,
    }
    if include_raw:
        row["raw_action"] = action_value
    return row


def write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ActorIndexTests(unittest.TestCase):
    def test_holdout_rows_are_rejected_before_any_episode_path_is_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "index.ndjson"
            write_jsonl(index, [{
                "schema": "excavation_v4_actor_data_index_record_v1",
                "episode_id": "sealed-example",
                "decision_step": 0,
                "subphase": "approach",
                "action": {
                    "axis_order": list(AXES),
                    "normalized_velocity": [0.0] * 4,
                },
                "axis_mask": [1.0] * 4,
                "training_eligible": True,
                "holdout": True,
            }])
            result = load_actor_index(index)
            self.assertEqual(result["issues"]["codes"]["holdout_row_forbidden"], 1)
            with self.assertRaisesRegex(ValueError, "unsafe_actor_index"):
                require_safe_actor_index(result)

    def test_unlisted_episode_is_rejected_before_source_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "index.ndjson"
            write_jsonl(index, [{
                "schema": "excavation_v4_actor_data_index_record_v1",
                "episode_id": "unexpected-training-episode",
                "decision_step": 0,
                "subphase": "approach",
                "action": {
                    "axis_order": list(AXES),
                    "normalized_velocity": [0.0] * 4,
                },
                "axis_mask": [1.0] * 4,
                "training_eligible": True,
                "holdout": False,
            }])
            result = load_actor_index(index)
            self.assertEqual(
                result["issues"]["codes"]["episode_not_in_explicit_allowlist"], 1)
            with self.assertRaisesRegex(ValueError, "unsafe_actor_index"):
                require_safe_actor_index(result)


class PolicyAuditTests(unittest.TestCase):
    def test_causal_chain_timing_and_physical_action_expansion(self):
        zero = [0.0] * 4
        boom_down = [0.0, -1.0, 0.0, 0.0]
        bucket_curl = [0.0, 0.0, 0.0, 1.0]
        obs0 = observation(0, zero)
        obs1 = observation(
            1, boom_down,
            velocity=[0.0, -1.0, 0.0, 0.0],
            cylinders=[-0.01, 0.0, 0.0],
            bucket=[5.002, 0.0, 0.99],
        )
        obs2 = observation(
            2, bucket_curl,
            velocity=[0.0, 0.0, 0.0, 1.0],
            cylinders=[-0.01, 0.0, 0.01],
            bucket=[5.002, 0.0, 0.989],
        )
        rows = [
            policy_row(0, obs0, obs1, boom_down),
            policy_row(1, obs1, obs2, bucket_curl),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.ndjson"
            write_jsonl(path, rows)
            result = analyze_policy_episode(path, indexed_steps={0, 1})
        self.assertEqual(result["issues"]["count"], 0)
        self.assertEqual(result["indexed_rows_observed"], 2)
        self.assertEqual(
            result["empirical_one_step_response"]["boom"]
            ["position_delta_sign_agreement_fraction"], 1.0)
        self.assertEqual(
            result["empirical_one_step_response"]["bucket"]
            ["position_delta_sign_agreement_fraction"], 1.0)

    def test_broken_observation_chain_is_reported(self):
        zero = [0.0] * 4
        first_action = [0.0, -1.0, 0.0, 0.0]
        obs0 = observation(0, zero)
        expected_obs1 = observation(1, first_action)
        broken_obs1 = observation(1, first_action, bucket=[9.0, 0.0, 1.0])
        obs2 = observation(2, zero)
        rows = [
            policy_row(0, obs0, expected_obs1, first_action),
            policy_row(1, broken_obs1, obs2, zero),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.ndjson"
            write_jsonl(path, rows)
            result = analyze_policy_episode(path)
        self.assertEqual(result["issues"]["codes"]["observation_chain_broken"], 1)


class CorrectionAuditTests(unittest.TestCase):
    def test_correlated_rows_are_segmented_and_matched_to_policy_log(self):
        zero = action([0.0] * 4)
        curl = action([0.0, 0.0, 0.0, 1.0])
        observations = [{"decision_step": step} for step in range(3)]
        policy_rows = []
        correction_rows = []
        for step, raw in enumerate((curl, curl, zero)):
            end_stop = {
                "bucket_action_projected": step == 1,
            }
            applied = zero if step == 1 else raw
            policy_rows.append({
                "decision_step": step,
                "action_source": "human_correction",
                "raw_action": raw,
                "action": applied,
                "end_stop_projection": end_stop,
                "observation": observations[step],
            })
            correction_rows.append({
                "schema": "excavation_dagger_correction_v1",
                "decision_step": step,
                "observation": observations[step],
                "raw_human_action": raw,
                "applied_action": applied,
                "end_stop_projection": end_stop,
                "takeover_first_nonzero_action": step == 0,
                "takeover_handshake_event": {} if step == 0 else None,
                "critic_eligible": False,
                "autonomous_success_eligible": False,
            })
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "policy.ndjson"
            corrections = Path(directory) / "corrections.ndjson"
            write_jsonl(policy, policy_rows)
            write_jsonl(corrections, correction_rows)
            result = analyze_corrections(corrections, policy)
        self.assertEqual(result["issues"]["count"], 0)
        self.assertEqual(result["matched_policy_rows"], 3)
        self.assertEqual(result["constant_sign_command_segment_count"], 2)
        self.assertEqual(result["bucket_end_stop_projected_rows"], 1)
        self.assertEqual(result["raw_action_sign_counts"]["bucket"]["positive"], 2)
        self.assertEqual(result["applied_action_sign_counts"]["bucket"]["positive"], 1)


class KinematicMappingTests(unittest.TestCase):
    MODEL = """<mujoco><worldbody><body><joint type="free"/>
      <body><joint name="chassis" type="hinge"/>
        <body><joint name="chassis piston cylinder" type="hinge"/>
          <body><joint name="chassis piston rod" type="slide"/></body></body>
        <body><joint name="boom" type="hinge"/>
          <body><joint name="boom piston cylinder" type="hinge"/>
            <body><joint name="boom piston rod" type="slide"/></body></body>
          <body><joint name="arm" type="hinge"/>
            <body><joint name="arm piston cylinder" type="hinge"/>
              <body><joint name="arm piston rod" type="slide"/>
                <body><joint name="H link" type="hinge"/></body>
              </body></body>
            <body><joint name="bucket" type="hinge"/>
              <body><joint name="side link" type="hinge"/></body>
            </body>
          </body>
        </body>
      </body>
    </body></worldbody></mujoco>"""

    def test_joint_addresses_and_pose_dependent_ratios_come_from_recorded_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "mechanism.xml"
            model.write_text(self.MODEL, encoding="utf-8")
            addresses = joint_addresses_from_xml(model)
            self.assertEqual(addresses["nq_derived"], 19)
            self.assertEqual(addresses["nv_derived"], 18)
            self.assertEqual(
                addresses["joints"]["arm piston rod"]["dof_address"], 14)
            episode = root / "training" / "ep"
            episode.mkdir(parents=True)
            rows = []
            for step, scale in ((1, 1.0), (2, 1.1)):
                qpos = [0.0] * 19
                qvel = [0.0] * 18
                for qpos_index, value in ((9, 0.4), (12, 0.5), (15, 0.6),
                                           (10, 0.1), (13, 0.2), (17, 0.3)):
                    qpos[qpos_index] = value
                qvel[8], qvel[9] = 0.01 * scale, -0.02 * scale
                qvel[11], qvel[12] = 0.01 * scale, 0.025 * scale
                qvel[14], qvel[16] = 0.02 * scale, 0.08 * scale
                rows.append({
                    "mechanism_step": step,
                    "time_s": step / 120.0,
                    "qpos": qpos,
                    "qvel": qvel,
                    "closure_m": 0.0,
                })
            log = episode / "mechanism-steps.ndjson"
            write_jsonl(log, rows)
            result = analyze_kinematic_logs(
                root / "training", ["ep"], model)
        self.assertEqual(result["issues"]["count"], 0)
        axes = result["cylinder_to_relative_joint_mapping"]
        self.assertAlmostEqual(
            axes["boom"]["d_joint_angle_d_cylinder_length_rad_per_m"]
            ["median"], -2.0)
        self.assertAlmostEqual(
            axes["arm"]["d_joint_angle_d_cylinder_length_rad_per_m"]
            ["median"], 2.5)
        self.assertAlmostEqual(
            axes["bucket"]["d_joint_angle_d_cylinder_length_rad_per_m"]
            ["median"], 4.0)


if __name__ == "__main__":
    unittest.main()
