import copy
import json
import unittest

from contracts import (ACTION_SCHEMA, AXES, OBSERVATION_SCHEMA, FourAxisAction,
                       validate_observation)


def valid_observation():
    known = [[True] * 38 for _ in range(30)]
    height = [[0.5] * 38 for _ in range(30)]
    known[0][0] = False
    height[0][0] = None
    zero_action = {
        "schema": ACTION_SCHEMA,
        "axis_order": list(AXES),
        "normalized_velocity": [0.0, 0.0, 0.0, 0.0],
    }
    return {
        "schema": OBSERVATION_SCHEMA,
        "decision_step": 4,
        "sim_time_s": 0.2,
        "terrain": {
            "height_m": height,
            "known_mask": known,
            "valid": True,
            "capture_sim_s": 0.0,
            "age_s": 0.2,
            "sequence": 0,
        },
        "target_world_m": [5.0, 0.0, 0.5],
        "target_valid": True,
        "proprioception": {
            "swing_sin_cos": [0.0, 1.0],
            "cylinder_position_normalized": [0.0, 0.0, 0.0],
            "axis_velocity_normalized": [0.0, 0.0, 0.0, 0.0],
            "axis_valid_mask": [True, True, True, True],
        },
        "bucket": {
            "position_world_m": [5.0, 0.0, 1.0],
            "orientation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "relative_to_target_m": [0.0, 0.0, 0.5],
            "mouth_angle_deg": 6.5,
            "valid": True,
        },
        "wrench_history": {
            "values_world_si": [[1.0] * 6, [0.0] * 6, None, None],
            "age_s": [0.0, 0.05, 0.1, 0.15],
            "valid_mask": [True, True, False, False],
        },
        "previous_action": zero_action,
        "elapsed_fraction": 0.2 / 60.0,
        "digging_interaction_observed": True,
    }


class ActionContractTests(unittest.TestCase):
    def test_simultaneous_action_maps_to_current_physical_speeds(self):
        action = FourAxisAction.parse({
            "schema": ACTION_SCHEMA,
            "axis_order": list(AXES),
            "normalized_velocity": [1.0, -1.0, 0.5, 0.25],
        })
        self.assertEqual(action.physical_velocity(), (0.12, -0.0225, 0.01125, 0.016875))

    def test_unknown_duplicate_nonfinite_and_range_are_rejected(self):
        base = {"schema": ACTION_SCHEMA, "axis_order": list(AXES),
                "normalized_velocity": [0.0] * 4}
        for key, value in (("extra", 1),):
            bad = dict(base)
            bad[key] = value
            with self.assertRaisesRegex(ValueError, "unknown_fields"):
                FourAxisAction.parse(bad)
        with self.assertRaisesRegex(ValueError, "duplicate_field"):
            FourAxisAction.loads(
                '{"schema":"%s","schema":"%s","axis_order":%s,"normalized_velocity":[0,0,0,0]}'
                % (ACTION_SCHEMA, ACTION_SCHEMA, json.dumps(list(AXES))))
        for bad_value in (float("nan"), float("inf"), 1.01, True):
            bad = copy.deepcopy(base)
            bad["normalized_velocity"][2] = bad_value
            with self.assertRaises(ValueError):
                FourAxisAction.parse(bad)

    def test_axis_order_is_frozen(self):
        bad = {"schema": ACTION_SCHEMA, "axis_order": list(reversed(AXES)),
               "normalized_velocity": [0.0] * 4}
        with self.assertRaisesRegex(ValueError, "axis_order"):
            FourAxisAction.parse(bad)


class ObservationContractTests(unittest.TestCase):
    def test_valid_observation(self):
        value = valid_observation()
        self.assertIs(validate_observation(value), value)

    def test_stale_map_and_out_of_scale_measured_state_remain_recordable(self):
        value = valid_observation()
        value["terrain"]["valid"] = False
        value["proprioception"]["axis_velocity_normalized"][0] = 1.2
        value["proprioception"]["cylinder_position_normalized"][2] = 1.01
        self.assertIs(validate_observation(value), value)

    def test_unknown_terrain_is_not_zero_filled(self):
        value = valid_observation()
        value["terrain"]["height_m"][0][0] = 0.0
        with self.assertRaisesRegex(ValueError, "unknown_terrain_must_be_null"):
            validate_observation(value)

    def test_shape_time_and_wrench_causality_metadata_are_strict(self):
        value = valid_observation()
        value["terrain"]["height_m"].pop()
        with self.assertRaisesRegex(ValueError, "invalid_shape"):
            validate_observation(value)
        value = valid_observation()
        value["terrain"]["age_s"] = 0.1
        with self.assertRaisesRegex(ValueError, "terrain_time_mismatch"):
            validate_observation(value)
        value = valid_observation()
        value["wrench_history"]["age_s"] = [0.05, 0.0, 0.1, 0.15]
        with self.assertRaisesRegex(ValueError, "not_newest_first"):
            validate_observation(value)

    def test_actor_excluded_medium_or_scoring_truth_is_rejected(self):
        for field in ("medium_id", "fill_ratio_truth", "particle_id"):
            value = valid_observation()
            value[field] = "forbidden"
            with self.assertRaisesRegex(ValueError, "unknown_fields"):
                validate_observation(value)


if __name__ == "__main__":
    unittest.main()
