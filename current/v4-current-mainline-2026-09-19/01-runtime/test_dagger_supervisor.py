import copy
import unittest

from contracts import TASK_PHASES, validate_observation_v2, validate_observation_v3
from dagger_supervisor import (
    BucketEndStopProjector, ClosedBucketBoomPauseSupervisor,
    CurlReversePauseSupervisor, HumanTakeoverHandshake, TakeoverSupervisor,
    augment_observation, augment_observation_v3,
    safety_state, task_phase_from_termination_state,
)
from observation_vector_v2 import (
    FEATURE_COUNT, FEATURE_NAMES, NORMALIZED_INDICES, encode_observation,
)
from observation_vector_v3 import (
    FEATURE_COUNT as V3_FEATURE_COUNT,
    FEATURE_NAMES as V3_FEATURE_NAMES,
    NORMALIZED_INDICES as V3_NORMALIZED_INDICES,
    encode_observation as encode_observation_v3,
)
from test_contracts import valid_observation


class DaggerSupervisorTests(unittest.TestCase):
    def test_safety_v2_vector_adds_explicit_validity(self):
        state = safety_state(
            bucket_clearance_m=0.8,
            previous_bucket_clearance_m=None,
            dt_s=0.05,
            bucket_end_stop_confirmed=False,
        )
        value = augment_observation(valid_observation(), state)
        self.assertIs(validate_observation_v2(value), value)
        vector = encode_observation(value)
        self.assertEqual(len(vector), FEATURE_COUNT)
        self.assertEqual(FEATURE_COUNT, 2354)
        self.assertEqual(vector[-3:], (0.0, 0.0, 0.0))
        normalized = set(NORMALIZED_INDICES)
        self.assertNotIn(FEATURE_NAMES.index(
            "safety_state.predicted_ground_ttc_valid"), normalized)
        self.assertNotIn(FEATURE_NAMES.index(
            "safety_state.bucket_end_stop_confirmed"), normalized)

    def test_ttc_uses_descent_and_pause_latches_before_contact(self):
        state = safety_state(
            bucket_clearance_m=0.4,
            previous_bucket_clearance_m=0.45,
            dt_s=0.05,
            bucket_end_stop_confirmed=False,
        )
        self.assertAlmostEqual(state["bucket_descent_rate_mps"], 1.0)
        self.assertAlmostEqual(state["predicted_ground_ttc_s"], 0.4)
        supervisor = TakeoverSupervisor()
        event = supervisor.evaluate(
            state=state,
            digging_interaction_observed=True,
            decision_step=10,
            sim_time_s=0.5,
        )
        self.assertEqual(event["reasons"], ["predicted_ground_ttc"])
        self.assertFalse(event["critic_eligible"])
        safe = safety_state(
            bucket_clearance_m=1.0,
            previous_bucket_clearance_m=0.9,
            dt_s=0.05,
            bucket_end_stop_confirmed=False,
        )
        self.assertIsNone(supervisor.evaluate(
            state=safe, digging_interaction_observed=True,
            decision_step=11, sim_time_s=0.55))
        self.assertTrue(supervisor.intervention_latched)

    def test_v3_adds_fill_blind_task_phase_one_hot(self):
        state = safety_state(
            bucket_clearance_m=0.8,
            previous_bucket_clearance_m=None,
            dt_s=0.05,
            bucket_end_stop_confirmed=False,
        )
        value = augment_observation_v3(
            valid_observation(), state, "off_soil_confirming")
        self.assertIs(validate_observation_v3(value), value)
        vector = encode_observation_v3(value)
        self.assertEqual(V3_FEATURE_COUNT, 2358)
        self.assertEqual(len(vector), V3_FEATURE_COUNT)
        self.assertEqual(vector[-4:], (0.0, 1.0, 0.0, 0.0))
        normalized = set(V3_NORMALIZED_INDICES)
        for phase in TASK_PHASES:
            self.assertNotIn(V3_FEATURE_NAMES.index(
                "task_phase." + phase), normalized)
        self.assertEqual(task_phase_from_termination_state(
            "LOAD_CORRECTION"), "carry")
        self.assertEqual(task_phase_from_termination_state(
            "EVALUATING"), "carry")

    def test_no_pause_before_digging(self):
        state = safety_state(
            bucket_clearance_m=0.05,
            previous_bucket_clearance_m=0.10,
            dt_s=0.05,
            bucket_end_stop_confirmed=False,
        )
        supervisor = TakeoverSupervisor()
        self.assertIsNone(supervisor.evaluate(
            state=state, digging_interaction_observed=False,
            decision_step=2, sim_time_s=0.1))
        self.assertFalse(supervisor.intervention_latched)

    def test_end_stop_requires_confirmation_and_allows_reverse(self):
        projector = BucketEndStopProjector()
        observation = valid_observation()
        observation["proprioception"]["cylinder_position_normalized"][2] = 0.97
        observation["proprioception"]["axis_velocity_normalized"][3] = 0.0
        first, first_meta = projector.project([0.0, 0.0, 0.0, 1.0], observation)
        second, second_meta = projector.project([0.0, 0.0, 0.0, 1.0], observation)
        self.assertEqual(first[3], 1.0)
        self.assertFalse(first_meta["bucket_action_projected"])
        self.assertEqual(second[3], 0.0)
        self.assertTrue(second_meta["bucket_action_projected"])
        reverse, reverse_meta = projector.project([0.0, 0.0, 0.0, -1.0], observation)
        self.assertEqual(reverse[3], -1.0)
        self.assertFalse(reverse_meta["bucket_end_stop_confirmed"])

    def test_closed_bucket_negative_boom_latches_full_action_pause(self):
        projector = BucketEndStopProjector()
        pause = ClosedBucketBoomPauseSupervisor()
        observation = valid_observation()
        observation["proprioception"]["cylinder_position_normalized"][2] = 0.97
        observation["proprioception"]["axis_velocity_normalized"][3] = 0.0
        raw = [0.0, -0.8, 0.4, 1.0]
        _, first_meta = projector.project(raw, observation)
        self.assertIsNone(pause.evaluate(
            raw_action=raw, end_stop=first_meta,
            decision_step=8, sim_time_s=0.4))
        _, second_meta = projector.project(raw, observation)
        event = pause.evaluate(
            raw_action=raw, end_stop=second_meta,
            decision_step=9, sim_time_s=0.45)
        self.assertEqual(event["decision_step"], 9)
        self.assertEqual(event["execution"], "pause_full_action_before_physics")
        self.assertFalse(event["critic_eligible"])
        self.assertTrue(pause.intervention_latched)

    def test_closed_bucket_nonnegative_boom_does_not_pause(self):
        pause = ClosedBucketBoomPauseSupervisor()
        end_stop = {"bucket_end_stop_confirmed": True}
        self.assertIsNone(pause.evaluate(
            raw_action=[0.0, 0.0, 1.0, 0.0], end_stop=end_stop,
            decision_step=10, sim_time_s=0.5))
        self.assertFalse(pause.intervention_latched)

    def test_curl_reverse_requires_started_curl_and_two_ticks(self):
        pause = CurlReversePauseSupervisor()
        reverse = [0.0, 1.0, -1.0, -0.8]
        self.assertIsNone(pause.evaluate(
            raw_action=reverse, curl_started=False,
            decision_step=20, sim_time_s=1.0))
        self.assertIsNone(pause.evaluate(
            raw_action=reverse, curl_started=True,
            decision_step=21, sim_time_s=1.05, routed_phase="curl_load"))
        event = pause.evaluate(
            raw_action=reverse, curl_started=True,
            decision_step=22, sim_time_s=1.1, routed_phase="curl_load")
        self.assertEqual(event["first_reverse_decision_step"], 21)
        self.assertEqual(event["decision_step"], 22)
        self.assertEqual(event["execution"], "pause_full_action_before_physics")
        self.assertFalse(event["critic_eligible"])

    def test_curl_reverse_confirmation_resets_on_supported_action(self):
        pause = CurlReversePauseSupervisor()
        reverse = [0.0, 0.0, 0.0, -0.2]
        supported = [0.0, 0.0, 0.0, 0.0]
        self.assertIsNone(pause.evaluate(
            raw_action=reverse, curl_started=True,
            decision_step=1, sim_time_s=0.05))
        self.assertIsNone(pause.evaluate(
            raw_action=supported, curl_started=True,
            decision_step=2, sim_time_s=0.1))
        self.assertEqual(pause.candidate_ticks, 0)
        self.assertIsNone(pause.evaluate(
            raw_action=reverse, curl_started=True,
            decision_step=3, sim_time_s=0.15))

    def test_v2_rejects_nonnull_invalid_ttc(self):
        state = safety_state(
            bucket_clearance_m=0.8,
            previous_bucket_clearance_m=None,
            dt_s=0.05,
            bucket_end_stop_confirmed=False,
        )
        value = augment_observation(valid_observation(), state)
        value = copy.deepcopy(value)
        value["safety_state"]["predicted_ground_ttc_s"] = 3.0
        with self.assertRaisesRegex(ValueError, "invalid_ttc_must_be_null"):
            validate_observation_v2(value)

    def test_takeover_handshake_keeps_exact_state_until_nonzero_action(self):
        handshake = HumanTakeoverHandshake()
        handshake.request(decision_step=478, sim_time_s=23.9)
        armed = handshake.arm(decision_step=478, sim_time_s=23.9)
        self.assertEqual(handshake.state, "human_correction_armed")
        self.assertFalse(armed["physics_advanced"])
        self.assertIsNone(handshake.activate(
            raw_action=[0.0, 0.0, 0.0, 0.0],
            decision_step=478,
            sim_time_s=23.9,
        ))
        self.assertEqual(handshake.state, "human_correction_armed")
        activated = handshake.activate(
            raw_action=[0.0, 1.0, -1.0, 1.0],
            decision_step=478,
            sim_time_s=23.9,
        )
        self.assertEqual(handshake.state, "human_correction")
        self.assertEqual(activated["decision_step"], 478)
        self.assertEqual(
            activated["raw_human_action"], [0.0, 1.0, -1.0, 1.0])
        self.assertFalse(activated["physics_advanced_before_action"])
        self.assertTrue(activated["record_as_direct_expert_label"])
        buffered, is_first = handshake.action_for_step(
            [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(buffered, [0.0, 1.0, -1.0, 1.0])
        self.assertTrue(is_first)
        handshake.mark_first_action_executed()
        live, is_first = handshake.action_for_step([1.0, 0.0, 0.0, 0.0])
        self.assertEqual(live, [1.0, 0.0, 0.0, 0.0])
        self.assertFalse(is_first)

    def test_takeover_handshake_rejects_state_advance_and_bad_action(self):
        handshake = HumanTakeoverHandshake()
        handshake.request(decision_step=12, sim_time_s=0.6)
        with self.assertRaisesRegex(
                RuntimeError, "state_advanced_before_arm"):
            handshake.arm(decision_step=13, sim_time_s=0.65)
        handshake.arm(decision_step=12, sim_time_s=0.6)
        with self.assertRaisesRegex(ValueError, "action_shape"):
            handshake.activate(
                raw_action=[1.0, 0.0], decision_step=12, sim_time_s=0.6)
        with self.assertRaisesRegex(ValueError, "invalid_takeover_handshake_action"):
            handshake.activate(
                raw_action=[0.0, 0.0, 0.0, 1.1],
                decision_step=12,
                sim_time_s=0.6,
            )
        with self.assertRaisesRegex(
                RuntimeError, "state_advanced_before_first_action"):
            handshake.activate(
                raw_action=[0.0, 1.0, 0.0, 0.0],
                decision_step=13,
                sim_time_s=0.65,
            )


if __name__ == "__main__":
    unittest.main()
