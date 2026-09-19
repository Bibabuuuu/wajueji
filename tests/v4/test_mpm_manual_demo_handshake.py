import unittest

from dagger_supervisor import HumanTakeoverHandshake
from mpm_manual_demo import _select_control_action


class ManualDemoHandshakeIntegrationTests(unittest.TestCase):
    def test_actor_first_step_defines_non_handshake_flag(self):
        action, source, first = _select_control_action(
            dagger_mode=True,
            control_mode="actor",
            actor_action=[0.0, -0.2, 0.3, 0.0],
            human_action=[0.0, 0.0, 0.0, 0.0],
            takeover_handshake=HumanTakeoverHandshake(),
        )
        self.assertEqual(action, [0.0, -0.2, 0.3, 0.0])
        self.assertEqual(source, "actor_prefix")
        self.assertFalse(first)

    def test_buffered_first_human_action_overrides_released_key_state(self):
        handshake = HumanTakeoverHandshake()
        handshake.request(decision_step=478, sim_time_s=23.9)
        handshake.arm(decision_step=478, sim_time_s=23.9)
        handshake.activate(
            raw_action=[0.0, 1.0, -1.0, 1.0],
            decision_step=478,
            sim_time_s=23.9,
        )
        action, source, first = _select_control_action(
            dagger_mode=True,
            control_mode="human_correction",
            actor_action=None,
            human_action=[0.0, 0.0, 0.0, 0.0],
            takeover_handshake=handshake,
        )
        self.assertEqual(action, [0.0, 1.0, -1.0, 1.0])
        self.assertEqual(source, "human_correction")
        self.assertTrue(first)

    def test_plain_manual_action_is_never_a_takeover_first(self):
        action, source, first = _select_control_action(
            dagger_mode=False,
            control_mode="human",
            actor_action=None,
            human_action=[1.0, 0.0, 0.0, 0.0],
            takeover_handshake=HumanTakeoverHandshake(),
        )
        self.assertEqual(action, [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(source, "human_keyboard")
        self.assertFalse(first)


if __name__ == "__main__":
    unittest.main()
