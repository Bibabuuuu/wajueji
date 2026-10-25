import unittest

from dagger_supervisor import HumanTakeoverHandshake
from mpm_manual_demo import (
    _requires_controller_enumeration, _select_control_action,
    _select_gamepad_controller,
)


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

    def test_plain_gamepad_action_keeps_its_device_source(self):
        action, source, first = _select_control_action(
            dagger_mode=False,
            control_mode="human",
            actor_action=None,
            human_action=[0.0, 0.5, 0.0, -0.5],
            human_source="human_gamepad",
            takeover_handshake=HumanTakeoverHandshake(),
        )
        self.assertEqual(action, [0.0, 0.5, 0.0, -0.5])
        self.assertEqual(source, "human_gamepad")
        self.assertFalse(first)

    def test_gamepad_dagger_first_action_is_buffered_once(self):
        handshake = HumanTakeoverHandshake()
        handshake.request(decision_step=5, sim_time_s=0.25)
        handshake.arm(decision_step=5, sim_time_s=0.25)
        first_event = handshake.activate(
            raw_action=[0.0, 0.0, 1.0, 0.0], decision_step=5, sim_time_s=0.25)
        self.assertIsNotNone(first_event)
        with self.assertRaisesRegex(RuntimeError, "not_armed"):
            handshake.activate(
                raw_action=[1.0, 0.0, 0.0, 0.0], decision_step=5, sim_time_s=0.25)
        action, source, first = _select_control_action(
            dagger_mode=True,
            control_mode="human_correction",
            actor_action=None,
            human_action=[0.0, 0.0, 0.0, 0.0],
            human_source="human_gamepad",
            takeover_handshake=handshake,
        )
        self.assertEqual(action, [0.0, 0.0, 1.0, 0.0])
        self.assertEqual(source, "human_correction")
        self.assertTrue(first)

    def test_keyboard_mode_does_not_touch_controller_inventory(self):
        class ExplodingInventory:
            def __len__(self):
                raise AssertionError("controller_inventory_should_not_be_read")

        self.assertIsNone(_select_gamepad_controller(
            mode="keyboard", controllers=ExplodingInventory(), index=0))
        self.assertFalse(_requires_controller_enumeration("keyboard"))
        self.assertTrue(_requires_controller_enumeration("gamepad"))
        self.assertTrue(_requires_controller_enumeration("auto"))

    def test_auto_requires_exactly_one_controller(self):
        controller = object()
        self.assertIs(_select_gamepad_controller(
            mode="auto", controllers=[controller], index=0), controller)
        self.assertIsNone(_select_gamepad_controller(
            mode="auto", controllers=[], index=0))
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            _select_gamepad_controller(mode="auto", controllers=[object(), object()], index=0)


if __name__ == "__main__":
    unittest.main()
