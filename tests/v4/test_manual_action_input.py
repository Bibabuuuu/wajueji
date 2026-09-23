import json
from pathlib import Path
import tempfile
import unittest

from manual_action_input import (
    CONFIG_SCHEMA,
    GamepadFourAxisState,
    KeyboardFourAxisState,
    ManualInputSample,
    apply_symmetric_deadzone,
    load_manual_input_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "v4" / "manual-input-v1.json"


class ManualActionInputTests(unittest.TestCase):
    def load_config(self):
        return load_manual_input_config(CONFIG)

    def write_config(self, value):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "manual-input.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        self.addCleanup(directory.cleanup)
        return path

    def test_symmetric_deadzone_center_boundary_rescale_and_clamp(self):
        self.assertEqual(apply_symmetric_deadzone(0.0, 0.12), 0.0)
        self.assertEqual(apply_symmetric_deadzone(0.12, 0.12), 0.0)
        self.assertEqual(apply_symmetric_deadzone(-0.12, 0.12), 0.0)
        self.assertAlmostEqual(apply_symmetric_deadzone(0.56, 0.12), 0.5)
        self.assertAlmostEqual(apply_symmetric_deadzone(-0.56, 0.12), -0.5)
        self.assertEqual(apply_symmetric_deadzone(1.0, 0.12), 1.0)
        self.assertEqual(apply_symmetric_deadzone(-1.0, 0.12), -1.0)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            apply_symmetric_deadzone(float("nan"), 0.12)
        with self.assertRaisesRegex(ValueError, "invalid_gamepad_deadzone"):
            apply_symmetric_deadzone(0.0, 0.5)

    def test_keyboard_legacy_eight_key_mapping_and_focus_zeroing(self):
        state = KeyboardFourAxisState()
        state.press("L")
        state.press("K")
        state.press("O")
        state.press("N")
        self.assertEqual(state.action(window_focused=True), [1.0, -1.0, 1.0, -1.0])
        self.assertEqual(state.action(window_focused=False), [0.0, 0.0, 0.0, 0.0])
        state.release("L")
        state.press("J")
        self.assertEqual(state.action(window_focused=True), [-1.0, -1.0, 1.0, -1.0])
        self.assertEqual(state.raw_controls(), {"pressed_keys": ["J", "K", "N", "O"]})

    def test_gamepad_order_sign_disconnect_and_latched_emergency_stop(self):
        config = self.load_config()
        config["gamepad"]["axes"]["boom"]["sign"] = -1.0
        state = GamepadFourAxisState(config)
        state.set_stick("leftstick", 1.0, 0.56)
        state.set_stick("rightstick", -0.56, 1.0)
        for actual, expected in zip(
                state.action(window_focused=True), [1.0, -0.5, 1.0, -0.5]):
            self.assertAlmostEqual(actual, expected)
        state.disconnect()
        self.assertEqual(state.action(window_focused=True), [0.0, 0.0, 0.0, 0.0])
        self.assertFalse(state.connected)

        state = GamepadFourAxisState(config)
        state.set_stick("leftstick", 1.0, 1.0)
        state.press_button("back")
        state.set_stick("leftstick", -1.0, -1.0)
        self.assertTrue(state.emergency_stop_latched)
        self.assertEqual(state.action(window_focused=True), [0.0, 0.0, 0.0, 0.0])

    def test_config_rejects_schema_axis_deadzone_button_and_unknown_fields(self):
        valid = json.loads(CONFIG.read_text(encoding="utf-8"))
        cases = []
        missing = dict(valid)
        missing.pop("status")
        cases.append((missing, "missing_fields"))
        extra = dict(valid)
        extra["unexpected"] = True
        cases.append((extra, "unknown_fields"))
        bad_order = json.loads(json.dumps(valid))
        bad_order["axis_order"] = ["boom", "swing", "arm", "bucket"]
        cases.append((bad_order, "axis_order"))
        bad_deadzone = json.loads(json.dumps(valid))
        bad_deadzone["gamepad"]["deadzone"] = 0.5
        cases.append((bad_deadzone, "deadzone"))
        bad_button = json.loads(json.dumps(valid))
        bad_button["gamepad"]["emergency_stop_button"] = "start"
        cases.append((bad_button, "emergency_stop_button"))
        bad_sign = json.loads(json.dumps(valid))
        bad_sign["gamepad"]["axes"]["swing"]["sign"] = float("inf")
        cases.append((bad_sign, "nonfinite"))
        for value, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                load_manual_input_config(self.write_config(value))

    def test_manual_input_sample_preserves_pre_projection_action(self):
        sample = ManualInputSample(
            episode_id="manual-001",
            decision_step=3,
            sim_time_s=0.15,
            sample_wall_monotonic_s=7.0,
            acquisition_domain="local-rtx4070-reconstructed-47",
            input_device="gamepad",
            device_name="Example Controller",
            device_index=0,
            config_sha256="f" * 64,
            window_focused=True,
            connected=True,
            raw_controls={"leftstick": {"x": 0.5, "y": 0.0}},
            normalized_before_projection=(0.5, 0.0, 0.0, 0.0),
            emergency_stop_latched=False,
        ).as_dict()
        self.assertEqual(sample["schema"], "excavation_manual_input_sample_v1")
        self.assertEqual(sample["normalized_before_projection"], [0.5, 0.0, 0.0, 0.0])

    def test_config_schema_is_frozen(self):
        config = self.load_config()
        self.assertEqual(config["schema"], CONFIG_SCHEMA)
        self.assertEqual(config["axis_order"], ["swing", "boom", "arm", "bucket"])


if __name__ == "__main__":
    unittest.main()
