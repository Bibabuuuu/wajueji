"""Side-effect-free keyboard and gamepad normalization for manual demos."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from contracts import ACTION_SCHEMA, AXES


CONFIG_SCHEMA = "excavation_manual_input_config_v1"
SAMPLE_SCHEMA = "excavation_manual_input_sample_v1"
_STICKS = ("leftstick", "rightstick")
_COMPONENTS = ("x", "y")
_KEYBOARD_BINDINGS = {
    "swing": ("L", "J"),
    "boom": ("I", "K"),
    "arm": ("O", "U"),
    "bucket": ("M", "N"),
}


def _finite(value, name, *, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("not_number:" + name)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite:" + name)
    if low is not None and result < low:
        raise ValueError("below_range:" + name)
    if high is not None and result > high:
        raise ValueError("above_range:" + name)
    return result


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_field:" + str(key))
        result[key] = value
    return result


def _exact_fields(value, expected, name):
    if not isinstance(value, dict):
        raise ValueError("not_object:" + name)
    missing = set(expected) - set(value)
    unknown = set(value) - set(expected)
    if missing:
        raise ValueError("missing_fields:%s:%s" % (name, ",".join(sorted(missing))))
    if unknown:
        raise ValueError("unknown_fields:%s:%s" % (name, ",".join(sorted(unknown))))


def apply_symmetric_deadzone(value, deadzone) -> float:
    """Apply the frozen symmetric deadzone without hiding invalid input."""
    value = _finite(value, "gamepad_axis", low=-1.0, high=1.0)
    deadzone = _finite(deadzone, "gamepad_deadzone", low=0.0)
    if deadzone >= 0.5:
        raise ValueError("invalid_gamepad_deadzone")
    if abs(value) <= deadzone:
        return 0.0
    scaled = math.copysign((abs(value) - deadzone) / (1.0 - deadzone), value)
    return max(-1.0, min(1.0, scaled))


def load_manual_input_config(path) -> dict:
    """Load and strictly validate the versioned device mapping configuration."""
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("nonfinite_json_constant:" + token)),
        )
    except json.JSONDecodeError as error:
        raise ValueError("invalid_manual_input_json") from error
    _exact_fields(value, ("schema", "status", "axis_order", "gamepad"), "manual_input_config")
    if value["schema"] != CONFIG_SCHEMA:
        raise ValueError("unsupported_manual_input_schema")
    if value["status"] != "keyboard_and_local_gamepad_candidate":
        raise ValueError("unsupported_manual_input_status")
    if value["axis_order"] != list(AXES):
        raise ValueError("manual_input_axis_order_changed")
    gamepad = value["gamepad"]
    _exact_fields(gamepad, ("deadzone", "axes", "emergency_stop_button"), "gamepad")
    deadzone = _finite(gamepad["deadzone"], "gamepad.deadzone", low=0.0)
    if deadzone >= 0.5:
        raise ValueError("invalid_gamepad_deadzone")
    if gamepad["emergency_stop_button"] != "back":
        raise ValueError("unsupported_emergency_stop_button")
    axes = gamepad["axes"]
    _exact_fields(axes, AXES, "gamepad.axes")
    normalized_axes = {}
    for axis in AXES:
        mapping = axes[axis]
        _exact_fields(mapping, ("stick", "component", "sign"), "gamepad.axes." + axis)
        if mapping["stick"] not in _STICKS:
            raise ValueError("unsupported_gamepad_stick:" + axis)
        if mapping["component"] not in _COMPONENTS:
            raise ValueError("unsupported_gamepad_component:" + axis)
        sign = _finite(mapping["sign"], "gamepad.axes." + axis + ".sign")
        if sign not in (-1.0, 1.0):
            raise ValueError("invalid_gamepad_sign:" + axis)
        normalized_axes[axis] = {
            "stick": mapping["stick"],
            "component": mapping["component"],
            "sign": sign,
        }
    return {
        "schema": CONFIG_SCHEMA,
        "status": value["status"],
        "axis_order": list(AXES),
        "gamepad": {
            "deadzone": deadzone,
            "axes": normalized_axes,
            "emergency_stop_button": "back",
        },
    }


class KeyboardFourAxisState:
    """The legacy J/L, I/K, U/O and N/M mapping without GUI imports."""

    def __init__(self):
        self._held = set()

    def press(self, key) -> None:
        if key not in {item for pair in _KEYBOARD_BINDINGS.values() for item in pair}:
            raise ValueError("unsupported_keyboard_motion_key")
        self._held.add(key)

    def release(self, key) -> None:
        self._held.discard(key)

    def clear(self) -> None:
        self._held.clear()

    def raw_controls(self) -> dict:
        return {"pressed_keys": sorted(self._held)}

    def action(self, *, window_focused: bool) -> list[float]:
        if not window_focused:
            return [0.0] * len(AXES)
        return [
            float(int(positive in self._held) - int(negative in self._held))
            for positive, negative in (_KEYBOARD_BINDINGS[axis] for axis in AXES)
        ]


class GamepadFourAxisState:
    """Controller state fed by normalized Pyglet events, with fail-safe zeroing."""

    def __init__(self, config: dict):
        self._config = config
        self._raw = {stick: {component: 0.0 for component in _COMPONENTS} for stick in _STICKS}
        self.connected = True
        self.emergency_stop_latched = False

    def set_stick(self, stick, x, y) -> None:
        if stick not in _STICKS:
            raise ValueError("unsupported_gamepad_stick_event")
        if not self.connected:
            return
        self._raw[stick]["x"] = _finite(x, stick + ".x", low=-1.0, high=1.0)
        self._raw[stick]["y"] = _finite(y, stick + ".y", low=-1.0, high=1.0)

    def press_button(self, button) -> None:
        if button == self._config["gamepad"]["emergency_stop_button"]:
            self.emergency_stop_latched = True

    def disconnect(self) -> None:
        self.connected = False
        for values in self._raw.values():
            values["x"] = 0.0
            values["y"] = 0.0

    def raw_controls(self) -> dict:
        return {stick: dict(values) for stick, values in self._raw.items()}

    def action(self, *, window_focused: bool) -> list[float]:
        if not window_focused or not self.connected or self.emergency_stop_latched:
            return [0.0] * len(AXES)
        gamepad = self._config["gamepad"]
        output = []
        for axis in AXES:
            mapping = gamepad["axes"][axis]
            raw = self._raw[mapping["stick"]][mapping["component"]]
            output.append(apply_symmetric_deadzone(raw * mapping["sign"], gamepad["deadzone"]))
        return output


@dataclass(frozen=True)
class ManualInputSample:
    """Strict, JSON-ready manual-input evidence for one actual decision step."""

    episode_id: str
    decision_step: int
    sim_time_s: float
    sample_wall_monotonic_s: float
    acquisition_domain: str
    input_device: str
    device_name: str | None
    device_index: int | None
    config_sha256: str | None
    window_focused: bool
    connected: bool
    raw_controls: dict
    normalized_before_projection: tuple
    emergency_stop_latched: bool

    def as_dict(self) -> dict:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("invalid_episode_id")
        if isinstance(self.decision_step, bool) or not isinstance(self.decision_step, int) or self.decision_step < 0:
            raise ValueError("invalid_decision_step")
        if self.input_device not in ("keyboard", "gamepad"):
            raise ValueError("invalid_input_device")
        if not isinstance(self.acquisition_domain, str) or not self.acquisition_domain:
            raise ValueError("invalid_acquisition_domain")
        if not isinstance(self.window_focused, bool) or not isinstance(self.connected, bool):
            raise ValueError("invalid_input_safety_flag")
        if not isinstance(self.emergency_stop_latched, bool):
            raise ValueError("invalid_emergency_stop_latched")
        values = tuple(
            _finite(value, "normalized_before_projection[%d]" % index, low=-1.0, high=1.0)
            for index, value in enumerate(self.normalized_before_projection))
        if len(values) != len(AXES):
            raise ValueError("invalid_action_shape")
        return {
            "schema": SAMPLE_SCHEMA,
            "episode_id": self.episode_id,
            "decision_step": self.decision_step,
            "sim_time_s": _finite(self.sim_time_s, "sim_time_s", low=0.0),
            "sample_wall_monotonic_s": _finite(
                self.sample_wall_monotonic_s, "sample_wall_monotonic_s", low=0.0),
            "acquisition_domain": self.acquisition_domain,
            "input_device": self.input_device,
            "device_name": self.device_name,
            "device_index": self.device_index,
            "config_sha256": self.config_sha256,
            "window_focused": self.window_focused,
            "connected": self.connected,
            "raw_controls": self.raw_controls,
            "normalized_before_projection": list(values),
            "emergency_stop_latched": self.emergency_stop_latched,
        }
