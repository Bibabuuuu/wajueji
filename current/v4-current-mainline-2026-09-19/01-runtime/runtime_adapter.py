"""Versioned 20 Hz to 120/60 Hz four-axis runtime wiring.

This module owns timing, request freshness, causal wrench history and lifecycle.
The injected backend owns physics.  Parsing always completes before a backend
is constructed or a physical step is requested.
"""
from dataclasses import dataclass
import math

from contracts import ACTION_SCHEMA, AXES, FourAxisAction


RESET_SCHEMA = "excavation_four_axis_reset_request_v1"
STEP_SCHEMA = "excavation_four_axis_step_request_v1"
RUNTIME_SCHEMA = "excavation_four_axis_runtime_v1"
POLICY_HZ = 20
MECHANISM_HZ = 120
SOIL_HZ = 60
MECHANISM_STEPS_PER_ACTION = MECHANISM_HZ // POLICY_HZ
SOIL_STEPS_PER_ACTION = SOIL_HZ // POLICY_HZ
TERRAIN_HZ = 5
TERRAIN_ACTION_STRIDE = POLICY_HZ // TERRAIN_HZ
MECHANISM_DT_S = 1.0 / MECHANISM_HZ
POLICY_DT_S = 1.0 / POLICY_HZ
TIME_TOLERANCE_S = 1e-8


def _exact_fields(value, expected, name):
    if not isinstance(value, dict):
        raise ValueError("not_object:" + name)
    missing = set(expected) - set(value)
    unknown = set(value) - set(expected)
    if missing:
        raise ValueError("missing_fields:%s:%s" % (name, ",".join(sorted(missing))))
    if unknown:
        raise ValueError("unknown_fields:%s:%s" % (name, ",".join(sorted(unknown))))


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("invalid_integer:" + name)
    return value


def _finite(value, name, *, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("not_number:" + name)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite:" + name)
    if low is not None and value < low:
        raise ValueError("below_range:" + name)
    if high is not None and value > high:
        raise ValueError("above_range:" + name)
    return value


def _vector(value, length, name):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError("invalid_shape:" + name)
    return tuple(_finite(item, "%s[%d]" % (name, index))
                 for index, item in enumerate(value))


@dataclass(frozen=True)
class ResetRequest:
    episode_id: str
    target_world_m: tuple

    @classmethod
    def parse(cls, value):
        _exact_fields(value, ("schema", "episode_id", "target_world_m"), "reset")
        if value["schema"] != RESET_SCHEMA:
            raise ValueError("unsupported_reset_schema")
        episode_id = value["episode_id"]
        if not isinstance(episode_id, str) or not episode_id or len(episode_id) > 128:
            raise ValueError("invalid_episode_id")
        return cls(episode_id=episode_id,
                   target_world_m=_vector(value["target_world_m"], 3, "target_world_m"))


@dataclass(frozen=True)
class StepRequest:
    decision_step: int
    observation_sequence: int
    observation_sim_time_s: float
    action: FourAxisAction

    @classmethod
    def parse(cls, value):
        _exact_fields(value, (
            "schema", "decision_step", "observation_sequence",
            "observation_sim_time_s", "action",
        ), "step")
        if value["schema"] != STEP_SCHEMA:
            raise ValueError("unsupported_step_schema")
        return cls(
            decision_step=_integer(value["decision_step"], "decision_step"),
            observation_sequence=_integer(
                value["observation_sequence"], "observation_sequence"),
            observation_sim_time_s=_finite(
                value["observation_sim_time_s"], "observation_sim_time_s",
                low=0.0, high=60.0),
            action=FourAxisAction.parse(value["action"]),
        )


@dataclass(frozen=True)
class MechanismCommand:
    substep: int
    expected_start_sim_s: float
    dt_s: float
    physical_velocity_si: tuple
    soil_step_after: bool

    def as_dict(self):
        return {
            "substep": self.substep,
            "expected_start_sim_s": self.expected_start_sim_s,
            "dt_s": self.dt_s,
            "axis_order": list(AXES),
            "physical_velocity_si": list(self.physical_velocity_si),
            "soil_step_after": self.soil_step_after,
        }


@dataclass(frozen=True)
class StepPlan:
    decision_step: int
    start_sim_s: float
    action: FourAxisAction
    mechanism_commands: tuple
    refresh_terrain_after: bool

    def as_dict(self):
        return {
            "schema": RUNTIME_SCHEMA,
            "decision_step": self.decision_step,
            "start_sim_s": self.start_sim_s,
            "end_sim_s": self.start_sim_s + POLICY_DT_S,
            "action": self.action.as_dict(),
            "mechanism_commands": [item.as_dict() for item in self.mechanism_commands],
            "soil_steps": sum(item.soil_step_after for item in self.mechanism_commands),
            "refresh_terrain_after": self.refresh_terrain_after,
        }


def plan_step(decision_step, start_sim_s, action):
    """Expand one validated policy action without touching a backend."""
    decision_step = _integer(decision_step, "decision_step")
    start_sim_s = _finite(start_sim_s, "start_sim_s", low=0.0, high=60.0)
    if not isinstance(action, FourAxisAction):
        raise TypeError("action_must_be_validated")
    physical = action.physical_velocity()
    commands = tuple(
        MechanismCommand(
            substep=index + 1,
            expected_start_sim_s=start_sim_s + index * MECHANISM_DT_S,
            dt_s=MECHANISM_DT_S,
            physical_velocity_si=physical,
            soil_step_after=(index + 1) % (MECHANISM_HZ // SOIL_HZ) == 0,
        )
        for index in range(MECHANISM_STEPS_PER_ACTION)
    )
    return StepPlan(
        decision_step=decision_step,
        start_sim_s=start_sim_s,
        action=action,
        mechanism_commands=commands,
        refresh_terrain_after=(decision_step + 1) % TERRAIN_ACTION_STRIDE == 0,
    )


class FourAxisRuntime:
    """Small synchronous adapter with one backend instance per reset.

    Backend interface:
      reset(reset_request), observe(), capture_terrain(), mechanism_step(command),
      soil_step(), current_wrench(), termination_inputs(), close().
    Each state dictionary returned by ``observe`` must contain ``sim_time_s``.
    """

    def __init__(self, backend_factory):
        self._factory = backend_factory
        self._backend = None
        self._lifecycle = "new"
        self._fault = None

    @property
    def lifecycle(self):
        return self._lifecycle

    @property
    def counters(self):
        return dict(self._counters)

    def _backend_time(self):
        state = self._backend.observe()
        if not isinstance(state, dict) or "sim_time_s" not in state:
            raise RuntimeError("backend_observation_contract")
        return _finite(state["sim_time_s"], "backend.sim_time_s", low=0.0)

    def _capture_terrain(self):
        before = self._backend_time()
        terrain = self._backend.capture_terrain()
        after = self._backend_time()
        if abs(after - before) > TIME_TOLERANCE_S:
            raise RuntimeError("terrain_capture_advanced_physics")
        self._terrain = terrain
        self._terrain_capture_sim_s = before
        self._terrain_sequence += 1
        self._counters["terrain_captures"] += 1

    def _frame(self):
        before = self._backend_time()
        state = self._backend.observe()
        after = self._backend_time()
        if abs(after - before) > TIME_TOLERANCE_S:
            self._trip(RuntimeError("observe_advanced_physics"))
            raise self._fault
        return {
            "schema": RUNTIME_SCHEMA,
            "lifecycle": self._lifecycle,
            "episode_id": self._reset.episode_id,
            "decision_step": self._decision_step,
            "observation_sequence": self._decision_step,
            "sim_time_s": after,
            "backend_state": state,
            "terrain": self._terrain,
            "terrain_sequence": self._terrain_sequence,
            "terrain_capture_sim_s": self._terrain_capture_sim_s,
            "terrain_age_s": after - self._terrain_capture_sim_s,
            "wrench_history_world_si": list(self._wrench_history),
            "previous_action": self._previous_action.as_dict(),
            "termination_inputs": self._backend.termination_inputs(),
            "counters": self.counters,
        }

    def reset(self, value):
        request = ResetRequest.parse(value)
        if self._lifecycle not in ("new", "closed"):
            raise RuntimeError("reset_requires_new_or_closed_runtime")
        backend = self._factory(request)
        self._backend = backend
        self._reset = request
        self._decision_step = 0
        self._terrain_sequence = -1
        self._terrain = None
        self._terrain_capture_sim_s = 0.0
        self._wrench_history = []
        self._previous_action = FourAxisAction((0.0, 0.0, 0.0, 0.0))
        self._counters = {
            "policy_steps": 0,
            "mechanism_steps": 0,
            "soil_steps": 0,
            "terrain_captures": 0,
        }
        self._fault = None
        try:
            backend.reset(request)
            self._lifecycle = "ready"
            self._capture_terrain()
            return self._frame()
        except BaseException as error:
            self._trip(error)
            raise

    def observe(self):
        if self._lifecycle != "ready":
            raise RuntimeError("observe_requires_ready_runtime")
        return self._frame()

    def _trip(self, error):
        self._fault = error
        if self._lifecycle != "closed":
            self._lifecycle = "faulted"

    def step(self, value):
        try:
            request = StepRequest.parse(value)
        except BaseException as error:
            if self._lifecycle == "ready":
                self._trip(error)
            raise
        if self._lifecycle != "ready":
            raise RuntimeError("step_requires_ready_runtime")
        try:
            now = self._backend_time()
            if request.decision_step != self._decision_step:
                raise ValueError("decision_step_mismatch")
            if request.observation_sequence != self._decision_step:
                raise ValueError("stale_observation_sequence")
            if abs(request.observation_sim_time_s - now) > TIME_TOLERANCE_S:
                raise ValueError("stale_observation_time")
            plan = plan_step(self._decision_step, now, request.action)
            for command in plan.mechanism_commands:
                current = self._backend_time()
                if abs(current - command.expected_start_sim_s) > TIME_TOLERANCE_S:
                    raise RuntimeError("mechanism_time_mismatch_before_step")
                self._backend.mechanism_step(command)
                self._counters["mechanism_steps"] += 1
                if command.soil_step_after:
                    self._backend.soil_step()
                    self._counters["soil_steps"] += 1
            end = self._backend_time()
            if abs(end - (now + POLICY_DT_S)) > TIME_TOLERANCE_S:
                raise RuntimeError("policy_time_mismatch_after_step")
            wrench = _vector(list(self._backend.current_wrench()), 6, "current_wrench")
            self._wrench_history.insert(0, list(wrench))
            del self._wrench_history[4:]
            self._decision_step += 1
            self._counters["policy_steps"] += 1
            self._previous_action = request.action
            if plan.refresh_terrain_after:
                self._capture_terrain()
            return {"plan": plan.as_dict(), "observation": self._frame()}
        except BaseException as error:
            self._trip(error)
            raise

    def close(self):
        if self._lifecycle == "closed":
            return
        try:
            if self._backend is not None:
                self._backend.close()
        finally:
            self._lifecycle = "closed"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def action_dict(values):
    return {
        "schema": ACTION_SCHEMA,
        "axis_order": list(AXES),
        "normalized_velocity": list(values),
    }


def step_request(frame, values):
    """Test/smoke helper that binds an action to one exact observation."""
    return {
        "schema": STEP_SCHEMA,
        "decision_step": frame["decision_step"],
        "observation_sequence": frame["observation_sequence"],
        "observation_sim_time_s": frame["sim_time_s"],
        "action": action_dict(values),
    }
