"""Pure safety-state, end-stop projection and DAgger takeover logic.

The supervisor never advances physics.  It latches an intervention request;
the interactive runner owns pausing and the human-control handoff.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from contracts import (OBSERVATION_SCHEMA_V2, OBSERVATION_SCHEMA_V3,
                       TASK_PHASES)


SAFETY_STATE_SCHEMA = "excavation_actor_safety_state_v1"
BUCKET_END_STOP_POSITION = 0.95
BUCKET_END_STOP_RELEASE = 0.94
BUCKET_END_STOP_SPEED = 0.01
BUCKET_END_STOP_CONFIRM_TICKS = 2
ACTIVE_DESCENT_RATE_MPS = 0.02
DEFAULT_TAKEOVER_TTC_S = 2.0
DEFAULT_TAKEOVER_CLEARANCE_M = 0.10


def safety_state(*, bucket_clearance_m, previous_bucket_clearance_m,
                 dt_s, bucket_end_stop_confirmed):
    values = (bucket_clearance_m, dt_s)
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool)
               and math.isfinite(float(item)) for item in values):
        raise ValueError("invalid_safety_state_input")
    if dt_s <= 0.0:
        raise ValueError("invalid_safety_state_dt")
    clearance = float(bucket_clearance_m)
    if previous_bucket_clearance_m is None:
        descent = 0.0
    else:
        previous = float(previous_bucket_clearance_m)
        if not math.isfinite(previous):
            raise ValueError("invalid_previous_bucket_clearance")
        descent = max(0.0, (previous - clearance) / float(dt_s))
    ttc_valid = descent > ACTIVE_DESCENT_RATE_MPS and clearance >= 0.0
    ttc = clearance / descent if ttc_valid else None
    return {
        "bucket_clearance_m": clearance,
        "bucket_descent_rate_mps": descent,
        "predicted_ground_ttc_s": ttc,
        "predicted_ground_ttc_valid": ttc_valid,
        "bucket_end_stop_confirmed": bool(bucket_end_stop_confirmed),
    }


def augment_observation(legacy_observation, state):
    value = dict(legacy_observation)
    value["schema"] = OBSERVATION_SCHEMA_V2
    value["safety_state"] = dict(state)
    return value


def task_phase_from_termination_state(state):
    """Collapse termination states without revealing fill qualification."""
    mapping = {
        "DIGGING": "digging",
        "OFF_SOIL_CONFIRMING": "off_soil_confirming",
        "WAITING_FOR_BUCKET_CLOSURE": "waiting_for_bucket_closure",
        "LOAD_CORRECTION": "carry",
        "EVALUATING": "carry",
        "SUCCESS": "carry",
        "LOAD_FAILURE": "carry",
        "CARRY_FAILURE": "carry",
        "HARD_FAILURE": "carry",
        "TIME_LIMIT": "carry",
    }
    try:
        phase = mapping[state]
    except KeyError as exc:
        raise ValueError("unsupported_termination_state:" + str(state)) from exc
    if phase not in TASK_PHASES:
        raise RuntimeError("invalid_task_phase_mapping")
    return phase


def augment_observation_v3(legacy_observation, state, task_phase):
    value = augment_observation(legacy_observation, state)
    value["schema"] = OBSERVATION_SCHEMA_V3
    if not isinstance(task_phase, str) or task_phase not in TASK_PHASES:
        raise ValueError("unsupported_task_phase")
    value["task_phase"] = task_phase
    return value


@dataclass
class BucketEndStopProjector:
    position_threshold: float = BUCKET_END_STOP_POSITION
    release_threshold: float = BUCKET_END_STOP_RELEASE
    speed_threshold: float = BUCKET_END_STOP_SPEED
    confirmation_ticks: int = BUCKET_END_STOP_CONFIRM_TICKS
    confirmed: bool = False
    candidate_ticks: int = 0
    projection_count: int = 0

    def __post_init__(self):
        if not (0.0 < self.release_threshold < self.position_threshold < 1.0):
            raise ValueError("invalid_end_stop_position_thresholds")
        if self.speed_threshold <= 0.0 or self.confirmation_ticks <= 0:
            raise ValueError("invalid_end_stop_confirmation")

    def project(self, raw_action, observation, *, record_projection=True):
        if len(raw_action) != 4:
            raise ValueError("invalid_end_stop_action_shape")
        action = [float(item) for item in raw_action]
        if not all(math.isfinite(item) and -1.0 <= item <= 1.0 for item in action):
            raise ValueError("invalid_end_stop_action")
        position = float(observation["proprioception"]
                         ["cylinder_position_normalized"][2])
        speed = float(observation["proprioception"]
                      ["axis_velocity_normalized"][3])
        request = action[3]

        if request < 0.0 or position < self.release_threshold:
            self.confirmed = False
            self.candidate_ticks = 0
        elif (request > 0.0 and position >= self.position_threshold
              and abs(speed) < self.speed_threshold):
            self.candidate_ticks += 1
            if self.candidate_ticks >= self.confirmation_ticks:
                self.confirmed = True
        elif not self.confirmed:
            self.candidate_ticks = 0

        projected = self.confirmed and request > 0.0
        if projected:
            action[3] = 0.0
            if record_projection:
                self.projection_count += 1
        return action, {
            "bucket_end_stop_confirmed": self.confirmed,
            "bucket_action_projected": projected,
            "raw_bucket_action": request,
            "applied_bucket_action": action[3],
            "candidate_ticks": self.candidate_ticks,
        }


@dataclass
class TakeoverSupervisor:
    ttc_threshold_s: float = DEFAULT_TAKEOVER_TTC_S
    clearance_threshold_m: float = DEFAULT_TAKEOVER_CLEARANCE_M
    intervention_latched: bool = False
    first_event: dict | None = None

    def __post_init__(self):
        if self.ttc_threshold_s <= 0.0 or self.clearance_threshold_m <= 0.0:
            raise ValueError("invalid_takeover_threshold")

    def evaluate(self, *, state, digging_interaction_observed,
                 decision_step, sim_time_s):
        reasons = []
        if digging_interaction_observed:
            if (state["predicted_ground_ttc_valid"]
                    and state["predicted_ground_ttc_s"] < self.ttc_threshold_s):
                reasons.append("predicted_ground_ttc")
            if (state["bucket_clearance_m"] < self.clearance_threshold_m
                    and state["bucket_descent_rate_mps"] > ACTIVE_DESCENT_RATE_MPS):
                reasons.append("low_clearance_descending")
        if reasons and not self.intervention_latched:
            self.intervention_latched = True
            self.first_event = {
                "schema": "excavation_dagger_takeover_event_v1",
                "decision_step": int(decision_step),
                "sim_time_s": float(sim_time_s),
                "reasons": reasons,
                "safety_state": dict(state),
                "episode_autonomous_success_eligible": False,
                "critic_eligible": False,
            }
        return self.first_event if reasons else None


@dataclass
class ClosedBucketBoomPauseSupervisor:
    """Latch a takeover before executing boom-down at a closed bucket.

    The caller must pass metadata produced by ``BucketEndStopProjector`` for
    the same raw action.  This object only decides whether the full action
    must be paused; it never advances physics and never substitutes a
    partially projected action.
    """

    intervention_latched: bool = False
    first_event: dict | None = None

    def evaluate(self, *, raw_action, end_stop, decision_step, sim_time_s,
                 safety_state=None):
        if len(raw_action) != 4:
            raise ValueError("invalid_closed_bucket_boom_action_shape")
        action = [float(item) for item in raw_action]
        if not all(math.isfinite(item) and -1.0 <= item <= 1.0
                   for item in action):
            raise ValueError("invalid_closed_bucket_boom_action")
        if not isinstance(end_stop, dict):
            raise ValueError("invalid_closed_bucket_end_stop_metadata")
        confirmed = end_stop.get("bucket_end_stop_confirmed")
        if not isinstance(confirmed, bool):
            raise ValueError("invalid_closed_bucket_end_stop_confirmation")
        step = int(decision_step)
        time_s = float(sim_time_s)
        if step < 0 or not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError("invalid_closed_bucket_boom_event_position")

        should_pause = confirmed and action[1] < 0.0
        if should_pause and not self.intervention_latched:
            if safety_state is not None and not isinstance(safety_state, dict):
                raise ValueError("invalid_closed_bucket_boom_safety_state")
            self.intervention_latched = True
            self.first_event = {
                "schema": "excavation_closed_bucket_boom_pause_event_v1",
                "decision_step": step,
                "sim_time_s": time_s,
                "reason": "negative_boom_after_confirmed_bucket_end_stop",
                "reasons": ["negative_boom_after_confirmed_bucket_end_stop"],
                "raw_action": action,
                "safety_state": (
                    dict(safety_state) if safety_state is not None else None),
                "bucket_end_stop_confirmed": True,
                "execution": "pause_full_action_before_physics",
                "human_takeover_required": True,
                "episode_autonomous_success_eligible": False,
                "critic_eligible": False,
            }
        return self.first_event if should_pause else None


@dataclass
class CurlReversePauseSupervisor:
    """Pause the full action when a started curl reverses for two ticks."""

    reverse_threshold: float = -0.1
    confirmation_ticks: int = 2
    candidate_ticks: int = 0
    intervention_latched: bool = False
    first_event: dict | None = None

    def __post_init__(self):
        if (not math.isfinite(self.reverse_threshold)
                or not -1.0 < self.reverse_threshold < 0.0):
            raise ValueError("invalid_curl_reverse_threshold")
        if self.confirmation_ticks < 2:
            raise ValueError("curl_reverse_requires_two_ticks")

    def evaluate(self, *, raw_action, curl_started, decision_step, sim_time_s,
                 routed_phase=None, safety_state=None):
        if len(raw_action) != 4:
            raise ValueError("invalid_curl_reverse_action_shape")
        action = [float(item) for item in raw_action]
        if not all(math.isfinite(item) and -1.0 <= item <= 1.0
                   for item in action):
            raise ValueError("invalid_curl_reverse_action")
        if not isinstance(curl_started, bool):
            raise ValueError("invalid_curl_started")
        reverse = curl_started and action[3] < self.reverse_threshold
        self.candidate_ticks = self.candidate_ticks + 1 if reverse else 0
        should_pause = (
            self.candidate_ticks >= self.confirmation_ticks
            and not self.intervention_latched)
        if should_pause:
            step = int(decision_step)
            time_s = float(sim_time_s)
            if step < 0 or not math.isfinite(time_s) or time_s < 0.0:
                raise ValueError("invalid_curl_reverse_event_position")
            if safety_state is not None and not isinstance(safety_state, dict):
                raise ValueError("invalid_curl_reverse_safety_state")
            self.intervention_latched = True
            self.first_event = {
                "schema": "excavation_curl_reverse_pause_event_v1",
                "decision_step": step,
                "first_reverse_decision_step": (
                    step - self.confirmation_ticks + 1),
                "sim_time_s": time_s,
                "reason": "negative_bucket_after_confirmed_curl_start",
                "reasons": ["negative_bucket_after_confirmed_curl_start"],
                "raw_action": action,
                "routed_phase": routed_phase,
                "curl_started": True,
                "reverse_threshold": self.reverse_threshold,
                "confirmation_ticks": self.confirmation_ticks,
                "safety_state": (
                    None if safety_state is None else dict(safety_state)),
                "execution": "pause_full_action_before_physics",
                "human_takeover_required": True,
                "episode_autonomous_success_eligible": False,
                "critic_eligible": False,
            }
        return self.first_event if should_pause else None


@dataclass
class HumanTakeoverHandshake:
    """Keep physics frozen until an armed operator supplies a real action.

    ``request`` is called at the exact pre-physics safety boundary, ``arm``
    records the operator's Enter acknowledgement without releasing that
    boundary, and ``activate`` releases it only for a finite, bounded,
    non-zero four-axis command.  The caller is responsible for executing the
    returned first action exactly once before consulting live keyboard state.
    """

    state: str = "idle"
    requested_decision_step: int | None = None
    requested_sim_time_s: float | None = None
    armed_event: dict | None = None
    activation_event: dict | None = None
    pending_first_action: list[float] | None = None

    @staticmethod
    def _event_position(decision_step, sim_time_s):
        step = int(decision_step)
        time_s = float(sim_time_s)
        if step < 0 or not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError("invalid_takeover_handshake_event_position")
        return step, time_s

    @staticmethod
    def _action(raw_action):
        if len(raw_action) != 4:
            raise ValueError("invalid_takeover_handshake_action_shape")
        action = [float(item) for item in raw_action]
        if not all(math.isfinite(item) and -1.0 <= item <= 1.0
                   for item in action):
            raise ValueError("invalid_takeover_handshake_action")
        return action

    def request(self, *, decision_step, sim_time_s):
        if self.state != "idle":
            raise RuntimeError("takeover_handshake_already_requested")
        step, time_s = self._event_position(decision_step, sim_time_s)
        self.state = "takeover_pending"
        self.requested_decision_step = step
        self.requested_sim_time_s = time_s

    def arm(self, *, decision_step, sim_time_s):
        if self.state != "takeover_pending":
            raise RuntimeError("takeover_handshake_not_pending")
        step, time_s = self._event_position(decision_step, sim_time_s)
        if (step != self.requested_decision_step
                or time_s != self.requested_sim_time_s):
            raise RuntimeError("takeover_handshake_state_advanced_before_arm")
        self.state = "human_correction_armed"
        self.armed_event = {
            "schema": "excavation_takeover_handshake_event_v1",
            "event": "human_takeover_armed",
            "decision_step": step,
            "sim_time_s": time_s,
            "physics_advanced": False,
            "release_condition": "first_nonzero_human_action",
        }
        return self.armed_event

    def activate(self, *, raw_action, decision_step, sim_time_s):
        if self.state != "human_correction_armed":
            raise RuntimeError("takeover_handshake_not_armed")
        action = self._action(raw_action)
        step, time_s = self._event_position(decision_step, sim_time_s)
        if (step != self.requested_decision_step
                or time_s != self.requested_sim_time_s):
            raise RuntimeError(
                "takeover_handshake_state_advanced_before_first_action")
        if not any(item != 0.0 for item in action):
            return None
        self.state = "human_correction"
        self.pending_first_action = list(action)
        self.activation_event = {
            "schema": "excavation_takeover_handshake_event_v1",
            "event": "first_nonzero_human_action",
            "decision_step": step,
            "sim_time_s": time_s,
            "raw_human_action": action,
            "physics_advanced_before_action": False,
            "record_as_direct_expert_label": True,
        }
        return self.activation_event

    def action_for_step(self, live_action):
        action = self._action(live_action)
        if self.pending_first_action is not None:
            return list(self.pending_first_action), True
        return action, False

    def mark_first_action_executed(self):
        if self.pending_first_action is None:
            raise RuntimeError("takeover_first_action_not_buffered")
        self.pending_first_action = None

    def as_dict(self):
        return {
            "schema": "excavation_takeover_handshake_summary_v1",
            "state": self.state,
            "requested_decision_step": self.requested_decision_step,
            "requested_sim_time_s": self.requested_sim_time_s,
            "armed_event": self.armed_event,
            "activation_event": self.activation_event,
            "first_action_buffered": self.pending_first_action is not None,
            "physics_frozen_until_first_nonzero_human_action": True,
        }
