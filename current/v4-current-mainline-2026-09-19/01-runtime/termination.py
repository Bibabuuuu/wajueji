"""Single-scoop termination shared by automatic and manual acquisition."""
from dataclasses import dataclass
import math

SCHEMA = "excavation_single_scoop_termination_v4"
OFF_SOIL_CONFIRM_S = 0.5
LOAD_CORRECTION_S = 2.0
LOAD_CORRECTION_REENTRY_CONFIRM_TICKS = 3
CARRY_WINDOW_S = 3.0
FILL_SAMPLE_PERIOD_S = 0.25
FILL_SAMPLE_COUNT = 13
MIN_FILL_FRACTION = 0.95
MIN_BUCKET_CLOSURE_NORMALIZED = 0.95
FAST_TIER_MAX_ABS_ERROR_FRACTION = 0.02164705882352913
EPISODE_LIMIT_S = 60.0

HARD_FAILURE_PRIORITY = (
    "nonfinite_state", "solver_failure", "collision", "physical_joint_limit",
    "closed_chain_failure", "unexplained_ledger_residual",
    "confirmed_physics_domain_failure", "invalid_fill_measurement",
)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("not_number:" + name)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite:" + name)
    return value


def classify_fast_tier_window(fill_fractions):
    """Classify a fast-tier window without pretending it is reference truth."""
    values = tuple(_finite(value, "fill_fraction") for value in fill_fractions)
    if len(values) != FILL_SAMPLE_COUNT or any(value < 0.0 for value in values):
        raise ValueError("invalid_fast_tier_window")
    minimum = min(values)
    if minimum >= MIN_FILL_FRACTION + FAST_TIER_MAX_ABS_ERROR_FRACTION:
        return "provisional_pass_high_margin"
    if minimum < MIN_FILL_FRACTION - FAST_TIER_MAX_ABS_ERROR_FRACTION:
        return "provisional_fail_high_margin"
    return "ambiguous_requires_balanced_or_reference_upgrade"


@dataclass(frozen=True)
class TerminationResult:
    state: str
    terminated: bool
    truncated: bool
    success: bool
    reason: str | None
    sim_time_s: float
    off_soil_elapsed_s: float
    evaluation_start_s: float | None
    evaluation_samples: tuple
    load_correction_start_s: float | None
    load_correction_elapsed_s: float
    load_correction_samples: tuple
    load_correction_used: bool
    load_correction_reentry_ticks: int
    digging_interaction_observed: bool
    trigger_bucket_closure_normalized: float | None

    def as_dict(self):
        return {
            "schema": SCHEMA, "state": self.state,
            "terminated": self.terminated, "truncated": self.truncated,
            "success": self.success, "reason": self.reason,
            "sim_time_s": self.sim_time_s,
            "off_soil_elapsed_s": self.off_soil_elapsed_s,
            "evaluation_start_s": self.evaluation_start_s,
            "evaluation_samples": [
                {"index": index, "offset_s": offset, "fill_fraction": fill}
                for index, offset, fill in self.evaluation_samples],
            "load_correction_start_s": self.load_correction_start_s,
            "load_correction_elapsed_s": self.load_correction_elapsed_s,
            "load_correction_samples": [
                {"index": index, "offset_s": offset, "fill_fraction": fill}
                for index, offset, fill in self.load_correction_samples],
            "load_correction_used": self.load_correction_used,
            "load_correction_reentry_ticks": self.load_correction_reentry_ticks,
            "load_correction_reentry_confirm_ticks": (
                LOAD_CORRECTION_REENTRY_CONFIRM_TICKS),
            "digging_interaction_observed": self.digging_interaction_observed,
            "trigger_bucket_closure_normalized": self.trigger_bucket_closure_normalized,
            "result_sealed": self.terminated or self.truncated,
        }


class SingleScoopTermination:
    """20 Hz evaluator; mouth angle is diagnostic, never a hard gate."""

    def __init__(self):
        self.state = "DIGGING"
        self.last_time_s = None
        self.off_soil_start_s = None
        self.evaluation_start_s = None
        self.samples = []
        self.next_sample_index = 0
        self.load_correction_start_s = None
        self.load_correction_samples = []
        self.next_load_correction_sample_index = 0
        self.load_correction_used = False
        self.load_correction_reentry_ticks = 0
        self.formal_evaluation_started = False
        self.digging_interaction_observed = False
        self.trigger_bucket_closure_normalized = None
        self.final = None

    @staticmethod
    def _closure_ok(value):
        return value >= MIN_BUCKET_CLOSURE_NORMALIZED

    def _result(self, now_s, *, terminated=False, truncated=False,
                success=False, reason=None):
        off_elapsed = (0.0 if self.off_soil_start_s is None else
                       max(0.0, now_s - self.off_soil_start_s))
        correction_elapsed = (0.0 if self.load_correction_start_s is None else
                              max(0.0, now_s - self.load_correction_start_s))
        result = TerminationResult(
            self.state, terminated, truncated, success, reason, now_s,
            off_elapsed, self.evaluation_start_s, tuple(self.samples),
            self.load_correction_start_s, correction_elapsed,
            tuple(self.load_correction_samples), self.load_correction_used,
            self.load_correction_reentry_ticks,
            self.digging_interaction_observed,
            self.trigger_bucket_closure_normalized)
        if terminated or truncated:
            self.final = result
        return result

    def _finish(self, now_s, state, reason, *, success=False, truncated=False):
        self.state = state
        return self._result(now_s, terminated=not truncated, truncated=truncated,
                            success=success, reason=reason)

    def _start_evaluation(self, now_s):
        self.state = "EVALUATING"
        self.formal_evaluation_started = True
        self.evaluation_start_s = now_s
        self.samples = []
        self.next_sample_index = 0

    def _start_load_correction(self, now_s):
        self.state = "LOAD_CORRECTION"
        self.load_correction_start_s = now_s
        self.load_correction_samples = []
        self.next_load_correction_sample_index = 0
        self.load_correction_used = True
        self.load_correction_reentry_ticks = 0

    def _take_load_correction_sample(self, now_s, fill_fraction):
        elapsed = now_s - self.load_correction_start_s
        maximum = int(round(LOAD_CORRECTION_S / FILL_SAMPLE_PERIOD_S))
        due = []
        while (self.next_load_correction_sample_index <= maximum
               and elapsed >= self.next_load_correction_sample_index
               * FILL_SAMPLE_PERIOD_S - 1e-9):
            due.append(self.next_load_correction_sample_index)
            self.next_load_correction_sample_index += 1
        if not due:
            if fill_fraction is not None:
                _finite(fill_fraction, "fill_fraction")
            return None
        if len(due) != 1:
            return self._finish(now_s, "HARD_FAILURE", "missed_load_correction_samples")
        if fill_fraction is None:
            return self._finish(now_s, "HARD_FAILURE", "invalid_fill_measurement")
        fill = _finite(fill_fraction, "fill_fraction")
        if fill < 0.0:
            return self._finish(now_s, "HARD_FAILURE", "invalid_fill_measurement")
        index = due[0]
        self.load_correction_samples.append((index, index * FILL_SAMPLE_PERIOD_S, fill))
        if fill >= MIN_FILL_FRACTION:
            self._start_evaluation(now_s)
            return self._take_due_sample(now_s, fill)
        if elapsed >= LOAD_CORRECTION_S - 1e-9:
            return self._finish(now_s, "LOAD_FAILURE",
                                "load_correction_timeout_below_95_percent")
        return None

    def _take_due_sample(self, now_s, fill_fraction):
        elapsed = now_s - self.evaluation_start_s
        due = []
        while (self.next_sample_index < FILL_SAMPLE_COUNT
               and elapsed >= self.next_sample_index * FILL_SAMPLE_PERIOD_S - 1e-9):
            due.append(self.next_sample_index)
            self.next_sample_index += 1
        if not due:
            if fill_fraction is not None:
                _finite(fill_fraction, "fill_fraction")
            return None
        if len(due) != 1:
            return self._finish(now_s, "HARD_FAILURE", "missed_fill_samples")
        if fill_fraction is None:
            return self._finish(now_s, "HARD_FAILURE", "invalid_fill_measurement")
        fill = _finite(fill_fraction, "fill_fraction")
        if fill < 0.0:
            return self._finish(now_s, "HARD_FAILURE", "invalid_fill_measurement")
        index = due[0]
        self.samples.append((index, index * FILL_SAMPLE_PERIOD_S, fill))
        if self.next_sample_index == FILL_SAMPLE_COUNT:
            if min(item[2] for item in self.samples) >= MIN_FILL_FRACTION:
                return self._finish(
                    now_s, "SUCCESS", "carry_window_passed", success=True)
            return self._finish(
                now_s, "LOAD_FAILURE", "carry_window_minimum_below_95_percent")
        return None

    def fill_sample_due(self, sim_time_s):
        now_s = _finite(sim_time_s, "sim_time_s")
        if self.final is not None:
            return False
        if self.state == "EVALUATING":
            elapsed = now_s - self.evaluation_start_s
            return (self.next_sample_index < FILL_SAMPLE_COUNT
                    and elapsed >= self.next_sample_index * FILL_SAMPLE_PERIOD_S - 1e-9)
        if self.state == "LOAD_CORRECTION":
            elapsed = now_s - self.load_correction_start_s
            maximum = int(round(LOAD_CORRECTION_S / FILL_SAMPLE_PERIOD_S))
            return (self.next_load_correction_sample_index <= maximum
                    and elapsed >= self.next_load_correction_sample_index
                    * FILL_SAMPLE_PERIOD_S - 1e-9)
        return False

    def sample_required(self, sim_time_s, *, digging_contact, terrain_contact,
                        bucket_closure_normalized):
        now_s = _finite(sim_time_s, "sim_time_s")
        closure = _finite(bucket_closure_normalized, "bucket_closure_normalized")
        if self.final is not None or terrain_contact:
            return False
        if self.state in ("EVALUATING", "LOAD_CORRECTION"):
            return self.fill_sample_due(now_s)
        interaction = self.digging_interaction_observed or digging_contact
        if not interaction:
            return False
        start = self.off_soil_start_s if self.off_soil_start_s is not None else now_s
        return (now_s - start >= OFF_SOIL_CONFIRM_S - 1e-9
                and self._closure_ok(closure))

    def update(self, sim_time_s, *, digging_contact, terrain_contact,
               bucket_closure_normalized, fill_fraction=None, hard_failures=()):
        if self.final is not None:
            return self.final
        now_s = _finite(sim_time_s, "sim_time_s")
        closure = _finite(bucket_closure_normalized, "bucket_closure_normalized")
        if now_s < 0.0 or (self.last_time_s is not None
                           and now_s < self.last_time_s - 1e-12):
            raise ValueError("invalid_or_reversed_sim_time")
        if not isinstance(digging_contact, bool) or not isinstance(terrain_contact, bool):
            raise ValueError("contact_flags_must_be_boolean")
        if digging_contact and not terrain_contact:
            raise ValueError("digging_contact_requires_terrain_contact")
        if not isinstance(hard_failures, (tuple, list, set, frozenset)):
            raise ValueError("hard_failures_must_be_sequence")
        unknown = set(hard_failures) - set(HARD_FAILURE_PRIORITY)
        if unknown:
            raise ValueError("unknown_hard_failure:" + ",".join(sorted(unknown)))
        self.last_time_s = now_s
        for failure in HARD_FAILURE_PRIORITY:
            if failure in hard_failures:
                return self._finish(now_s, "HARD_FAILURE", failure)
        if digging_contact:
            self.digging_interaction_observed = True
        if self.formal_evaluation_started and terrain_contact:
            return self._finish(now_s, "CARRY_FAILURE",
                                "reentered_soil_after_evaluation_started")
        if self.state == "LOAD_CORRECTION":
            if terrain_contact:
                self.load_correction_reentry_ticks += 1
                if (self.load_correction_reentry_ticks
                        >= LOAD_CORRECTION_REENTRY_CONFIRM_TICKS):
                    return self._finish(
                        now_s, "CARRY_FAILURE",
                        "confirmed_reentry_during_load_correction")
            else:
                self.load_correction_reentry_ticks = 0
                outcome = self._take_load_correction_sample(now_s, fill_fraction)
                if outcome is not None:
                    return outcome
        elif self.state == "EVALUATING":
            outcome = self._take_due_sample(now_s, fill_fraction)
            if outcome is not None:
                return outcome
        elif not self.formal_evaluation_started:
            if not self.digging_interaction_observed or terrain_contact:
                self.state = "DIGGING"
                self.off_soil_start_s = None
            else:
                if self.off_soil_start_s is None:
                    self.off_soil_start_s = now_s
                elapsed = now_s - self.off_soil_start_s
                self.state = ("OFF_SOIL_CONFIRMING" if elapsed < OFF_SOIL_CONFIRM_S - 1e-9
                              else "WAITING_FOR_BUCKET_CLOSURE")
                if (elapsed >= OFF_SOIL_CONFIRM_S - 1e-9
                        and self._closure_ok(closure)):
                    self.trigger_bucket_closure_normalized = closure
                    if fill_fraction is None:
                        return self._finish(now_s, "HARD_FAILURE",
                                            "invalid_fill_measurement")
                    fill = _finite(fill_fraction, "fill_fraction")
                    if fill < 0.0:
                        return self._finish(now_s, "HARD_FAILURE",
                                            "invalid_fill_measurement")
                    if fill >= MIN_FILL_FRACTION:
                        self._start_evaluation(now_s)
                        outcome = self._take_due_sample(now_s, fill)
                    else:
                        self._start_load_correction(now_s)
                        outcome = self._take_load_correction_sample(now_s, fill)
                    if outcome is not None:
                        return outcome
        if now_s >= EPISODE_LIMIT_S - 1e-9:
            return self._finish(now_s, "TIME_LIMIT", "single_scoop_60_second_limit",
                                truncated=True)
        return self._result(now_s)
