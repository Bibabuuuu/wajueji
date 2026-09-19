"""D-067 causal entry phases; observation-only, no physics or model imports.

Qualified scope: the existing fixed digging plane and bucket geometry. The
target-relative world x/z corridor is not a general rotated-workface planner.
Interaction is diagnostic, never an actuator or phase-advance gate.
"""

import math


CONTRACT = "target_relative_entry_two_tick_v1"
PHASES = ("approach", "penetrate_deepen", "curl_load", "lift_carry")


def contract_spec():
    return {
        "version": CONTRACT, "confirmation_ticks": 2,
        "entry_relative_x_min_m": -0.30,
        "entry_relative_z_range_m": [0.85, 1.05],
        "effective_force_min_n": 1.0,
        "curl_action_min": 0.5, "lift_boom_action_min": 0.5,
        "lift_bucket_position_min": 0.90,
        "label_boundary": "confirmation_tick_no_backdating",
        "interaction_role": "audit_only_required_strictly_before_lift",
        "scope": "fixed_world_x_digging_plane_existing_bucket_geometry",
    }


def finite_vector(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float))
           or not math.isfinite(item) for item in value):
        return None
    return tuple(float(item) for item in value)


def valid_pose(observation):
    bucket = observation.get("bucket", {})
    if not isinstance(bucket, dict):
        return False
    return (observation.get("target_valid") is True
            and bucket.get("valid") is True
            and finite_vector(observation.get("target_world_m"), 3) is not None
            and finite_vector(bucket.get("relative_to_target_m"), 3) is not None)


def entry_pose_ready(observation):
    if not valid_pose(observation):
        return False
    x, _, z = observation["bucket"]["relative_to_target_m"]
    return x >= -0.30 and 0.85 <= z <= 1.05


def current_force_norm(observation):
    """Never fall back to an older valid history slot or nonzero-age sample."""
    history = observation.get("wrench_history", {})
    if not isinstance(history, dict):
        return None
    try:
        if history["valid_mask"][0] is not True:
            return None
        age = history["age_s"][0]
        if isinstance(age, bool) or not isinstance(age, (int, float)) or age != 0:
            return None
        sample = finite_vector(history["values_world_si"][0], 6)
    except (KeyError, IndexError, TypeError):
        return None
    return None if sample is None else math.hypot(*sample[:3])


class EntryMilestonePhaseRouter:
    """One forward phase at most per tick; explicit episode reset required."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.phase_id = 0
        self.advance_confirmation_ticks = 0
        self.last_step = None
        self.target = None
        self.events = {}
        self.force_ticks = 0
        self.effective_interaction_confirmed_step = None
        self.effective_interaction_before_lift = None

    def update(self, observation, routed_action):
        if not isinstance(observation, dict):
            raise ValueError("invalid_entry_observation")
        step = observation.get("decision_step")
        if type(step) is not int or step < 0:
            raise ValueError("invalid_entry_decision_step")
        if self.last_step is not None and step != self.last_step + 1:
            raise ValueError("nonsequential_entry_observation_reset_required")
        action = finite_vector(routed_action, 4)
        if action is None or any(abs(value) > 1 for value in action):
            raise ValueError("invalid_entry_action")
        pose_valid = valid_pose(observation)
        target = finite_vector(observation.get("target_world_m"), 3)
        if pose_valid and self.target is not None and target != self.target:
            raise ValueError("entry_target_changed_reset_required")
        if pose_valid:
            self.target = target
        self.last_step = step

        # Force confirmation starts strictly after entry confirmation.
        if self.phase_id >= 1 and self.effective_interaction_confirmed_step is None:
            force = current_force_norm(observation)
            self.force_ticks = (self.force_ticks + 1
                                if pose_valid and force is not None and force >= 1.0
                                else 0)
            if self.force_ticks >= 2:
                self.effective_interaction_confirmed_step = step

        if self.phase_id == 0:
            condition = entry_pose_ready(observation)
        elif self.phase_id == 1:
            condition = pose_valid and action[3] >= 0.5
        elif self.phase_id == 2:
            proprio = observation.get("proprioception", {})
            positions = (finite_vector(proprio.get("cylinder_position_normalized"), 3)
                         if isinstance(proprio, dict) else None)
            condition = (pose_valid and positions is not None
                         and all(abs(value) <= 1 for value in positions)
                         and positions[2] >= 0.90 and action[1] >= 0.5)
        else:
            self.advance_confirmation_ticks = 0
            return self.phase_id, False
        self.advance_confirmation_ticks = (self.advance_confirmation_ticks + 1
                                           if condition else 0)
        advanced = self.advance_confirmation_ticks >= 2
        if advanced:
            self.phase_id += 1
            self.advance_confirmation_ticks = 0
            self.events[PHASES[self.phase_id]] = {
                "onset_decision_step": step - 1,
                "confirmed_at_decision_step": step,
                "decision_step": step,
            }
            if self.phase_id == 3:
                effective = self.effective_interaction_confirmed_step
                self.effective_interaction_before_lift = (
                    effective is not None and effective < step)
        return self.phase_id, advanced
