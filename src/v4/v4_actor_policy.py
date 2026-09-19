"""CUDA inference loader for the D-058 compact-history v4 actor."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np

# Explicit D-067 alternative for zero-physics validation. Legacy checkpoint
# loading below keeps its original router until a new checkpoint is qualified.
from entry_phase_contract import EntryMilestonePhaseRouter, CONTRACT, contract_spec

from observation_vector_v4 import (
    FEATURE_COUNT, FEATURE_SPEC_SHA256, HISTORY_TICKS, encode_observation,
)


CHECKPOINT_SCHEMA = "excavation_phase_head_actor_bc_checkpoint_v4"
ENTRY_CHECKPOINT_SCHEMA = "excavation_phase_head_actor_entry_bc_checkpoint_v1"
PHASES = ("approach", "penetrate_deepen", "curl_load", "lift_carry")


def checkpoint_runtime_route(checkpoint):
    """Bind new semantics explicitly; old weights keep the legacy route."""
    if checkpoint.get("schema") == ENTRY_CHECKPOINT_SCHEMA:
        if (checkpoint.get("phase_route") != CONTRACT
                or checkpoint.get("entry_contract") != contract_spec()):
            raise ValueError("entry_checkpoint_route_contract_mismatch")
        return CONTRACT
    if (checkpoint.get("schema") == CHECKPOINT_SCHEMA
            and checkpoint.get("phase_route") == "monotonic_two_tick_episode_reset"
            and "entry_contract" not in checkpoint):
        return "causal_milestone_two_tick_episode_reset"
    raise ValueError("v4_actor_checkpoint_contract_mismatch")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MonotonicPhaseLatch:
    """Episode-local phase latch with two consecutive advance signals."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.phase_id = 0
        self.advance_confirmation_ticks = 0

    def update(self, raw_phase_id):
        raw_phase_id = int(raw_phase_id)
        if not 0 <= raw_phase_id < len(PHASES):
            raise ValueError("invalid_v4_raw_phase_id")
        if raw_phase_id > self.phase_id and self.phase_id < len(PHASES) - 1:
            self.advance_confirmation_ticks += 1
            if self.advance_confirmation_ticks >= 2:
                self.phase_id += 1
                self.advance_confirmation_ticks = 0
        else:
            self.advance_confirmation_ticks = 0
        return self.phase_id


class MilestonePhaseRouter:
    """Route phase heads only after causal physical/action milestones.

    The classifier remains diagnostic.  Runtime routing is owned by observable
    interaction, current routed action intent, and bucket position, matching
    the causal portions of the D-055 label contract without future truth.
    """

    def __init__(self, *, confirmation_ticks=2, curl_action_threshold=0.5,
                 lift_boom_threshold=0.5, lift_bucket_position_min=0.90):
        self.confirmation_ticks = int(confirmation_ticks)
        self.curl_action_threshold = float(curl_action_threshold)
        self.lift_boom_threshold = float(lift_boom_threshold)
        self.lift_bucket_position_min = float(lift_bucket_position_min)
        if self.confirmation_ticks < 2:
            raise ValueError("milestone_router_requires_two_ticks")
        self.reset()

    def reset(self):
        self.phase_id = 0
        self.advance_confirmation_ticks = 0

    def update(self, observation, routed_action):
        if not isinstance(observation, dict):
            raise ValueError("invalid_milestone_observation")
        if not isinstance(routed_action, (list, tuple)) or len(routed_action) != 4:
            raise ValueError("invalid_milestone_action")
        action = [float(value) for value in routed_action]
        if self.phase_id == 0:
            condition = observation.get("digging_interaction_observed") is True
        elif self.phase_id == 1:
            condition = (
                observation.get("digging_interaction_observed") is True
                and action[3] >= self.curl_action_threshold)
        elif self.phase_id == 2:
            try:
                bucket_position = float(observation["proprioception"]
                                        ["cylinder_position_normalized"][2])
            except (KeyError, TypeError, IndexError) as exc:
                raise ValueError("missing_milestone_bucket_position") from exc
            condition = (
                bucket_position >= self.lift_bucket_position_min
                and action[1] >= self.lift_boom_threshold)
        else:
            self.advance_confirmation_ticks = 0
            return self.phase_id, False
        self.advance_confirmation_ticks = (
            self.advance_confirmation_ticks + 1 if condition else 0)
        advanced = self.advance_confirmation_ticks >= self.confirmation_ticks
        if advanced:
            self.phase_id += 1
            self.advance_confirmation_ticks = 0
        return self.phase_id, advanced


class EpisodeHistory:
    """Strictly causal compact-history source that cannot cross an episode."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._observations = []
        self.last_decision_step = None

    def priors(self, current_observation):
        current_step = int(current_observation["decision_step"])
        if (self.last_decision_step is not None
                and current_step != self.last_decision_step + 1):
            raise ValueError("nonsequential_v4_runtime_observation")
        newest_first = list(reversed(self._observations))
        return newest_first + [None] * (HISTORY_TICKS - len(newest_first))

    def commit(self, current_observation):
        self.last_decision_step = int(current_observation["decision_step"])
        self._observations.append(copy.deepcopy(current_observation))
        self._observations = self._observations[-HISTORY_TICKS:]

    @property
    def valid_count(self):
        return len(self._observations)


def build_actor(torch):
    nn = torch.nn

    class PhaseActor(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(FEATURE_COUNT, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU())
            self.phase_head = nn.Linear(256, len(PHASES))
            self.action_heads = nn.ModuleList(
                nn.Linear(256, 4) for _ in PHASES)

        def deterministic(self, value):
            latent = self.trunk(value)
            logits = self.phase_head(latent)
            actions = torch.stack([
                torch.tanh(head(latent)) for head in self.action_heads], dim=1)
            actions = torch.cat(
                (torch.zeros_like(actions[:, :, :1]), actions[:, :, 1:]), dim=2)
            return logits, actions

    return PhaseActor()


class LoadedV4ActorPolicy:
    def __init__(self, *, checkpoint_path, device="cuda"):
        import torch

        self.path = Path(checkpoint_path).resolve()
        self.checkpoint_sha256 = sha256(self.path)
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("v4_actor_policy_requires_cuda")
        checkpoint = torch.load(
            self.path, map_location=self.device, weights_only=False)
        runtime_route = checkpoint_runtime_route(checkpoint)
        if (checkpoint.get("feature_count") != FEATURE_COUNT
                or checkpoint.get("feature_spec_sha256") != FEATURE_SPEC_SHA256
                or checkpoint.get("phase_names") != list(PHASES)):
            raise ValueError("v4_actor_checkpoint_contract_mismatch")
        self.actor = build_actor(torch).to(self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        self.actor.eval()
        self.mean = checkpoint["normalization_mean"].to(
            device=self.device, dtype=torch.float32)
        self.scale = checkpoint["normalization_scale"].to(
            device=self.device, dtype=torch.float32)
        self.frozen_mask = checkpoint["normalization_frozen_mask"].to(
            device=self.device, dtype=torch.bool)
        self.action_frozen_mask = checkpoint["action_frozen_mask"].to(
            device=self.device, dtype=torch.bool)
        self.action_frozen_values = checkpoint["action_frozen_values"].to(
            device=self.device, dtype=torch.float32)
        if (self.mean.shape != (FEATURE_COUNT,)
                or self.scale.shape != (FEATURE_COUNT,)
                or self.frozen_mask.shape != (FEATURE_COUNT,)
                or self.action_frozen_mask.shape != (4,)
                or self.action_frozen_values.shape != (4,)
                or not torch.isfinite(self.mean).all()
                or not torch.isfinite(self.scale).all()
                or not torch.all(self.scale > 0.0)
                or self.action_frozen_mask.cpu().tolist()
                != [True, False, False, False]
                or self.action_frozen_values.cpu().tolist() != [0.0] * 4):
            raise ValueError("invalid_v4_actor_checkpoint_tensors")
        counters = checkpoint.get("counters", {})
        self.metadata = {
            "schema": checkpoint["schema"],
            "checkpoint_sha256": self.checkpoint_sha256,
            "policy_stage": "v4_compact_history_phase_actor_bc",
            "feature_count": FEATURE_COUNT,
            "feature_spec_sha256": FEATURE_SPEC_SHA256,
            "history_ticks": HISTORY_TICKS,
            "phase_names": list(PHASES),
            "phase_route": checkpoint["phase_route"],
            "runtime_phase_route": runtime_route,
            "runtime_phase_milestones": {
                "approach_to_penetrate": "interaction_observed_two_ticks",
                "penetrate_to_curl": (
                    "interaction_and_bucket_action_ge_0p5_two_ticks"),
                "curl_to_lift": (
                    "bucket_position_ge_0p90_and_boom_action_ge_0p5_two_ticks"),
            },
            "train_episodes": list(checkpoint["train_episodes"]),
            "validation_episodes": list(checkpoint["validation_episodes"]),
            "sealed_holdout_episode_ids": list(
                checkpoint["sealed_holdout_episode_ids"]),
            "actor_behavior_cloning_updates": int(
                counters.get("actor_behavior_cloning_updates", 0)),
            "critic_updates": int(counters.get("critic_updates", 0)),
            "sac_gradient_steps": int(counters.get("sac_gradient_steps", 0)),
            "environment_steps_at_training": int(
                counters.get("environment_steps", 0)),
            "action_frozen_mask": self.action_frozen_mask.cpu().tolist(),
            "action_frozen_values": self.action_frozen_values.cpu().tolist(),
        }
        if runtime_route == CONTRACT:
            self.metadata["entry_contract"] = contract_spec()
            self.metadata["runtime_phase_milestones"] = {
                "approach_to_penetrate": "valid_target_relative_pose_two_ticks",
                "penetrate_to_curl": "post_entry_bucket_action_ge_0p5_two_ticks",
                "curl_to_lift": "bucket_position_ge_0p90_and_boom_action_ge_0p5_two_ticks",
                "effective_interaction": "post_entry_current_force_ge_1N_two_ticks_audit_only",
            }
        self.phase_router = (EntryMilestonePhaseRouter() if runtime_route == CONTRACT
                             else MilestonePhaseRouter())
        self.history = EpisodeHistory()
        self.last_inference = None

    def reset_episode_state(self):
        self.phase_router.reset()
        self.history.reset()
        self.last_inference = None

    def action(self, observation):
        import torch

        priors = self.history.priors(observation)
        history_valid_before = self.history.valid_count
        vector = np.asarray(
            encode_observation(observation, priors), dtype=np.float32)
        value = torch.from_numpy(vector).to(self.device)
        normalized = (value - self.mean) / self.scale
        normalized.masked_fill_(self.frozen_mask, 0.0)
        with torch.no_grad():
            logits, all_actions = self.actor.deterministic(
                normalized.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)[0]
        raw_phase_id = int(torch.argmax(logits[0]).item())
        routed_phase_before = self.phase_router.phase_id
        provisional = all_actions[0, routed_phase_before].cpu().numpy()
        routed_phase_id, milestone_advanced = self.phase_router.update(
            observation, provisional.tolist())
        output = all_actions[0, routed_phase_id].cpu().numpy()
        mask = self.action_frozen_mask.cpu().numpy()
        output[mask] = self.action_frozen_values.cpu().numpy()[mask]
        output = np.clip(output, -1.0, 1.0)
        if not np.isfinite(output).all():
            raise RuntimeError("nonfinite_v4_actor_action")
        self.last_inference = {
            "raw_phase_id": raw_phase_id,
            "raw_phase": PHASES[raw_phase_id],
            "latched_phase_id": routed_phase_id,
            "latched_phase": PHASES[routed_phase_id],
            "routed_phase_id": routed_phase_id,
            "routed_phase": PHASES[routed_phase_id],
            "routed_phase_before_id": routed_phase_before,
            "routed_phase_before": PHASES[routed_phase_before],
            "milestone_advanced": milestone_advanced,
            "advance_confirmation_ticks": (
                self.phase_router.advance_confirmation_ticks),
            "runtime_phase_route": self.metadata["runtime_phase_route"],
            "curl_started": routed_phase_id >= 2,
            "phase_probabilities": probabilities.cpu().tolist(),
            "history_valid_count": history_valid_before,
            "decision_step": int(observation["decision_step"]),
        }
        if self.metadata["runtime_phase_route"] == CONTRACT:
            self.last_inference["effective_interaction_confirmed_step"] = (
                self.phase_router.effective_interaction_confirmed_step)
            self.last_inference["effective_interaction_before_lift"] = (
                self.phase_router.effective_interaction_before_lift)
        self.history.commit(observation)
        return output.astype(float).tolist()
