"""Metadata-only contract for the future contact-only behavior-cloning actor.

This module deliberately imports no model weights and performs no training or
physics.  It separates the complete offline label taxonomy from the three
action heads that are allowed to supervise a contact actor.  A reviewed label
receipt and a separate runtime-admission receipt are both mandatory; legacy
four-head checkpoints cannot satisfy this contract.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import re

from audit_contact_data_contract import TRAINING_EPISODE_ALLOWLIST
from contracts import ACTION_SCHEMA, AXES
from observation_vector_v4 import FEATURE_COUNT, FEATURE_SPEC_SHA256


CONTRACT_VERSION = "contact_actor_checkpoint_contract_v1"
CONTACT_CHECKPOINT_SCHEMA = "excavation_contact_actor_bc_checkpoint_v1"
REVIEWED_SEGMENT_SCHEMA = "excavation_contact_segment_reviewed_v1"
REVIEWED_LABEL_VERSION = "contact_segments_v1_reviewed"
REVIEW_RECEIPT_SCHEMA = "excavation_contact_label_review_receipt_v1"
RUNTIME_RECEIPT_SCHEMA = "excavation_contact_runtime_admission_receipt_v1"
RUNTIME_ROUTE_CONTRACT = "external_contact_handoff_router_v1"

LABEL_PHASES = (
    "approach",
    "penetrate_deepen",
    "curl_load",
    "lift_clear",
    "off_soil_confirm",
    "uncertain/recovery",
)
ACTOR_HEAD_PHASES = ("penetrate_deepen", "curl_load", "lift_clear")
GEOMETRIC_PHASES = ("approach", "off_soil_confirm")
SUPERVISOR_ONLY_PHASES = ("uncertain/recovery",)
SEALED_HOLDOUT_EPISODE_IDS = (
    "v4-demo-attempt-20260918-103349",
    "v4-demo-attempt-20260918-103536",
)
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def contract_spec():
    """Return the canonical JSON-serializable contract."""
    return {
        "version": CONTRACT_VERSION,
        "checkpoint_schema": CONTACT_CHECKPOINT_SCHEMA,
        "checkpoint_role": (
            "actor_training_artifact_requires_separate_runtime_admission"),
        "observation_contract": {
            "feature_count": FEATURE_COUNT,
            "feature_spec_sha256": FEATURE_SPEC_SHA256,
        },
        "action_contract": {
            "schema": ACTION_SCHEMA,
            "axis_order": list(AXES),
            "output": "normalized_velocity",
            "swing_output_frozen_zero": True,
        },
        "label_contract": {
            "schema": REVIEWED_SEGMENT_SCHEMA,
            "version": REVIEWED_LABEL_VERSION,
            "all_phases": list(LABEL_PHASES),
            "actor_head_phases": list(ACTOR_HEAD_PHASES),
            "geometric_phases": list(GEOMETRIC_PHASES),
            "supervisor_only_phases": list(SUPERVISOR_ONLY_PHASES),
            "pending_review_is_training_ineligible": True,
        },
        "data_contract": {
            "allowed_episode_ids": list(TRAINING_EPISODE_ALLOWLIST),
            "sealed_holdout_episode_ids": list(SEALED_HOLDOUT_EPISODE_IDS),
            "normalization_scope": "train_split_only",
        },
        "runtime_contract": {
            "route": RUNTIME_ROUTE_CONTRACT,
            "phase_classifier_role": "diagnostic_only",
            "runtime_admission": "separate_receipt_required",
            "recovery_owner": "safety_supervisor",
        },
    }


def _canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


CONTRACT_SPEC_SHA256 = _canonical_sha256(contract_spec())


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _is_sha256(value):
    return isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None


def validate_reviewed_segment_records(records):
    """Validate a fully resolved label set before any training can use it."""
    records = list(records)
    _require(records, "empty_contact_label_set")
    episodes = set()
    phase_rows = Counter()
    eligible_rows = 0
    for row in records:
        _require(isinstance(row, dict), "invalid_contact_label_record")
        _require(
            row.get("schema") == REVIEWED_SEGMENT_SCHEMA,
            "contact_label_schema_not_reviewed",
        )
        _require(
            row.get("label_version") == REVIEWED_LABEL_VERSION,
            "contact_label_version_not_reviewed",
        )
        phase = row.get("phase")
        _require(phase in LABEL_PHASES, "unknown_contact_label_phase")
        _require(row.get("holdout") is False, "holdout_contact_label_forbidden")
        review_status = row.get("review_status")
        _require(
            review_status in {"approved", "rejected"},
            "contact_label_review_unresolved",
        )
        training_eligible = row.get("training_eligible_now") is True
        axis_mask = row.get("axis_mask_after_review")
        actor_eligible = row.get("actor_action_eligible_after_review") is True
        if phase not in ACTOR_HEAD_PHASES or review_status == "rejected":
            _require(not training_eligible, "excluded_phase_training_enabled")
            _require(not actor_eligible, "excluded_phase_actor_enabled")
            _require(axis_mask == [0.0] * 4, "excluded_phase_axis_mask_nonzero")
        else:
            _require(training_eligible, "approved_actor_phase_training_disabled")
            _require(actor_eligible, "approved_actor_phase_not_actor_eligible")
            _require(axis_mask == [1.0] * 4, "approved_actor_phase_axis_mask_mismatch")
            rows = row.get("row_count")
            _require(type(rows) is int and rows > 0, "invalid_contact_segment_rows")
            phase_rows[phase] += rows
            eligible_rows += rows
        episode_id = row.get("episode_id")
        _require(
            episode_id in TRAINING_EPISODE_ALLOWLIST,
            "unlisted_contact_training_episode",
        )
        episodes.add(episode_id)
    _require(
        episodes == set(TRAINING_EPISODE_ALLOWLIST),
        "reviewed_contact_episode_coverage_mismatch",
    )
    _require(
        set(phase_rows) == set(ACTOR_HEAD_PHASES),
        "reviewed_contact_actor_phase_coverage_mismatch",
    )
    return {
        "reviewed_segments": len(records),
        "eligible_rows": eligible_rows,
        "eligible_rows_by_phase": dict(sorted(phase_rows.items())),
        "episode_ids": list(TRAINING_EPISODE_ALLOWLIST),
    }


def validate_contact_checkpoint_metadata(checkpoint):
    """Reject semantic mismatches before tensors or a device are touched."""
    _require(isinstance(checkpoint, dict), "invalid_contact_checkpoint")
    _require(
        checkpoint.get("schema") == CONTACT_CHECKPOINT_SCHEMA,
        "contact_checkpoint_schema_mismatch",
    )
    _require(
        checkpoint.get("contract_version") == CONTRACT_VERSION
        and checkpoint.get("contract_spec_sha256") == CONTRACT_SPEC_SHA256,
        "contact_checkpoint_contract_mismatch",
    )
    _require(
        checkpoint.get("feature_count") == FEATURE_COUNT
        and checkpoint.get("feature_spec_sha256") == FEATURE_SPEC_SHA256,
        "contact_checkpoint_feature_contract_mismatch",
    )
    _require(
        checkpoint.get("action_schema") == ACTION_SCHEMA
        and checkpoint.get("axis_order") == list(AXES)
        and checkpoint.get("action_output") == "normalized_velocity",
        "contact_checkpoint_action_contract_mismatch",
    )
    _require(
        checkpoint.get("actor_head_phases") == list(ACTOR_HEAD_PHASES),
        "contact_checkpoint_actor_head_mismatch",
    )
    _require(
        checkpoint.get("training_label_schema") == REVIEWED_SEGMENT_SCHEMA
        and checkpoint.get("training_label_version") == REVIEWED_LABEL_VERSION,
        "contact_checkpoint_label_contract_mismatch",
    )
    label_hash = checkpoint.get("training_label_manifest_sha256")
    _require(_is_sha256(label_hash), "invalid_contact_label_manifest_sha256")

    review = checkpoint.get("manual_review_receipt")
    _require(isinstance(review, dict), "missing_contact_manual_review_receipt")
    _require(
        review.get("schema") == REVIEW_RECEIPT_SCHEMA
        and review.get("status") == "approved"
        and review.get("label_manifest_sha256") == label_hash,
        "contact_manual_review_receipt_mismatch",
    )
    _require(
        review.get("reviewed_episode_ids") == list(TRAINING_EPISODE_ALLOWLIST)
        and review.get("continuous_visual_review") is True
        and review.get("takeover_boundary_visual_review") is True,
        "contact_manual_visual_review_incomplete",
    )

    train = checkpoint.get("train_episodes")
    validation = checkpoint.get("validation_episodes")
    _require(
        isinstance(train, list) and train
        and isinstance(validation, list) and validation,
        "contact_checkpoint_split_missing",
    )
    _require(
        len(train) == len(set(train)) and len(validation) == len(set(validation))
        and set(train).isdisjoint(validation)
        and set(train) | set(validation) == set(TRAINING_EPISODE_ALLOWLIST),
        "contact_checkpoint_split_mismatch",
    )
    _require(
        checkpoint.get("sealed_holdout_episode_ids")
        == list(SEALED_HOLDOUT_EPISODE_IDS)
        and checkpoint.get("sealed_holdout_files_opened") == 0,
        "contact_checkpoint_holdout_contract_mismatch",
    )
    normalization = checkpoint.get("normalization")
    _require(
        isinstance(normalization, dict)
        and normalization.get("scope") == "train_split_only"
        and normalization.get("source_episode_ids") == train
        and normalization.get("feature_spec_sha256") == FEATURE_SPEC_SHA256,
        "contact_checkpoint_normalization_scope_mismatch",
    )
    phase_rows = checkpoint.get("training_rows_by_actor_phase")
    _require(
        isinstance(phase_rows, dict)
        and set(phase_rows) == set(ACTOR_HEAD_PHASES)
        and all(type(value) is int and value > 0 for value in phase_rows.values()),
        "contact_checkpoint_phase_rows_mismatch",
    )
    _require(
        checkpoint.get("action_frozen_mask") == [True, False, False, False]
        and checkpoint.get("action_frozen_values") == [0.0] * 4,
        "contact_checkpoint_frozen_action_mismatch",
    )
    counters = checkpoint.get("counters")
    _require(
        isinstance(counters, dict)
        and type(counters.get("actor_behavior_cloning_updates")) is int
        and counters["actor_behavior_cloning_updates"] > 0
        and counters.get("critic_updates") == 0
        and counters.get("sac_gradient_steps") == 0
        and counters.get("environment_steps") == 0,
        "contact_checkpoint_training_counter_mismatch",
    )
    _require(
        checkpoint.get("checkpoint_role")
        == "actor_training_artifact_requires_separate_runtime_admission"
        and checkpoint.get("runtime_route_contract") == RUNTIME_ROUTE_CONTRACT
        and checkpoint.get("runtime_admission_embedded") is False,
        "contact_checkpoint_runtime_boundary_mismatch",
    )
    for forbidden in (
            "critic_state_dict", "optimizer_state_dict", "replay_buffer",
            "runtime_admission_receipt"):
        _require(forbidden not in checkpoint, "forbidden_contact_checkpoint_field")
    return checkpoint


def validate_runtime_admission_receipt(
        checkpoint, receipt, *, checkpoint_sha256):
    """Require a separate qualified handoff/router receipt for deployment."""
    validate_contact_checkpoint_metadata(checkpoint)
    _require(_is_sha256(checkpoint_sha256), "invalid_contact_checkpoint_sha256")
    _require(isinstance(receipt, dict), "missing_contact_runtime_receipt")
    _require(
        receipt.get("schema") == RUNTIME_RECEIPT_SCHEMA
        and receipt.get("status") == "passed"
        and receipt.get("checkpoint_sha256") == checkpoint_sha256
        and receipt.get("runtime_route_contract") == RUNTIME_ROUTE_CONTRACT,
        "contact_runtime_receipt_mismatch",
    )
    audit_hash = receipt.get("runtime_audit_sha256")
    _require(_is_sha256(audit_hash), "invalid_contact_runtime_audit_sha256")
    return receipt
