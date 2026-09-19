import copy
import unittest

from contact_checkpoint_contract import (
    ACTOR_HEAD_PHASES,
    CONTACT_CHECKPOINT_SCHEMA,
    CONTRACT_SPEC_SHA256,
    CONTRACT_VERSION,
    LABEL_PHASES,
    REVIEWED_LABEL_VERSION,
    REVIEWED_SEGMENT_SCHEMA,
    REVIEW_RECEIPT_SCHEMA,
    RUNTIME_RECEIPT_SCHEMA,
    RUNTIME_ROUTE_CONTRACT,
    SEALED_HOLDOUT_EPISODE_IDS,
    validate_contact_checkpoint_metadata,
    validate_reviewed_segment_records,
    validate_runtime_admission_receipt,
)
from audit_contact_data_contract import TRAINING_EPISODE_ALLOWLIST
from contracts import ACTION_SCHEMA, AXES
from observation_vector_v4 import FEATURE_COUNT, FEATURE_SPEC_SHA256
from v4_actor_policy import checkpoint_runtime_route


def valid_checkpoint():
    label_hash = "1" * 64
    train = list(TRAINING_EPISODE_ALLOWLIST[:-1])
    validation = [TRAINING_EPISODE_ALLOWLIST[-1]]
    return {
        "schema": CONTACT_CHECKPOINT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "contract_spec_sha256": CONTRACT_SPEC_SHA256,
        "feature_count": FEATURE_COUNT,
        "feature_spec_sha256": FEATURE_SPEC_SHA256,
        "action_schema": ACTION_SCHEMA,
        "axis_order": list(AXES),
        "action_output": "normalized_velocity",
        "actor_head_phases": list(ACTOR_HEAD_PHASES),
        "training_label_schema": REVIEWED_SEGMENT_SCHEMA,
        "training_label_version": REVIEWED_LABEL_VERSION,
        "training_label_manifest_sha256": label_hash,
        "manual_review_receipt": {
            "schema": REVIEW_RECEIPT_SCHEMA,
            "status": "approved",
            "label_manifest_sha256": label_hash,
            "reviewed_episode_ids": list(TRAINING_EPISODE_ALLOWLIST),
            "continuous_visual_review": True,
            "takeover_boundary_visual_review": True,
        },
        "train_episodes": train,
        "validation_episodes": validation,
        "sealed_holdout_episode_ids": list(SEALED_HOLDOUT_EPISODE_IDS),
        "sealed_holdout_files_opened": 0,
        "normalization": {
            "scope": "train_split_only",
            "source_episode_ids": train,
            "feature_spec_sha256": FEATURE_SPEC_SHA256,
        },
        "training_rows_by_actor_phase": {
            phase: 1 for phase in ACTOR_HEAD_PHASES
        },
        "action_frozen_mask": [True, False, False, False],
        "action_frozen_values": [0.0, 0.0, 0.0, 0.0],
        "counters": {
            "actor_behavior_cloning_updates": 1,
            "critic_updates": 0,
            "sac_gradient_steps": 0,
            "environment_steps": 0,
        },
        "checkpoint_role": (
            "actor_training_artifact_requires_separate_runtime_admission"),
        "runtime_route_contract": RUNTIME_ROUTE_CONTRACT,
        "runtime_admission_embedded": False,
    }


def valid_reviewed_segments():
    records = []
    for episode_id in TRAINING_EPISODE_ALLOWLIST:
        for segment_index, phase in enumerate(LABEL_PHASES):
            actor_phase = phase in ACTOR_HEAD_PHASES
            records.append({
                "schema": REVIEWED_SEGMENT_SCHEMA,
                "label_version": REVIEWED_LABEL_VERSION,
                "episode_id": episode_id,
                "segment_index": segment_index,
                "phase": phase,
                "row_count": 2,
                "holdout": False,
                "review_status": "approved",
                "training_eligible_now": actor_phase,
                "actor_action_eligible_after_review": actor_phase,
                "axis_mask_after_review": [1.0 if actor_phase else 0.0] * 4,
            })
    return records


class ContactCheckpointTests(unittest.TestCase):
    def test_valid_reviewed_checkpoint_metadata_passes_without_tensors(self):
        value = valid_checkpoint()
        self.assertIs(validate_contact_checkpoint_metadata(value), value)

    def test_legacy_four_head_checkpoint_is_rejected(self):
        value = valid_checkpoint()
        value["schema"] = "excavation_phase_head_actor_bc_checkpoint_v4"
        value["actor_head_phases"] = [
            "approach", "penetrate_deepen", "curl_load", "lift_carry"]
        with self.assertRaisesRegex(ValueError, "schema_mismatch"):
            validate_contact_checkpoint_metadata(value)

    def test_candidate_or_pending_labels_cannot_enter_training(self):
        record = {
            "schema": "excavation_contact_segment_candidate_v1",
            "label_version": "contact_segments_v1_candidate",
            "episode_id": TRAINING_EPISODE_ALLOWLIST[0],
            "phase": "penetrate_deepen",
            "row_count": 3,
            "holdout": False,
            "review_status": "visual_manual_review_pending",
            "training_eligible_now": False,
            "actor_action_eligible_after_review": True,
            "axis_mask_after_review": [1.0] * 4,
        }
        with self.assertRaisesRegex(ValueError, "schema_not_reviewed"):
            validate_reviewed_segment_records([record])

    def test_reviewed_labels_keep_geometry_and_recovery_out_of_actor(self):
        result = validate_reviewed_segment_records(valid_reviewed_segments())
        self.assertEqual(result["eligible_rows"], 36)
        self.assertEqual(
            set(result["eligible_rows_by_phase"]), set(ACTOR_HEAD_PHASES))
        bad = valid_reviewed_segments()
        recovery = next(
            row for row in bad if row["phase"] == "uncertain/recovery")
        recovery["training_eligible_now"] = True
        with self.assertRaisesRegex(ValueError, "excluded_phase_training_enabled"):
            validate_reviewed_segment_records(bad)

    def test_review_receipt_and_split_are_mandatory(self):
        for mutation, error in (
            (lambda value: value["manual_review_receipt"].update(
                {"continuous_visual_review": False}), "visual_review_incomplete"),
            (lambda value: value.update({"validation_episodes": []}),
             "split_missing"),
            (lambda value: value["normalization"].update(
                {"scope": "all_six_episodes"}), "normalization_scope_mismatch"),
        ):
            value = valid_checkpoint()
            mutation(value)
            with self.assertRaisesRegex(ValueError, error):
                validate_contact_checkpoint_metadata(value)

    def test_new_contact_checkpoint_is_rejected_by_legacy_loader(self):
        with self.assertRaisesRegex(ValueError, "v4_actor_checkpoint_contract_mismatch"):
            checkpoint_runtime_route(valid_checkpoint())

    def test_runtime_requires_separate_hash_bound_receipt(self):
        checkpoint = valid_checkpoint()
        receipt = {
            "schema": RUNTIME_RECEIPT_SCHEMA,
            "status": "passed",
            "checkpoint_sha256": "2" * 64,
            "runtime_route_contract": RUNTIME_ROUTE_CONTRACT,
            "runtime_audit_sha256": "3" * 64,
        }
        self.assertIs(validate_runtime_admission_receipt(
            checkpoint, receipt, checkpoint_sha256="2" * 64), receipt)
        bad = copy.deepcopy(receipt)
        bad["checkpoint_sha256"] = "4" * 64
        with self.assertRaisesRegex(ValueError, "runtime_receipt_mismatch"):
            validate_runtime_admission_receipt(
                checkpoint, bad, checkpoint_sha256="2" * 64)


if __name__ == "__main__":
    unittest.main()
