import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import run_layered_mound_checkpoint_hold_audit as audit
import run_layered_mound_fortyseven_rock_four_layer as continuation


ROOT = Path(__file__).resolve().parents[2]


class LayeredMoundCheckpointContinuationTest(unittest.TestCase):
    def test_solver_restore_report_accepts_current_and_legacy_layouts(self):
        solver = {
            "grid_readback_exact": {"grid": True},
            "solver_field_readback_exact": {"warmstart": True},
        }
        self.assertIs(
            continuation._solver_restore_report({"solver_restore": solver}),
            solver)
        self.assertIs(
            continuation._solver_restore_report({
                "restore": {"solver_continuation": solver}}),
            solver)
        with self.assertRaisesRegex(
                KeyError, "solver_restore_or_solver_continuation"):
            continuation._solver_restore_report({"restore": {}})

    def test_recovery_paths_do_not_claim_the_formal_checkpoint_name(self):
        output = Path("/tmp/four-layer-report.json")
        checkpoint = Path("/tmp/four-layer-checkpoint.npz")
        recovery_checkpoint, recovery_metadata, evidence = (
            continuation._recovery_paths(output, checkpoint))
        self.assertEqual(
            recovery_checkpoint.name, "four-layer-checkpoint.recovery.npz")
        self.assertEqual(
            recovery_metadata.name,
            "four-layer-checkpoint.recovery.npz.json")
        self.assertEqual(evidence.name, "four-layer-report.recovery.json")

    def test_saved_recovery_can_finalize_without_physics_rerun(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "report.json"
            checkpoint_out = root / "checkpoint.npz"
            metadata_out = root / "checkpoint.npz.json"
            source = root / "source.npz"
            source_metadata = root / "source.npz.json"
            config_path = root / "config.json"
            source.write_bytes(b"unchanged-source")
            source_metadata.write_text("{}\n", encoding="utf-8")
            config_path.write_text("{}\n", encoding="utf-8")
            source_identity = continuation.checkpoint_identity(source)
            config = {
                "input_checkpoint": {"sha256": source_identity["sha256"]},
                "construction": {"required_new_pair_sat_gap_m": 0.001},
                "probe": {
                    "release_stable_window_frames": 120,
                    "maximum_rock_linear_speed_m_s": 5.0,
                    "maximum_rock_angular_speed_rad_s": 25.0,
                    "maximum_particle_speed_m_s": 5.0,
                },
            }
            recovery_checkpoint, recovery_metadata_path, evidence_path = (
                continuation._recovery_paths(output, checkpoint_out))
            recovery_checkpoint.write_bytes(b"validated-recovery-state")
            solver = {
                "grid_readback_exact": {"grid": True},
                "solver_field_readback_exact": {"warmstart": True},
            }
            evidence = {
                "schema": continuation.REPORT_SCHEMA,
                "status": continuation.RECOVERY_STATUS,
                "checks": {},
                "source_checkpoint": source_identity,
                "source_checkpoint_after": source_identity,
                "restore": {"restore": {
                    "field_readback_exact": {"particle_q": True},
                    "rigid_qpos_readback_within_tolerance": True,
                    "rigid_qvel_readback_within_tolerance": True,
                    "solver_continuation": solver,
                }},
                "checkpoint_self_check": {"checks": {"continuity": True}},
                "expanded_fixture_continuity": {"passed": True},
                "geometry": {
                    "post_cover_runtime_sat_passed": True,
                    "minimum_new_vs_existing_sat_gap_m": 0.002,
                    "minimum_new_vs_new_sat_gap_m": 0.002,
                    "minimum_pre_cover_runtime_sat_gap_m": 0.002,
                    "minimum_post_cover_runtime_sat_gap_m": 0.002,
                },
                "particles": {
                    "upper_cover_active_count": 1,
                    "activation_report": {"dormant_history_restored": True},
                },
                "construction": {
                    "finite": True,
                    "parked_force_max_n": 0.0,
                    "settle": {"passed": True},
                    "external_rigid_support_substeps": 0,
                    "maximum_forbidden_contacts": 0,
                },
                "release": {
                    "executed": True,
                    "all_finite": True,
                    "stable_window_frames": 120,
                    "external_rigid_support_substeps": 0,
                    "maximum_forbidden_contacts": 0,
                    "maximum_global_momentum_residual_ns": 0.0,
                    "maximum_rock_linear_speed_m_s": 0.0,
                    "maximum_rock_angular_speed_rad_s": 0.0,
                    "maximum_particle_speed_m_s": 0.0,
                },
                "additional_uncontrolled_hold": {
                    "all_frames_stable": True},
                "execution": {"recovery_saved_wall_seconds": 12.0},
                "next_gate": "finalize_saved_four_layer_recovery",
            }
            continuation._write_json(evidence_path, evidence)
            recovery_identity = continuation.checkpoint_identity(
                recovery_checkpoint)
            recovery = {
                "schema": continuation.RECOVERY_SCHEMA,
                "status": continuation.RECOVERY_STATUS,
                "recovery_checkpoint_file": recovery_checkpoint.name,
                "recovery_checkpoint_sha256": recovery_identity["sha256"],
                "recovery_evidence_file": evidence_path.name,
                "recovery_evidence_sha256": continuation.sha256(evidence_path),
                "intended_checkpoint_file": checkpoint_out.name,
                "intended_report_file": output.name,
                "source_checkpoint": str(source),
                "source_checkpoint_sha256": source_identity["sha256"],
                "source_metadata_sha256": continuation.sha256(source_metadata),
                "source_config_sha256": continuation.sha256(config_path),
                "physics_runner_sha256": "a" * 64,
                "continuation_point": "test-recovery",
                "newton_version": "1.5.1",
                "warp_version": "1.17.0",
                "solver_grid_type": "fixed",
                "solver_warmstart_mode": "grid",
                "collider_velocity_mode": "forward",
                "array_count": 18,
                "fields": {"particle_q": {
                    "shape": [1, 3], "dtype": "float32"}},
                "particle_count": 1,
                "rock_count": 47,
                "simulation_time_s": 1.0,
                "size_bytes": recovery_identity["size_bytes"],
                "data_domain": "test-local-domain",
                "requires_finalization": True,
            }
            continuation._write_json(recovery_metadata_path, recovery)

            report = continuation._finalize_recovery(
                output=output,
                checkpoint_out=checkpoint_out,
                metadata_out=metadata_out,
                recovery_checkpoint=recovery_checkpoint,
                recovery_metadata_path=recovery_metadata_path,
                recovery_evidence_path=evidence_path,
                config=config,
                config_path=config_path,
                source_checkpoint=source,
                source_metadata_path=source_metadata,
                data_domain="test-local-domain")

            self.assertEqual(report["status"], "four_layer_checkpoint_saved")
            self.assertTrue(all(report["checks"].values()))
            self.assertTrue(checkpoint_out.is_file())
            self.assertTrue(metadata_out.is_file())
            self.assertTrue(output.is_file())
            self.assertFalse(recovery_checkpoint.exists())
            promoted = json.loads(
                recovery_metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(
                promoted["status"],
                "four_layer_recovery_promoted_to_formal")

    def test_audit_contract_is_read_only_and_uses_gpu_independent_scope(self):
        config = audit.load_config(
            ROOT / "configs/v4/layered-mound-checkpoint-hold-audit-v1.json", ROOT)
        self.assertFalse(config["audit"]["writes_checkpoint"])
        self.assertTrue(config["scope"]["independent_gpu_audit"])
        self.assertEqual(config["audit"]["hold_frames"], 120)

    def test_four_layer_config_starts_from_frozen_platform_checkpoint(self):
        config = continuation.load_config(
            ROOT / "configs/v4/layered-mound-fortyseven-rock-four-layer-v1.json",
            ROOT)
        self.assertEqual(config["input_checkpoint"]["rock_count"], 33)
        self.assertTrue(config["input_checkpoint"]["read_only"])
        self.assertEqual(config["scope"]["output_rock_count"], 47)
        self.assertEqual(len(config["targets"]), 14)
        source_bank = Path(config["source"]["source_bank_path"])
        self.assertFalse(source_bank.is_absolute())
        self.assertEqual(source_bank, Path("data/initial-bank.npz"))
        self.assertTrue((ROOT / source_bank).is_file())

    def test_local_rebuild_can_supply_an_explicit_separate_source_contract(self):
        source_path = (
            ROOT / "configs/v4/layered-mound-fortyseven-rock-four-layer-v1.json"
        )
        raw = json.loads(source_path.read_text(encoding="utf-8"))
        local_contract = {
            **continuation.FORMAL_INPUT_CHECKPOINT_CONTRACT,
            "default_path": "/local-data/reconstructed-33.npz",
            "sha256": "1" * 64,
            "source_config_sha256": "2" * 64,
            "source_runner_sha256": "3" * 64,
        }
        raw["input_checkpoint"] = local_contract
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "local-four-layer.json"
            candidate.write_text(json.dumps(raw), encoding="utf-8")
            config = continuation.load_config(
                candidate,
                ROOT,
                expected_input_checkpoint=local_contract)
            with self.assertRaisesRegex(
                    ValueError, "fortyseven_input_checkpoint_contract"):
                continuation.load_config(candidate, ROOT)

        self.assertEqual(config["input_checkpoint"], local_contract)

    def test_upper_geometry_is_derived_against_restored_pose_contract(self):
        config = continuation.load_config(
            ROOT / "configs/v4/layered-mound-fortyseven-rock-four-layer-v1.json",
            ROOT)
        manifest = json.loads((
            ROOT / config["physical_geometry"]["manifest_path"]
        ).read_text(encoding="utf-8"))
        source = [rock for rock in manifest["rocks"]
                  if rock["layer_id"] in continuation.SOURCE_LAYERS]
        upper = [rock for rock in manifest["rocks"]
                 if rock["layer_id"] == continuation.NEW_LAYER]
        qpos = np.asarray([
            [*rock["position_world_m"], *rock["orientation_quaternion_wxyz"]]
            for rock in source], dtype=np.float64)
        prepared = continuation._prepare_upper(config, ROOT, source, upper, qpos)
        self.assertEqual(prepared["raw_count"], 39056)
        self.assertGreater(prepared["active_count"], 0)
        self.assertGreaterEqual(
            prepared["minimum_new_vs_existing_sat_gap_m"], 0.001)
        self.assertGreaterEqual(prepared["minimum_new_vs_new_sat_gap_m"], 0.001)
        self.assertEqual(len(prepared["starts"]), 14)


if __name__ == "__main__":
    unittest.main()
