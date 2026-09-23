import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "v4"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from build_layered_mixed_scene_local_preflight import (  # noqa: E402
    build, load_and_validate_config,
)


CONFIG = ROOT / "configs" / "v4" / "layered-mixed-scene-local-preflight-v1.json"


class LayeredMixedSceneLocalPreflightTests(unittest.TestCase):
    def test_config_freezes_geometry_only_three_layer_route(self):
        config = load_and_validate_config(CONFIG, ROOT)
        self.assertEqual([layer["layer_id"] for layer in config["layers"]], [
            "lower_skeleton", "middle_small_dominant", "top_exposed_small"])
        self.assertEqual(config["construction"]["embed_measure"],
                         "true_transformed_mesh_vertical_span_at_stage_fill_surface")
        self.assertFalse(config["scope"]["physics_step"])
        self.assertFalse(config["scope"]["stable_scene_claimed"])

    def test_rejects_scope_bank_and_embed_attacks(self):
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        attacks = []
        candidate = copy.deepcopy(base)
        candidate["scope"]["stable_scene_claimed"] = True
        attacks.append((candidate, "scope_expanded"))
        candidate = copy.deepcopy(base)
        candidate["frozen_bank"]["q_count"] = 1
        attacks.append((candidate, "bank_contract_drift"))
        candidate = copy.deepcopy(base)
        candidate["layers"][2]["rocks"][0]["embed_ratio"] = 0.9
        attacks.append((candidate, "rock_pose_or_embed"))
        with tempfile.TemporaryDirectory() as temporary:
            for index, (candidate, message) in enumerate(attacks):
                path = Path(temporary) / f"attack-{index}.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(message=message), self.assertRaisesRegex(
                        ValueError, message):
                    load_and_validate_config(path, ROOT)

    def test_build_is_deterministic_partitioned_and_truthfully_scoped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first", root / "second"
            first_report = build(CONFIG, ROOT, first)
            second_report = build(CONFIG, ROOT, second)
            self.assertEqual(first_report, second_report)
            for name in ("layered-local-bank.npz", "scene-manifest.json",
                         "metrics.json", "report.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            metrics = json.loads((first / "metrics.json").read_text(encoding="utf-8"))
            manifest = json.loads((first / "scene-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["rock_count"], 22)
            self.assertTrue(all(metrics["checks"].values()))
            self.assertEqual(metrics["retained_particle_center_inside_rock_count"], 0)
            self.assertGreater(metrics["removed_particle_count"], 0)
            self.assertLessEqual(metrics["maximum_embed_ratio_error"], 1e-9)
            with np.load(first / "layered-local-bank.npz", allow_pickle=False) as archive:
                self.assertEqual(len(archive["q"]), metrics["retained_particle_count"])
                self.assertEqual(set(archive["activation_stage"].tolist()), {0, 1, 2, 3})
            self.assertFalse(first_report["physics_step_executed"])
            self.assertFalse(first_report["stable_scene_claimed"])

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "output_dir_exists"):
                build(CONFIG, ROOT, output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
