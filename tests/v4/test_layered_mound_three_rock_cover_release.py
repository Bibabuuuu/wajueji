import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "v4"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from run_layered_mound_three_rock_cover_release import (  # noqa: E402
    _prepare_particles,
    load_config,
)


CONFIG = ROOT / "configs" / "v4" / "layered-mound-three-rock-cover-release-v1.json"


class LayeredMoundThreeRockCoverReleaseTests(unittest.TestCase):
    def test_cover_partition_and_user_acceptance(self):
        value = load_config(CONFIG, ROOT)
        manifest = json.loads((ROOT / value["physical_geometry"]["manifest_path"])
                              .read_text(encoding="utf-8"))
        by_id = {rock["rock_id"]: rock for rock in manifest["rocks"]}
        records = [by_id[item["manifest_rock_id"]] for item in value["targets"]]
        prepared = _prepare_particles(value, ROOT, records)
        self.assertEqual(prepared["q"].shape, (133079, 3))
        self.assertEqual(int((prepared["activation"] == 0).sum()), 85620)
        self.assertEqual(int((prepared["activation"] == 1).sum()), 47459)
        self.assertTrue(value["acceptance"]["rock_rotation_during_release_allowed"])
        self.assertFalse(value["acceptance"]["final_manifest_pose_required"])
        self.assertTrue(value["acceptance"]["cover_must_settle_before_release"])

    def test_rejects_failure_binding_and_construction_tampering(self):
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        attacks = []
        candidate = copy.deepcopy(base)
        candidate["failed_bare_release_gate"]["report_sha256"] = "0" * 64
        attacks.append((candidate, "failed_bare_release_gate_report_binding"))
        candidate = copy.deepcopy(base)
        candidate["construction"]["rocks_held_during_cover_settle"] = False
        attacks.append((candidate, "construction_contract"))
        candidate = copy.deepcopy(base)
        candidate["particles"]["combined_particle_count"] += 1
        attacks.append((candidate, "particle_contract"))
        with tempfile.TemporaryDirectory() as temporary:
            for index, (candidate, message) in enumerate(attacks):
                path = Path(temporary) / f"attack-{index}.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(message=message), self.assertRaisesRegex(
                        ValueError, message):
                    load_config(path, ROOT)


if __name__ == "__main__":
    unittest.main()
