import copy
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src" / "v4"))

from build_scene_assembly_preflight import (  # noqa: E402
    _tree_hash,
    _tree_snapshot,
    _validate_config,
    build,
    sha256,
    write_json,
)


CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "v4" / "scene-assembly-preflight-v1.json"
)
INPUT_SCENE_ROOT = (
    REPOSITORY_ROOT / "reports" / "v4" / "scene-reconstruction-2026-09-19-v1"
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class ConfigurationValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(CONFIG_PATH)

    def test_rejects_schema_contract_capacity_and_scope_drift(self):
        cases = []
        invalid = copy.deepcopy(self.config)
        invalid["schema"] = "wrong"
        cases.append((invalid, "invalid_assembly_preflight_config_schema"))
        invalid = copy.deepcopy(self.config)
        invalid["current_bridge"]["maximum_rock_count"] = 3
        cases.append((invalid, "current_bridge_capacity_must_match_two_rock_evidence"))
        invalid = copy.deepcopy(self.config)
        invalid["future_candidate"]["unit_regression_batch_capacity"] = 3
        cases.append((invalid, "unit_regression_capacity_must_match_current_bridge"))
        invalid = copy.deepcopy(self.config)
        invalid["future_candidate"]["maximum_capacity"] = 13
        cases.append((invalid, "maximum_capacity_must_be_20"))
        invalid = copy.deepcopy(self.config)
        invalid["scope"]["physics_run"] = True
        cases.append((invalid, "scope_must_remain_false:physics_run"))
        invalid = copy.deepcopy(self.config)
        invalid["current_bridge"]["qpos_per_rock"] = float("nan")
        cases.append((invalid, "non_finite_value"))
        for candidate, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    _validate_config(candidate)


class OutputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.first_output = cls.root / "first"
        cls.report = build(CONFIG_PATH, cls.first_output, REPOSITORY_ROOT)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_nine_scenes_and_all_rocks_are_bound_exactly_once(self):
        self.assertEqual(self.report["status"], "scene_assembly_preflight_candidate")
        self.assertEqual(self.report["scene_count"], 9)
        self.assertEqual(self.report["rock_count"], 159)
        self.assertEqual(self.report["unit_regression_batch_count"], 81)
        self.assertEqual(self.report["maximum_capacity"], 20)
        self.assertEqual(self.report["first_round_admission_scene_count"], 6)
        self.assertEqual(self.report["deferred_boundary_scene_count"], 3)
        self.assertTrue(self.report["input_tree_unchanged"])
        self.assertTrue(all(self.report["checks"].values()))

        seen = set()
        counts = []
        for record in self.report["scenes"]:
            path = self.first_output / record["scene_id"] / "assembly-plan.json"
            self.assertEqual(record["assembly_plan_sha256"], sha256(path))
            plan = read_json(path)
            counts.append(plan["rock_count"])
            flattened = []
            for batch in plan["unit_regression_batches"]:
                self.assertIn(batch["rock_count"], (1, 2))
                self.assertTrue(batch["does_not_qualify_full_scene_physics"])
                flattened.extend(member["source_rock_id"] for member in batch["members"])
            expected = [rock["rock_id"] for rock in plan["rocks"]]
            self.assertEqual(flattened, expected)
            self.assertEqual(len(flattened), len(set(flattened)))
            self.assertTrue(seen.isdisjoint(flattened))
            seen.update(flattened)
        self.assertEqual(sorted(counts), [13, 13, 13, 20, 20, 20, 20, 20, 20])
        self.assertEqual(len(seen), 159)

    def test_capacity_addresses_messages_and_contact_upper_bounds(self):
        compatibility = read_json(self.first_output / "compatibility-report.json")
        decision = compatibility["maximum_capacity_decision"]
        self.assertEqual(decision["maximum_capacity"], 20)
        maximum_contract = decision["maximum_capacity_contract"]
        self.assertEqual((maximum_contract["nq"], maximum_contract["nv"]), (159, 138))
        self.assertEqual(
            maximum_contract["estimated_state_named_fields_including_rock_records"], 99
        )
        self.assertEqual(maximum_contract["state_numeric_values"], 411)
        self.assertEqual(decision["decision_status"],
                         "user_confirmed_unique_maximum_not_implemented")
        self.assertTrue(decision["padding_forbidden"])
        self.assertEqual(compatibility["first_round_admission"]["scene_count"], 6)
        self.assertEqual(compatibility["deferred_boundary_diagnostics"]["scene_count"], 3)
        bounds = compatibility["contact_pair_upper_bounds"]
        self.assertEqual(bounds["13"]["total_candidate_geom_pairs"], 390)
        self.assertEqual(bounds["20"]["total_candidate_geom_pairs"], 670)

        plan_13 = next(
            read_json(self.first_output / item["scene_id"] / "assembly-plan.json")
            for item in self.report["scenes"] if item["rock_count"] == 13
        )
        self.assertEqual(plan_13["maximum_capacity"], 20)
        self.assertEqual(plan_13["loaded_model_rock_count"], 13)
        self.assertEqual(plan_13["runtime_contract"]["rock_capacity"], 13)
        self.assertEqual(plan_13["runtime_contract"]["nq"], 110)
        self.assertEqual(plan_13["runtime_contract"]["nv"], 96)
        self.assertEqual(
            plan_13["runtime_contract"][
                "estimated_state_named_fields_including_rock_records"
            ],
            71,
        )
        self.assertEqual(plan_13["runtime_contract"]["state_numeric_values"], 299)
        self.assertTrue(plan_13["padding_forbidden"])
        self.assertFalse(plan_13["first_round_admission"])
        self.assertEqual(plan_13["qualification"], "deferred_boundary_diagnostic")
        self.assertNotIn("inactive_fixed_slots", plan_13)
        plan_20 = next(
            read_json(self.first_output / item["scene_id"] / "assembly-plan.json")
            for item in self.report["scenes"] if item["rock_count"] == 20
        )
        self.assertEqual(plan_20["runtime_contract"]["nq"], 159)
        self.assertEqual(plan_20["runtime_contract"]["nv"], 138)
        self.assertTrue(plan_20["first_round_admission"])
        for slot, rock in enumerate(plan_13["rocks"]):
            binding = rock["full_scene_binding"]
            self.assertEqual(binding["slot"], slot)
            self.assertEqual(binding["body_name"], f"r2-rock-{slot:03d}")
            self.assertEqual(binding["freejoint_name"], f"r2_rock_free_{slot:03d}")
            self.assertEqual(binding["qpos_address"], 19 + 7 * slot)
            self.assertEqual(binding["qvel_address"], 18 + 6 * slot)

    def test_mass_inertia_pose_source_and_r1_contract_are_explicit(self):
        plan = read_json(
            self.first_output / self.report["scenes"][0]["scene_id"] / "assembly-plan.json"
        )
        for rock in plan["rocks"]:
            self.assertGreater(rock["mass_kg"], 0.0)
            self.assertGreater(rock["volume_m3"], 0.0)
            self.assertEqual(len(rock["position_world_m"]), 3)
            self.assertEqual(len(rock["orientation_quaternion_wxyz"]), 4)
            self.assertTrue(math.isclose(
                sum(value * value for value in rock["orientation_quaternion_wxyz"]),
                1.0, rel_tol=0.0, abs_tol=2e-9,
            ))
            self.assertEqual(len(rock["inertia_about_com_body_kg_m2"]), 3)
            self.assertEqual(len(rock["inertia_about_com_world_kg_m2"]), 3)
            self.assertIn(rock["parameter_source"]["status"], {
                "measured", "human_reviewed_estimate", "synthetic_assumption", "unknown"
            })
        self.assertEqual(
            self.report["r1_evidence"]["two_rock_contract"],
            {"nq": 33, "nv": 30, "nu": 12, "nplugin": 0, "rock_count": 2},
        )

    def test_two_fresh_builds_are_byte_identical(self):
        second_output = self.root / "second"
        build(CONFIG_PATH, second_output, REPOSITORY_ROOT)
        first_files = sorted(
            path.relative_to(self.first_output) for path in self.first_output.rglob("*")
            if path.is_file()
        )
        second_files = sorted(
            path.relative_to(second_output) for path in second_output.rglob("*")
            if path.is_file()
        )
        self.assertEqual(first_files, second_files)
        for relative in first_files:
            self.assertEqual(
                (self.first_output / relative).read_bytes(),
                (second_output / relative).read_bytes(),
                relative,
            )


class FailureAndBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.scene_root = self.root / "input-scenes"
        shutil.copytree(INPUT_SCENE_ROOT, self.scene_root)
        self.config = read_json(CONFIG_PATH)
        self.config["input"]["scene_root"] = str(self.scene_root)

    def tearDown(self):
        self.temporary.cleanup()

    def _root_report(self):
        return read_json(self.scene_root / "report.json")

    def _rewrite_root_and_config(self, root_report):
        write_json(self.scene_root / "report.json", root_report)
        self.config["input"]["root_report_sha256"] = sha256(
            self.scene_root / "report.json"
        )
        config_path = self.root / "config.json"
        write_json(config_path, self.config)
        return config_path

    def _repair_scene_chain(self, scene_id):
        directory = self.scene_root / scene_id
        report = read_json(directory / "report.json")
        report["scene_manifest_sha256"] = sha256(directory / "scene-manifest.json")
        report["scene_metrics_sha256"] = sha256(directory / "scene-metrics.json")
        report["preview_sha256"] = sha256(directory / "preview.svg")
        write_json(directory / "report.json", report)
        root_report = self._root_report()
        record = next(item for item in root_report["scenes"] if item["scene_id"] == scene_id)
        record["report_sha256"] = sha256(directory / "report.json")
        return self._rewrite_root_and_config(root_report)

    def _assert_fails_without_output(self, config_path, expected):
        output = self.root / f"must-not-exist-{len(list(self.root.glob('must-not-exist-*')))}"
        with self.assertRaisesRegex((ValueError, FileNotFoundError), expected):
            build(config_path, output, REPOSITORY_ROOT)
        self.assertFalse(output.exists())

    def test_changed_valid_input_hash_changes_output_binding(self):
        baseline_tree_hash = _tree_hash(_tree_snapshot(self.scene_root))
        scene_id = self._root_report()["scenes"][0]["scene_id"]
        preview = self.scene_root / scene_id / "preview.svg"
        preview.write_text(
            preview.read_text(encoding="utf-8") + "<!-- binding change -->\n",
            encoding="utf-8", newline="\n",
        )
        config_path = self._repair_scene_chain(scene_id)
        output = self.root / "changed-output"
        report = build(config_path, output, REPOSITORY_ROOT)
        self.assertNotEqual(
            report["input_tree_sha256_before"],
            baseline_tree_hash,
        )
        plan = read_json(output / scene_id / "assembly-plan.json")
        self.assertEqual(plan["source_input_hashes"]["preview_sha256"], sha256(preview))

    def test_bad_hash_and_missing_file_fail_before_output(self):
        bad_hash = copy.deepcopy(self.config)
        bad_hash["input"]["root_report_sha256"] = "0" * 64
        bad_hash_path = self.root / "bad-hash.json"
        write_json(bad_hash_path, bad_hash)
        self._assert_fails_without_output(bad_hash_path, "scene_root_report_hash_mismatch")

        scene_id = self._root_report()["scenes"][0]["scene_id"]
        (self.scene_root / scene_id / "preview.svg").unlink()
        missing_path = self.root / "missing.json"
        write_json(missing_path, self.config)
        self._assert_fails_without_output(missing_path, "missing_scene_input")

    def test_duplicate_id_unknown_prototype_bad_inertia_and_bad_status_fail(self):
        mutations = (
            ("duplicate", lambda manifest: manifest["rocks"][1].__setitem__(
                "rock_id", manifest["rocks"][0]["rock_id"]), "duplicate_rock_id"),
            ("prototype", lambda manifest: manifest["rocks"][0].__setitem__(
                "prototype_id", "unknown"), "unknown_prototype"),
            ("inertia", lambda manifest: manifest["rocks"][0][
                "inertia_about_com_body_kg_m2"][0].__setitem__(0, 999.0),
             "numeric_contract_mismatch:body_inertia"),
        )
        for label, mutate, expected in mutations:
            with self.subTest(label=label):
                case_root = self.root / label
                shutil.copytree(INPUT_SCENE_ROOT, case_root)
                scene_id = read_json(case_root / "report.json")["scenes"][0]["scene_id"]
                manifest_path = case_root / scene_id / "scene-manifest.json"
                manifest = read_json(manifest_path)
                mutate(manifest)
                write_json(manifest_path, manifest)
                old_root = self.scene_root
                old_config = self.config
                try:
                    self.scene_root = case_root
                    self.config = read_json(CONFIG_PATH)
                    self.config["input"]["scene_root"] = str(case_root)
                    config_path = self._repair_scene_chain(scene_id)
                    self._assert_fails_without_output(config_path, expected)
                finally:
                    self.scene_root = old_root
                    self.config = old_config

        root_report = self._root_report()
        root_report["status"] = "passed"
        status_path = self._rewrite_root_and_config(root_report)
        self._assert_fails_without_output(status_path, "scene_root_status_mismatch")

    def test_non_finite_value_fails_before_output(self):
        scene_id = self._root_report()["scenes"][0]["scene_id"]
        directory = self.scene_root / scene_id
        manifest_path = directory / "scene-manifest.json"
        manifest = read_json(manifest_path)
        manifest["rocks"][0]["mass_kg"] = float("nan")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=True) + "\n",
            encoding="utf-8", newline="\n",
        )
        config_path = self._repair_scene_chain(scene_id)
        self._assert_fails_without_output(config_path, "non_finite_value")


if __name__ == "__main__":
    unittest.main()
