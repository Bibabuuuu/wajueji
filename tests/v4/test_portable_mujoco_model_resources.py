import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "v4"))

from run_layered_mixed_scene_local_dynamics_smoke import (  # noqa: E402
    _make_model_resources_portable,
)


class PortableMujocoModelResourceTests(unittest.TestCase):
    def test_offline_overrides_and_repository_asset_rewrite(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            repository = base / "repo"
            mesh_dir = base / "offline" / "mesh"
            texture_dir = base / "offline" / "texture"
            repo_mesh = repository / "assets" / "v4" / "fixture" / "side.obj"
            for path in (mesh_dir / "machine.obj",
                         texture_dir / "machine.png", repo_mesh):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            root = ET.fromstring("""
                <mujoco>
                  <compiler meshdir="/missing/mesh" texturedir="/missing/texture"/>
                  <asset>
                    <mesh file="machine.obj"/>
                    <mesh file="/old/worktree/assets/v4/fixture/side.obj"/>
                    <texture file="machine.png"/>
                  </asset>
                </mujoco>
            """)
            with mock.patch.dict(os.environ, {
                    "EXCAVATOR_MUJOCO_MESH_DIR": str(mesh_dir),
                    "EXCAVATOR_MUJOCO_TEXTURE_DIR": str(texture_dir)}, clear=False):
                report = _make_model_resources_portable(root, repository)
            self.assertTrue(report["offline_override_applied"])
            self.assertEqual(report["absolute_repository_paths_rewritten"], 1)
            self.assertEqual(report["external_files_checked"], 3)
            self.assertEqual(root.find("compiler").get("meshdir"), str(mesh_dir))
            files = [item.get("file") for item in root.findall("./asset/mesh")]
            self.assertEqual(files[1], str(repo_mesh))

    def test_override_pair_is_required(self):
        root = ET.fromstring("<mujoco><compiler/><asset/></mujoco>")
        with mock.patch.dict(os.environ, {
                "EXCAVATOR_MUJOCO_MESH_DIR": "/tmp/mesh"}, clear=True):
            with self.assertRaisesRegex(ValueError, "override_pair"):
                _make_model_resources_portable(root, Path("/tmp/repo"))


if __name__ == "__main__":
    unittest.main()
