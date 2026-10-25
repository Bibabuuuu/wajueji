import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/v4'))
sys.path.insert(0, '/home/qunwang/codex-work/small-soil-20260910-v17-performance')

from legacy_cad_bucket_collision_candidate import build_geometry, public_summary
from scene_assembly import Assembler


class LegacyCadBucketCollisionCandidateTest(unittest.TestCase):
    def setUp(self):
        source = Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
        self.assembly = Assembler(
            source/'scene/input/model/A.xml', source/'scene/input/mesh')
        self.qpos = np.asarray(json.loads(Path(
            '/home/qunwang/codex-work/compact-bank-2026-09-11/operator-start.json'
        ).read_text())['qpos'])

    def test_extracts_open_finite_bucket_mesh(self):
        geometry = build_geometry(self.assembly, self.qpos)
        self.assertEqual(geometry['schema'],
                         'legacy_cad_bucket_collision_candidate_v1')
        self.assertFalse(geometry['is_solid'])
        self.assertEqual(geometry['vertices_bucket_m'].shape, (3017, 3))
        self.assertEqual(geometry['faces'].shape, (5922, 3))
        self.assertEqual(geometry['boundary_edges'], 80)
        self.assertEqual(geometry['nonmanifold_edges'], 0)
        self.assertEqual(geometry['source_mesh_sha256'],
                         'e24093436a3eef4e8f4fba21152e6f15ac349a837df8d13ebcdb23df860ca79b')

    def test_public_summary_excludes_large_arrays(self):
        summary = public_summary(build_geometry(self.assembly, self.qpos))
        self.assertNotIn('vertices_bucket_m', summary)
        self.assertNotIn('faces', summary)
        self.assertEqual(summary['faces_count'], 5922)


if __name__ == '__main__':
    unittest.main()
