import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import compare_47_rigid_frequency as comparison


class RigidComparisonTest(unittest.TestCase):
    def pair(self):
        base=dict(status='practice_completed',mode='combined_smoke',frames=270,checks={'safe':True},
            protocol={'rigid_internal_hz':4800},source_checkpoint={'sha256':'same'},input_bindings={},
            runner_sha256='same',device='cuda:0',acquisition_domain='diagnostic',scope={},diagnostic_snapshots=[])
        candidate=copy.deepcopy(base)
        candidate['protocol']['rigid_internal_hz']=2400
        rows=[dict(motion_guard=True,strict=True,phase='hold',command_si=[0]*4,
                   axis_positions=[0]*4,bucket_vertical_clearance_m=.4) for _ in range(270)]
        return base,candidate,rows

    def test_rejects_failed_or_incomplete_windows(self):
        b,c,rows=self.pair(); c['status']='practice_failed'
        with patch.object(comparison,'read_run',side_effect=[(b,rows),(c,rows)]):
            with self.assertRaisesRegex(ValueError,'qualified_windows'):
                comparison.compare(Path('/base'),Path('/candidate'))

    def test_rejects_different_source_or_physics_protocol(self):
        for kind in ('source','protocol','actions'):
            b,c,rows=self.pair(); cr=copy.deepcopy(rows)
            if kind=='source':c['source_checkpoint']['sha256']='other'
            elif kind=='protocol':c['protocol']['hz']=120
            else:cr[0]['command_si']=[1,0,0,0]
            with patch.object(comparison,'read_run',side_effect=[(b,rows),(c,cr)]):
                with self.assertRaisesRegex(ValueError,'mismatch'):
                    comparison.compare(Path('/base'),Path('/candidate'))

    def test_requires_snapshot_evidence(self):
        b,c,rows=self.pair()
        with patch.object(comparison,'read_run',side_effect=[(b,rows),(c,rows)]):
            with self.assertRaisesRegex(ValueError,'five_diagnostic'):
                comparison.compare(Path('/base'),Path('/candidate'))


if __name__=='__main__':
    unittest.main()
