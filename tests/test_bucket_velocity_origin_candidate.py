"""Regression contract for the bucket velocity reference-point correction."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/v4'))
from bucket_velocity_origin_candidate import OLD, correct_source

class BucketVelocityOriginTest(unittest.TestCase):
    def test_corrects_bucket_reference_without_changing_rock_query(self):
        rock='mj_objectVelocity(model.get(), data.get(), mjOBJ_BODY, rock_bodies[0], rock_velocity, 0);'
        result=correct_source(OLD+'\n'+rock)
        self.assertIn('mjOBJ_XBODY, bucket_body',result)
        self.assertIn('bucket_com_linear_velocity_world_m_s',result)
        self.assertIn(rock,result)
        self.assertEqual(result.count('mjOBJ_XBODY'),1)

    def test_source_contract_rejects_missing_or_ambiguous_patch_point(self):
        for source in ['',OLD+OLD]:
            with self.assertRaisesRegex(ValueError,'source_contract'):
                correct_source(source)

if __name__=='__main__':unittest.main()
