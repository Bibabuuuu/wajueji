"""Cheap regression for the explicit terrain exemption and remaining guards."""
import ast,unittest
from pathlib import Path
from clearance import FixedRouteClearance
class Tests(unittest.TestCase):
    def checker(self,**changes):
        c=FixedRouteClearance.__new__(FixedRouteClearance)
        result=dict(bucket_truck_m=.1,bucket_terrain_m=None,mechanism_truck_m=.1,mechanism_ground_m=.1,closure_m=0.)
        result.update(changes);c.measure_clearance=lambda q:(result,'boom');return c
    def test_explicit_unmeasured_terrain(self):
        self.assertIsNone(self.checker().check(None)['bucket_terrain_m'])
        self.assertIsNone(self.checker().material_clearance(None))
    def test_static_guards_remain(self):
        for key,value in [('bucket_truck_m',.039),('mechanism_truck_m',.019),('mechanism_ground_m',.019),('closure_m',.0011),('bucket_truck_m',float('nan'))]:
            with self.subTest(key=key),self.assertRaises(ValueError):self.checker(**{key:value}).check(None,physical=True)
    def test_original_closure_tolerances(self):
        self.checker(closure_m=.0005).check(None,physical=True)
        with self.assertRaises(ValueError):self.checker(closure_m=.0005).check(None)
    def test_syntax(self):
        for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
