import unittest
import numpy as np
from terrain.solid_bank import SolidBank
from terrain.tool_region import swept_bounds,require_active_corridor


class ToolRegionTests(unittest.TestCase):
    def test_crossing_solid_rejects_before_activation_then_accepts(self):
        bank=SolidBank(np.ones((4,4,4),bool),[0,0,0])
        a=np.array([[-.2,.11,.11],[-.1,.19,.19]])
        b=a+[.7,0,0]
        with self.assertRaisesRegex(ValueError,'unconverted'):require_active_corridor(bank,a,b)
        self.assertEqual(bank.version,0)
        lo,hi=swept_bounds(a,b,.02)
        bank.activate(lo,hi,lambda p,s:None)
        require_active_corridor(bank,a,b)
        self.assertGreater(bank.solid.sum(),0)

    def test_padding_rejects_nearby_boundary(self):
        bank=SolidBank(np.ones((1,1,1),bool),[0,0,0])
        a=np.array([[.105,.02,.02],[.12,.08,.08]])
        require_active_corridor(bank,a,a,0)
        with self.assertRaises(ValueError):require_active_corridor(bank,a,a,.02)

    def test_invalid_input(self):
        for a,b,p in [([[0,0,0]],[[float('nan'),0,0]],0),([],[],0),([[0,0,0]],[[0,0,0]],-1)]:
            with self.assertRaises(ValueError):swept_bounds(a,b,p)
