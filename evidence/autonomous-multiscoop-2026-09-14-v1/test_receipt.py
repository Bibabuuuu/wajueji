import unittest
import numpy as np
from receipt_observer import StableWindow

class WindowTests(unittest.TestCase):
    def test_requires_full_window(self):
        w=StableWindow(2)
        for t in np.arange(0,5,.25):self.assertFalse(w.update(t,np.array([True,False])).any())
        self.assertEqual(w.update(5,np.array([True,False])).tolist(),[True,False])
    def test_departure_resets_identity(self):
        w=StableWindow(1)
        for t in np.arange(0,5,.25):w.update(t,np.array([True]))
        w.update(5,np.array([False]));self.assertFalse(w.update(5.25,np.array([True])).any())
    def test_missing_sample_resets(self):
        w=StableWindow(1)
        for t in np.arange(0,5,.25):w.update(t,np.array([True]))
        self.assertFalse(w.update(5.5,np.array([True])).any())
    def test_reversal_rejected(self):
        w=StableWindow(1);w.update(1,np.array([True]))
        with self.assertRaises(ValueError):w.update(.5,np.array([True]))
if __name__=='__main__':unittest.main()
