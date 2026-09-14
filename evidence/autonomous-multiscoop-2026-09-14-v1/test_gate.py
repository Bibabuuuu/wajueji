from stage_gate import StageGate
from pathlib import Path
import hashlib,json,unittest

class Gates(unittest.TestCase):
    def test_elapsed_time_does_not_complete_motion(self):
        g=StageGate('CUT',0,1)
        self.assertFalse(g.update(2,reached=False,fresh=True))
    def test_confirmation_resets_on_motion(self):
        g=StageGate('CUT',0,1)
        self.assertFalse(g.update(.1,reached=True,fresh=True))
        self.assertFalse(g.update(.2,reached=False,fresh=True))
        self.assertFalse(g.update(.3,reached=True,fresh=True))
        self.assertFalse(g.update(.5,reached=True,fresh=True))
        self.assertTrue(g.update(.55,reached=True,fresh=True))
    def test_empty_curl_does_not_advance(self):
        g=StageGate('CURL',0,1)
        for t in (.1,.5,1.):self.assertFalse(g.update(t,reached=True,fresh=True,cavity_kg=0))
    def test_lift_requires_clearance(self):
        g=StageGate('LIFT',0,1)
        self.assertFalse(g.update(.1,reached=True,fresh=True,cavity_kg=100,clear=False))
        self.assertFalse(g.update(.5,reached=True,fresh=True,cavity_kg=100,clear=True))
        self.assertTrue(g.update(.75,reached=True,fresh=True,cavity_kg=100,clear=True))
    def test_expired_and_duplicate_observations_reject(self):
        with self.assertRaises(ValueError):StageGate('CUT',0,1).update(0,reached=True,fresh=False)
        g=StageGate('CUT',0,1);g.update(0,reached=True,fresh=True)
        with self.assertRaises(ValueError):g.update(0,reached=True,fresh=True)
    def test_timeout_is_failure(self):
        with self.assertRaises(TimeoutError):StageGate('CUT',0,1).update(10,reached=True,fresh=True)

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Gates));a=Path(__file__).resolve().parent
    (a/'unit-tests.json').write_text(json.dumps({'passed':r.wasSuccessful(),'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'gate_sha256':hashlib.sha256((a/'stage_gate.py').read_bytes()).hexdigest()},indent=2)+'\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
