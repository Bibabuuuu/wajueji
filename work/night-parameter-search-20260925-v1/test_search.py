"""CPU-only scope, ranking and budget-contract checks."""
import unittest,sys,ast,tempfile,json
from pathlib import Path
import numpy as np
from parameters import BASELINE,CANDIDATES,modify,validate
from score import better
W=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_identity(self):
  rng=np.random.default_rng(925)
  for a in rng.uniform(-1,1,(100,4)):
   self.assertTrue(np.array_equal(modify(a,'curl_load',.8,BASELINE),a))
 def test_scope(self):
  p={k:v for k,v in CANDIDATES[-1].items() if k!='name'};a=np.array([.3,.4,-.6,.8])
  for phase in ('approach_contact','penetrate','lift','off_soil','CLEAR_PILE','ALIGN','DUMP'):
   self.assertTrue(np.array_equal(modify(a,phase,.8,p),a))
  self.assertTrue(np.array_equal(modify(a,'curl_load',.95,p),a))
  changed=modify(a,'curl_load',.8,p)
  self.assertEqual(changed[0],a[0]);self.assertTrue(np.all(abs(changed)<=1))
 def test_reject_bad_scale(self):
  with self.assertRaises(ValueError):validate({**BASELINE,'curl_scale':2})
 def test_volume_priority(self):
  b=dict(eligible=True,working_fill_percent=50.,seconds=50.,command_total_variation=10.,receipt_fraction=.99)
  self.assertFalse(better({**b,'eligible':False,'working_fill_percent':1000},b))
  self.assertFalse(better({**b,'working_fill_percent':40.,'seconds':10.},b))
  self.assertTrue(better({**b,'working_fill_percent':55.,'receipt_fraction':None},b))
  self.assertFalse(better({**b,'working_fill_percent':55.,'receipt_fraction':.5},b))
 def test_syntax(self):
  for folder in (W,W.parent/'night-search-tail-20260925-v1'):
   for p in folder.glob('*.py'):ast.parse(p.read_text())
 def test_failure_does_not_start_another_child(self):
  from batch import run_process,environment
  with tempfile.TemporaryDirectory() as directory:
   log=Path(directory)/'failure.log'
   with self.assertRaisesRegex(RuntimeError,'candidate_process_stopped'):
    run_process([sys.executable,'-c','raise SystemExit(7)'],environment(),log,5)
   self.assertTrue(log.exists())
 def test_failure_artifact_keeps_candidate_unconfirmed(self):
  from artifacts import export_comparison
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'baseline-replay.json').write_text('{"label":"baseline"}')
   report=dict(status='stopped',failure='guard',trials=[],
    best=dict(name='baseline',parameters=BASELINE,selection_status='baseline'),
    unconfirmed_candidate=dict(name='candidate'))
   export_comparison(root,report)
   self.assertEqual(json.loads((root/'selected-replay.json').read_text())['label'],'baseline')
   self.assertIn('not selected for deployment',(root/'comparison.md').read_text())
 def test_failed_baseline_never_starts_physics(self):
  import batch
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);tail=root/'tail';tail.mkdir();(tail/'report.json').write_text('{"status":"failed"}')
   work=root/'work';work.mkdir();(work/'manifest.json').write_text('{"bindings":{}}')
   with patch.object(batch,'W',work),patch.object(batch,'BASE_TAIL',tail),patch.object(batch,'RUN',root/'batch'),patch.object(batch,'run_process') as run,patch.object(sys,'argv',['batch','--run']):
    with self.assertRaisesRegex(ValueError,'baseline_unload_not_completed'):batch.main()
    run.assert_not_called();self.assertFalse((root/'batch').exists())
if __name__=='__main__':unittest.main()
