"""CPU-only scope, ranking and budget-contract checks."""
import unittest,sys,ast,tempfile,json
from pathlib import Path
import numpy as np
from parameters import BASELINE,CANDIDATES,modify,validate
from score import better,digging_metrics
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
 def test_split_baseline_time_and_transition_variation(self):
  acts=[dict(applied_command_si=[0.,0.,0.,0.])]*2
  prefix=[dict(command=[1.,0.,0.,0.])]*439
  tail=[dict(command=[0.,0.,0.,0.])]*8
  seconds,variation=digging_metrics(acts,prefix,tail,6,np.ones(4))
  self.assertAlmostEqual(seconds,447/60);self.assertEqual(variation,2.)
 def test_three_segment_replay(self):
  from artifacts import export_replay
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);parts=[]
   for name in ('contact','prefix','tail'):
    p=root/name/'display';p.mkdir(parents=True);(p/'reconstruction.json').write_text('{"cache_sha256":{}}');parts.append(p.parent)
   export_replay(root/'replay.json',parts[0],parts[2],'baseline','test',prefix=parts[1])
   self.assertEqual([x['path'] for x in json.loads((root/'replay.json').read_text())['sources']],[str(p/'display') for p in parts])
 def test_syntax(self):
  for folder in (W,W.parent/'night-search-tail-20260925-v2'):
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
   with patch.object(batch,'W',work),patch.object(batch,'baseline_readiness',return_value=['baseline_running']),patch.object(batch,'RUN',root/'batch'),patch.object(batch,'run_process') as run,patch.object(sys,'argv',['batch','--run']):
    with self.assertRaisesRegex(ValueError,'baseline_not_admitted'):batch.main()
    run.assert_not_called();self.assertFalse((root/'batch').exists())
 def test_check_only_never_starts_candidates(self):
  import batch
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);work=root/'work';work.mkdir()
   (work/'manifest.json').write_text('{"bindings":{},"status":"prepared_not_launched"}')
   with patch.object(batch,'W',work),patch.object(batch,'RUN',root/'run'),patch.object(batch,'baseline_readiness',return_value=[]),patch.object(batch,'precheck',return_value={'status':'passed'}) as check,patch.object(batch,'trial') as trial,patch.object(sys,'argv',['batch','--check-only']):
    batch.main();check.assert_called_once();trial.assert_not_called()
    self.assertFalse((root/'run/batch.json').exists())
 def test_precheck_reuse_requires_unchanged_evidence(self):
  import batch
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'manifest.json').write_text('{}');evidence=root/'evidence';evidence.write_text('original')
   review=dict(status='passed',manifest_sha256=batch.sha(root/'manifest.json'),bindings={str(evidence):batch.sha(evidence)},wall_seconds=60)
   batch.save(root/'precheck.json',review)
   with patch.object(batch,'W',root),patch.object(batch,'RUN',root),patch.object(batch,'run_process') as run:
    self.assertEqual(batch.precheck({}),review);run.assert_not_called()
    evidence.write_text('changed')
    with self.assertRaisesRegex(ValueError,'precheck_evidence_changed'):batch.precheck({})
    review['status']='stopped';batch.save(root/'precheck.json',review)
    with self.assertRaisesRegex(ValueError,'no automatic retry'):batch.precheck({})
    run.assert_not_called()
if __name__=='__main__':unittest.main()
