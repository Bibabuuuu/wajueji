"""CPU checks of scope, fail-closed classification and unattended scheduling."""
import ast,json,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import batch
from parameters import BASELINE,CANDIDATES,modify,validate
from failures import TrialRejected,classify,message
from paired_score import better

def score(fill=50.):
 return dict(eligible=True,working_fill_percent=fill,seconds=50.,command_total_variation=5.,receipt_fraction=.99,unload_validated=True)
def pair(fill=50.):return {k:score(fill) for k in ('first','second')}

class Tests(unittest.TestCase):
 def test_identity_and_scope(self):
  for action in np.random.default_rng(925).uniform(-1,1,(30,4)):
   self.assertTrue(np.array_equal(modify(action,'curl_load',.8,BASELINE),action))
   p={k:v for k,v in CANDIDATES[-1].items() if k!='name'}
   for phase in ('approach_contact','penetrate','lift','off_soil','CLEAR_PILE','ALIGN','DUMP'):
    self.assertTrue(np.array_equal(modify(action,phase,.8,p),action))
   self.assertTrue(np.array_equal(modify(action,'curl_load',.95,p),action))
   self.assertEqual(modify(action,'curl_load',.8,p)[0],action[0])
 def test_bounds(self):
  with self.assertRaises(ValueError):validate({**BASELINE,'boom_scale':.1})
 def test_paired_fill_not_one_source_tradeoff(self):
  b=pair();self.assertTrue(better(pair(55),b,b))
  self.assertFalse(better(dict(first=score(80),second=score(40)),b,b))
  with self.assertRaises(ValueError):better(dict(first=score()),b,b)
 def test_confirm_both_sources(self):
  self.assertTrue(batch.confirmation_passed(pair(55),pair(55),pair()))
  x=pair(55);x['second']['receipt_fraction']=.5
  self.assertFalse(batch.confirmation_passed(x,pair(55),pair()))
 def test_confirmation_rejects_fill_loss(self):
  self.assertFalse(batch.confirmation_passed(pair(51),pair(55),pair()))
 def test_syntax(self):
  for folder in (batch.W,batch.TAIL):
   for p in folder.glob('*.py'):ast.parse(p.read_text())
 def fixture(self,root,reason):
  p=root/'contact';(p/'physics').mkdir(parents=True)
  checkpoint=p/'physics/failure.npz';checkpoint.write_bytes(b'fixture; not physical evidence')
  batch.save(str(checkpoint)+'.json',dict(checkpoint_sha256=batch.sha(checkpoint)))
  batch.save(p/'report.json',dict(status='failed',failure=repr(ValueError(reason)),zero_step_arrays_exact={'fixture':True},source_unchanged=True))
  (p/'physics/frames.ndjson').write_text(json.dumps(dict(hard_guard_checks=dict(finite=True,joint_bounds=True)))+'\n')
  return p
 def test_guard_rejection_requires_evidence(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=self.fixture(root,'feedback_lateral_corridor_exceeded')
   self.assertEqual(classify(root)['stage'],'contact')
   (p/'physics/failure.npz').write_bytes(b'changed')
   self.assertIsNone(classify(root))
 def test_unknown_error_fatal(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.fixture(root,'program_bug');self.assertIsNone(classify(root))
 def test_nonfinite_fatal_even_with_action_error(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=self.fixture(root,'feedback_lateral_corridor_exceeded')
   (p/'physics/frames.ndjson').write_text('{"hard_guard_checks":{"finite":false}}\n')
   self.assertIsNone(classify(root))
 def test_bad_restore_fatal(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=self.fixture(root,'feedback_lateral_corridor_exceeded')
   r=batch.read(p/'report.json');r['zero_step_arrays_exact']['fixture']=False;batch.save(p/'report.json',r)
   self.assertIsNone(classify(root))
 def test_hard_action_contact_rejected_but_closure_fatal(self):
  for key,allowed in [('joint_bounds',True),('physical_closure',False),('hencky_material_state_safe',False)]:
   with tempfile.TemporaryDirectory() as d:
    root=Path(d);self.fixture(root,'cycle_hard_guard:'+str({key:False,'finite':True}))
    self.assertEqual(classify(root) is not None,allowed)
 def test_no_execution_of_exception_text(self):
  self.assertEqual(message("ValueError('x')"),'x')
  self.assertEqual(message("__import__('os').getcwd()"),"__import__('os').getcwd()")
 def test_failed_child_returns_nonzero(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(RuntimeError,'trial_process_failed'):
    batch.child([sys.executable,'-c','raise SystemExit(7)'],batch.environment(),Path(d)/'log',3,time.monotonic()+100)
 def run_mock(self,root,trial):
  w=root/'work';w.mkdir();run=root/'run'
  batch.save(w/'manifest.json',dict(status='authorized_current_physics_paired_search',bindings={},remaining_seconds=28000,prior_check_seconds=121,physics='fixture',candidates=batch.CANDIDATES[:2]))
  with patch.object(batch,'W',w),patch.object(batch,'RUN',run),patch.object(batch,'preparation_admission'),patch.object(batch,'trial',side_effect=trial):batch.run()
  return batch.read(run/'batch.json')
 def test_guard_candidate_skipped_next_pair_and_confirmation_run(self):
  calls=[]
  def trial(name,kind,params,deadline,**kw):
   calls.append((name,kind))
   if name==batch.CANDIDATES[0]['name']:raise TrialRejected(dict(reason='motion_guard',trial=name+'/'+kind))
   return score(50 if name=='baseline' else 60)
  with tempfile.TemporaryDirectory() as d:
   r=self.run_mock(Path(d),trial)
   self.assertEqual(r['status'],'completed');self.assertEqual(len(r['rejections']),1)
   self.assertNotIn((batch.CANDIDATES[0]['name'],'second'),calls)
   self.assertIn((batch.CANDIDATES[1]['name'],'second'),calls)
   self.assertIn(('confirm-best','second'),calls)
   self.assertEqual(r['best']['selection_status'],'confirmed_two_source_development_candidate')
 def test_unknown_error_stops_no_next_candidate(self):
  calls=[]
  def trial(name,kind,params,deadline,**kw):
   calls.append((name,kind))
   if name!='baseline':raise RuntimeError('unknown')
   return score()
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   with self.assertRaisesRegex(RuntimeError,'unknown'):self.run_mock(root,trial)
   r=batch.read(root/'run/batch.json');self.assertEqual(r['status'],'stopped');self.assertEqual(r['best']['name'],'baseline')
   self.assertFalse(any(n==batch.CANDIDATES[1]['name'] for n,k in calls))
 def test_confirmation_guard_keeps_baseline(self):
  def trial(name,kind,params,deadline,**kw):
   if name=='confirm-best':raise TrialRejected(dict(reason='motion_guard',trial=name+'/'+kind))
   return score(50 if name=='baseline' else 60)
  with tempfile.TemporaryDirectory() as d:
   r=self.run_mock(Path(d),trial);self.assertEqual(r['best']['name'],'baseline');self.assertTrue(r['confirmation_rejected'])

if __name__=='__main__':unittest.main()
