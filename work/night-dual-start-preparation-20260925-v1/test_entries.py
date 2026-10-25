import ast,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from paired_score import assess,better
import check_entries as entry
class Tests(unittest.TestCase):
 def setUp(self):
  self.baseline={n:dict(eligible=True,working_fill_percent=f,seconds=70.,command_total_variation=90.,receipt_fraction=.999) for n,f in [('first',45.53),('second',30.21)]}
 def candidate(self,first,second):
  return {n:{**s,'working_fill_percent':s['working_fill_percent']+g} for (n,s),g in zip(self.baseline.items(),(first,second))}
 def test_reject_tradeoff(self):
  self.assertFalse(assess(self.candidate(-5,20),self.baseline)['eligible'])
 def test_both_improve(self):
  self.assertTrue(better(self.candidate(4,8),self.baseline,self.baseline))
 def test_invalid_source_never_ranked(self):
  candidate=self.candidate(5,20);candidate['first']['eligible']=False
  self.assertFalse(assess(candidate,self.baseline)['eligible'])
 def test_missing_source(self):
  with self.assertRaises(ValueError):assess({'second':self.baseline['second']},self.baseline)
 def test_nonfinite_metric(self):
  candidate=self.candidate(5,8);candidate['second']['seconds']=float('nan')
  with self.assertRaises(ValueError):assess(candidate,self.baseline)
 def test_worse_source_priority(self):
  self.assertTrue(better(self.candidate(5,5),self.candidate(0,20),self.baseline))
 def test_syntax(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
 def test_changed_input_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'input';p.write_text('original');manifest={'bindings':{str(p):entry.sha(p)}}
   entry.verify(manifest);p.write_text('changed')
   with self.assertRaisesRegex(ValueError,'frozen_entry_input_changed'):entry.verify(manifest)
 def test_failed_zero_never_starts_120(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'manifest.json').write_text('{"bindings":{}}')
   with patch.object(entry,'W',root),patch.object(entry,'RUN',root/'run'),patch.object(entry,'first_source'),patch.object(entry,'second_evidence',return_value={}),patch.object(entry,'trial',side_effect=RuntimeError('zero_failure')) as trial,patch('sys.argv',['entry','--check-first']):
    with self.assertRaisesRegex(RuntimeError,'zero_failure'):entry.main()
    trial.assert_called_once_with(0)
    self.assertEqual(json.loads((root/'run/entry-review.json').read_text())['status'],'stopped')
    with self.assertRaisesRegex(ValueError,'no automatic retry'):entry.main()
if __name__=='__main__':unittest.main()
