import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

import layered_47_rigid_frequency_candidate as candidate


class RigidFrequencyCandidateTest(unittest.TestCase):
    def test_frequency_mapping_preserves_wrench_impulse_and_half_diagnostics(self):
        for hz in (2400,3600):
            bridge=candidate.RigidFrequencyCommandBridge(SimpleNamespace(),hz)
            bridge.command=np.array([.12,-.0225,.0225,.0675])
            wrench={'bucket':{'force':[3,4,5]}}
            states=[dict(closure_m=.0004,forbidden_contact_count=1,contact_substeps={'a':2}),
                    dict(closure_m=.0001,forbidden_contact_count=0,contact_substeps={'a':3,'b':1})]
            with patch.object(candidate,'_bridge_step',side_effect=states) as step:
                result=bridge.step(1/60,hz//60,wrench)
            self.assertEqual(step.call_count,2)
            for call in step.call_args_list:
                self.assertEqual(call.args[1:3],(1/120,hz//120))
                np.testing.assert_array_equal(call.args[3],bridge.command)
                self.assertIs(call.args[4],wrench)
            self.assertAlmostEqual(sum(c.args[1] for c in step.call_args_list),1/60)
            self.assertEqual(result['closure_m'],.0004)
            self.assertEqual(result['forbidden_contact_count'],1)
            self.assertEqual(result['contact_substeps'],{'a':5,'b':1})

    def test_baseline_delegates_frozen_bridge(self):
        bridge=candidate.RigidFrequencyCommandBridge(SimpleNamespace(),4800)
        with patch.object(candidate,'subcycled_step',return_value={'baseline':True}) as step:
            self.assertEqual(bridge.step(1/60,80,{}),{'baseline':True})
        self.assertEqual(step.call_count,1)

    def test_bad_contract_or_command_rejected_without_advancing(self):
        bridge=candidate.RigidFrequencyCommandBridge(SimpleNamespace(),2400)
        with patch.object(candidate,'_bridge_step') as step:
            for dt,n in [(1/60,80),(1/30,80),(float('inf'),40)]:
                with self.assertRaises((ValueError,OverflowError)):
                    bridge.step(dt,n,{})
            for command in ([.12001,0,0,0],[0,float('nan'),0,0],[0,0,0]):
                bridge.command=np.asarray(command)
                with self.assertRaisesRegex(ValueError,'motion_command_limit'):
                    bridge.step(1/60,40,{})
            step.assert_not_called()

    def test_first_half_nonfinite_stops_before_second_half(self):
        bridge=candidate.RigidFrequencyCommandBridge(SimpleNamespace(),3600)
        with patch.object(candidate,'_bridge_step',return_value={'closure_m':float('nan')}) as step:
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                bridge.step(1/60,60,{})
            self.assertEqual(step.call_count,1)

    def test_low_frequency_cannot_enter_manual_or_ui_mode(self):
        for hz in (2400,3600):
            with self.assertRaisesRegex(ValueError,'headless_only'):
                candidate.validate_run_mode(hz,smoke=False,combined_smoke=False)
            candidate.validate_run_mode(hz,smoke=False,combined_smoke=True)
        candidate.validate_run_mode(4800,smoke=False,combined_smoke=False)
        with self.assertRaisesRegex(ValueError,'candidate_hz'):
            candidate.RigidFrequencyCommandBridge(None,1200)


if __name__=='__main__':
    unittest.main()
