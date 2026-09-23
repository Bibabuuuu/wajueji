from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/v4'))
import layered_mound_checkpoint_runtime as runtime
from layered_47_optimized_bridge import (
    optimized_bridge, CANDIDATE_SHA256, instrument_drive_load)
import run_layered_47_optimized_practice as practice


class OptimizedIntegrationTest(unittest.TestCase):
    def test_bridge_scope_and_restoration_after_exception(self):
        original=runtime.render_capacity_bridge
        source = '''#include "inertia_controller.h"
    std::array<double, 3> bucket_interval_reference_point{};
      array("controller_target_velocity", controller.target_velocity.data(), 4);
        const auto result = feedforward.Evaluate(
            data->qpos, controller, controller.target_velocity, acceleration);
        const double before = data->time;
        mj_step(model.get(), data.get());
'''
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runtime,'render_capacity_bridge',return_value=(source,{})) as baseline:
                with self.assertRaisesRegex(RuntimeError,'test_error'):
                    with optimized_bridge(ROOT/'evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8',Path(directory)/'build') as provenance:
                        code,manifest=runtime.render_capacity_bridge(None,None)
                        self.assertIn('inertia_candidate.h',code)
                        self.assertIn('equivalent_drive_load', code)
                        self.assertEqual(manifest['optimized_inertia']['candidate_header_sha256'],CANDIDATE_SHA256)
                        self.assertTrue(
                            provenance['drive_load_observation_only'])
                        self.assertFalse(
                            provenance['external_load_compensation_applied'])
                        self.assertIn('generated_bridge_sha256',provenance)
                        raise RuntimeError('test_error')
                self.assertIs(runtime.render_capacity_bridge,baseline)
        self.assertIs(runtime.render_capacity_bridge,original)

    def test_drive_load_instrumentation_is_observation_only(self):
        source = '''    std::array<double, 3> bucket_interval_reference_point{};
      array("controller_target_velocity", controller.target_velocity.data(), 4);
        const auto result = feedforward.Evaluate(
            data->qpos, controller, controller.target_velocity, acceleration);
        const double before = data->time;
        mj_step(model.get(), data.get());
'''
        result = instrument_drive_load(source)
        self.assertIn('equivalent_drive_load', result)
        self.assertIn('data->qfrc_applied[dof]', result)
        self.assertNotIn('external_load_compensation', result)
        self.assertNotIn('data->ctrl[controller.hold_actuator[axis]] -=', result)

    def test_neutral_rearm_after_pause_and_focus_loss(self):
        gate=practice.InputGate()
        np.testing.assert_array_equal(gate.command([1,0,0,0],[0]*4),[0]*4)
        gate.command([0]*4,[0]*4)
        self.assertGreater(gate.command([1,0,0,0],[0]*4)[0],0)
        for flag in ({'paused':True},{'focused':False}):
            np.testing.assert_array_equal(gate.command([1,0,0,0],[0]*4,**flag),[0]*4)
            np.testing.assert_array_equal(gate.command([1,0,0,0],[0]*4),[0]*4)
            gate.command([0]*4,[0]*4)

    def test_estop_and_disconnect_are_latched(self):
        for flag,reason in [({'estop':True},'manual_emergency_stop'),({'connected':False},'gamepad_disconnect')]:
            gate=practice.InputGate()
            gate.command([0]*4,[0]*4)
            np.testing.assert_array_equal(gate.command([1]*4,[0]*4,**flag),[0]*4)
            self.assertEqual(gate.stop_reason,reason)
            np.testing.assert_array_equal(gate.command([1]*4,[0]*4),[0]*4)

    def test_single_axis_and_rate_caps(self):
        gate=practice.InputGate()
        gate.command([0]*4,[0]*4)
        command=gate.command([.5,-1,.3,.7],[0]*4)
        np.testing.assert_array_equal(command,[0,-.003,0,0])

    def test_envelope_blocks_outward_not_return(self):
        gate=practice.InputGate()
        gate.command([0]*4,[0]*4)
        np.testing.assert_array_equal(gate.command([1,0,0,0],[.025,0,0,0]),[0]*4)
        self.assertLess(gate.command([-1,0,0,0],[.025,0,0,0])[0],0)

    def test_invalid_input_rejected(self):
        for value in ([float('nan'),0,0,0],[1.01,0,0,0],[0,0,0]):
            with self.assertRaisesRegex(ValueError,'invalid_input'):
                practice.InputGate().command(value,[0]*4)

    def test_bidirectional_smoke_is_bounded(self):
        phases=list(practice.smoke_schedule())
        self.assertEqual(len(phases),17)
        self.assertEqual(sum(p[3]==1 for p in phases),4)
        self.assertEqual(sum(p[3]==-1 for p in phases),4)
        np.testing.assert_array_equal(sum((p[1] for p in phases),np.zeros(4)),np.zeros(4))

    def test_normal_rates_do_not_use_millimetre_test_envelope(self):
        gate=practice.InputGate('normal')
        gate.command([0]*4,[0]*4)
        for axis,rate in enumerate([.12,.0225,.0225,.0675]):
            action=np.zeros(4); action[axis]=1
            command=gate.command(action,[.2,.03,.03,.03])
            self.assertEqual(command[axis],rate)
            self.assertEqual(np.count_nonzero(command),1)

    def test_normal_schedule_boom_retracts_first(self):
        phases=list(practice.smoke_schedule('normal'))
        self.assertEqual(len(phases),17)
        self.assertEqual([p[3] for p in phases if p[2]==1 and p[3]],[-1,1])
        np.testing.assert_array_equal(sum((p[1] for p in phases),np.zeros(4)),np.zeros(4))

    def test_normal_keyboard_chord_and_partial_release(self):
        keyboard=practice.KeyboardFourAxisState()
        gate=practice.InputGate('normal')
        gate.command(keyboard.action(window_focused=True),[0]*4)
        keyboard.press('K'); keyboard.press('O')
        np.testing.assert_array_equal(gate.command(keyboard.action(window_focused=True),[0]*4),[0,-.0225,.0225,0])
        keyboard.release('K')
        np.testing.assert_array_equal(gate.command(keyboard.action(window_focused=True),[0]*4),[0,0,.0225,0])
        keyboard.release('O')
        np.testing.assert_array_equal(gate.command(keyboard.action(window_focused=True),[0]*4),[0]*4)

    def test_normal_gamepad_both_sticks_and_independent_scales(self):
        config=practice.load_manual_input_config(ROOT/'configs/v4/manual-input-v1.json')
        pad=practice.GamepadFourAxisState(config)
        gate=practice.InputGate('normal')
        gate.command(pad.action(window_focused=True),[0]*4)
        pad.set_stick('leftstick',.5,-.8)
        pad.set_stick('rightstick',-.7,.9)
        action=np.asarray(pad.action(window_focused=True))
        self.assertTrue(np.all(action!=0))
        np.testing.assert_array_equal(gate.command(action,[0]*4),action*practice.NORMAL_RATES)
        np.testing.assert_array_equal(gate.command([1,-1,1,-1],[0]*4),practice.NORMAL_RATES*[1,-1,1,-1])
        pad.set_stick('leftstick',0,0)
        self.assertEqual(np.count_nonzero(gate.command(pad.action(window_focused=True),[0]*4)),2)

    def test_normal_all_axis_pause_focus_estop_disconnect(self):
        for flag in ({'paused':True},{'focused':False},{'estop':True},{'connected':False}):
            gate=practice.InputGate('normal')
            gate.command([0]*4,[0]*4)
            self.assertEqual(np.count_nonzero(gate.command([1]*4,[0]*4)),4)
            np.testing.assert_array_equal(gate.command([1]*4,[0]*4,**flag),[0]*4)
            np.testing.assert_array_equal(gate.command([1]*4,[0]*4),[0]*4)

    def test_combined_smoke_has_no_net_requested_displacement(self):
        phases=list(practice.combined_smoke_schedule())
        self.assertEqual(len(phases),9)
        np.testing.assert_array_equal(phases[1][1],[0,-1,1,0])
        self.assertEqual(np.count_nonzero(phases[5][1]),4)
        np.testing.assert_array_equal(sum((action for _,action in phases),np.zeros(4)),np.zeros(4))

    def test_preview_blocks_clearance_and_joint_bounds_without_state_mutation(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        q=np.zeros(19)
        chain=SimpleNamespace(limits={9:np.array([-.1,.1])},q=None)
        chain.advance=Mock(return_value=(q.copy(),0))
        pose=np.eye(4); pose[2,3]=1
        assembly=SimpleNamespace(assemble=Mock(return_value=(None,{'bucket':pose})))
        guard=practice.NoncontactGuard.__new__(practice.NoncontactGuard)
        guard.chain,guard.assembly,guard.vertices,guard.reason=chain,assembly,np.zeros((1,3)),None
        command=np.array([.12,-.0225,.0225,.0675])
        np.testing.assert_array_equal(guard.project(q,command,.8),command)
        self.assertEqual(chain.advance.call_count,3)
        for call in chain.advance.call_args_list:
            np.testing.assert_array_equal(call.args[0],command)
        np.testing.assert_array_equal(guard.project(q,command,.84),np.zeros(4))
        self.assertEqual(guard.reason,'noncontact_clearance_preview')
        bad=q.copy();bad[9]=.11;chain.advance.return_value=(bad,0)
        np.testing.assert_array_equal(guard.project(q,command,0),np.zeros(4))
        self.assertEqual(guard.reason,'joint_working_range')
        np.testing.assert_array_equal(q,np.zeros(19))
        chain.advance.side_effect=ValueError('joint_margin')
        np.testing.assert_array_equal(guard.project(q,command,0),np.zeros(4))
        self.assertEqual(guard.reason,'joint_margin')


if __name__=='__main__':
    unittest.main()
