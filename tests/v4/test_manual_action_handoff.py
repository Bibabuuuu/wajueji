"""Exercise the real env.step with scripted CPU-only state/physics doubles."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from manual_residual_sac_env import ManualResidualEnv, LIMITS_SI, OBS_SIZE


class HandoffTests(unittest.TestCase):
    def make_env(self, initial, modes, *, legacy=False, cursor=10., hold=None):
        temporary=TemporaryDirectory();self.addCleanup(temporary.cleanup)
        env=ManualResidualEnv.__new__(ManualResidualEnv)
        env.directory=Path(temporary.name)
        env.budget=SimpleNamespace(check=lambda:None)
        env.cursor=cursor;env.hold=hold;env.max_frames=1560
        env.commands=np.tile(LIMITS_SI*.2,(1170,1))
        env.waiting_action_mode='signed_residual_v2' if legacy else 'error_strength_v1'
        env.cap_fraction=.1;env.collect=True;env.transitions=[];env.zero_check=False
        env.previous_action=np.zeros(5);env.previous_applied=np.zeros(4)
        env.alignment=SimpleNamespace(measured=0.)
        def state(mode):
            return dict(trusted=True,reference_limit=env.cursor+(0 if mode=='wait' else 6),
                        wait_target_error_si=[.01,-.01,.01,-.01],endpoint_ready=False,
                        mode=mode)
        env.report=dict(status='running',alignment_wait_frames=0,last_alignment=state(initial))
        env.runtime=SimpleNamespace(frames=0,bridge=SimpleNamespace(state={'mechanism_qpos':[0.]*16}))
        env.runtime.projector=SimpleNamespace(project=lambda q,a:a.copy(),reason=None)
        requests=[]
        def advance(a):
            requests.append(a.copy());env.runtime.frames+=1
        env.runtime.step=advance
        env.ground=SimpleNamespace(measure=lambda q:{'safe':True})
        iterator=iter(modes)
        def alignment(stage):
            value=state(next(iterator));env.report['last_alignment']=value;return value
        env.alignment_state=alignment
        env.observation=lambda:np.zeros(OBS_SIZE,dtype=np.float32)
        env.finish=lambda status:env.report.update(status=status)
        return env,requests

    def rows(self,env):
        return [json.loads(s) for s in (env.directory/'actions.ndjson').read_text().splitlines()]

    def test_follow_to_wait_stops_until_next_decision(self):
        env,requests=self.make_env('follow',['follow','wait','wait','wait','wait','wait',
                                                      'wait','wait','wait','wait','wait','wait'])
        action=np.zeros(5);_,_,done,truncated,info=env.step(action)
        np.testing.assert_array_equal(requests[0],env.commands[10])
        np.testing.assert_array_equal(requests[1:],np.zeros((2,4)))
        self.assertEqual([r['action_invalidated'] for r in self.rows(env)],[False,True,True])
        self.assertFalse(done or truncated);self.assertEqual(info['physical_frames'],3)
        self.assertEqual(len(env.transitions),1)
        np.testing.assert_array_equal(env.transitions[0][2],action)
        env.step(np.zeros(5))
        np.testing.assert_allclose(requests[3:],np.tile(np.array([1,-1,1,-1])*LIMITS_SI*.05,(3,1)))
        self.assertTrue(all(not r['action_invalidated'] for r in self.rows(env)[3:]))
        self.assertEqual(env.runtime.frames,6);self.assertEqual(len(env.transitions),2)

    def test_wait_to_follow_cannot_reinterpret_stop_or_full_or_pace(self):
        for strength_action in [-1.,1.]:
            env,requests=self.make_env('wait',['wait','follow','follow','follow','follow','follow'])
            env.step(np.r_[np.full(4,strength_action),1.])
            np.testing.assert_array_equal(requests[1:],env.commands[10:12])
            rows=self.rows(env)
            self.assertEqual([r['pace'] for r in rows],[0.,1.,1.])
            self.assertEqual([r['action_invalidated'] for r in rows],[False,True,True])
            for row in rows[1:]:self.assertEqual(row['residual_si'],[0.]*4)

    def test_return_to_original_mode_does_not_revive_action(self):
        env,requests=self.make_env('follow',['follow','wait','wait','follow','follow','follow'])
        env.step(np.ones(5))
        rows=self.rows(env)
        self.assertEqual([r['action_invalidated'] for r in rows],[False,True,True])
        self.assertEqual(rows[1]['request_si'],[0.]*4)
        self.assertEqual(rows[2]['residual_si'],[0.]*4)
        self.assertEqual(rows[2]['pace'],1.)

    def test_after_observation_transition_not_lost_when_before_changes_back(self):
        env,_=self.make_env('follow',['follow','wait','follow','follow','follow','follow'])
        env.step(np.ones(5))
        self.assertEqual([r['action_invalidated'] for r in self.rows(env)],[False,True,True])

    def test_change_before_first_frame_invalidates_entire_action(self):
        env,requests=self.make_env('follow',['wait']*6)
        env.step(np.zeros(5))
        np.testing.assert_array_equal(requests,np.zeros((3,4)))
        self.assertTrue(all(r['action_invalidated'] for r in self.rows(env)))

    def test_zero_strength_same_mode_and_endpoint_wait_are_zero(self):
        for cursor in [10.,1170.]:
            env,requests=self.make_env('wait',['wait']*6,cursor=cursor)
            env.step(np.r_[-np.ones(4),0.])
            np.testing.assert_array_equal(requests,np.zeros((3,4)))
            self.assertEqual(env.cursor,cursor)

    def test_hold_commands_stay_zero(self):
        env,requests=self.make_env('wait',['wait']*6,cursor=1170.,hold=0)
        env.step(np.ones(5))
        np.testing.assert_array_equal(requests,np.zeros((3,4)))
        self.assertEqual(env.hold,3)
        self.assertTrue(all(r['execution_mode']=='hold' for r in self.rows(env)))

    def test_legacy_keeps_existing_cross_mode_behavior(self):
        env,_=self.make_env('wait',['wait','follow','follow','follow','follow','follow'],legacy=True)
        env.step(np.ones(5))
        rows=self.rows(env)
        self.assertTrue(all(not r['action_invalidated'] for r in rows))
        np.testing.assert_allclose(rows[1]['residual_si'],LIMITS_SI*.1)
        self.assertEqual(rows[1]['pace'],1.5)


if __name__=='__main__':unittest.main()
