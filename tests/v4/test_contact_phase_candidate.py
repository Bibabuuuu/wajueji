import unittest

from contact_phase_candidate import ContactPhaseCandidate, ContactPhaseConfig


class ContactPhaseCandidateTest(unittest.TestCase):
    def router(self, angle=20):
        return ContactPhaseCandidate(ContactPhaseConfig(-5.881154296396837), initial_mouth_angle_deg=angle)

    def push(self, router, force=0, rock=0, angle=20):
        return router.observe_end(frame_end=router.last_frame_end+1,
                                  soil_force_n=force, rock_contact_substeps=rock,
                                  mouth_angle_deg=angle)

    def test_noise_threshold_equality_and_isolated_spikes(self):
        r=self.router()
        for force in (7e-8, 10, 100, 0, 100, 100, 0):
            self.assertIsNone(self.push(r, force)['event'])
        self.assertEqual(r.phase, 'approach_contact')

    def test_confirmation_only_changes_next_decision(self):
        r=self.router()
        for _ in range(2):self.push(r, 11)
        self.assertEqual(r.phase_for_decision(3), 'approach_contact')
        result=self.push(r, 11)
        self.assertEqual(result['event']['evidence'][0]['onset_frame_end'], 1)
        self.assertEqual(result['event']['confirmation_frame_end'], 3)
        self.assertEqual(r.phase_for_decision(4), 'penetrate')
        with self.assertRaises(ValueError):r.phase_for_decision(3)

    def test_stone_only_and_sources_do_not_mix_streaks(self):
        r=self.router()
        self.push(r, force=100)
        self.push(r, rock=1)
        self.push(r, force=100)
        self.push(r, rock=1)
        self.assertIsNone(self.push(r, rock=1)['event'])
        event=self.push(r, rock=1)['event']
        self.assertEqual(event['evidence'], [dict(kind='rock_contact_pair_presence', onset_frame_end=4)])

    def test_pause_force_loss_and_angle_recoil_do_not_rewind(self):
        r=self.router()
        for _ in range(3):self.push(r, 20)
        for _ in range(6):self.push(r)
        self.assertEqual(r.phase, 'penetrate')
        self.assertEqual(self.push(r, angle=-6)['event']['transition'], 'penetrate_to_curl_load')
        for _ in range(5):self.assertIsNone(self.push(r, angle=0)['event'])
        self.assertEqual(r.phase, 'curl_load')

    def test_airborne_crossing_is_not_latched_and_below_at_contact_flagged(self):
        r=self.router()
        self.assertIsNone(self.push(r, angle=-6)['event'])
        for _ in range(3):out=self.push(r, force=20, angle=-6)
        self.assertEqual(r.phase, 'penetrate')
        self.assertTrue(out['curl_boundary_unresolved'])
        self.assertIsNone(self.push(r, angle=-7)['event'])
        self.push(r, angle=0)
        self.assertEqual(self.push(r, angle=-6)['event']['transition'], 'penetrate_to_curl_load')

    def test_invalid_input_does_not_change_state(self):
        for bad in (dict(frame_end=2), dict(soil_force_n=float('nan')),
                    dict(mouth_angle_deg=float('inf')), dict(rock_contact_substeps=-1),
                    dict(rock_contact_substeps=81), dict(rock_contact_substeps=True)):
            r=self.router();before=r.__dict__.copy()
            row=dict(frame_end=1, soil_force_n=0, rock_contact_substeps=0, mouth_angle_deg=20);row.update(bad)
            with self.assertRaises(ValueError):r.observe_end(**row)
            self.assertEqual(r.__dict__,before)

    def test_prefix_invariance_and_new_episode_reset(self):
        a=self.router();b=self.router()
        prefix=[(0,20),(20,20),(20,10),(20,0),(0,-6)]
        for force,angle in prefix:self.assertEqual(self.push(a,force,angle=angle),self.push(b,force,angle=angle))
        phase=b.phase_for_decision(6)
        for _ in range(5):self.push(a,100,angle=-90)
        self.assertEqual(phase,'curl_load')
        self.assertEqual(self.router().phase_for_decision(1),'approach_contact')

    def test_invalid_config(self):
        for values in (dict(soil_force_threshold_n=0),dict(soil_confirmation_frames=0),
                       dict(rock_confirmation_frames=True),dict(curl_angle_threshold_deg=float('nan'))):
            args=dict(curl_angle_threshold_deg=-5.88);args.update(values)
            with self.assertRaises(ValueError):ContactPhaseConfig(**args)


if __name__ == '__main__':
    unittest.main()
