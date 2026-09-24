import copy
import unittest

from contact_phase_candidate import ContactPhaseConfig
from contact_phase_completion import CompletionConfig, FivePhaseCandidate


class PhaseCompletionTest(unittest.TestCase):
    def router(self):
        return FivePhaseCandidate(ContactPhaseConfig(-5), CompletionConfig(
            lift_motion_window_frames=2,lift_confirmation_frames=3,
            lift_window_origin_rise_m=.01,lift_window_boom_extension_m=.0005,
            lift_total_origin_rise_m=.08),initial_mouth_angle_deg=20)

    def push(self,r,*,z=0,boom=0,angle=-6,gap=None,force=20):
        f=r.last_frame_end+1
        return r.observe_end(frame_end=f,soil_force_n=force,rock_contact_substeps=0,
            mouth_angle_deg=angle,actual_boom_cylinder_m=boom,bucket_origin_z_m=z,
            remaining_soil_gap_m=gap,gap_frame_end=f if gap is not None else None)

    def curl(self,r):
        for _ in range(3):self.push(r,angle=20)
        self.push(r)
        self.assertEqual(r.phase,'curl_load')

    def lift(self,r):
        self.curl(r)
        for i in range(1,5):out=self.push(r,z=i*.025,boom=i*.001)
        self.assertEqual(r.phase,'lift')
        return out

    def test_no_skipping_curl_or_lift(self):
        r=self.router()
        for i in range(1,20):self.push(r,angle=20,z=i*.1,boom=i*.01,gap=1)
        self.assertEqual(r.phase,'penetrate')
        self.push(r,z=2,boom=.2)
        self.assertEqual(r.phase,'curl_load')
        for _ in range(10):self.push(r,z=2,boom=.2,gap=1)
        self.assertEqual(r.phase,'curl_load')

    def test_sustained_motion_and_next_decision(self):
        r=self.router();out=self.lift(r)
        self.assertEqual(out['phase_before_observation'],'curl_load')
        self.assertEqual(out['event']['confirmation_frame_end'],8)
        self.assertEqual(r.phase_for_decision(9),'lift')
        self.assertEqual(out['off_soil_clear_frames'],0)

    def test_small_adjustment_pause_and_restart(self):
        r=self.router();self.curl(r)
        for i in range(1,5):self.push(r,z=i*.011,boom=i*.001)
        self.assertEqual(r.phase,'curl_load') # persistent but less than 8cm
        for _ in range(4):self.push(r,z=.044,boom=.004)
        self.assertEqual(r.lift_streak,0)
        for i in range(1,5):self.push(r,z=.044+i*.025,boom=.004+i*.001)
        self.assertEqual(r.phase,'lift')

    def test_bucket_rotation_or_command_without_actual_boom_not_lift(self):
        r=self.router();self.curl(r)
        for i in range(1,20):self.push(r,z=i*.1,boom=0)
        self.assertEqual(r.phase,'curl_load')
        r=self.router();self.curl(r)
        for i in range(1,20):self.push(r,z=0,boom=i*.01)
        self.assertEqual(r.phase,'curl_load')

    def test_gap_equality_five_fresh_frames_and_reset(self):
        r=self.router();self.lift(r)
        for gap in (.003,.003,.002999,.003,None,.003,.003,.003,.003):
            self.assertIsNone(self.push(r,gap=gap)['event'])
        out=self.push(r,gap=.003)
        self.assertEqual(out['phase_before_observation'],'lift')
        self.assertEqual(r.phase,'off_soil')
        self.assertEqual(out['event']['evidence'][0]['consecutive_frames'],5)
        self.push(r,gap=0)
        self.assertEqual(r.phase,'off_soil')

    def test_invalid_stale_or_missing_frame_atomic(self):
        for bad in (dict(actual_boom_cylinder_m=float('nan')),
                    dict(remaining_soil_gap_m=.004,gap_frame_end=1),
                    dict(remaining_soil_gap_m=float('inf'),gap_frame_end=9),
                    dict(frame_end=10),dict(soil_force_n=-1)):
            r=self.router();self.lift(r);before=copy.deepcopy(r.__dict__)
            row=dict(frame_end=9,soil_force_n=0,rock_contact_substeps=0,mouth_angle_deg=-6,
                actual_boom_cylinder_m=.1,bucket_origin_z_m=.1);row.update(bad)
            with self.assertRaises(ValueError):r.observe_end(**row)
            self.assertEqual(r.contact.__dict__,before.pop('contact').__dict__)
            actual=r.__dict__.copy();actual.pop('contact');self.assertEqual(actual,before)

    def test_prefix_consistency_and_reset(self):
        a=self.router();b=self.router();self.assertEqual(self.lift(a),self.lift(b))
        phase=b.phase_for_decision(9)
        for _ in range(5):self.push(a,gap=.003)
        self.assertEqual(phase,'lift')
        self.assertEqual(self.router().phase,'approach_contact')

    def test_invalid_config(self):
        for kw in (dict(lift_confirmation_frames=0),dict(off_soil_confirmation_frames=True),
                   dict(off_soil_gap_m=float('nan')),dict(lift_total_origin_rise_m=0)):
            with self.assertRaises(ValueError):CompletionConfig(**kw)


if __name__=='__main__':unittest.main()
