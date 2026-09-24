import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest import TestCase,main
from unittest.mock import patch
import numpy as np
import candidate_runtime as c

class CandidateTests(TestCase):
    def row(self):
        return dict(frame=1,wall_seconds=.1,assignment_wall_seconds=.01,mpm_wall_seconds=.05,mujoco_wall_seconds=.04,
            bucket_mpm_contact_samples=1,rock_bucket_contact_substeps=2,external_rigid_support_substeps=0,forbidden_contact_count=0,
            closure_m=.0001,maximum_rock_linear_speed_m_s=.1,maximum_rock_angular_speed_rad_s=.2,maximum_rock_position_step_m=.001,
            finite=True,stable=True,hard_guard=True,hard_guard_checks={'safe':True},rigid_contact_substeps={'rock_bucket':2},
            hencky_material_guard={'safe':True},moving_bucket_sweep={'safe':True,'balance':1})
    def test_half_aggregation_does_not_hide_bad_first_half(self):
        a=self.row();b=self.row();a['hard_guard_checks']['safe']=False;a['hencky_material_guard']['safe']=False;a['closure_m']=.002
        r=c.aggregate([a,b]);self.assertFalse(r['hard_guard']);self.assertFalse(r['hencky_material_guard']['safe'])
        self.assertEqual(r['closure_m'],.002);self.assertEqual(r['rock_bucket_contact_substeps'],4)
        self.assertEqual(r['moving_bucket_sweep']['balance'],1)
        self.assertTrue(b['hencky_material_guard']['safe'])
    def runtime(self,path):
        r=c.CandidateRuntime.__new__(c.CandidateRuntime);r.output=path;r.frames=0;r.timings=[];r.coupling_hz=120;r.truck_enabled=False
        r.exchange_config={'probe':{'output_dt_s':1/120,'mujoco_internal_substeps':40}}
        r.bridge=NS(state={'mechanism_qpos':[0]*4,'contact_substeps':{}})
        r.guard=NS(prepare_frame=lambda:None,summarize=lambda s:{'safe':True})
        r.fixture=NS(state=NS(particle_q=NS(numpy=lambda:np.zeros((1,3))),particle_qd=NS(numpy=lambda:np.zeros((1,3)))))
        r.sweep=NS(finish_frame=lambda:{'safe':True});r.projector=NS(physical_bounds_ok=lambda q:True)
        r.exchange_ground=NS(measure=lambda q:{'safe':True})
        r.captured={'wrenches':{'bucket':{'force':np.zeros(3),'torque':np.zeros(3)}}}
        return r
    def run_fake(self,bad=False):
        with TemporaryDirectory() as d:
            r=self.runtime(Path(d));seen=[]
            def advance(f,b,config,frame):
                seen.append(b.state['mechanism_qpos'][0]);b.state['mechanism_qpos'][0]+=1
                self.assertEqual(config['probe']['output_dt_s'],1/120)
                self.assertEqual(config['probe']['mujoco_internal_substeps'],40)
                return self.row(),.1
            with patch.object(c.motion.probe.bench,'_advance_frame',advance),patch.object(c.manual,'compact_state',lambda s:copy.deepcopy(s)),patch.object(c.manual,'hard_guard_checks',lambda *a,**kw:{'safe':not bad}),patch.object(c.wp,'synchronize',lambda:None):
                if bad:
                    with self.assertRaises(RuntimeError):r.step(np.zeros(4))
                    self.assertEqual(seen,[0]);self.assertEqual(r.frames,0)
                else:
                    row=r.step(np.zeros(4));self.assertEqual(seen,[0,1]);self.assertEqual(r.frames,1)
                    self.assertEqual(row['after_state']['mechanism_qpos'][0],2)
                    self.assertEqual(len((r.output/'exchanges.ndjson').read_text().splitlines()),2)
    def test_second_half_uses_updated_rigid_state(self):self.run_fake()
    def test_first_half_guard_stops_without_second_half(self):self.run_fake(True)
    def test_clock_preserves_partial_frame(self):
        r=c.CandidateRuntime.__new__(c.CandidateRuntime);r.coupling_hz=120;r.fixture=object();r.bridge=NS(state={})
        r.arrays={'simulation_time_s':np.array([10.])};r.metadata={'controller_state':{'native_time_s':20.}}
        with patch.object(c,'controller_snapshot',lambda s:{'native_time_s':20.+1/120}),patch.object(c.motion.probe.snapshots,'capture_arrays',lambda f,s,t:t):
            self.assertAlmostEqual(r.capture(),10.+1/120)
if __name__=='__main__':main()
