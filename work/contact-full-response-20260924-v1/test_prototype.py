import copy,json,time,unittest
import numpy as np
import warp as wp
from predict_response import W,OUT,PRIOR,predict,errors
from shadow import Shadow
from full_response_operator import patch_bucket

class PrototypeTests(unittest.TestCase):
    def test_response_both_probe_amplitudes(self):
        raw=json.loads((PRIOR/'exact-response.json').read_text());review=json.loads((W/'response-review.json').read_text())
        for context,result in zip(raw['contexts'],review['contexts'],strict=True):
            C=predict(result['coefficients'])
            for ref in context['matrices'].values():
                e=errors(C,np.array(ref));self.assertLess(e['relative'],.2);self.assertLess(max(e['column_relative']),.3)
            self.assertLess(np.max(np.abs(C-C.T)),1e-12);self.assertGreater(np.linalg.eigvalsh(C).min(),-1e-12)
    def test_shadow_restores_and_predicts_frozen_zero_cases(self):
        raw=json.loads((PRIOR/'exact-response.json').read_text());shadow=Shadow(OUT);timings=[]
        try:
            for c in (raw['contexts'][0],raw['contexts'][2],raw['contexts'][0]):
                trial=c['trials'][0];initial=copy.deepcopy(trial['states'][0]);before=copy.deepcopy(initial)
                C,v,detail=shadow.forecast(initial,c['command']);self.assertEqual(initial,before)
                for actual,expected in zip(detail['predicted_states'],trial['states'][1:],strict=True):
                    for key in ('mechanism_qpos','mechanism_qvel','rocks','controller_target','controller_target_velocity','cycle_native_warmstart','cycle_native_qacc','cycle_native_ctrl','bucket_integral_effort','bucket_active_curl_effort'):
                        self.assertEqual(actual[key],expected[key],key)
                expected=np.array(trial['exact_endpoint_twist_scaled_at_initial_wrench_origin']);scaled=v.copy();scaled[3:]*=.6
                np.testing.assert_array_equal(scaled,expected)
                timings.append(detail['wall_seconds'])
        finally:shadow.close()
        (W/'shadow-verification.json').write_text(json.dumps(dict(three_predictions_exact=True,source_object_unchanged=True,MPM_steps=0,native_steps=240,wall_seconds=timings),indent=2)+'\n')
    def test_known_pending_impulse_predicted_once(self):
        raw=json.loads((PRIOR/'exact-response.json').read_text());c=raw['contexts'][0];trial=next(t for t in c['trials'] if t['label'].endswith('a100-axis0-sign1'))
        pending=dict(reaction_impulse_ns=[100/60,0,0],reaction_angular_impulse_nms=[0,0,0],point_world_m=c['origin_world_m'])
        old=copy.deepcopy(pending);shadow=Shadow(OUT)
        try:
            _,_,detail=shadow.forecast(trial['states'][0],c['command'],pending)
            for actual,expected in zip(detail['predicted_states'],trial['states'][1:],strict=True):
                self.assertEqual(actual['mechanism_qpos'],expected['mechanism_qpos']);self.assertEqual(actual['mechanism_qvel'],expected['mechanism_qvel'])
            self.assertEqual(pending,old)
        finally:shadow.close()
    def test_warp_operator_work_and_impulse_mapping(self):
        wp.init();wp.set_device('cuda:0');rng=np.random.default_rng(142);A=rng.normal(size=(6,4));C=A@A.T*.003
        points=np.array([[.3,-.4,1.2],[.8,.1,.5],[-.1,.7,1.]],np.float32);origin=np.array([.1,.2,.3],np.float32);ids=np.array([7,3,7],np.int32)
        rows=wp.full(9,-2,dtype=int);cols=wp.full(9,-2,dtype=int);J=wp.zeros(9,dtype=wp.mat33);IJ=wp.zeros(9,dtype=wp.mat33)
        cq=wp.array(np.r_[origin,[0,0,0,1]].reshape(1,7),dtype=wp.transform);blocks=np.array([C[:3,:3],C[:3,3:],C[3:,:3],C[3:,3:]],np.float32);vol=.001
        wp.launch(patch_bucket,dim=3,inputs=[wp.array(points,dtype=wp.vec3),cq,wp.array(ids,dtype=int),7,0,vol,wp.array(blocks,dtype=wp.mat33),rows,cols,J,IJ]);wp.synchronize()
        j=J.numpy();ij=IJ.numpy();r=rows.numpy();co=cols.numpy();p=rng.normal(size=(3,3));total=np.zeros(6)
        for i in (0,2):total+=np.r_[p[i],np.cross(points[i]-origin,p[i])]*vol
        perm=np.r_[3:6,0:3];indirect=np.zeros(6)
        for i in (0,2):
            np.testing.assert_array_equal(r[3*i:3*i+2],[i,i]);np.testing.assert_array_equal(co[3*i:3*i+2],[0,1]);indirect+=np.r_[ij[3*i]@p[i],ij[3*i+1]@p[i]]
        np.testing.assert_allclose(indirect,(C@total)[perm],rtol=2e-6,atol=1e-11)
        for i in (0,2):
            nodal=j[3*i]@indirect[:3]+j[3*i+1]@indirect[3:]
            v=C@total;expected=v[:3]+np.cross(v[3:],points[i]-origin)
            np.testing.assert_allclose(nodal,expected,rtol=2e-6,atol=1e-11)
        np.testing.assert_array_equal(r[3:6],[-2]*3);np.testing.assert_array_equal(j[3:6],np.zeros((3,3,3)))
        twist=rng.normal(size=6)
        work=sum(np.dot(p[i]*vol,twist[:3]+np.cross(twist[3:],points[i]-origin)) for i in (0,2))
        self.assertAlmostEqual(work,np.dot(total,twist),places=9)
        self.assertGreaterEqual(float(total@C@total),0)

if __name__=='__main__':unittest.main(verbosity=2)
