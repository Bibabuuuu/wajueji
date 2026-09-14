import io
from types import SimpleNamespace
import unittest
from perception.core import LatestSchedule
try:
    from perception.online import OnlinePerception, CAMERA_POSES
except ImportError:
    OnlinePerception = None


@unittest.skipIf(OnlinePerception is None,'Warp runtime required')
class OnlineFailureTests(unittest.TestCase):
    def test_sensor_and_storage_failure_do_not_escape_into_control(self):
        observer=OnlinePerception.__new__(OnlinePerception)
        def unavailable():raise ValueError('sensor unavailable')
        observer.soil=SimpleNamespace(time=2.,state=SimpleNamespace(particle_q=SimpleNamespace(numpy=unavailable)))
        observer.latest=None;observer.schedule=LatestSchedule();observer.error='';observer.failures=0
        observer.sequence=0;observer.poses=CAMERA_POSES;observer.trace=io.StringIO()
        def storage_unavailable(packet):raise OSError('storage unavailable')
        observer._write_latest=storage_unavailable
        self.assertFalse(observer.poll([],{}))
        self.assertEqual(observer.failures,1);self.assertIn('sensor unavailable',observer.status()['error'])
        self.assertFalse(observer.status()['valid']);self.assertEqual(observer.soil.time,2.)


if __name__=='__main__':unittest.main()
