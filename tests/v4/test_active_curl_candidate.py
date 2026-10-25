from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/v4"))
import layered_47_active_curl_candidate as candidate
import layered_47_bucket_integral_candidate as integral
import layered_mound_checkpoint_runtime as runtime
from tests.v4.test_bucket_integral_manual_candidate import SOURCE


class ActiveCurlCandidateTest(unittest.TestCase):
    def test_disabled_is_noop(self):
        original = runtime.render_capacity_bridge
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "build"
            with candidate.active_curl_bridge(False, target):
                self.assertIs(runtime.render_capacity_bridge, original)
            self.assertFalse(target.exists())

    def test_composition_provenance_restore_and_missing_integral_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(runtime, "render_capacity_bridge",
                              return_value=(SOURCE, {"native": True})) as original:
                with self.assertRaisesRegex(RuntimeError, "deliberate"):
                    with integral.bucket_integral_bridge(True, Path(tmp) / "integral"):
                        with candidate.active_curl_bridge(True, Path(tmp) / "active",
                                                          activation_time_s=10) as policy:
                            source, manifest = runtime.render_capacity_bridge(None)
                            self.assertTrue(manifest["native"])
                            self.assertEqual(manifest["active_curl_candidate"], policy)
                            self.assertIn("data->time >= 10 - 1e-9", source)
                            self.assertIn("active curl loaded restore unqualified", source)
                            raise RuntimeError("deliberate")
                self.assertIs(runtime.render_capacity_bridge, original)
        with self.assertRaisesRegex(ValueError, "candidate_anchor"):
            candidate.instrument_active_curl(SOURCE, candidate.HEADER)
        for time in [-1, float("inf"), float("nan")]:
            with self.assertRaisesRegex(ValueError, "activation_time"):
                candidate.instrument_active_curl(SOURCE, candidate.HEADER, time)

    @unittest.skipUnless(shutil.which("g++"), "C++ compiler required")
    def test_cpp_control_envelope_release_and_constant_load_response(self):
        code = r'''
#include "bucket_active_curl_candidate.h"
#include <cassert>
#include <limits>
int main() {
  constexpr double dt=1.0/4800;
  BucketActiveCurl b;
  for(int i=0;i<4800;++i) assert(b.Step(0, .02, dt)==0);
  double last=0;
  for(int i=0;i<4800;++i) {
    double v=b.Step(.06,.02,dt);
    assert(v>=0 && v<=b.cap && v-last<=b.rise*dt+1e-8); last=v;
  }
  assert(b.effort==b.cap);
  for(double command: {0.0, -.06}) {
    b.effort=b.cap;
    for(int i=0;i<480;++i) b.Step(command,.02,dt);
    assert(b.effort==0);
  }
  b.effort=b.cap;
  for(int i=0;i<480;++i) b.Step(.06,-.001,dt);
  assert(b.effort==0);
  for(int i=0;i<4800;++i) assert(b.Step(.06,.0005,dt)==0);
  for(int i=0;i<4800;++i) b.Step(.06,.003,dt);
  assert(std::abs(b.effort-12000)<1e-6);
  const auto before=b.effort;
  assert(b.Step(.06,.02,0)==before);
  bool rejected=false;
  try { b.Step(0,std::numeric_limits<double>::quiet_NaN(),dt); }
  catch(const std::runtime_error&) {rejected=true;}
  assert(rejected && b.effort==before);
  rejected=false;
  try { b.Step(0,0,-dt); } catch(const std::runtime_error&) {rejected=true;}
  assert(rejected);
  // Toy constant-load plant checks direction and active-only effect, not MPM.
  double error[2]={.008,.008}, velocity[2]={0,0};
  BucketActiveCurl a;
  for(int i=0;i<9600;++i) for(int mode=0;mode<2;++mode) {
    double extra=mode ? a.Step(.06,error[mode],dt) : 0;
    double force=3e6*error[mode]+120000+extra-144000-100000*velocity[mode];
    velocity[mode]+=force/1000*dt;
    error[mode]-=velocity[mode]*dt;
    assert(error[mode]>0 && std::isfinite(error[mode]));
  }
  assert(error[1]<.5*error[0]);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "check.cc"
            binary = Path(tmp) / "check"
            source.write_text(code)
            subprocess.run(["g++", "-std=c++17", "-O2", "-I", str(candidate.HEADER.parent),
                            str(source), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
