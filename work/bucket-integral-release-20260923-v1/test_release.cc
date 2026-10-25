#include "bucket_integral.h"
#include "previous_integral.h"
#include <cassert>
#include <iostream>
#include <limits>
int main() {
  const double dt=1.0/4800;
  BucketIntegral now; PreviousBucketIntegral old;
  // Existing growth, neutral hold, error unwinding and saturation stay exact.
  for(int i=0;i<100000;i++) {
    const double cmd=i<48000?.02:0;
    const double error=i<80000?.04:-.003;
    assert(now.Step(cmd,error,dt)==old.Step(cmd,error,dt));
  }
  for(double sign:{-1.,1.}) {
    BucketIntegral s; s.effort=sign*104971.195;
    for(int i=0;i<2400;i++) {
      double before=s.effort;
      s.Step(-sign*.02,sign*.01,dt); // old error persists but reversal releases
      assert(std::abs(s.effort)<std::abs(before));
      assert(std::abs(s.last_rate)<=30000+1e-7);
    }
    assert(std::abs(s.effort-sign*89971.195)<1e-6);
    double held=s.effort;
    assert(s.Step(0,sign*.01,dt)==held); // stop command stops forced release
    s.Step(-sign*.001,sign*.01,.1);
    assert(std::abs(s.effort-(held-sign*300))<1e-6); // tiny command proportional
    assert(s.Step(-sign*.02,sign*.01,100)==0); // release cannot change sign
    assert(s.Step(-sign*.02,-sign*.01,dt)*sign<0); // may subsequently learn new support
  }
  BucketIntegral s; s.effort=100;
  assert(s.Step(-.02,.1,0)==100);
  bool bad=false;
  try{s.Step(std::numeric_limits<double>::quiet_NaN(),0,dt);}catch(...){bad=true;}
  assert(bad);
  std::cout<<"RELEASE_TEST_PASS prefix-equivalence symmetry rate neutral small-command no-crossing new-direction invalid\n";
}
