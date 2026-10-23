#include "/home/qunwang/wajueji/src/v4/bucket_integral_candidate.h"
#include <cassert>
#include <limits>
#include <iostream>
int main(){
 BucketIntegral i;
 const double dt=1.0/4800;
 double before=0;
 for(int k=0;k<48000;++k){double now=i.Step(.02,.04,dt);assert(now>=before&&now-before<=BucketIntegral::slew*dt+1e-8);before=now;}
 assert(i.effort==BucketIntegral::cap);
 before=i.effort;
 for(int k=0;k<4800;++k)assert(i.Step(0,.04,dt)==before); // neutral support, no extra growth
 assert(i.Step(-.02,.04,dt)==before); // reverse must not add old-direction effort
 assert(i.Step(-.02,-.02,dt)<before); // immediate unwind; no saturation backlog
 before=i.effort;assert(i.Step(0,-.02,dt)<before); // unload/overshoot can unwind at neutral
 before=i.effort;assert(i.Step(.02,.0003,dt)==before); // deadband
 assert(i.Step(.02,.03,0)==before);
 bool rejected=false;try{i.Step(0,std::numeric_limits<double>::quiet_NaN(),dt);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 BucketIntegral j;for(int k=0;k<48000;++k)j.Step(-.02,-.04,dt);assert(j.effort==-BucketIntegral::cap);
 // Constant-load overdamped surrogate: sustained command removes most error,
 // then a neutral command preserves load support. This is not a physics gate.
 BucketIntegral s;double q=.817;const double target=.852,kp=3e6,load=105000,b=500000;
 for(int k=0;k<48000;++k){double force=s.Step(.02,target-q,dt);q+=(kp*(target-q)+force-load)/b*dt;}
 assert(target-q<.002);
 const double held=s.effort;for(int k=0;k<4800;++k)s.Step(0,.001,dt);assert(s.effort==held);
 std::cout<<"INTEGRAL_TEST_PASS rate/cap/neutral/reverse/unwind/deadband/invalid/surrogate\n";
}
