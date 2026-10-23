#include "loaded_support.h"
#include <cassert>
#include <iostream>
int main() {
  assert(LoadedSupportStep(90000,-80000,0,.01,120000,30000)==90000);
  assert(LoadedSupportStep(90000,-90000,1,.01,120000,30000)==90000);
  assert(LoadedSupportStep(90000,-80000,1,.01,120000,30000)==89700);
  assert(LoadedSupportStep(-90000,80000,1,.01,120000,30000)==-89700);
  double effort=91594.706155;
  for(int i=0;i<100000;i++) {
    const double old=effort, dt=1.0/4800;
    effort=LoadedSupportStep(effort,-83000+1000*std::sin(i*.001),1,dt,120000,30000);
    assert(std::isfinite(effort) && std::abs(effort)<=120000);
    assert(std::abs(effort-old)<=30000*dt+1e-9);
  }
  assert(std::abs(effort-83000)<1100);
  bool rejected=false;
  try {LoadedSupportStep(0,0,1,-1,120000,30000);}catch(...){rejected=true;}
  assert(rejected);
  std::cout << "support_checks_passed 100000_bounded_substeps\n";
}
