#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
// Isolated experiment: output is an extra bucket-cylinder effort, not a
// physical hydraulic capacity. No position/reference/geometry writes.
struct BucketIntegral {
  double effort = 0.0;
  double last_rate = 0.0;
  bool limited = false;
  static constexpr double ki = 1500000.0; // N/(m*s); nominal kp/ki = 2 s
  static constexpr double cap = 120000.0; // N, experiment accumulator envelope
  static constexpr double slew = 30000.0; // N/s
  static constexpr double deadband = 0.0005; // m
  double Step(double command, double error, double dt) {
    if(!std::isfinite(command)||!std::isfinite(error)||!std::isfinite(dt)||dt<0)
      throw std::runtime_error("integral nonfinite/negative time input");
    last_rate=0; limited=false;
    if(dt==0)return effort;
    const double e=std::copysign(std::max(0.0,std::abs(error)-deadband),error);
    // Grow only toward an actively requested motion. At neutral keep the
    // learned supporting effort, but allow overshoot/error reversal to unwind.
    const bool requested_correction=command*e>0;
    const bool unwinding=effort*e<0;
    if(requested_correction||unwinding){
      double rate=std::clamp(ki*e,-slew,slew);
      const double next=std::clamp(effort+rate*dt,-cap,cap);
      limited=(std::abs(ki*e)>slew || next!=effort+rate*dt);
      last_rate=(next-effort)/dt;
      effort=next; // accumulator is the bounded output; no hidden windup
    }
    return effort;
  }
};
