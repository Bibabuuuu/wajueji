#ifndef EXCAVATOR_BOUNDED_VELOCITY_H_
#define EXCAVATOR_BOUNDED_VELOCITY_H_

#include <algorithm>
#include <cmath>

namespace excavator {

// Reference generator only: does not edit physical qpos/qvel or actuator gains.
// Acceleration is a commanded reference bound, not a guaranteed plant braking rate.
struct VelocityBounds {
  double max_speed = 0.5;
  double acceleration = 1.0;
  double braking_acceleration = 0.25;
  bool limited = false;
  double lower = 0;
  double upper = 0;
  double margin = 0.003;
};

struct BoundedVelocity {
  double target = 0;
  double velocity = 0;
  bool valid = true;
  bool safety_limited = false;
  bool emergency_clamp = false;

  void Reset(double q) {
    target = q;
    velocity = 0;
    valid = std::isfinite(q);
    safety_limited = emergency_clamp = false;
  }

  void Step(double request, double q, double actual_velocity, double dt,
            const VelocityBounds& limits) {
    safety_limited = emergency_clamp = false;
    valid = std::isfinite(target) && std::isfinite(velocity) &&
            std::isfinite(q) && std::isfinite(actual_velocity) &&
            std::isfinite(dt) && dt >= 0 &&
            std::isfinite(limits.max_speed) && limits.max_speed > 0 &&
            std::isfinite(limits.acceleration) && limits.acceleration > 0 &&
            std::isfinite(limits.braking_acceleration) &&
            limits.braking_acceleration > 0 &&
            limits.braking_acceleration <= limits.acceleration &&
            (!limits.limited || (std::isfinite(limits.lower) &&
              std::isfinite(limits.upper) && limits.lower < limits.upper &&
              std::isfinite(limits.margin) && limits.margin >= 0 &&
              2*limits.margin < limits.upper-limits.lower));
    if (!valid) { velocity = 0; return; }
    if (!std::isfinite(request)) request = 0;
    request = std::clamp(request, -limits.max_speed, limits.max_speed);
    if (dt == 0) return;

    double positive = limits.max_speed, negative = limits.max_speed;
    if (limits.limited) {
      const double upper_distance = limits.upper-limits.margin -
          std::max(target,q) - std::max(0.0,actual_velocity)*dt;
      const double lower_distance = std::min(target,q) -
          (limits.lower+limits.margin) - std::max(0.0,-actual_velocity)*dt;
      // v*dt + v^2/(2*a) <= remaining: include one discrete-step advance.
      const auto cap = [&](double distance) {
        const double a_dt = limits.braking_acceleration*dt;
        return std::sqrt(a_dt*a_dt + 2*limits.braking_acceleration*
                         std::max(0.0,distance))-a_dt;
      };
      positive = std::min(positive,cap(upper_distance));
      negative = std::min(negative,cap(lower_distance));
    }
    const double desired = std::clamp(request,-negative,positive);
    safety_limited = desired != request;
    double next = velocity + std::clamp(desired-velocity,
                         -limits.acceleration*dt,limits.acceleration*dt);
    const double bounded = std::clamp(next,-negative,positive);
    emergency_clamp = std::abs(bounded-velocity) > limits.acceleration*dt+1e-12;
    velocity = bounded;
    const double proposed = target+velocity*dt;
    target = limits.limited ? std::clamp(proposed,limits.lower,limits.upper) : proposed;
    if (target != proposed) { velocity=0; emergency_clamp=true; }
  }
};
}  // namespace excavator
#endif
