#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Experimental active-curl correction, additional to the unchanged D120
// support integrator. No hydraulic capacity claim and no reference writes.
struct BucketActiveCurl {
  double effort = 0.0;
  static constexpr double gain = 6000000.0; // N/m
  static constexpr double cap = 60000.0; // N, supplemental effort only
  static constexpr double deadband = 0.001; // m
  static constexpr double rise = 120000.0; // N/s
  static constexpr double release = 600000.0; // N/s; <=0.1 s from cap

  double Step(double command, double error, double dt) {
    if (!std::isfinite(command) || !std::isfinite(error) ||
        !std::isfinite(dt) || dt < 0.0)
      throw std::runtime_error("active curl invalid input");
    const double desired = command > 0.0
        ? std::clamp(gain * (error - deadband), 0.0, cap) : 0.0;
    // Track a bounded error-dependent force, never integrate position error.
    // Neutral/reverse/overshoot release this correction; D120 holds support.
    effort += std::clamp(desired - effort, -release * dt, rise * dt);
    return effort;
  }
};
