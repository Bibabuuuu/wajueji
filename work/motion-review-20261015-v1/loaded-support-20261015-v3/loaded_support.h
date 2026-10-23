#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
// Neutral loaded transport: support the measured generalized external load.
// The accumulator remains the saved controller state; no state reset occurs.
inline double LoadedSupportStep(double effort, double external, double blend,
                                double dt, double cap, double slew) {
  if (!std::isfinite(effort) || !std::isfinite(external) ||
      !std::isfinite(blend) || !std::isfinite(dt) || dt < 0 ||
      blend < 0 || blend > 1 || cap <= 0 || slew <= 0)
    throw std::runtime_error("invalid loaded support input");
  const double target = std::clamp(-external, -cap, cap);
  const double rate = blend * std::clamp((target-effort)/0.25, -slew, slew);
  return std::clamp(effort + rate*dt, -cap, cap);
}
