"""Measured neutral support plus inertia-derived dissipation; isolated candidate."""
import json
from pathlib import Path
from damping_candidate import instrument as damping, instrument_observation


def instrument(source, activation_clock):
    changed=damping(source,activation_clock)
    old='''        bucket_integral_effort[3] = bucket_integrator.Step(
            command[3], bucket_error, model->opt.timestep);'''
    new='''        const double support_s=std::clamp((double(data->time)-ACTIVATION_CLOCK)/0.25,0.0,1.0);
        const double support_blend=support_s*support_s*(3.0-2.0*support_s);
        if (support_blend>0 && command[3]==0.0) {
          const double previous_effort=bucket_integrator.effort;
          bucket_integrator.effort=LoadedSupportStep(previous_effort,equivalent_drive_load[3],
              support_blend,model->opt.timestep,BucketIntegral::cap,BucketIntegral::slew);
          bucket_integrator.last_rate=(bucket_integrator.effort-previous_effort)/model->opt.timestep;
          bucket_integrator.limited=std::abs(bucket_integrator.last_rate)>=BucketIntegral::slew-1e-6;
          bucket_integral_effort[3]=bucket_integrator.effort;
        } else {
          bucket_integral_effort[3] = bucket_integrator.Step(
              command[3], bucket_error, model->opt.timestep);
        }'''.replace('ACTIVATION_CLOCK',format(activation_clock,'.17g'))
    assert changed.count(old)==1
    header=json.dumps(str(Path(__file__).with_name('loaded_support.h').resolve()))
    return changed.replace(old,new).replace('#include <mujoco/mujoco.h>','#include <mujoco/mujoco.h>\n#include '+header)
