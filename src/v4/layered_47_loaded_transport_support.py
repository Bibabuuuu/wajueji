"""Opt-in, finite-window neutral bucket support for loaded transport.

Original commands and native states are retained.  The existing accumulator
tracks the projected external load, while reflected inertia sets dissipation.
Outside the explicit window the original controller is used without changes.
"""
import hashlib
import json
import math
from pathlib import Path

import layered_mound_checkpoint_runtime as generator
from layered_47_cycle_runtime import CycleRuntime

POLICY='measured_loaded_support_v1'
RAMP_SECONDS=.05


def validate_window(start_s,end_s):
    if not (math.isfinite(start_s) and math.isfinite(end_s) and start_s>=0 and end_s-start_s>2*RAMP_SECONDS):
        raise ValueError('loaded_support_invalid_window')


def validate_activation_clock(native_time_s,start_s):
    if not math.isfinite(native_time_s) or not -1e-8<=start_s-native_time_s<=.1:
        raise ValueError('loaded_support_start_must_use_current_native_time')


def instrument(source, start_s, end_s):
    validate_window(start_s,end_s)
    header=json.dumps(str(Path(__file__).with_name('loaded_transport_support.h').resolve()))
    anchor='#include <mujoco/mujoco.h>'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+'\n#include '+header)
    anchor='''        bucket_integral_effort[3] = bucket_integrator.Step(
            command[3], bucket_error, model->opt.timestep);'''
    assert source.count(anchor)==1
    block='''        const auto support_smooth=[](double x) {
          const double s=std::clamp(x,0.0,1.0); return s*s*(3.0-2.0*s);
        };
        const double loaded_blend=support_smooth((double(data->time)-START)/0.05)*
            support_smooth((END-double(data->time))/0.05);
        if (loaded_blend>0 && command[3]!=0.0)
          throw std::runtime_error("loaded_support_requires_neutral_bucket");
        if (loaded_blend>0) {
          const double previous_effort=bucket_integrator.effort;
          bucket_integrator.effort=LoadedSupportStep(previous_effort,equivalent_drive_load[3],
              loaded_blend,model->opt.timestep,BucketIntegral::cap,BucketIntegral::slew);
          bucket_integrator.last_rate=(bucket_integrator.effort-previous_effort)/model->opt.timestep;
          bucket_integrator.limited=std::abs(bucket_integrator.last_rate)>=BucketIntegral::slew-1e-6;
          bucket_integral_effort[3]=bucket_integrator.effort;
        } else {
          bucket_integral_effort[3] = bucket_integrator.Step(
              command[3], bucket_error, model->opt.timestep);
        }'''.replace('START',format(start_s,'.17g')).replace('END',format(end_s,'.17g'))
    source=source.replace(anchor,block)
    anchor='        const double before = data->time;\n        mj_step(model.get(), data.get());'
    assert source.count(anchor)==1
    block='''        const double damper_kp=model->actuator_gainprm[mjNGAIN*bucket_hold];
        const double damper_kv=model->actuator_gainprm[mjNGAIN*controller.drive_actuator[3]];
        const double damper_mass=result.mass[15];
        if (!(damper_mass>0 && damper_kp>0 && damper_kv>=0))
          throw std::runtime_error("loaded_support_invalid_coefficients");
        const double extra_damping=loaded_blend*std::max(0.0,2.0*std::sqrt(damper_mass*damper_kp)-damper_kv);
        const double bucket_speed=data->qvel[controller.dof_address[3]];
        const double damping_effort=std::clamp(-extra_damping*bucket_speed,-BucketIntegral::cap,BucketIntegral::cap);
        if (!std::isfinite(damping_effort) || damping_effort*bucket_speed>1e-9)
          throw std::runtime_error("loaded_support_not_dissipative");
        data->ctrl[bucket_hold] += damping_effort/damper_kp;
'''
    return source.replace(anchor,block+anchor)


class LoadedTransportRuntime(CycleRuntime):
    supports_loaded_transport_policy=True

    def __init__(self, source, output, *, support_start_s=None, support_end_s=None, **kwargs):
        super().__init__(source,output,**kwargs)
        previous=self.metadata.get('loaded_transport_support')
        if previous is not None:
            if previous['policy']!=POLICY:
                raise ValueError('loaded_support_unknown_policy')
            if support_start_s is None: support_start_s=previous['start_s']
            if support_end_s is None: support_end_s=previous['end_s']
            if support_start_s!=previous['start_s'] or support_end_s!=previous['end_s']:
                raise ValueError('loaded_support_restore_window_changed')
        if support_start_s is None or support_end_s is None:
            raise ValueError('loaded_support_requires_explicit_window')
        validate_window(support_start_s,support_end_s)
        if previous is None:
            validate_activation_clock(self.metadata['controller_state']['native_time_s'],support_start_s)
        self.support_policy=dict(policy=POLICY,clock='native_time_s',start_s=float(support_start_s),end_s=float(support_end_s),ramp_s=RAMP_SECONDS)
        if previous is not None:
            if previous.get('bindings')!={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('layered_47_loaded_transport_support.py','loaded_transport_support.h')}:
                raise ValueError('loaded_support_restore_binding_changed')
        self.support_policy['bindings']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('layered_47_loaded_transport_support.py','loaded_transport_support.h')}
        self.metadata['loaded_transport_support']=self.support_policy
        self.bindings.update(self.support_policy['bindings'])

    def _initialize_fixture(self,*args):
        original=generator.render_capacity_bridge
        def render(*a):
            source,manifest=original(*a)
            return instrument(source,self.support_policy['start_s'],self.support_policy['end_s']),manifest
        generator.render_capacity_bridge=render
        try:return super()._initialize_fixture(*args)
        finally:generator.render_capacity_bridge=original
