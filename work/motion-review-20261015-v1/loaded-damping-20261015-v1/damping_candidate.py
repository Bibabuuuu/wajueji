"""Isolated loaded-port damping, derived from existing reflected inertia."""


def instrument(source, activation_clock):
    anchor = '        const double before = data->time;\n        mj_step(model.get(), data.get());'
    assert source.count(anchor) == 1
    block = '''        // Experimental actuator feedback only; no target/model/state writes.
        if (command[3] != 0.0) throw std::runtime_error("loaded_damper_nonzero_bucket_command");
        const double damper_kp=model->actuator_gainprm[mjNGAIN*bucket_hold];
        const double damper_kv=model->actuator_gainprm[mjNGAIN*controller.drive_actuator[3]];
        const double damper_mass=result.mass[15];
        if (!(damper_mass>0 && damper_kp>0 && damper_kv>=0))
          throw std::runtime_error("loaded_damper_invalid_coefficients");
        const double damper_s=std::clamp((double(data->time)-ACTIVATION_CLOCK)/0.25,0.0,1.0);
        const double damper_blend=damper_s*damper_s*(3.0-2.0*damper_s);
        const double extra_damping=damper_blend*std::max(0.0,2.0*std::sqrt(damper_mass*damper_kp)-damper_kv);
        const double bucket_speed=data->qvel[controller.dof_address[3]];
        const double damping_effort=std::clamp(-extra_damping*bucket_speed,-BucketIntegral::cap,BucketIntegral::cap);
        if (!std::isfinite(damping_effort) || damping_effort*bucket_speed>1e-9)
          throw std::runtime_error("loaded_damper_not_dissipative");
        data->ctrl[bucket_hold] += damping_effort/damper_kp;
'''.replace('ACTIVATION_CLOCK', format(activation_clock,'.17g'))
    changed=source.replace(anchor,block+anchor)
    assert changed.replace(block+anchor,anchor)==source
    # All force additions enter the real actuator, and therefore native dynamics.
    return changed


def instrument_observation(source):
    assert source.count('qacc_axis,ncon') == 1
    source=source.replace('qacc_axis,ncon','qacc_axis,ncon,mass_bucket,extra_damping,damping_effort')
    anchor="<< data->qacc[controller.dof_address[axis]] << ',' << data->ncon << '\\n';"
    assert source.count(anchor)==1
    return source.replace(anchor,"<< data->qacc[controller.dof_address[axis]] << ',' << data->ncon << ',' << result.mass[15] << ',' << extra_damping << ',' << damping_effort << '\\n';")
