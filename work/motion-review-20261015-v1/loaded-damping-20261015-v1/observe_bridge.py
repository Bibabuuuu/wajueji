"""Observe existing native substeps without modifying the physical update."""
import json
from pathlib import Path


def instrument(source, output):
    replacements = []
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)
        replacements.append((old, new))
    replace('#include <mujoco/mujoco.h>', '#include <mujoco/mujoco.h>\n#include <fstream>')
    header = 'time_before,time_after,axis,q_before,v_before,q_after,v_after,target,target_velocity,command,hold_ctrl,hold_kp,hold_bias_position,hold_bias_velocity,drive_kv,hold_effort,drive_effort,ff,gravity_ff,accel_ff,velocity_ff,integral,external,actuator_projected,passive_projected,bias_projected,constraint_projected,qacc_axis,ncon'
    replace('    InertiaFeedforward feedforward(model.get());',
            '    InertiaFeedforward feedforward(model.get());\n'
            '    std::ofstream observation(' + json.dumps(str(Path(output).resolve())) + ');\n'
            '    if (!observation) throw std::runtime_error("observation_open");\n'
            '    observation << std::setprecision(17) << ' + json.dumps(header + '\n') + ';')
    anchor = '        const double before = data->time;\n        mj_step(model.get(), data.get());'
    replace(anchor, '''        std::array<double,4> observed_q{}, observed_v{};
        for (int axis=0; axis<4; ++axis) {
          observed_q[axis] = data->qpos[controller.qpos_address[axis]];
          observed_v[axis] = data->qvel[controller.dof_address[axis]];
        }
''' + anchor + '''
        for (int axis=0; axis<4; ++axis) {
          double actuator=0, passive=0, bias=0, constraint=0;
          for (int dof=0; dof<model->nv; ++dof) {
            const double tangent=result.tangent[dof*4+axis];
            actuator += tangent*data->qfrc_actuator[dof];
            passive += tangent*data->qfrc_passive[dof];
            bias += tangent*data->qfrc_bias[dof];
            constraint += tangent*data->qfrc_constraint[dof];
          }
          const int hold=controller.hold_actuator[axis];
          const int drive=controller.drive_actuator[axis];
          observation << before << ',' << data->time << ',' << axis << ','
            << observed_q[axis] << ',' << observed_v[axis] << ','
            << data->qpos[controller.qpos_address[axis]] << ','
            << data->qvel[controller.dof_address[axis]] << ','
            << controller.target[axis] << ',' << controller.target_velocity[axis] << ','
            << command[axis] << ',' << data->ctrl[hold] << ','
            << model->actuator_gainprm[mjNGAIN*hold] << ','
            << model->actuator_biasprm[mjNBIAS*hold+1] << ','
            << model->actuator_biasprm[mjNBIAS*hold+2] << ','
            << model->actuator_gainprm[mjNGAIN*drive] << ','
            << data->actuator_force[hold] << ',' << data->actuator_force[drive] << ','
            << result.force[axis] << ',' << result.gravity.force[axis] << ','
            << result.acceleration_force[axis] << ',' << result.velocity_bias[axis] << ','
            << bucket_integral_effort[axis] << ',' << equivalent_drive_load[axis] << ','
            << actuator << ',' << passive << ',' << bias << ',' << constraint << ','
            << data->qacc[controller.dof_address[axis]] << ',' << data->ncon << '\\n';
        }
        if (!observation) throw std::runtime_error("observation_write");''')
    recovered = source
    for old, new in reversed(replacements):
        assert recovered.count(new) == 1
        recovered = recovered.replace(new, old)
    return source, recovered


def self_check(path):
    original = Path(path).read_text()
    changed, recovered = instrument(original, '/tmp/observation-cpu-check.csv')
    assert original == recovered and original != changed
    assert changed.count('mj_step(model.get(), data.get());') == original.count('mj_step(model.get(), data.get());')
    try:
        instrument('', '/tmp/unused.csv')
    except AssertionError:
        pass
    else:
        raise AssertionError('invalid source accepted')
    return changed
