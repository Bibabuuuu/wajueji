"""Observation-only native instrumentation. No mjData/model/control writes."""
import json


def instrument(source, log_path):
    edits = []

    def add(anchor, extra, before=False):
        nonlocal source
        if source.count(anchor) != 1:
            raise ValueError('response_observer_anchor:' + anchor[:90])
        replacement = extra + anchor if before else anchor + extra
        source = source.replace(anchor, replacement)
        edits.append((anchor, replacement))

    add('#include <iomanip>', '\n#include <fstream>')
    header = ('frame half substep axis time_before time_after command q_before v_before '
              'q_after v_after target target_velocity hold_ctrl drive_ctrl hold_force drive_force '
              'feedforward integral active external_projected rock_projected actuator_projected '
              'constraint_projected passive_projected bias_projected qacc mass0 mass1 mass2 mass3 '
              'rock_contact constraint_count kp kv hold_force_limited drive_force_limited').split()
    add('    std::array<double, 4> equivalent_drive_load{};',
        '\n    int response_block = 0;\n    std::ofstream response_log(' + json.dumps(str(log_path)) + ');\n'
        '    if (!response_log) throw std::runtime_error("response_log_open");\n'
        '    response_log << std::setprecision(17) << ' + json.dumps('\t'.join(header) + '\n') + ';\n')
    add('      model->opt.timestep = dt / substeps_value;', '''
      ++response_block;
      const int response_frame = (response_block + 1) / 2;
      const bool response_record = response_frame >= 940 && response_frame <= 972;
      if (response_record && (int(substeps_value) != 40 || std::abs(dt-1.0/120)>1e-12))
        throw std::runtime_error("response_block_contract");
''')
    add('        const double before = data->time;', '''        std::array<double,4> response_q{}, response_v{};
        if (response_record) for (int axis=0; axis<4; ++axis) {
          response_q[axis] = data->qpos[controller.qpos_address[axis]];
          response_v[axis] = data->qvel[controller.dof_address[axis]];
        }
''', before=True)
    add('        last_bucket_rock_force = substep_bucket_force;', '''
        if (response_record) {
          // mj_applyFT only accumulates into our private vector. No live force write.
          std::vector<mjtNum> response_rock(model->nv,0);
          mj_applyFT(model.get(),data.get(),substep_bucket_force.data(),substep_bucket_torque.data(),
              bucket_interval_reference_point.data(),bucket_body,response_rock.data());
          const auto response_project = [&](const mjtNum* force,int axis) {
            double total=0;
            for(int dof=0;dof<model->nv;++dof) total+=result.tangent[dof*4+axis]*force[dof];
            return total;
          };
          for(int axis=0;axis<4;++axis) {
            const int hold=controller.hold_actuator[axis],drive=controller.drive_actuator[axis];
            response_log << response_frame << '\\t' << (response_block-1)%2 << '\\t' << substep << '\\t' << axis
              << '\\t' << before << '\\t' << data->time << '\\t' << command[axis]
              << '\\t' << response_q[axis] << '\\t' << response_v[axis]
              << '\\t' << data->qpos[controller.qpos_address[axis]] << '\\t' << data->qvel[controller.dof_address[axis]]
              << '\\t' << controller.target[axis] << '\\t' << controller.target_velocity[axis]
              << '\\t' << data->ctrl[hold] << '\\t' << data->ctrl[drive]
              << '\\t' << data->actuator_force[hold] << '\\t' << data->actuator_force[drive]
              << '\\t' << unloaded_feedforward[axis] << '\\t' << bucket_integral_effort[axis] << '\\t' << bucket_active_curl_effort[axis]
              << '\\t' << equivalent_drive_load[axis] << '\\t' << response_project(response_rock.data(),axis)
              << '\\t' << response_project(data->qfrc_actuator,axis) << '\\t' << response_project(data->qfrc_constraint,axis)
              << '\\t' << response_project(data->qfrc_passive,axis) << '\\t' << response_project(data->qfrc_bias,axis)
              << '\\t' << data->qacc[controller.dof_address[axis]];
            for(int j=0;j<4;++j) response_log << '\\t' << result.mass[axis*4+j];
            response_log << '\\t' << bucket_this_substep << '\\t' << data->nefc
              << '\\t' << model->actuator_gainprm[mjNGAIN*hold] << '\\t' << model->actuator_gainprm[mjNGAIN*drive]
              << '\\t' << int(model->actuator_forcelimited[hold]) << '\\t' << int(model->actuator_forcelimited[drive]) << '\\n';
          }
          if (substep==int(substeps_value)-1) response_log.flush();
        }
''')
    # Cheap invariant: removing all inserted observations yields original bytes.
    stripped = source
    for anchor, replacement in reversed(edits):
        assert stripped.count(replacement) == 1
        stripped = stripped.replace(replacement, anchor)
    return source, stripped
