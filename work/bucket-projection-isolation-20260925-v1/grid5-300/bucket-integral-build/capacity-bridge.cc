#include "/home/qunwang/wajueji/work/bucket-projection-isolation-20260925-v1/grid5-300/bucket-integral-build/bucket_integral_candidate.h"
// Generated isolated R3-C two-rock MPM bridge; frozen R1/R2/R3-A/R3-B are untouched.
#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "/home/qunwang/wajueji/work/bucket-projection-isolation-20260925-v1/grid5-300/optimized-build/inertia_candidate.h"

namespace {

bool StartsWith(const char* value, const char* prefix) {
  if (!value) return false;
  return std::string(value).rfind(prefix, 0) == 0;
}

void ContactVectorWorld(const mjContact& contact, const mjtNum local[6],
                        mjtNum force[3], mjtNum torque[3]) {
  for (int axis = 0; axis < 3; ++axis) {
    force[axis] = torque[axis] = 0;
    for (int basis = 0; basis < 3; ++basis) {
      force[axis] += contact.frame[3 * basis + axis] * local[basis];
      torque[axis] += contact.frame[3 * basis + axis] * local[3 + basis];
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: mixed_coupled_bridge_capacity model.xml");
    char error[2048] = {};
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
        mj_loadXML(argv[1], nullptr, error, sizeof(error)), mj_deleteModel);
    if (!model) throw std::runtime_error(error);
    const int rock_count = (model->nq - 19) / 7;
    if (mj_version() != 238 || rock_count < 1 || rock_count > 1024 ||
        model->nq != 19 + 7 * rock_count ||
        model->nv != 18 + 6 * rock_count || model->nu != 12 ||
        model->nplugin != 0) {
      throw std::runtime_error(
          "capacity model contract: MuJoCo 238, mechanism 19/18/12 plus dynamic rocks");
    }
    InertiaFeedforward feedforward(model.get());
    std::unique_ptr<mjData, decltype(&mj_deleteData)> data(
        mj_makeData(model.get()), mj_deleteData);
    excavator::HoldController controller;
    const int bucket_body = mj_name2id(model.get(), mjOBJ_BODY, "bucket");
    std::vector<int> rock_bodies;
    std::vector<int> rock_joints;
    std::unordered_map<std::string, int> rock_index_by_name;
    for (int index = 0; index < rock_count; ++index) {
      char body_name[64], joint_name[64];
      std::snprintf(body_name, sizeof(body_name), "r1-rock-%03d", index);
      std::snprintf(joint_name, sizeof(joint_name), "r1_rock_free_%03d", index);
      const int body = mj_name2id(model.get(), mjOBJ_BODY, body_name);
      const int joint = mj_name2id(model.get(), mjOBJ_JOINT, joint_name);
      if (body < 0 || joint < 0 ||
          model->jnt_qposadr[joint] != 19 + 7 * index ||
          model->jnt_dofadr[joint] != 18 + 6 * index) {
        throw std::runtime_error("rock body and address contract");
      }
      rock_bodies.push_back(body);
      rock_joints.push_back(joint);
      rock_index_by_name.emplace(body_name, index);
    }
    if (bucket_body < 0) {
      throw std::runtime_error("bucket body contract");
    }
    auto rock_index_for_body = [&](int body) {
      for (int index = 0; index < rock_count; ++index) {
        if (rock_bodies[index] == body) return index;
      }
      return -1;
    };

    bool ready = false;
    double closure = 0;
    int rock_ground_contact_substeps = 0;
    int rock_ramp_contact_substeps = 0;
    int rock_bucket_contact_substeps = 0;
    int rock_rock_contact_substeps = 0;
    int rock_truck_contact_substeps = 0;
    int forbidden_contact_count = 0;
    std::array<double, 3> bucket_rock_impulse{};
    std::array<double, 3> bucket_rock_angular_impulse{};
    std::array<double, 3> bucket_rock_contact_torque_impulse{};
    std::array<double, 3> bucket_rock_moment_arm_impulse{};
    std::array<double, 3> last_bucket_rock_force{};
    std::array<double, 3> last_bucket_rock_torque{};

    struct ExternalWrench {
      int body = -1;
      std::array<double, 3> force{};
      std::array<double, 3> torque{};
      std::array<double, 3> point{};
    };
    std::vector<ExternalWrench> external_wrenches;
    std::array<double, 3> applied_bucket_mpm_force{};
    std::array<double, 3> applied_bucket_mpm_torque{};
    std::vector<std::array<double, 3>> applied_rock_mpm_force(rock_count);
    std::vector<std::array<double, 3>> applied_rock_mpm_torque(rock_count);
    std::array<double, 3> bucket_interval_reference_point{};
    std::array<double, 4> equivalent_drive_load{};
    std::array<double, 4> unloaded_feedforward{};
    std::array<double,4> bucket_integral_effort{};
    BucketIntegral bucket_integrator;
    std::array<double,4> restored_controller_target{};
    std::array<double,4> restored_controller_target_velocity{};
    double restored_bucket_integral_effort = 0.0;

    auto array = [&](const char* name, const auto* values, int count) {
      std::cout << ",\"" << name << "\":[";
      for (int index = 0; index < count; ++index) {
        if (index) std::cout << ',';
        if (!std::isfinite(double(values[index]))) {
          throw std::runtime_error("nonfinite output");
        }
        std::cout << values[index];
      }
      std::cout << ']';
    };
    auto emit = [&]() {
      mj_kinematics(model.get(), data.get());
      mj_comPos(model.get(), data.get());
      std::cout << std::setprecision(17)
                << "@STATE {\"schema\":\"excavation_mixed_coupled_bridge_capacity_state_v1\""
                << ",\"time_s\":" << data->time;
      array("mechanism_qpos", data->qpos, 19);
      array("mechanism_qvel", data->qvel, 18);
      array("rock_qpos", data->qpos + 19, 7);
      array("rock_qvel", data->qvel + 18, 6);
      array("bucket_origin", data->xpos + 3 * bucket_body, 3);
      array("bucket_rotation", data->xmat + 9 * bucket_body, 9);
      array("rock_position", data->xpos + 3 * rock_bodies[0], 3);
      std::cout << ",\"rocks\":[";
      for (int index = 0; index < rock_count; ++index) {
        if (index) std::cout << ',';
        std::cout << "{\"rock_id\":\"r1-rock-" << std::setw(3)
                  << std::setfill('0') << index << std::setfill(' ') << "\"";
        array("qpos", data->qpos + 19 + 7 * index, 7);
        array("qvel", data->qvel + 18 + 6 * index, 6);
        array("position", data->xpos + 3 * rock_bodies[index], 3);
        mjtNum rock_world_velocity[6] = {};
        mj_objectVelocity(model.get(), data.get(), mjOBJ_BODY,
                          rock_bodies[index], rock_world_velocity, 0);
        array("linear_velocity_world_m_s", rock_world_velocity + 3, 3);
        array("angular_velocity_world_rad_s", rock_world_velocity, 3);
        std::cout << '}';
      }
      std::cout << ']';
      array("bucket_rock_impulse_world_ns", bucket_rock_impulse.data(), 3);
      array("bucket_rock_angular_impulse_world_nms",
            bucket_rock_angular_impulse.data(), 3);
      array("bucket_rock_contact_torque_impulse_world_nms",
            bucket_rock_contact_torque_impulse.data(), 3);
      array("bucket_rock_moment_arm_impulse_world_nms",
            bucket_rock_moment_arm_impulse.data(), 3);
      array("last_bucket_rock_force_world_n", last_bucket_rock_force.data(), 3);
      array("last_bucket_rock_torque_about_bucket_origin_world_nm",
            last_bucket_rock_torque.data(), 3);


      mjtNum bucket_velocity[6] = {};
      mj_objectVelocity(model.get(), data.get(), mjOBJ_BODY, bucket_body,
                        bucket_velocity, 0);
      array("bucket_com_linear_velocity_world_m_s", bucket_velocity + 3, 3);
      double bucket_inertial_offset[3];
      for (int k = 0; k < 3; ++k)
        bucket_inertial_offset[k] = data->xipos[3*bucket_body+k] - data->xpos[3*bucket_body+k];
      array("bucket_inertial_offset_world_m", bucket_inertial_offset, 3);
      // MPM body_com=0 and moving sweeps use the bucket's regular body origin.
      mj_objectVelocity(model.get(), data.get(), mjOBJ_XBODY, bucket_body,
                        bucket_velocity, 0);
      array("bucket_linear_velocity_world_m_s", bucket_velocity + 3, 3);
      array("bucket_angular_velocity_world_rad_s", bucket_velocity, 3);
      mjtNum rock_velocity[6] = {};
      mj_objectVelocity(model.get(), data.get(), mjOBJ_BODY, rock_bodies[0],
                        rock_velocity, 0);
      array("rock_linear_velocity_world_m_s", rock_velocity + 3, 3);
      array("rock_angular_velocity_world_rad_s", rock_velocity, 3);
      array("controller_target", controller.target.data(), 4);
      array("controller_target_velocity", controller.target_velocity.data(), 4);
      array("actuator_force", data->actuator_force, model->nu);
      array("equivalent_drive_load", equivalent_drive_load.data(), 4);
      array("unloaded_feedforward", unloaded_feedforward.data(), 4);
      array("bucket_integral_effort", bucket_integral_effort.data(), 4);
      array("applied_bucket_mpm_force_world_n", applied_bucket_mpm_force.data(), 3);
      array("applied_bucket_mpm_torque_world_nm", applied_bucket_mpm_torque.data(), 3);
      std::cout << ",\"applied_rocks_mpm\":[";
      for (int index = 0; index < rock_count; ++index) {
        if (index) std::cout << ',';
        std::cout << "{\"rock_id\":\"r1-rock-" << std::setw(3)
                  << std::setfill('0') << index << std::setfill(' ') << "\"";
        array("force_world_n", applied_rock_mpm_force[index].data(), 3);
        array("torque_world_nm", applied_rock_mpm_torque[index].data(), 3);
        std::cout << '}';
      }
      std::cout << ']';
      array("bucket_interval_reference_point_world_m",
            bucket_interval_reference_point.data(), 3);
      std::cout << ",\"contact_substeps\":{\"rock_ground\":"
                << rock_ground_contact_substeps << ",\"rock_bucket\":"
                << rock_bucket_contact_substeps << ",\"rock_rock\":"
                << rock_rock_contact_substeps << ",\"rock_truck\":"
                << rock_truck_contact_substeps << ",\"rock_ramp\":"
                << rock_ramp_contact_substeps << "}"
                << ",\"forbidden_contact_count\":" << forbidden_contact_count
                << ",\"closure_m\":" << closure << "}\n";
    };

    std::cout << "@READY {\"schema\":\"excavation_mixed_coupled_bridge_capacity_v1\""
              << ",\"mechanism_nq\":19,\"mechanism_nv\":18"
              << ",\"rock_count\":" << rock_count
              << ",\"maximum_external_wrench_count\":" << rock_count + 1
              << ",\"allowed_external_wrench_bodies\":[\"bucket\"";
    for (int index = 0; index < rock_count; ++index) {
      std::cout << ",\"r1-rock-" << std::setw(3) << std::setfill('0')
                << index << std::setfill(' ') << "\"";
    }
    std::cout << "]"
              << ",\"external_wrench_input\":\"named_world_wrench_about_interval_start_body_origin\"}"
              << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
      std::istringstream input(line);
      std::string operation;
      input >> operation;
      if (operation == "close") break;
      auto number = [&]() {
        double value;
        if (!(input >> value) || !std::isfinite(value)) {
          throw std::runtime_error("invalid finite input");
        }
        return value;
      };
      if (operation == "init_default" || operation == "init" ||
          operation == "init_loaded") {
        if (ready) throw std::runtime_error("init once only");
        const bool restore_loaded = operation == "init_loaded";
        if (operation == "init" || restore_loaded) {
          for (int index = 0; index < model->nq; ++index) data->qpos[index] = number();
          for (int index = 0; index < model->nv; ++index) data->qvel[index] = number();
        }
        if (restore_loaded) {
          for (double& value : restored_controller_target) value = number();
          for (double& value : restored_controller_target_velocity) value = number();
          restored_bucket_integral_effort = number();
          if (std::abs(restored_bucket_integral_effort) > BucketIntegral::cap)
            throw std::runtime_error("loaded integral outside envelope");
        }
        std::string extra;
        if (input >> extra) throw std::runtime_error("extra init fields");
        mj_forward(model.get(), data.get());
        if (!controller.Configure(model.get(), data.get())) {
          throw std::runtime_error("controller init");
        }
        for (int axis = 0; axis < 4; ++axis) {
          if (controller.limited[axis]) {
            controller.lower_limit[axis] += .005;
            controller.upper_limit[axis] -= .005;
          }
        }
        ConfigureCandidate(model.get(), controller, 1, 1);
        if (restore_loaded) {
          controller.target = restored_controller_target;
          controller.target_velocity = restored_controller_target_velocity;
          controller.effective_velocity = restored_controller_target_velocity;
          for (int axis = 0; axis < 4; ++axis) {
            controller.reference[axis].Reset(restored_controller_target[axis]);
            controller.reference[axis].velocity =
                restored_controller_target_velocity[axis];
          }
          bucket_integrator.effort = restored_bucket_integral_effort;
          bucket_integral_effort[3] = restored_bucket_integral_effort;
        }
        const auto initial = feedforward.Evaluate(data->qpos, controller, {}, {});
        for (int axis = 0; axis < 4; ++axis) {
          const int actuator = controller.hold_actuator[axis];
          data->ctrl[actuator] = controller.target[axis] +
              initial.force[axis] /
              model->actuator_gainprm[mjNGAIN * actuator];
        }
        mj_forward(model.get(), data.get());
        ready = true;
        std::copy(data->xpos + 3 * bucket_body,
                  data->xpos + 3 * bucket_body + 3,
                  bucket_interval_reference_point.begin());
        emit();
        continue;
      }
      if (operation != "step" || !ready) {
        throw std::runtime_error("invalid operation");
      }
      const double expected_time = number();
      const double dt = number();
      const double substeps_value = number();
      if (std::abs(expected_time - data->time) > 1e-8 || dt <= 0 ||
          dt > 1.0 / 120.0 + 1e-12 || substeps_value < 20 ||
          substeps_value > 160 || substeps_value != int(substeps_value)) {
        throw std::runtime_error("step time/substep contract");
      }
      std::array<double, 4> command{};
      for (double& value : command) value = number();
      constexpr double limits[4] = {.12, .0225, .0225, .0675};
      for (int axis = 0; axis < 4; ++axis) {
        if (std::abs(command[axis]) > limits[axis] + 1e-12) {
          throw std::runtime_error("command limit");
        }
      }

      const double external_wrench_count_value = number();
      if (external_wrench_count_value < 0 || external_wrench_count_value > rock_count + 1 ||
          external_wrench_count_value != int(external_wrench_count_value)) {
        throw std::runtime_error("capacity external wrench count contract");
      }
      external_wrenches.clear();
      applied_bucket_mpm_force.fill(0);
      applied_bucket_mpm_torque.fill(0);
      for (int index = 0; index < rock_count; ++index) {
        applied_rock_mpm_force[index].fill(0);
        applied_rock_mpm_torque[index].fill(0);
      }
      bool seen_bucket = false;
      std::vector<bool> seen_rocks(rock_count, false);
      for (int wrench_index = 0;
           wrench_index < int(external_wrench_count_value); ++wrench_index) {
        std::string body_name;
        if (!(input >> body_name)) throw std::runtime_error("missing capacity wrench body");
        ExternalWrench wrench;
        int rock_index = -1;
        if (body_name == "bucket") {
          if (seen_bucket) throw std::runtime_error("duplicate bucket wrench");
          seen_bucket = true;
          wrench.body = bucket_body;
        } else if (body_name.rfind("r1-rock-", 0) == 0) {
          const auto found = rock_index_by_name.find(body_name);
          if (found == rock_index_by_name.end()) {
            throw std::runtime_error("unknown or absent capacity rock wrench");
          }
          rock_index = found->second;
        } else {
          throw std::runtime_error("unapproved capacity wrench body");
        }
        if (rock_index >= 0) {
          if (rock_index >= rock_count || seen_rocks[rock_index]) {
            throw std::runtime_error("duplicate or absent capacity rock wrench");
          }
          seen_rocks[rock_index] = true;
          wrench.body = rock_bodies[rock_index];
        }
        for (double& value : wrench.force) value = number();
        for (double& value : wrench.torque) value = number();
        for (double& value : wrench.point) value = number();
        if (rock_index < 0) {
          applied_bucket_mpm_force = wrench.force;
          applied_bucket_mpm_torque = wrench.torque;
        } else {
          applied_rock_mpm_force[rock_index] = wrench.force;
          applied_rock_mpm_torque[rock_index] = wrench.torque;
        }
        external_wrenches.push_back(wrench);
      }
      std::string extra;
      if (input >> extra) throw std::runtime_error("extra step fields");

      std::copy(data->xpos + 3 * bucket_body,
                data->xpos + 3 * bucket_body + 3,
                bucket_interval_reference_point.begin());
      rock_ground_contact_substeps = rock_ramp_contact_substeps = 0;
      rock_bucket_contact_substeps = 0;
      rock_rock_contact_substeps = rock_truck_contact_substeps = 0;
      forbidden_contact_count = 0;
      bucket_rock_impulse.fill(0);
      bucket_rock_angular_impulse.fill(0);
      bucket_rock_contact_torque_impulse.fill(0);
      bucket_rock_moment_arm_impulse.fill(0);
      last_bucket_rock_force.fill(0);
      last_bucket_rock_torque.fill(0);
      model->opt.timestep = dt / substeps_value;
      for (int substep = 0; substep < int(substeps_value); ++substep) {
        const auto previous_velocity = controller.target_velocity;
        const double elapsed = std::max(0.0, double(data->time) - controller.last_time);
        for (int axis = 0; axis < 4; ++axis) {
          data->ctrl[controller.command_actuator[axis]] = command[axis];
        }
        if (!controller.Update(model.get(), data.get())) {
          throw std::runtime_error("controller update");
        }
        std::array<double, 4> acceleration{};
        for (int axis = 0; axis < 4; ++axis) {
          if (elapsed > 0) {
            acceleration[axis] =
                (controller.target_velocity[axis] - previous_velocity[axis]) / elapsed;
          }
        }
        const auto result = feedforward.Evaluate(
            data->qpos, controller, controller.target_velocity, acceleration);
        unloaded_feedforward = result.force;
        for (int axis = 0; axis < 4; ++axis) {
          const int actuator = controller.hold_actuator[axis];
          data->ctrl[actuator] = controller.target[axis] + result.force[axis] /
              model->actuator_gainprm[mjNGAIN * actuator];
        }

        mju_zero(data->qfrc_applied, model->nv);
        for (const auto& wrench : external_wrenches) {
          mj_applyFT(model.get(), data.get(), wrench.force.data(),
                     wrench.torque.data(), wrench.point.data(), wrench.body,
                     data->qfrc_applied);
        }
        // Observation only: project the already-applied external wrench through
        // the closed-chain tangent.  Do not add compensation or change ctrl.
        equivalent_drive_load.fill(0);
        for (int axis = 0; axis < 4; ++axis) {
          for (int dof = 0; dof < model->nv; ++dof) {
            equivalent_drive_load[axis] +=
                result.tangent[dof * 4 + axis] * data->qfrc_applied[dof];
          }
        }
        const double bucket_error = controller.target[3] -
            data->qpos[controller.qpos_address[3]];
        bucket_integral_effort[3] = bucket_integrator.Step(
            command[3], bucket_error, model->opt.timestep);
        const int bucket_hold = controller.hold_actuator[3];
        data->ctrl[bucket_hold] += bucket_integral_effort[3] /
            model->actuator_gainprm[mjNGAIN * bucket_hold];
        const double before = data->time;
        mj_step(model.get(), data.get());
        if (std::abs(data->time - before - model->opt.timestep) > 1e-10) {
          throw std::runtime_error("engine time reset");
        }
        for (int warning = 0; warning < mjNWARNING; ++warning) {
          if (data->warning[warning].number) throw std::runtime_error("engine warning");
        }

        bool ground_this_substep = false;
        bool ramp_this_substep = false;
        bool bucket_this_substep = false;
        bool rock_this_substep = false;
        bool truck_this_substep = false;
        std::array<double, 3> substep_bucket_force{};
        std::array<double, 3> substep_bucket_torque{};
        std::array<double, 3> substep_bucket_contact_torque{};
        std::array<double, 3> substep_bucket_moment_arm{};
        for (int contact_index = 0; contact_index < data->ncon; ++contact_index) {
          const mjContact& contact = data->contact[contact_index];
          const int body1 = model->geom_bodyid[contact.geom1];
          const int body2 = model->geom_bodyid[contact.geom2];
          const int first_rock = rock_index_for_body(body1);
          const int second_rock = rock_index_for_body(body2);
          if (first_rock < 0 && second_rock < 0) {
            ++forbidden_contact_count;
            continue;
          }
          if (first_rock >= 0 && second_rock >= 0) {
            rock_this_substep = true;
            continue;
          }
          const bool rock_is_first = first_rock >= 0;
          const int other_geom = rock_is_first ? contact.geom2 : contact.geom1;
          const int other_body = rock_is_first ? body2 : body1;
          const char* other_name = mj_id2name(model.get(), mjOBJ_GEOM, other_geom);
          if (other_body == bucket_body && StartsWith(other_name, "r1_bucket_")) {
            bucket_this_substep = true;
            mjtNum local[6] = {}, world_force[3] = {}, world_torque[3] = {};
            mj_contactForce(model.get(), data.get(), contact_index, local);
            ContactVectorWorld(contact, local, world_force, world_torque);
            // mj_contactForce is the wrench on geom2.  Flip it when bucket is geom1.
            const double sign = body2 == bucket_body ? 1.0 : -1.0;
            const double* bucket_origin =
                bucket_interval_reference_point.data();
            mjtNum lever[3], moment[3];
            mju_sub3(lever, contact.pos, bucket_origin);
            mju_cross(moment, lever, world_force);
            for (int axis = 0; axis < 3; ++axis) {
              substep_bucket_force[axis] += sign * world_force[axis];
              substep_bucket_contact_torque[axis] += sign * world_torque[axis];
              substep_bucket_moment_arm[axis] += sign * moment[axis];
              substep_bucket_torque[axis] +=
                  sign * (world_torque[axis] + moment[axis]);
            }
          } else if (other_name && std::string(other_name) == "ground_plane") {
            ground_this_substep = true;
          } else if (StartsWith(other_name, "r1_ground_")) {
            ground_this_substep = true;
            ramp_this_substep = true;
          } else if (StartsWith(other_name, "r1_truck_")) {
            truck_this_substep = true;
          } else {
            ++forbidden_contact_count;
          }
        }
        rock_ground_contact_substeps += ground_this_substep;
        rock_ramp_contact_substeps += ramp_this_substep;
        rock_bucket_contact_substeps += bucket_this_substep;
        rock_rock_contact_substeps += rock_this_substep;
        rock_truck_contact_substeps += truck_this_substep;
        last_bucket_rock_force = substep_bucket_force;
        last_bucket_rock_torque = substep_bucket_torque;
        for (int axis = 0; axis < 3; ++axis) {
          bucket_rock_impulse[axis] +=
              substep_bucket_force[axis] * model->opt.timestep;
          bucket_rock_angular_impulse[axis] +=
              substep_bucket_torque[axis] * model->opt.timestep;
          bucket_rock_contact_torque_impulse[axis] +=
              substep_bucket_contact_torque[axis] * model->opt.timestep;
          bucket_rock_moment_arm_impulse[axis] +=
              substep_bucket_moment_arm[axis] * model->opt.timestep;
        }
        if (forbidden_contact_count) {
          throw std::runtime_error("forbidden contact class");
        }
        for (int index = 0; index < model->nv; ++index) {
          if (!std::isfinite(data->qvel[index]) || !std::isfinite(data->qacc[index])) {
            throw std::runtime_error("nonfinite dynamics");
          }
        }
        mj_kinematics(model.get(), data.get());
        mj_comPos(model.get(), data.get());
        for (int equality = 0; equality < model->neq; ++equality) {
          if (model->eq_type[equality] != mjEQ_CONNECT) continue;
          const int a = model->eq_obj1id[equality];
          const int b = model->eq_obj2id[equality];
          mjtNum pa[3], pb[3];
          mju_rotVecMat(pa, model->eq_data + mjNEQDATA * equality,
                       data->xmat + 9 * a);
          mju_addTo3(pa, data->xpos + 3 * a);
          mju_rotVecMat(pb, model->eq_data + mjNEQDATA * equality + 3,
                       data->xmat + 9 * b);
          mju_addTo3(pb, data->xpos + 3 * b);
          closure = std::max(closure, double(mju_dist3(pa, pb)));
        }
        if (closure > .001) throw std::runtime_error("dynamic closure above 1 mm");
        for (int joint = 0; joint < model->njnt; ++joint) {
          if (!model->jnt_limited[joint]) continue;
          const double q = data->qpos[model->jnt_qposadr[joint]];
          if (q < model->jnt_range[2 * joint] ||
              q > model->jnt_range[2 * joint + 1]) {
            throw std::runtime_error("joint range");
          }
        }
      }
      emit();
    }
  } catch (const std::exception& error) {
    std::cerr << "mixed_coupled_bridge_capacity: " << error.what() << '\n';
    return 1;
  }
}
