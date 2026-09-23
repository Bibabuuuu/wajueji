// Short standalone MuJoCo probe for the generated R1 open-bucket fixtures.
// This is intentionally independent of the frozen 19/18/12 mechanism bridge.
#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

const mjtNum* Numeric(const mjModel* model, const char* name, int width) {
  const int id = mj_name2id(model, mjOBJ_NUMERIC, name);
  if (id < 0 || model->numeric_size[id] != width) {
    throw std::runtime_error(std::string("missing numeric: ") + name);
  }
  return model->numeric_data + model->numeric_adr[id];
}

bool StartsWith(const char* value, const char* prefix) {
  return value && std::strncmp(value, prefix, std::strlen(prefix)) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error("usage: mixed_material_r1_probe fixture.xml");
    }
    char error[2048] = {};
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
        mj_loadXML(argv[1], nullptr, error, sizeof(error)), mj_deleteModel);
    if (!model) throw std::runtime_error(error);
    if (mj_version() != 238 || model->nplugin != 0) {
      throw std::runtime_error("MuJoCo 2.3.8 and zero plugins required");
    }
    std::unique_ptr<mjData, decltype(&mj_deleteData)> data(
        mj_makeData(model.get()), mj_deleteData);
    const int rock_body = mj_name2id(model.get(), mjOBJ_BODY, "rock");
    const int rock_geom = mj_name2id(model.get(), mjOBJ_GEOM, "rock_geom");
    if (rock_body < 0 || rock_geom < 0 || model->nq != 7 || model->nv != 6) {
      throw std::runtime_error("single free-rock fixture contract");
    }
    const mjtNum* mouth_center = Numeric(model.get(), "mouth_center", 3);
    const mjtNum* inward = Numeric(model.get(), "mouth_inward", 3);
    const double duration = Numeric(model.get(), "probe_duration_s", 1)[0];
    const double diameter = Numeric(model.get(), "probe_diameter_m", 1)[0];
    if (!(duration > 0) || !(diameter > 0)) {
      throw std::runtime_error("invalid probe parameters");
    }

    auto depth = [&]() {
      const mjtNum* position = data->xpos + 3 * rock_body;
      double result = 0;
      for (int axis = 0; axis < 3; ++axis) {
        result += (position[axis] - mouth_center[axis]) * inward[axis];
      }
      return result;
    };

    mj_forward(model.get(), data.get());
    const double initial_depth = depth();
    double maximum_depth = initial_depth;
    int bucket_contact_steps = 0;
    int maximum_contacts = 0;
    while (data->time + 0.5 * model->opt.timestep < duration) {
      mj_step(model.get(), data.get());
      maximum_depth = std::max(maximum_depth, depth());
      maximum_contacts = std::max(maximum_contacts, data->ncon);
      bool bucket_contact = false;
      for (int index = 0; index < data->ncon; ++index) {
        const mjContact& contact = data->contact[index];
        const int other = contact.geom1 == rock_geom ? contact.geom2
                         : contact.geom2 == rock_geom ? contact.geom1 : -1;
        const char* name = other >= 0
                               ? mj_id2name(model.get(), mjOBJ_GEOM, other)
                               : nullptr;
        bucket_contact = bucket_contact || StartsWith(name, "bucket_");
      }
      if (bucket_contact) ++bucket_contact_steps;
      for (int index = 0; index < model->nq; ++index) {
        if (!std::isfinite(data->qpos[index])) {
          throw std::runtime_error("nonfinite qpos");
        }
      }
      for (int index = 0; index < model->nv; ++index) {
        if (!std::isfinite(data->qvel[index])) {
          throw std::runtime_error("nonfinite qvel");
        }
      }
      for (int warning = 0; warning < mjNWARNING; ++warning) {
        if (data->warning[warning].number) {
          throw std::runtime_error("MuJoCo warning");
        }
      }
    }
    const mjtNum* final_position = data->xpos + 3 * rock_body;
    const double final_speed = mju_norm3(data->qvel);
    const bool passed = maximum_depth > diameter / 2.0 && bucket_contact_steps > 0;
    std::cout << std::setprecision(17)
              << "{\"schema\":\"excavation_open_bucket_probe_result_v1\""
              << ",\"passed\":" << (passed ? "true" : "false")
              << ",\"diameter_m\":" << diameter
              << ",\"initial_mouth_depth_m\":" << initial_depth
              << ",\"maximum_mouth_depth_m\":" << maximum_depth
              << ",\"bucket_contact_steps\":" << bucket_contact_steps
              << ",\"maximum_contact_count\":" << maximum_contacts
              << ",\"final_speed_m_s\":" << final_speed
              << ",\"final_position_m\":[" << final_position[0] << ','
              << final_position[1] << ',' << final_position[2] << "]}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "mixed_material_r1_probe: " << error.what() << '\n';
    return 1;
  }
}
