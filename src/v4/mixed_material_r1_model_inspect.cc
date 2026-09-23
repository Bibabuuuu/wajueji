// Load-only contract probe for the isolated one-rock mechanism candidate.
#include <mujoco/mujoco.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool StartsWith(const char* value, const char* prefix) {
  return value && std::strncmp(value, prefix, std::strlen(prefix)) == 0;
}

int Id(const mjModel* model, mjtObj type, const char* name) {
  const int result = mj_name2id(model, type, name);
  if (result < 0) throw std::runtime_error(std::string("missing name: ") + name);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error("usage: mixed_material_r1_model_inspect model.xml");
    }
    char error[2048] = {};
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
        mj_loadXML(argv[1], nullptr, error, sizeof(error)), mj_deleteModel);
    if (!model) throw std::runtime_error(error);
    const int rock_count = (model->nq - 19) / 7;
    if (mj_version() != 238 || rock_count < 1 || rock_count > 2 ||
        model->nq != 19 + 7 * rock_count ||
        model->nv != 18 + 6 * rock_count ||
        model->nu != 12 || model->nplugin != 0) {
      throw std::runtime_error("candidate model count contract");
    }
    const int bucket_geom = Id(model.get(), mjOBJ_GEOM, "bucket");
    const int rock_ground = Id(model.get(), mjOBJ_GEOM, "ground_plane");
    if (model->geom_contype[bucket_geom] != 0 ||
        model->geom_conaffinity[bucket_geom] != 0 ||
        model->geom_contype[rock_ground] != (1 | 1024) ||
        model->geom_conaffinity[rock_ground] != (1 | 256)) {
      throw std::runtime_error("candidate collision or address contract");
    }
    std::vector<int> rock_bodies;
    std::vector<int> rock_joints;
    std::vector<double> rock_masses;
    double total_rock_mass = 0;
    for (int index = 0; index < rock_count; ++index) {
      char body_name[64], geom_name[64], joint_name[64];
      std::snprintf(body_name, sizeof(body_name), "r1-rock-%03d", index);
      std::snprintf(geom_name, sizeof(geom_name), "r1_rock_geom_%03d", index);
      std::snprintf(joint_name, sizeof(joint_name), "r1_rock_free_%03d", index);
      const int rock_body = Id(model.get(), mjOBJ_BODY, body_name);
      const int rock_geom = Id(model.get(), mjOBJ_GEOM, geom_name);
      const int rock_joint = Id(model.get(), mjOBJ_JOINT, joint_name);
      if (model->jnt_type[rock_joint] != mjJNT_FREE ||
          model->jnt_qposadr[rock_joint] != 19 + 7 * index ||
          model->jnt_dofadr[rock_joint] != 18 + 6 * index ||
          model->geom_contype[rock_geom] != 256 ||
          model->geom_conaffinity[rock_geom] != (256 | 512 | 1024 | 2048)) {
        throw std::runtime_error("rock collision or address contract");
      }
      rock_bodies.push_back(rock_body);
      rock_joints.push_back(rock_joint);
      rock_masses.push_back(model->body_mass[rock_body]);
      total_rock_mass += model->body_mass[rock_body];
    }
    int proxy_geoms = 0;
    for (int geom = 0; geom < model->ngeom; ++geom) {
      const char* name = mj_id2name(model.get(), mjOBJ_GEOM, geom);
      if (!StartsWith(name, "r1_bucket_")) continue;
      ++proxy_geoms;
      if (model->geom_contype[geom] != 512 ||
          model->geom_conaffinity[geom] != 256) {
        throw std::runtime_error("bucket proxy collision mask");
      }
    }
    if (proxy_geoms != 18) throw std::runtime_error("bucket proxy count");
    std::unique_ptr<mjData, decltype(&mj_deleteData)> data(
        mj_makeData(model.get()), mj_deleteData);
    mj_forward(model.get(), data.get());
    for (int index = 0; index < model->nq; ++index) {
      if (!std::isfinite(data->qpos[index])) throw std::runtime_error("nonfinite qpos");
    }
    for (int warning = 0; warning < mjNWARNING; ++warning) {
      if (data->warning[warning].number) throw std::runtime_error("MuJoCo warning");
    }
    std::cout << std::setprecision(17)
              << "{\"schema\":\"excavation_mixed_rigid_model_inspect_v1\""
              << ",\"passed\":true,\"nq\":" << model->nq
              << ",\"nv\":" << model->nv << ",\"nu\":" << model->nu
              << ",\"nplugin\":" << model->nplugin
              << ",\"rock_count\":" << rock_count
              << ",\"bucket_proxy_geom_count\":" << proxy_geoms
              << ",\"rock_qpos_address\":" << model->jnt_qposadr[rock_joints[0]]
              << ",\"rock_dof_address\":" << model->jnt_dofadr[rock_joints[0]]
              << ",\"rock_mass_kg\":" << rock_masses[0]
              << ",\"total_rock_mass_kg\":" << total_rock_mass
              << ",\"rock_masses_kg\":[";
    for (int index = 0; index < rock_count; ++index) {
      if (index) std::cout << ',';
      std::cout << rock_masses[index];
    }
    std::cout
              << "]"
              << ",\"initial_contact_count\":" << data->ncon << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "mixed_material_r1_model_inspect: " << error.what() << '\n';
    return 1;
  }
}
