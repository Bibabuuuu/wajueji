// Load-only structural inspection for an isolated R2 multi-rock model.
#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

bool StartsWith(const char* value, const char* prefix) {
  return value && std::strncmp(value, prefix, std::strlen(prefix)) == 0;
}

int Id(const mjModel* model, mjtObj type, const char* name) {
  const int result = mj_name2id(model, type, name);
  if (result < 0) throw std::runtime_error(std::string("missing name: ") + name);
  return result;
}

int ParseCount(const char* value) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || parsed < 1 || parsed > 20) {
    throw std::runtime_error("expected rock count must be in [1,20]");
  }
  return static_cast<int>(parsed);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      throw std::runtime_error(
          "usage: mixed_material_r2_model_inspect model.xml expected_rock_count");
    }
    const int expected_rock_count = ParseCount(argv[2]);
    char error[2048] = {};
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
        mj_loadXML(argv[1], nullptr, error, sizeof(error)), mj_deleteModel);
    if (!model) throw std::runtime_error(error);
    const int rock_count = (model->nq - 19) / 7;
    if (mj_version() != 238 || rock_count != expected_rock_count ||
        rock_count < 1 || rock_count > 20 ||
        model->nq != 19 + 7 * rock_count ||
        model->nv != 18 + 6 * rock_count || model->nu != 12 ||
        model->nplugin != 0) {
      throw std::runtime_error("candidate model count contract");
    }

    const int source_bucket = Id(model.get(), mjOBJ_GEOM, "bucket");
    const int ground = Id(model.get(), mjOBJ_GEOM, "ground_plane");
    const int terrain = Id(model.get(), mjOBJ_GEOM, "r2_surface_terrain");
    if (model->geom_contype[source_bucket] != 0 ||
        model->geom_conaffinity[source_bucket] != 0 ||
        model->geom_contype[ground] != 1 ||
        model->geom_conaffinity[ground] != 1 ||
        model->geom_type[terrain] != mjGEOM_HFIELD ||
        model->geom_contype[terrain] != 1024 ||
        model->geom_conaffinity[terrain] != 256 ||
        model->nhfield != 1 || model->hfield_nrow[0] != 65 ||
        model->hfield_ncol[0] != 65) {
      throw std::runtime_error("source collision mask contract");
    }
    float minimum_heightfield_value = 1.0f;
    float maximum_heightfield_value = 0.0f;
    for (int index = 0; index < model->nhfielddata; ++index) {
      minimum_heightfield_value =
          std::min(minimum_heightfield_value, model->hfield_data[index]);
      maximum_heightfield_value =
          std::max(maximum_heightfield_value, model->hfield_data[index]);
    }
    if (minimum_heightfield_value != 0.0f ||
        maximum_heightfield_value != 1.0f) {
      throw std::runtime_error("heightfield normalization contract");
    }

    int proxy_geoms = 0;
    for (int geom = 0; geom < model->ngeom; ++geom) {
      const char* name = mj_id2name(model.get(), mjOBJ_GEOM, geom);
      if (!StartsWith(name, "r2_bucket_")) continue;
      ++proxy_geoms;
      if (model->geom_contype[geom] != 512 ||
          model->geom_conaffinity[geom] != 256) {
        throw std::runtime_error("bucket proxy collision mask");
      }
    }
    if (proxy_geoms != 18) throw std::runtime_error("bucket proxy count");

    double total_rock_mass = 0.0;
    for (int index = 0; index < rock_count; ++index) {
      char body_name[64], geom_name[64], joint_name[64];
      std::snprintf(body_name, sizeof(body_name), "r2-rock-%03d", index);
      std::snprintf(geom_name, sizeof(geom_name), "r2_rock_geom_%03d", index);
      std::snprintf(joint_name, sizeof(joint_name), "r2_rock_free_%03d", index);
      const int body = Id(model.get(), mjOBJ_BODY, body_name);
      const int geom = Id(model.get(), mjOBJ_GEOM, geom_name);
      const int joint = Id(model.get(), mjOBJ_JOINT, joint_name);
      if (model->jnt_type[joint] != mjJNT_FREE ||
          model->jnt_qposadr[joint] != 19 + 7 * index ||
          model->jnt_dofadr[joint] != 18 + 6 * index ||
          model->geom_contype[geom] != 256 ||
          model->geom_conaffinity[geom] != (256 | 512 | 1024 | 2048)) {
        throw std::runtime_error("rock collision or address contract");
      }
      if (!std::isfinite(model->body_mass[body]) || model->body_mass[body] <= 0) {
        throw std::runtime_error("invalid rock mass");
      }
      for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(model->body_inertia[3 * body + axis]) ||
            model->body_inertia[3 * body + axis] <= 0) {
          throw std::runtime_error("invalid rock principal inertia");
        }
      }
      total_rock_mass += model->body_mass[body];
    }
    for (int index = 0; index < model->nq; ++index) {
      if (!std::isfinite(model->qpos0[index])) {
        throw std::runtime_error("nonfinite qpos0");
      }
    }

    std::cout << std::setprecision(17)
              << "{\"schema\":\"excavation_mixed_material_r2_load_inspect_v1\""
              << ",\"status\":\"passed\",\"load_only\":true"
              << ",\"forward_dynamics_called\":false"
              << ",\"time_step_called\":false"
              << ",\"physics_qualified\":false"
              << ",\"nq\":" << model->nq << ",\"nv\":" << model->nv
              << ",\"nu\":" << model->nu
              << ",\"nplugin\":" << model->nplugin
              << ",\"rock_count\":" << rock_count
              << ",\"maximum_rock_capacity\":20"
              << ",\"placeholder_rock_count\":0"
              << ",\"bucket_proxy_geom_count\":" << proxy_geoms
              << ",\"heightfield_count\":" << model->nhfield
              << ",\"heightfield_nrow\":" << model->hfield_nrow[0]
              << ",\"heightfield_ncol\":" << model->hfield_ncol[0]
              << ",\"source_ground_plane_rock_collision_enabled\":false"
              << ",\"total_rock_mass_kg\":" << total_rock_mass << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "mixed_material_r2_model_inspect: " << error.what() << '\n';
    return 1;
  }
}
