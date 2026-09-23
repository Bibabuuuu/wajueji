// Rendering and keyboard input only. Physics remains in the frozen R2 bridge.
#include <mujoco/mujoco.h>

#define GLFW_INCLUDE_NONE
#include "glfw3.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/select.h>
#include <unistd.h>

namespace fs = std::filesystem;

struct Viewer {
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model{nullptr, mj_deleteModel};
  std::unique_ptr<mjData, decltype(&mj_deleteData)> data{nullptr, mj_deleteData};
  GLFWwindow* window = nullptr;
  mjvCamera camera{};
  mjvOption option{};
  mjvScene scene{};
  mjrContext context{};
  std::array<bool, GLFW_KEY_LAST + 1> keys{};
  std::array<double, 4> maximum{{.12, .0225, .0225, .0675}};
  fs::path output;
  bool initialized = false;
  bool paused = true;
  bool close_requested = false;
  bool smoke = false;
  double speed_scale = .5;
  double last_x = 0;
  double last_y = 0;
  double closure_m = 0;
  double fill_percent = 0;
  double sim_time_s = 0;
  int forbidden_contacts = 0;
  int nearby_off_ground = 0;
  int sequence = 0;
  int frame = 0;
  std::string phase = "unmarked";
  std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();

  Viewer(const fs::path& model_path, const fs::path& output_path, bool smoke_mode)
      : output(fs::absolute(output_path)), smoke(smoke_mode) {
    char error[2048] = {};
    model.reset(mj_loadXML(model_path.c_str(), nullptr, error, sizeof(error)));
    if (!model) throw std::runtime_error(error);
    const int rock_count = (model->nq - 19) / 7;
    if (mj_version() != 238 || rock_count != 20 || model->nq != 159 ||
        model->nv != 138 || model->nu != 12 || model->nplugin != 0) {
      throw std::runtime_error("manual viewer model contract");
    }
    data.reset(mj_makeData(model.get()));
    if (!data) throw std::runtime_error("manual viewer data allocation");
    mj_forward(model.get(), data.get());
    fs::create_directories(output);
    glfwSetErrorCallback([](int code, const char* message) {
      std::cerr << "GLFW " << code << ": " << message << '\n';
    });
    if (!glfwInit()) throw std::runtime_error("glfw_init");
    if (smoke) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window = glfwCreateWindow(1440, 900,
        "R2 manual rock loading - PAUSED", nullptr, nullptr);
    if (!window) {
      glfwTerminate();
      throw std::runtime_error("glfw_window");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetWindowPos(window, 120, 60);
    mjv_defaultCamera(&camera);
    mjv_defaultOption(&option);
    mjv_defaultScene(&scene);
    mjr_defaultContext(&context);
    mjv_makeScene(model.get(), &scene, 12000);
    mjr_makeContext(model.get(), &context, mjFONTSCALE_150);
    initialized = true;
    set_view(1);
    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, [](GLFWwindow* target, int key, int, int action, int) {
      static_cast<Viewer*>(glfwGetWindowUserPointer(target))->on_key(key, action);
    });
    glfwSetWindowFocusCallback(window, [](GLFWwindow* target, int focused) {
      auto& viewer = *static_cast<Viewer*>(glfwGetWindowUserPointer(target));
      if (!focused) {
        viewer.keys.fill(false);
        viewer.paused = true;
        viewer.emit_event("focus_lost_pause");
      }
    });
    glfwSetCursorPosCallback(window, [](GLFWwindow* target, double x, double y) {
      auto& viewer = *static_cast<Viewer*>(glfwGetWindowUserPointer(target));
      const double dx = x - viewer.last_x;
      const double dy = y - viewer.last_y;
      viewer.last_x = x;
      viewer.last_y = y;
      const bool left = glfwGetMouseButton(target, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
      const bool right = glfwGetMouseButton(target, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
      if (!left && !right) return;
      int width, height;
      glfwGetWindowSize(target, &width, &height);
      if (height <= 0) return;
      const bool shift = glfwGetKey(target, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                         glfwGetKey(target, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
      const int mode = right ? (shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V)
                             : (shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V);
      mjv_moveCamera(viewer.model.get(), mode, dx / height, dy / height,
                     &viewer.scene, &viewer.camera);
    });
    glfwSetScrollCallback(window, [](GLFWwindow* target, double, double y) {
      auto& viewer = *static_cast<Viewer*>(glfwGetWindowUserPointer(target));
      mjv_moveCamera(viewer.model.get(), mjMOUSE_ZOOM, 0, -.05 * y,
                     &viewer.scene, &viewer.camera);
    });
  }

  ~Viewer() {
    if (initialized) {
      mjr_freeContext(&context);
      mjv_freeScene(&scene);
    }
    if (window) glfwDestroyWindow(window);
    glfwTerminate();
  }

  double wall_s() const {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
  }

  void emit_event(const std::string& name) {
    std::cout << "@EVENT " << name << ' ' << std::setprecision(17)
              << wall_s() << ' ' << sim_time_s << '\n' << std::flush;
  }

  void emit_marker(const std::string& name) {
    phase = name;
    std::cout << "@MARK " << name << ' ' << std::setprecision(17)
              << wall_s() << ' ' << sim_time_s << '\n' << std::flush;
  }

  void set_view(int number) {
    camera.type = mjCAMERA_FREE;
    camera.lookat[0] = 4.8;
    camera.lookat[1] = -.35;
    camera.lookat[2] = number == 1 ? 1.55 : .9;
    camera.distance = number == 1 ? 7.5 : 4.2;
    camera.azimuth = number == 1 ? 135 : number == 2 ? 90 : 0;
    camera.elevation = number == 1 ? -22 : -12;
  }

  void on_key(int key, int action) {
    if (key >= 0 && key <= GLFW_KEY_LAST && action != GLFW_REPEAT) {
      keys[key] = action == GLFW_PRESS;
    }
    if (action == GLFW_REPEAT) return;
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_X || key == GLFW_KEY_ESCAPE) {
      paused = true;
      keys.fill(false);
      close_requested = true;
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    } else if (key == GLFW_KEY_SPACE) {
      paused = !paused;
      keys.fill(false);
      emit_event(paused ? "pause" : "resume");
    } else if (key >= GLFW_KEY_1 && key <= GLFW_KEY_3) {
      set_view(key - GLFW_KEY_0);
    } else if (key == GLFW_KEY_MINUS) {
      speed_scale = std::max(.25, speed_scale / 2);
    } else if (key == GLFW_KEY_EQUAL) {
      speed_scale = std::min(1.0, speed_scale * 2);
    } else if (key == GLFW_KEY_F1) {
      emit_marker("approach_start");
    } else if (key == GLFW_KEY_F2) {
      emit_marker("curl_start");
    } else if (key == GLFW_KEY_F3) {
      emit_marker("lift_start");
    } else if (key == GLFW_KEY_F4) {
      emit_marker("evaluation_hold_start");
    } else if (key == GLFW_KEY_P) {
      save_frame("manual");
    }
  }

  std::array<double, 4> action() const {
    if (paused) return {};
    const int positive[4] = {GLFW_KEY_J, GLFW_KEY_I, GLFW_KEY_U, GLFW_KEY_M};
    const int negative[4] = {GLFW_KEY_L, GLFW_KEY_K, GLFW_KEY_O, GLFW_KEY_N};
    std::array<double, 4> result{};
    for (int axis = 0; axis < 4; ++axis) {
      result[axis] = maximum[axis] * speed_scale *
          (int(keys[positive[axis]]) - int(keys[negative[axis]]));
    }
    return result;
  }

  void emit_action() {
    const auto value = action();
    std::cout << "@ACTION " << ++sequence << ' ' << (paused ? 1 : 0) << ' '
              << std::setprecision(17) << speed_scale;
    for (double axis : value) std::cout << ' ' << axis;
    std::cout << '\n' << std::flush;
  }

  void synchronize(const std::string& line) {
    std::istringstream input(line);
    std::string operation;
    int nq = 0;
    input >> operation >> sim_time_s >> closure_m >> forbidden_contacts
          >> nearby_off_ground >> fill_percent >> nq;
    if (operation != "state" || nq != model->nq) {
      throw std::runtime_error("viewer state protocol");
    }
    for (int index = 0; index < nq; ++index) {
      if (!(input >> data->qpos[index]) || !std::isfinite(data->qpos[index])) {
        throw std::runtime_error("viewer nonfinite qpos");
      }
    }
    std::string extra;
    if (input >> extra) throw std::runtime_error("viewer extra state field");
    data->time = sim_time_s;
    mj_forward(model.get(), data.get());
  }

  bool read_pending_states() {
    std::string first;
    if (!std::getline(std::cin, first)) return false;
    synchronize(first);
    while (true) {
      fd_set descriptors;
      FD_ZERO(&descriptors);
      FD_SET(STDIN_FILENO, &descriptors);
      timeval timeout{0, 0};
      const int ready = select(STDIN_FILENO + 1, &descriptors, nullptr, nullptr,
                               &timeout);
      if (ready < 0) throw std::runtime_error("viewer stdin select");
      if (!ready) return true;
      std::string line;
      if (!std::getline(std::cin, line)) return false;
      synchronize(line);
    }
  }

  void save_frame(const std::string& label) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    if (width <= 0 || height <= 0) return;
    mjv_updateScene(model.get(), data.get(), &option, nullptr, &camera,
                    mjCAT_ALL, &scene);
    mjrRect viewport{0, 0, width, height};
    mjr_render(viewport, &scene, &context);
    std::vector<unsigned char> pixels(3 * width * height);
    mjr_readPixels(pixels.data(), nullptr, viewport, &context);
    std::ofstream file(output / ("frame-" + label + ".ppm"), std::ios::binary);
    file << "P6\n" << width << ' ' << height << "\n255\n";
    for (int row = height - 1; row >= 0; --row) {
      file.write(reinterpret_cast<const char*>(pixels.data() + 3 * width * row),
                 3 * width);
    }
  }

  void draw() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    if (width <= 0 || height <= 0) return;
    mjv_updateScene(model.get(), data.get(), &option, nullptr, &camera,
                    mjCAT_ALL, &scene);
    mjrRect viewport{0, 0, width, height};
    mjr_render(viewport, &scene, &context);
    std::ostringstream status;
    status << std::fixed << std::setprecision(3)
           << "R2 MANUAL / " << (paused ? "PAUSED" : "RUNNING")
           << " / command scale " << speed_scale << "x\n"
           << "sim " << sim_time_s << " s | phase " << phase << '\n'
           << "near+off-ground now " << nearby_off_ground
           << " | solid fill now " << fill_percent << "%\n"
           << "closure " << 1000 * closure_m << " mm | forbidden "
           << forbidden_contacts << '\n'
           << "Press F3 exactly when lifting starts; release all motion, press F4, "
              "hold >=1 s, then X.";
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport,
                status.str().c_str(), nullptr, &context);
    mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, viewport,
        "SPACE run/pause | X/ESC save+exit | J/L rotate | I/K boom up/down\n"
        "U/O arm in/out | M/N curl/open | release=hold | -/= speed\n"
        "F1 approach | F2 curl | F3 lift start | F4 evaluation hold | P screenshot\n"
        "1 wide | 2 side | 3 front | drag camera | wheel zoom",
        nullptr, &context);
    glfwSwapBuffers(window);
  }

  int run() {
    std::cout << "@READY " << model->nq << '\n' << std::flush;
    save_frame("initial");
    while (!glfwWindowShouldClose(window) && wall_s() < 1800.0) {
      const auto begin = std::chrono::steady_clock::now();
      if (!read_pending_states()) break;
      glfwPollEvents();
      if (smoke) {
        if (frame == 2) paused = false;
        if (frame >= 4 && frame < 8) keys[GLFW_KEY_I] = true;
        if (frame == 8) keys[GLFW_KEY_I] = false;
        if (frame == 10) emit_marker("lift_start");
        if (frame == 15) emit_marker("evaluation_hold_start");
        if (frame == 35) {
          close_requested = true;
          glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
      }
      emit_action();
      draw();
      ++frame;
      std::this_thread::sleep_until(begin + std::chrono::milliseconds(16));
    }
    paused = true;
    keys.fill(false);
    emit_action();
    save_frame("final");
    std::cout << "@CLOSE " << (close_requested ? "user" : "input_or_timeout")
              << ' ' << std::setprecision(17) << wall_s() << ' ' << sim_time_s
              << '\n' << std::flush;
    return 0;
  }
};

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 4) {
      throw std::runtime_error(
          "usage: mixed_material_r2_manual_viewer model.xml output [--smoke]");
    }
    const bool smoke = argc == 4 && std::string(argv[3]) == "--smoke";
    if (argc == 4 && !smoke) throw std::runtime_error("unknown viewer option");
    Viewer viewer(fs::absolute(argv[1]), fs::absolute(argv[2]), smoke);
    return viewer.run();
  } catch (const std::exception& error) {
    std::cerr << "r2 manual viewer: " << error.what() << '\n';
    return 1;
  }
}
