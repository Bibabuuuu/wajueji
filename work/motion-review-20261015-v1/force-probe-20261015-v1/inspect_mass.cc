// Offline mass evaluation only. No mj_forward, mj_step, soil or contact solve.
#include "/home/qunwang/wajueji/runs/loaded-force-probe-20261015-v1/physics/optimized/inertia_candidate.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  char error[2048] = {};
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
      mj_loadXML(argv[1], nullptr, error, sizeof(error)), mj_deleteModel);
  if (!model || model->nplugin != 0) { std::cerr << error; return 3; }
  InertiaFeedforward feedforward(model.get());
  std::unique_ptr<mjData, decltype(&mj_deleteData)> data(
      mj_makeData(model.get()), mj_deleteData);
  excavator::HoldController controller;
  if (!controller.Configure(model.get(), data.get())) return 4;
  std::ifstream input(argv[2]);
  int frame;
  std::cout << std::setprecision(17);
  while (input >> frame) {
    for (int i=0;i<19;++i) if (!(input >> data->qpos[i])) return 5;
    const auto result=feedforward.Evaluate(data->qpos,controller,{},{});
    std::cout << frame;
    for (double value:result.mass) std::cout << ',' << value;
    std::cout << '\n';
  }
  return 0;
}
