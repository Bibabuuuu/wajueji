// Read-only kinematic postprocessing of saved native diagnostic states.
#include <mujoco/mujoco.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
int main(int argc,char** argv){
 try {
  if(argc!=2)throw std::runtime_error("expected model");
  char error[2048]={};
  std::unique_ptr<mjModel,decltype(&mj_deleteModel)> m(mj_loadXML(argv[1],nullptr,error,sizeof(error)),mj_deleteModel);
  if(!m)throw std::runtime_error(error);
  std::unique_ptr<mjData,decltype(&mj_deleteData)> d(mj_makeData(m.get()),mj_deleteData);
  int body=mj_name2id(m.get(),mjOBJ_BODY,"bucket");
  if(body<0 || mj_version()!=238)throw std::runtime_error("native identity");
  std::vector<mjtNum> jp(3*m->nv),jr(3*m->nv);
  double point[3];
  while(std::cin>>point[0]){
   std::cin>>point[1]>>point[2];
   for(int i=0;i<m->nq;++i)std::cin>>d->qpos[i];
   for(int i=0;i<m->nv;++i)std::cin>>d->qvel[i];
   if(!std::cin)throw std::runtime_error("input length");
   mj_kinematics(m.get(),d.get());mj_comPos(m.get(),d.get());
   mj_jac(m.get(),d.get(),jp.data(),jr.data(),point,body);
   std::cout<<std::setprecision(17)<<'[';
   for(int row=0;row<6;++row){
    double v=0;const auto& j=row<3?jp:jr;
    for(int k=0;k<m->nv;++k)v+=j[(row%3)*m->nv+k]*d->qvel[k];
    if(!std::isfinite(v))throw std::runtime_error("nonfinite endpoint velocity");
    if(row) { std::cout<<','; }
    std::cout<<v;
   }
   std::cout<<"]\n";
  }
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
