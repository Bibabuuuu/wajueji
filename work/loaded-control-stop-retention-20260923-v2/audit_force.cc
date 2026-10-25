#include <mujoco/mujoco.h>
#include <iostream>
int main(int argc, char** argv) {
  char error[2048]{};
  mjModel* m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
  if(!m){std::cerr<<error;return 1;}
  std::cout<<"{\"mujoco_version\":\""<<mj_versionString()<<"\",\"actuators\":[";
  for(int a=0;a<m->nu;++a){
    if(a)std::cout<<',';
    std::cout<<"{\"id\":"<<a<<",\"name\":\""<<mj_id2name(m,mjOBJ_ACTUATOR,a)<<"\",\"ctrl_limited\":"<<int(m->actuator_ctrllimited[a])<<",\"force_limited\":"<<int(m->actuator_forcelimited[a])<<",\"force_range\":["<<m->actuator_forcerange[a*2]<<','<<m->actuator_forcerange[a*2+1]<<"],\"gear\":"<<m->actuator_gear[a*6]<<",\"gain\":"<<m->actuator_gainprm[a*mjNGAIN]<<",\"joint\":\""<<mj_id2name(m,mjOBJ_JOINT,m->actuator_trnid[a*2])<<"\"}";
  }
  std::cout<<"],\"joints\":[";
  for(int j=0;j<m->njnt;++j){if(j)std::cout<<',';std::cout<<"{\"name\":\""<<(mj_id2name(m,mjOBJ_JOINT,j)?mj_id2name(m,mjOBJ_JOINT,j):"unnamed")<<"\",\"actuator_force_limited\":"<<int(m->jnt_actfrclimited[j])<<",\"range\":["<<m->jnt_actfrcrange[2*j]<<','<<m->jnt_actfrcrange[2*j+1]<<"]}";}
  std::cout<<"]}\n";mj_deleteModel(m);
}
