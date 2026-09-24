// D152: pure kinematic/dynamic coefficient export. No mj_step or live mutation.
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <stdexcept>
#include "/home/qunwang/wajueji/runs/contact-geometric-probe-20260924-v9-handoff/contact1200/physics/optimized/inertia_candidate.h"
template<class A> void array(const A& x){std::cout<<'[';bool first=true;for(auto v:x){if(!first)std::cout<<',';first=false;if(!std::isfinite(v))throw std::runtime_error("nonfinite");std::cout<<v;}std::cout<<']';}
int main(int argc,char** argv){try{
 if(argc!=2)throw std::runtime_error("expected model");
 char err[2048]={};
 std::unique_ptr<mjModel,decltype(&mj_deleteModel)> m(mj_loadXML(argv[1],nullptr,err,sizeof(err)),mj_deleteModel);
 if(!m)throw std::runtime_error(err);
 if(mj_version()!=238||m->nplugin!=0)throw std::runtime_error("model version");
 InertiaFeedforward ff(m.get());std::unique_ptr<mjData,decltype(&mj_deleteData)> d(mj_makeData(m.get()),mj_deleteData);
 excavator::HoldController c;int bucket=mj_name2id(m.get(),mjOBJ_BODY,"bucket");double p[3];
 std::cout<<std::unitbuf;
 while(std::cin>>p[0]){std::cin>>p[1]>>p[2];for(int i=0;i<m->nq;++i)std::cin>>d->qpos[i];for(int i=0;i<m->nv;++i)std::cin>>d->qvel[i];if(!std::cin)throw std::runtime_error("input");
 if(!c.Configure(m.get(),d.get()))throw std::runtime_error("controller");
 ConfigureCandidate(m.get(),c,1,1);
 auto out=ff.Evaluate(d->qpos,c,std::array<double,4>{},std::array<double,4>{});
 mj_kinematics(m.get(),d.get());mj_comPos(m.get(),d.get());std::vector<mjtNum> jp(3*m->nv),jr(3*m->nv);mj_jac(m.get(),d.get(),jp.data(),jr.data(),p,bucket);
 std::array<double,24> J{};std::array<double,16> P{};std::array<double,4> K{},D{};
 for(int i=0;i<4;++i){int h=c.hold_actuator[i],a=c.drive_actuator[i];K[i]=-m->actuator_biasprm[mjNBIAS*h+1];D[i]=-m->actuator_biasprm[mjNBIAS*a+2];
 for(int j=0;j<4;++j)for(int v=0;v<m->nv;++v)P[i*4+j]+=out.tangent[v*4+i]*m->dof_damping[v]*out.tangent[v*4+j];}
 for(int r=0;r<6;++r)for(int a=0;a<4;++a)for(int v=0;v<m->nv;++v)J[r*4+a]+=(r<3?jp[r*m->nv+v]:jr[(r-3)*m->nv+v])*out.tangent[v*4+a];
 std::cout<<std::setprecision(17)<<"{\"mass\":";array(out.mass);std::cout<<",\"jacobian\":";array(J);std::cout<<",\"passive_damping\":";array(P);std::cout<<",\"stiffness\":";array(K);std::cout<<",\"drive_damping\":";array(D);std::cout<<",\"integrator\":"<<m->opt.integrator<<"}\n";
 }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
