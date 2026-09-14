// Independent MuJoCo mechanism. No soil plugin and no global callbacks.
#include <mujoco/mujoco.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <memory>
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>
#include "inertia_controller.h"

int main(int argc,char** argv) {
  try {
    if(argc!=2) throw std::runtime_error("usage: mpm_bridge plugin_free_model.xml");
    char error[2048]={};
    std::unique_ptr<mjModel,decltype(&mj_deleteModel)> m(mj_loadXML(argv[1],nullptr,error,2048),mj_deleteModel);
    if(!m) throw std::runtime_error(error);
    if(mj_version()!=238 || m->nq!=19 || m->nv!=18 || m->nu!=12 || m->nplugin!=0)
      throw std::runtime_error("model contract: MuJoCo 238, 19/18/12, zero plugins");
    std::unique_ptr<mjData,decltype(&mj_deleteData)> d(mj_makeData(m.get()),mj_deleteData);
    InertiaFeedforward ff(m.get());
    excavator::HoldController c;
    const int bucket=mj_name2id(m.get(),mjOBJ_BODY,"bucket");
    if(bucket<0) throw std::runtime_error("missing bucket");
    bool ready=false;
    std::array<double,4> drive_load{};
    std::vector<mjtNum> applied(m->nv);
    double power_residual=0,closure=0;
    auto emit=[&]() {
      std::cout<<std::setprecision(17)<<"@STATE {\"time_s\":"<<d->time;
      auto arr=[&](const char* name,const auto* p,int n) {
        std::cout<<",\""<<name<<"\":[";
        for(int i=0;i<n;++i){if(i)std::cout<<',';if(!std::isfinite(double(p[i])))throw std::runtime_error("nonfinite output");std::cout<<p[i];}
        std::cout<<']';
      };
      arr("qpos",d->qpos,m->nq);arr("qvel",d->qvel,m->nv);
      arr("bucket_origin",d->xpos+3*bucket,3);arr("bucket_rotation",d->xmat+9*bucket,9);
      arr("axis_qpos_address",c.qpos_address.data(),4);arr("axis_dof_address",c.dof_address.data(),4);
      arr("target",c.target.data(),4);arr("target_velocity",c.target_velocity.data(),4);
      arr("actuator_force",d->actuator_force,m->nu);
      arr("applied_generalized_force",applied.data(),m->nv);
      arr("equivalent_drive_load",drive_load.data(),4);
      std::cout<<",\"closure_m\":"<<closure<<",\"contact_count\":"<<d->ncon
        <<",\"virtual_power_residual_w\":"<<power_residual<<"}"<<std::endl;
    };
    std::cout<<"@READY"<<std::endl;
    std::string line;
    while(std::getline(std::cin,line)) {
      std::istringstream in(line);std::string op;in>>op;
      if(op=="close")break;
      auto number=[&]() {double x;if(!(in>>x)||!std::isfinite(x))throw std::runtime_error("invalid finite input");return x;};
      if(op=="init") {
        if(ready)throw std::runtime_error("init once only");
        for(int i=0;i<m->nq;++i)d->qpos[i]=number();
        std::string extra;if(in>>extra)throw std::runtime_error("extra init fields");
        mj_forward(m.get(),d.get());
        if(!c.Configure(m.get(),d.get()))throw std::runtime_error("controller init");
        ConfigureCandidate(m.get(),c,1,1);
        const auto initial_ff=ff.Evaluate(d->qpos,c,{},{});
        for(int i=0;i<4;++i)d->ctrl[c.hold_actuator[i]]=c.target[i]+initial_ff.force[i]/m->actuator_gainprm[mjNGAIN*c.hold_actuator[i]];
        mj_forward(m.get(),d.get());
        ready=true;emit();continue;
      }
      if(op!="step" || !ready)throw std::runtime_error("invalid operation");
      const double expected=number(),dt=number(),sub=number();
      if(std::abs(expected-d->time)>1e-8 || dt<=0 || dt>1./120+1e-12 || sub<20 || sub>160 || sub!=int(sub))
        throw std::runtime_error("step time/substep contract");
      std::array<double,4> command{};for(auto& x:command)x=number();
      constexpr double limits[4]={.12,.0225,.0225,.0675};
      for(int i=0;i<4;++i)if(std::abs(command[i])>limits[i]+1e-12)throw std::runtime_error("command limit");
      mjtNum force[3],torque[3],point[3];
      for(auto& x:force)x=number();
      for(auto& x:torque)x=number();
      for(auto& x:point)x=number();
      std::string extra;if(in>>extra)throw std::runtime_error("extra step fields");
      // The wrench is world-aligned about this FIXED world reference point.
      // Applying at that same point automatically preserves its net moment as
      // the bucket moves through the substeps. No COM/origin substitution.
      m->opt.timestep=dt/sub;
      for(int k=0;k<int(sub);++k) {
        const auto previous=c.target_velocity;
        const double elapsed=std::max(0.,double(d->time)-c.last_time);
        for(int i=0;i<4;++i)d->ctrl[c.command_actuator[i]]=command[i];
        if(!c.Update(m.get(),d.get()))throw std::runtime_error("controller update");
        std::array<double,4> acceleration{};
        for(int i=0;i<4;++i)if(elapsed>0)acceleration[i]=(c.target_velocity[i]-previous[i])/elapsed;
        const auto result=ff.Evaluate(d->qpos,c,c.target_velocity,acceleration);
        for(int i=0;i<4;++i)d->ctrl[c.hold_actuator[i]]=c.target[i]+result.force[i]/m->actuator_gainprm[mjNGAIN*c.hold_actuator[i]];
        mj_kinematics(m.get(),d.get());mj_comPos(m.get(),d.get());
        mju_zero(d->qfrc_applied,m->nv);
        mj_applyFT(m.get(),d.get(),force,torque,point,bucket,d->qfrc_applied);
        std::copy(d->qfrc_applied,d->qfrc_applied+m->nv,applied.begin());
        drive_load.fill(0);
        for(int i=0;i<4;++i)for(int v=0;v<m->nv;++v)drive_load[i]+=result.tangent[v*4+i]*applied[v];
        std::vector<mjtNum> jp(3*m->nv),jr(3*m->nv);
        mj_jac(m.get(),d.get(),jp.data(),jr.data(),point,bucket);
        double cartesian_power=0,generalized_power=0;
        for(int v=0;v<m->nv;++v) {
          generalized_power+=applied[v]*d->qvel[v];
          for(int axis=0;axis<3;++axis)cartesian_power+=(force[axis]*jp[axis*m->nv+v]+torque[axis]*jr[axis*m->nv+v])*d->qvel[v];
        }
        power_residual=std::max(power_residual,std::abs(cartesian_power-generalized_power));
        const double before=d->time;mj_step(m.get(),d.get());
        if(std::abs(d->time-before-m->opt.timestep)>1e-10)throw std::runtime_error("engine time reset");
        for(int i=0;i<mjNWARNING;++i)if(d->warning[i].number)throw std::runtime_error("engine warning");
        if(d->ncon)throw std::runtime_error("mechanism or ground collision");
        for(int i=0;i<m->nv;++i)if(!std::isfinite(d->qvel[i])||!std::isfinite(d->qacc[i]))throw std::runtime_error("nonfinite dynamics");
        // Refresh derived geometry at the returned q, without advancing time.
        mj_kinematics(m.get(),d.get());mj_comPos(m.get(),d.get());
        for(int e=0;e<m->neq;++e)if(m->eq_type[e]==mjEQ_CONNECT) {
          const int a=m->eq_obj1id[e],b=m->eq_obj2id[e];mjtNum pa[3],pb[3];
          mju_rotVecMat(pa,m->eq_data+mjNEQDATA*e,d->xmat+9*a);mju_addTo3(pa,d->xpos+3*a);
          mju_rotVecMat(pb,m->eq_data+mjNEQDATA*e+3,d->xmat+9*b);mju_addTo3(pb,d->xpos+3*b);
          closure=std::max(closure,double(mju_dist3(pa,pb)));
        }
        if(closure>.001)throw std::runtime_error("dynamic closure above 1 mm");
        for(int j=0;j<m->njnt;++j)if(m->jnt_limited[j]) {
          double q=d->qpos[m->jnt_qposadr[j]];
          if(q<m->jnt_range[2*j] || q>m->jnt_range[2*j+1])throw std::runtime_error("joint range");
        }
      }
      emit();
    }
  } catch(const std::exception& e) {std::cerr<<"mpm_bridge: "<<e.what()<<std::endl;return 1;}
}
