// Isolated, local-assembly self-weight feedforward. No passive-joint support.
#pragma once
#include <mujoco/mujoco.h>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "excavator_simulator/hold_controller.h"

struct GravityResult {
  std::array<double,4> force{};
  double residual=0;
  int rank=0;
};

inline GravityResult GravityEffort(const mjModel* m,mjData* d,const excavator::HoldController& c) {
  // These kinematic stages do not invoke soil/plugin callbacks or advance time.
  mj_kinematics(m,d);mj_comPos(m,d);
  std::vector<int> passive;
  for(int j=0;j<m->njnt;++j) {
    if(m->jnt_type[j]==mjJNT_FREE)continue; // base is treated as locally supported
    int v=m->jnt_dofadr[j];
    if(std::find(c.dof_address.begin(),c.dof_address.end(),v)==c.dof_address.end())passive.push_back(v);
  }
  if(passive.size()!=8)throw std::runtime_error("gravity: expected eight passive scalar coordinates");
  std::vector<mjtNum> J,ja(3*m->nv),jb(3*m->nv),gravity(m->nv,0);
  for(int e=0;e<m->neq;++e)if(m->eq_type[e]==mjEQ_CONNECT) {
    const int a=m->eq_obj1id[e],b=m->eq_obj2id[e];mjtNum pa[3],pb[3];
    mju_rotVecMat(pa,m->eq_data+mjNEQDATA*e,d->xmat+9*a);mju_addTo3(pa,d->xpos+3*a);
    mju_rotVecMat(pb,m->eq_data+mjNEQDATA*e+3,d->xmat+9*b);mju_addTo3(pb,d->xpos+3*b);
    mj_jac(m,d,ja.data(),nullptr,pa,a);mj_jac(m,d,jb.data(),nullptr,pb,b);
    for(int k=0;k<3*m->nv;++k)J.push_back(ja[k]-jb[k]);
  }
  if(J.size()!=size_t(12*m->nv))throw std::runtime_error("gravity: expected four connect constraints");
  for(int b=1;b<m->nbody;++b)if(m->body_mass[b]>0) {
    mj_jacBodyCom(m,d,ja.data(),nullptr,b);
    for(int v=0;v<m->nv;++v)for(int k=0;k<3;++k)
      gravity[v]-=m->body_mass[b]*m->opt.gravity[k]*ja[k*m->nv+v];
  }
  std::array<mjtNum,64> normal{};
  for(int i=0;i<8;++i)for(int j=0;j<8;++j)for(int r=0;r<12;++r)
    normal[i*8+j]+=J[r*m->nv+passive[i]]*J[r*m->nv+passive[j]];
  GravityResult out;out.rank=mju_cholFactor(normal.data(),8,1e-12);
  if(out.rank!=8)throw std::runtime_error("gravity: singular passive tangent");
  for(int axis=0;axis<4;++axis) {
    // Common chassis yaw rotates the whole linkage, including any soft anchor
    // error. In chassis coordinates that error is invariant. Passive tangent
    // rates are exactly zero; fixing a nonzero WORLD error vector would wrongly
    // reject this valid rigid rotation. Weight torque is still evaluated in world.
    if(axis==0){out.force[axis]=gravity[c.dof_address[axis]];continue;}
    std::array<mjtNum,8> rhs{},x{};
    for(int i=0;i<8;++i)for(int r=0;r<12;++r)
      rhs[i]-=J[r*m->nv+passive[i]]*J[r*m->nv+c.dof_address[axis]];
    mju_cholSolve(x.data(),normal.data(),rhs.data(),8);
    out.force[axis]=gravity[c.dof_address[axis]];
    for(int i=0;i<8;++i)out.force[axis]+=x[i]*gravity[passive[i]];
    for(int r=0;r<12;++r) {
      double residual=J[r*m->nv+c.dof_address[axis]];
      for(int i=0;i<8;++i)residual+=J[r*m->nv+passive[i]]*x[i];
      out.residual=std::max(out.residual,std::abs(residual));
    }
    if(!std::isfinite(out.force[axis])||std::abs(out.force[axis])>1e6)
      throw std::runtime_error("gravity: nonfinite or excessive effort");
  }
  if(out.residual>1e-6)throw std::runtime_error("gravity: inconsistent tangent");
  return out;
}

inline void ConfigureCandidate(mjModel* m,excavator::HoldController& c,int variant,double stiffness_factor) {
  if(m->nu!=12||!excavator::ValidGovernedPorts(m)||!(stiffness_factor>0&&stiffness_factor<=1))
    throw std::runtime_error("candidate contract");
  for(int i=0;i<4;++i) {
    int hold=c.hold_actuator[i],drive=c.drive_actuator[i];
    if(m->actuator_gear[6*hold]!=1||m->actuator_gear[6*drive]!=1||m->actuator_ctrllimited[hold])
      throw std::runtime_error("candidate requires unbounded unit-gear hold effort path");
    if(variant==2) {
      // Continuous force is unchanged by damping relocation; Euler treats this implicitly.
      const double kv=m->actuator_gainprm[mjNGAIN*drive];
      if(m->actuator_biasprm[mjNBIAS*drive+2]!=-kv)throw std::runtime_error("drive bias contract");
      m->dof_damping[c.dof_address[i]]+=kv;
      m->actuator_biasprm[mjNBIAS*drive+2]=0;
      m->actuator_gainprm[mjNGAIN*hold]*=stiffness_factor;
      m->actuator_biasprm[mjNBIAS*hold+1]*=stiffness_factor;
    } else if(stiffness_factor!=1)throw std::runtime_error("C0 gains must remain unchanged");
  }
}
