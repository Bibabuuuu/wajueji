#ifndef EXCAVATOR_HOLD_CONTROLLER_H_
#define EXCAVATOR_HOLD_CONTROLLER_H_

#include <array>
#include <mujoco/mujoco.h>
#include "bounded_velocity.h"

namespace excavator {
inline constexpr std::array<const char*,4> kCommandActuatorNames={"Rotation","Boom","Arm","Bucket"};
inline constexpr std::array<const char*,4> kHoldActuatorNames={"Rotation hold","Boom hold","Arm hold","Bucket hold"};
inline constexpr std::array<const char*,4> kDriveActuatorNames={"Rotation drive","Boom drive","Arm drive","Bucket drive"};
inline constexpr std::array<const char*,4> kControlledJointNames={"chassis","chassis piston rod","boom piston rod","arm piston rod"};
inline constexpr std::array<mjtNum,4> kMaxTargetAcceleration={1.5,1,1,1};

inline bool ValidGovernedPorts(const mjModel* m) {
  for(int i=0;i<4;++i) {
    const int input=mj_name2id(m,mjOBJ_ACTUATOR,kCommandActuatorNames[i]);
    const int drive=mj_name2id(m,mjOBJ_ACTUATOR,kDriveActuatorNames[i]);
    const int hold=mj_name2id(m,mjOBJ_ACTUATOR,kHoldActuatorNames[i]);
    const int joint=mj_name2id(m,mjOBJ_JOINT,kControlledJointNames[i]);
    if(input<0||drive<0||hold<0||joint<0)return false;
    if(m->actuator_gaintype[input]!=mjGAIN_FIXED||m->actuator_gainprm[mjNGAIN*input]!=0||m->actuator_biastype[input]!=mjBIAS_NONE)return false;
    for(int id:{input,drive,hold})if(m->actuator_trntype[id]!=mjTRN_JOINT||m->actuator_trnid[2*id]!=joint)return false;
    if(m->actuator_gaintype[drive]!=mjGAIN_FIXED||m->actuator_biastype[drive]!=mjBIAS_AFFINE||m->actuator_gainprm[mjNGAIN*drive]<=0)return false;
  }
  return true;
}

// Per-environment state, shared by GUI and headless tests. No global callbacks.
struct HoldController {
  const mjModel* model=nullptr;
  const mjData* data=nullptr;
  std::array<int,4> command_actuator{},hold_actuator{},drive_actuator{},qpos_address{},dof_address{};
  std::array<mjtNum,4> lower_limit{},upper_limit{},target{},target_velocity{},effective_velocity{};
  std::array<bool,4> limited{};
  std::array<BoundedVelocity,4> reference{};
  mjtNum last_time=0;
  bool available=false, governed=false;

  bool Configure(const mjModel* m,mjData* d) {
    *this={}; model=m;data=d;
    if(!m||!d)return false;
    int drives=0;
    for(int i=0;i<4;++i) {
      command_actuator[i]=mj_name2id(m,mjOBJ_ACTUATOR,kCommandActuatorNames[i]);
      hold_actuator[i]=mj_name2id(m,mjOBJ_ACTUATOR,kHoldActuatorNames[i]);
      drive_actuator[i]=mj_name2id(m,mjOBJ_ACTUATOR,kDriveActuatorNames[i]);
      const int j=mj_name2id(m,mjOBJ_JOINT,kControlledJointNames[i]);
      if(command_actuator[i]<0||hold_actuator[i]<0||j<0)return false;
      if(drive_actuator[i]>=0)++drives;
      for(int actuator:{command_actuator[i],hold_actuator[i],drive_actuator[i]}) {
        if(actuator>=0&&(m->actuator_trntype[actuator]!=mjTRN_JOINT||m->actuator_trnid[2*actuator]!=j))return false;
      }
      qpos_address[i]=m->jnt_qposadr[j];dof_address[i]=m->jnt_dofadr[j];
      limited[i]=m->jnt_limited[j];lower_limit[i]=m->jnt_range[2*j];upper_limit[i]=m->jnt_range[2*j+1];
      target[i]=d->qpos[qpos_address[i]];reference[i].Reset(target[i]);
      if(!reference[i].valid)return false;
    }
    if(drives!=0&&drives!=4)return false;
    governed=drives==4;
    // In the governed model raw command ports must not apply any force.
    if(governed&&!ValidGovernedPorts(m))return false;
    for(int i=0;i<4;++i) {
      d->ctrl[hold_actuator[i]]=target[i];
      if(governed)d->ctrl[drive_actuator[i]]=0;
    }
    last_time=d->time;available=true;return true;
  }

  bool Update(const mjModel* m,mjData* d) {
    if(!m||!d)return false;
    if(model!=m||data!=d||d->time<last_time) {
      if(!Configure(m,d))return false;
    }
    if(!available)return false;
    const double dt=std::max(0.0,d->time-last_time);
    for(int i=0;i<4;++i) {
      const int input=command_actuator[i];
      double request=d->ctrl[input];
      if(!std::isfinite(request))request=0;
      if(m->actuator_ctrllimited[input])request=std::clamp(request,m->actuator_ctrlrange[2*input],m->actuator_ctrlrange[2*input+1]);
      if(governed) {
        VelocityBounds bounds;
        bounds.acceleration=kMaxTargetAcceleration[i];
        bounds.limited=limited[i];bounds.lower=lower_limit[i];bounds.upper=upper_limit[i];
        // Nominal governor must differentiate its own consistent state.
        // Measured q/qd remain in MuJoCo servo feedback and bridge guards.
        reference[i].Step(request,reference[i].target,reference[i].velocity,dt,bounds);
        if(!reference[i].valid) {
          for(int id:drive_actuator)d->ctrl[id]=0;
          return false;
        }
        target[i]=reference[i].target;target_velocity[i]=reference[i].velocity;
        effective_velocity[i]=target_velocity[i];d->ctrl[drive_actuator[i]]=effective_velocity[i];
      } else {
        // Eight-actuator historical models retain their original control law.
        const double delta=kMaxTargetAcceleration[i]*dt;
        target_velocity[i]+=std::clamp(request-target_velocity[i],-delta,delta);
        target[i]+=target_velocity[i]*dt;
        if(limited[i]) {
          const double t=std::clamp(target[i],lower_limit[i],upper_limit[i]);
          if(t!=target[i])target_velocity[i]=0;
          target[i]=t;
        }
        effective_velocity[i]=request;
      }
      d->ctrl[hold_actuator[i]]=target[i];
    }
    last_time=d->time;return true;
  }
};
}  // namespace excavator
#endif
