// Isolated C2 experiment. Four-port rigid-linkage inverse-dynamics feedforward.
#pragma once
#include "gravity_controller.h"

struct InertiaResult {
  GravityResult gravity;
  std::array<double,4> extra{},force{},velocity_bias{},acceleration_force{};
  std::array<double,16> mass{};
  std::vector<mjtNum> tangent;
};

class InertiaFeedforward {
 public:
  // Construct BEFORE the live mjData. The private model/data are used only for
  // kinematics/CRB/RNE, never forward/step/plugin compute. Plugin initialization
  // can seed the process-global RNG: the live instance must initialize last.
  explicit InertiaFeedforward(const mjModel* model) {
    m=mj_copyModel(nullptr,model);
    if(!m)throw std::runtime_error("inertia: copy model failed");
    d=mj_makeData(m);
    if(!d){mj_deleteModel(m);throw std::runtime_error("inertia: scratch allocation failed");}
  }
  ~InertiaFeedforward(){mj_deleteData(d);mj_deleteModel(m);}
  InertiaFeedforward(const InertiaFeedforward&)=delete;
  InertiaFeedforward& operator=(const InertiaFeedforward&)=delete;

  std::vector<mjtNum> Tangent(const mjtNum* q,const excavator::HoldController& c) {
    mju_copy(d->qpos,q,m->nq);mj_kinematics(m,d);mj_comPos(m,d);
    std::vector<int> passive;
    for(int j=0;j<m->njnt;++j)if(m->jnt_type[j]!=mjJNT_FREE) {
      int v=m->jnt_dofadr[j];
      if(std::find(c.dof_address.begin(),c.dof_address.end(),v)==c.dof_address.end())passive.push_back(v);
    }
    if(passive.size()!=8)throw std::runtime_error("inertia: passive count");
    std::vector<mjtNum> J,ja(3*m->nv),jb(3*m->nv),T(m->nv*4,0);
    for(int e=0;e<m->neq;++e)if(m->eq_type[e]==mjEQ_CONNECT) {
      int a=m->eq_obj1id[e],b=m->eq_obj2id[e];mjtNum pa[3],pb[3];
      mju_rotVecMat(pa,m->eq_data+mjNEQDATA*e,d->xmat+9*a);mju_addTo3(pa,d->xpos+3*a);
      mju_rotVecMat(pb,m->eq_data+mjNEQDATA*e+3,d->xmat+9*b);mju_addTo3(pb,d->xpos+3*b);
      mj_jac(m,d,ja.data(),nullptr,pa,a);mj_jac(m,d,jb.data(),nullptr,pb,b);
      for(int k=0;k<3*m->nv;++k)J.push_back(ja[k]-jb[k]);
    }
    if(J.size()!=size_t(12*m->nv))throw std::runtime_error("inertia: connect count");
    std::array<mjtNum,64> A{};
    for(int i=0;i<8;++i)for(int j=0;j<8;++j)for(int r=0;r<12;++r)
      A[i*8+j]+=J[r*m->nv+passive[i]]*J[r*m->nv+passive[j]];
    if(mju_cholFactor(A.data(),8,1e-12)!=8)throw std::runtime_error("inertia: tangent rank");
    for(int axis=0;axis<4;++axis) {
      T[c.dof_address[axis]*4+axis]=1;
      if(axis==0)continue; // C0 convention: common yaw rotates soft world error.
      std::array<mjtNum,8> rhs{},x{};
      for(int i=0;i<8;++i)for(int r=0;r<12;++r)rhs[i]-=J[r*m->nv+passive[i]]*J[r*m->nv+c.dof_address[axis]];
      mju_cholSolve(x.data(),A.data(),rhs.data(),8);
      for(int i=0;i<8;++i)T[passive[i]*4+axis]=x[i];
      for(int r=0;r<12;++r){double residual=0;for(int v=0;v<m->nv;++v)residual+=J[r*m->nv+v]*T[v*4+axis];
        if(!std::isfinite(residual)||std::abs(residual)>1e-6)throw std::runtime_error("inertia: tangent residual");}
    }
    return T;
  }

  InertiaResult Evaluate(const mjtNum* q,const excavator::HoldController& c,
                        const std::array<double,4>& velocity,
                        const std::array<double,4>& acceleration,double epsilon=1e-5) {
    InertiaResult out;out.tangent=Tangent(q,c);const auto& T=out.tangent;
    std::vector<mjtNum> M(m->nv*m->nv),v(m->nv,0),a(m->nv,0),curve(m->nv,0),bias(m->nv),zero_bias(m->nv);
    mj_crb(m,d);mj_fullM(m,M.data(),d->qM);
    for(int i=0;i<4;++i)for(int j=0;j<4;++j)for(int u=0;u<m->nv;++u)for(int w=0;w<m->nv;++w)
      out.mass[i*4+j]+=T[u*4+i]*M[u*m->nv+w]*T[w*4+j];
    auto factor=out.mass;if(mju_cholFactor(factor.data(),4,1e-8)!=4)throw std::runtime_error("inertia: reduced mass not positive definite");
    for(int u=0;u<m->nv;++u)for(int i=0;i<4;++i){v[u]+=T[u*4+i]*velocity[i];a[u]+=T[u*4+i]*acceleration[i];}
    if(mju_norm(v.data(),m->nv)>0) {
      std::vector<mjtNum> plus(q,q+m->nq),minus=plus;
      mj_integratePos(m,plus.data(),v.data(),epsilon);mj_integratePos(m,minus.data(),v.data(),-epsilon);
      auto Tp=Tangent(plus.data(),c),Tm=Tangent(minus.data(),c);
      for(int u=0;u<m->nv;++u)for(int i=0;i<4;++i)curve[u]+=(Tp[u*4+i]-Tm[u*4+i])/(2*epsilon)*velocity[i];
    }
    // Rebuild the nominal kinematic state after directional differencing.
    mju_copy(d->qpos,q,m->nq);mj_kinematics(m,d);mj_comPos(m,d);
    mju_zero(d->qvel,m->nv);mj_comVel(m,d);mj_rne(m,d,0,zero_bias.data());
    out.gravity=GravityEffort(m,d,c);
    mju_copy(d->qvel,v.data(),m->nv);mj_comVel(m,d);mj_rne(m,d,0,bias.data());
    for(int i=0;i<4;++i)for(int u=0;u<m->nv;++u) {
      double inertia=0,centripetal=bias[u]-zero_bias[u];
      for(int w=0;w<m->nv;++w){inertia+=M[u*m->nv+w]*a[w];centripetal+=M[u*m->nv+w]*curve[w];}
      out.acceleration_force[i]+=T[u*4+i]*inertia;
      out.velocity_bias[i]+=T[u*4+i]*centripetal;
    }
    for(int i=0;i<4;++i){out.extra[i]=out.acceleration_force[i]+out.velocity_bias[i];out.force[i]=out.gravity.force[i]+out.extra[i];
      if(!std::isfinite(out.force[i])||std::abs(out.force[i])>1e6)throw std::runtime_error("inertia: effort nonfinite or over 1e6");}
    return out;
  }
 private:
  mjModel* m=nullptr; mjData* d=nullptr;
};
