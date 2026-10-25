#include <mujoco/mujoco.h>
#include <cstdio>
#include <cmath>
int main(int argc, char** argv) {
  char error[1024] = {};
  mjModel* m = mj_loadXML(argv[1], nullptr, error, sizeof(error));
  if (!m) {std::fprintf(stderr,"%s\n", error);return 1;}
  mjData* d = mj_makeData(m);
  int id = mj_name2id(m, mjOBJ_BODY, "bucket");
  double max_fixed=0., max_old=0.;
  for(int i=0;i<40;i++) {
    d->qpos[0]=.013*i; d->qpos[1]=-.4+.02*i;
    d->qvel[0]=.3; d->qvel[1]=-.2;
    mj_forward(m,d);
    mjtNum jp[6]={}, jr[6]={}, oldv[6]={}, newv[6]={};
    mj_jacBody(m,d,jp,jr,id);
    mj_objectVelocity(m,d,mjOBJ_BODY,id,oldv,0);
    mj_objectVelocity(m,d,mjOBJ_XBODY,id,newv,0);
    for(int k=0;k<3;k++) {
      double expected=jp[2*k]*d->qvel[0]+jp[2*k+1]*d->qvel[1];
      max_fixed=fmax(max_fixed,fabs(newv[3+k]-expected));
      max_old=fmax(max_old,fabs(oldv[3+k]-expected));
    }
  }
  std::printf("{\"poses\":40,\"origin_jacobian_error\":%.17g,\"old_COM_error\":%.17g}\n",max_fixed,max_old);
  mj_deleteData(d);mj_deleteModel(m);
  return (max_fixed<1e-12 && max_old>.05)?0:2;
}
