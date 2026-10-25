#include <mujoco/mujoco.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
int main(int argc,char** argv) {
  if(argc!=3) return 2;
  char error[2048]={};
  mjModel* m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
  if(!m){std::cerr<<error;return 3;}
  mjData* d=mj_makeData(m);
  std::ifstream state(argv[2]);
  for(int i=0;i<m->nq;++i) if(!(state>>d->qpos[i]))return 4;
  for(int i=0;i<m->nv;++i) if(!(state>>d->qvel[i]))return 5;
  mj_forward(m,d); // collision/forward only; no integration
  std::map<std::string,int> pairs;
  for(int i=0;i<d->ncon;++i) {
    const auto& c=d->contact[i];
    const char* an=mj_id2name(m,mjOBJ_GEOM,c.geom1);
    const char* bn=mj_id2name(m,mjOBJ_GEOM,c.geom2);
    std::string a=an?an:"unnamed", b=bn?bn:"unnamed";
    if(a>b)std::swap(a,b);
    ++pairs[a+" | "+b];
  }
  std::cout<<"{\"version\":"<<std::quoted(mj_versionString())
    <<",\"nq\":"<<m->nq<<",\"nv\":"<<m->nv<<",\"nbody\":"<<m->nbody
    <<",\"ngeom\":"<<m->ngeom<<",\"ncon\":"<<d->ncon
    <<",\"shelf_geom_id\":"<<mj_name2id(m,mjOBJ_GEOM,"r1_ground_push_shelf")
    <<",\"pairs\":{";
  bool first=true;
  for(const auto& p:pairs) {
    if(!first)std::cout<<",";first=false;
    std::cout<<std::quoted(p.first)<<":"<<p.second;
  }
  std::cout<<"},\"physics_steps\":0}\n";
  mj_deleteData(d);mj_deleteModel(m);
}
