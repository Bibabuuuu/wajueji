"""GPU candidate checks against independent float64 spectral/rotation oracle."""
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
import warp.fem as fem
import soil_hencky_update_candidate as candidate
from newton._src.solvers.implicit_mpm import implicit_mpm_solver_kernels as original
from newton._src.solvers.implicit_mpm import solver_implicit_mpm as solver_module

OUT=Path(__file__).resolve().parent

def exp_sym(x):
    d,u=np.linalg.eigh(x);return (u*np.exp(d))@u.T

def rotation(v):
    angle=np.linalg.norm(v)
    if angle==0:return np.eye(3)
    x,y,z=v/angle;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)

@wp.kernel
def batch(F:wp.array[wp.mat33],E:wp.array[wp.mat33],W:wp.array[wp.mat33],out:wp.array[wp.mat33],code:wp.array[int]):
    i=wp.tid();value,status=candidate.hencky_proposal(F[i],E[i],W[i]);out[i]=value;code[i]=status

def main():
    wp.init();wp.set_device('cuda:0')
    inputs=[];targets=[];labels=[]
    def add(name,H,E,v,R):
        F=exp_sym(H)@R;x,y,z=v;W=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        inputs.append((F,E,W));targets.append(rotation(v)@exp_sym(H+E)@R);labels.append(name)
    for h in [-.02,0.,.3,np.log(1000)]:
        add('isotropic_unload_'+str(h),h*np.eye(3),-h*np.eye(3),np.zeros(3),np.eye(3))
    add('biaxial_unload',np.diag([np.log(1000),np.log(1000),0]),np.diag([-np.log(1000),-np.log(1000),0]),np.zeros(3),np.eye(3))
    add('noncoaxial_unload',np.diag([.3,-.2,.1]),np.array([[.001,.002,0],[.002,-.003,.001],[0,.001,.002]])-np.diag([.3,-.2,.1]),np.array([0,0,.2]),rotation(np.array([0,0,.4])))
    rng=np.random.default_rng(127)
    for k in range(128):
        H=rng.normal(size=(3,3))*.12;H=(H+H.T)/2
        E=rng.normal(size=(3,3))*.04;E=(E+E.T)/2
        add('random_'+str(k),H,E,rng.normal(size=3)*.3,rotation(rng.normal(size=3)))
    def run(values):
        n=len(values);arrays=[wp.array(np.array([v[k] for v in values]),dtype=wp.mat33) for k in range(3)]
        out=wp.zeros(n,dtype=wp.mat33);code=wp.zeros(n,dtype=int)
        wp.launch(batch,n,inputs=[*arrays,out,code]);return out.numpy().astype(float),code.numpy()
    actual,codes=run(inputs);targets=np.array(targets)
    relative=np.linalg.norm(actual-targets,axis=(1,2))/np.linalg.norm(targets,axis=(1,2))
    checks={'134_spectral_oracles':bool(np.max(relative)<3e-5 and np.all(codes==0))}
    guard_cases=[(np.eye(3),20*np.eye(3),np.zeros((3,3))),
                 (np.eye(3),-20*np.eye(3),np.zeros((3,3))),
                 (np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3)))]
    guarded,gcodes=run(guard_cases)
    checks['out_of_range_and_invalid_rejected']=bool(np.array_equal(gcodes,[2,2,1]) and np.array_equal(guarded,np.array([v[0] for v in guard_cases])))
    F=np.eye(3);W=np.array([[0,-.2,0],[.2,0,0],[0,0,0.]])
    for _ in range(100):F,code=run([(F,np.zeros((3,3)),W)]);F=F[0];assert code[0]==0
    rotation_error=float(np.linalg.norm(F-rotation(np.array([0,0,20.]))))
    checks['100_step_rotation']=rotation_error<3e-4
    # Production-signature FEM integration: original history, stress and bypasses.
    domain=fem.Cells(fem.Grid3D(res=wp.vec3i(1,1,1)))
    pic=fem.PicQuadrature(domain,wp.array([[.3,.3,.3],[.7,.7,.7]],dtype=wp.vec3))
    def interpolate(integrand,flag,density,young):
        material=original.MaterialParameters()
        for name,value in dict(young_modulus=young,poisson_ratio=.3,damping=0,friction=.68,yield_pressure=1e5,tensile_yield_ratio=.0025,yield_stress=12000,viscosity=0,hardening=0,hardening_rate=1,softening_rate=1,dilatancy=0).items():setattr(material,name,wp.full(2,value,dtype=float))
        out=wp.zeros(2,dtype=wp.mat33);jp=wp.zeros(2);stress=wp.zeros(2,dtype=wp.mat33)
        fem.interpolate(integrand,at=pic,fields={'grid_vel':fem.UniformField(domain,wp.vec3(0.)),
            'elastic_strain_delta':fem.UniformField(domain,wp.mat33(np.eye(3,dtype=np.float32)*.02)),
            'plastic_strain_delta':fem.UniformField(domain,wp.mat33(np.eye(3,dtype=np.float32)*.03)),
            'stress':fem.UniformField(domain,wp.mat33(np.eye(3,dtype=np.float32)*200000.))},
            values={'dt':1/60,'particle_flags':wp.full(2,flag,dtype=int),'particle_density':wp.full(2,density,dtype=float),
                'particle_volume':wp.full(2,.5,dtype=float),'material_parameters':material,
                'elastic_strain_prev':wp.array([np.eye(3)*np.exp(-.02)]*2,dtype=wp.mat33),'particle_Jp_prev':wp.full(2,1.2,dtype=float),
                'elastic_strain':out,'particle_Jp':jp,'particle_stress':stress})
        return out.numpy(),jp.numpy(),stress.numpy()
    for name,flag,density,young in [('active',1,1600,2e6),('inactive',0,1600,2e6),('kinematic',1,0,2e6),('stiff',1,1600,1e13)]:
        old=interpolate(original.update_particle_strains,flag,density,young)
        new=interpolate(candidate.update_particle_strains_hencky,flag,density,young)
        checks[name+'_plastic_and_stress_unchanged']=bool(np.array_equal(old[1],new[1]) and np.array_equal(old[2],new[2]))
        if name!='active':checks[name+'_F_unchanged']=bool(np.array_equal(old[0],new[0]))
        else:checks['fem_unload_matches_identity']=bool(np.max(np.abs(new[0]-np.eye(3)))<3e-5)
    before=solver_module.update_particle_strains
    try:
        with candidate.installed_candidate():
            checks['explicit_override_installed']=solver_module.update_particle_strains is candidate.update_particle_strains_hencky
            raise ValueError('intentional_context_exit')
    except ValueError:pass
    checks['override_restored_on_exception']=solver_module.update_particle_strains is before
    report={'checks':checks,'matrix_cases':len(inputs),'max_relative_error':float(np.max(relative)),
            'worst_case':labels[int(np.argmax(relative))],'rotation_100_step_F_error':rotation_error,
            'candidate_sha256':hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),'scene_steps':0}
    (OUT/'preflight-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    assert all(checks.values()),checks

if __name__=='__main__':main()
