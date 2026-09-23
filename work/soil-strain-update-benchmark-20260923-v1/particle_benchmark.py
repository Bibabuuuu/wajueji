"""One-cell calls to installed production particle integrand, including its guard."""
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
import warp.fem as fem
from newton._src.solvers.implicit_mpm.implicit_mpm_solver_kernels import MaterialParameters, update_particle_strains

OUT=Path(__file__).resolve().parent

def main():
    wp.init();wp.set_device('cuda:0')
    (OUT/'particle-binding-before.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})+'\n')
    domain=fem.Cells(fem.Grid3D(res=wp.vec3i(1,1,1)))
    points=wp.array([[.5,.5,.5]],dtype=wp.vec3)
    pic=fem.PicQuadrature(domain,points)
    material=MaterialParameters()
    for name,value in dict(young_modulus=2e6,poisson_ratio=.3,damping=0,friction=.68,yield_pressure=1e5,tensile_yield_ratio=.0025,yield_stress=12000,viscosity=0,hardening=0,hardening_rate=1,softening_rate=1,dilatancy=0).items():
        setattr(material,name,wp.full(1,value,dtype=float))
    def step(F,E,P=None,sigma=None):
        out=wp.zeros(1,dtype=wp.mat33);jp=wp.zeros(1);stress=wp.zeros(1,dtype=wp.mat33)
        P=np.zeros((3,3)) if P is None else P
        sigma=np.zeros((3,3)) if sigma is None else sigma
        fem.interpolate(update_particle_strains,at=pic,
            fields={'grid_vel':fem.UniformField(domain,wp.vec3(0.)),
                    'elastic_strain_delta':fem.UniformField(domain,wp.mat33(E.astype(np.float32))),
                    'plastic_strain_delta':fem.UniformField(domain,wp.mat33(P.astype(np.float32))),
                    'stress':fem.UniformField(domain,wp.mat33(sigma.astype(np.float32)))},
            values={'dt':1/60,'particle_flags':wp.ones(1,dtype=int),'particle_density':wp.full(1,1600.,dtype=float),
                    'particle_volume':wp.ones(1,dtype=float),'material_parameters':material,
                    'elastic_strain_prev':wp.array([F],dtype=wp.mat33),'particle_Jp_prev':wp.ones(1),
                    'elastic_strain':out,'particle_Jp':jp,'particle_stress':stress})
        return out.numpy()[0].astype(float),float(jp.numpy()[0])
    rows=[]
    for name,F,E in [('biaxial_large_unload',np.diag([1000.,1000.,1.]),np.diag([-np.log(1000),-np.log(1000),0.])),
                     ('isotropic_large_unload',1000*np.eye(3),-np.log(1000)*np.eye(3)),
                     ('small_compression',np.eye(3),-.02*np.eye(3)),
                     ('small_exact_unload',np.exp(-.02)*np.eye(3),.02*np.eye(3))]:
        value,jp=step(F,E)
        rows.append({'name':name,'F_new':value.tolist(),'singular_values':np.linalg.svd(value,compute_uv=False).tolist(),'retained_previous':bool(np.array_equal(value,F)),'Jp':jp})
    # Feed actual postprocessed homogeneous hydrostatic output to production update.
    a=np.load(OUT/'homogeneous-kernel-input.npz');kr=json.loads((OUT/'kernel-report.json').read_text())
    coupled=[]
    for i,row in enumerate(kr['rows']):
        F=np.diag(np.exp(np.diag(a['H'][i])))
        new,jp=step(F,np.array(row['elastic_increment']),np.array(row['plastic_increment']),a['stress'][i])
        coupled.append({'name':row['name'],'F_singular_values':np.linalg.svd(new,compute_uv=False).tolist(),'Jp_after_one_step':jp,'retained_previous':bool(np.allclose(new,F,rtol=0,atol=1e-6))})
    checks={'production_guard_allows_biaxial_flip_amplification':rows[0]['singular_values'][0]>5900 and not rows[0]['retained_previous'],
            'production_guard_rejects_isotropic_orientation_reversal':rows[1]['retained_previous'],
            'small_unload_has_nonzero_residual':abs(rows[3]['singular_values'][0]-1)>1e-5}
    report={'scope':'Actual installed production particle integrand in one unit cell; prescribed fields, no full MPM step',
            'rows':rows,'postprocess_to_particle_update':coupled,'checks':checks}
    (OUT/'particle-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));assert all(checks.values())

if __name__=='__main__':main()
