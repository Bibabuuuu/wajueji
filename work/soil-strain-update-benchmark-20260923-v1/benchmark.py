"""Bounded offline constitutive-update comparison; no scene or solver mutation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm

OUT = Path(__file__).resolve().parent
AUDIT = OUT.parent / 'soil-local-onset-audit-20260923-v1'
I = np.eye(3)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def polar_log(F):
    U, s, Vt = np.linalg.svd(F)
    assert np.linalg.det(F) > 0 and np.min(s) > 0
    return (U * np.log(s)) @ U.T, U @ Vt

def sym_exp(H):
    v, U = np.linalg.eigh((H + H.T) / 2)
    return (U * np.exp(v)) @ U.T

def update(F, E, W, mode):
    if mode == 'linear':
        return (I + E + W) @ F
    Q = expm(W)
    if mode == 'finite_rotation_only':
        return Q @ (I + E) @ F
    if mode == 'exp_increment':
        return Q @ sym_exp(E) @ F
    H, R = polar_log(F)
    return Q @ sym_exp(H + E) @ R

MODES = ['linear', 'finite_rotation_only', 'exp_increment', 'log_reconstruction']

def metrics(F, target):
    s = np.linalg.svd(F, compute_uv=False)
    H = polar_log(F)[0] if np.linalg.det(F) > 0 else None
    return {'singular_values': s.tolist(), 'determinant': float(np.linalg.det(F)),
            'F_error': float(np.linalg.norm(F-target)),
            'Hencky_error': None if H is None else float(np.linalg.norm(H-polar_log(target)[0]))}

def main():
    old_bindings = json.loads((AUDIT/'protocol.json').read_text())['bindings']
    # Bind the physical sources, frozen prepoint, actual observation input and this experiment.
    paths = [p for p in old_bindings if '/site-packages/' in p or 'approach-handoff.npz' in p or p.endswith('run_layered_47_sparse_manual_contact_episode.py')]
    paths += [str(AUDIT/'cohort-history.npz'), str(OUT/'protocol.json'), str(OUT/'benchmark.py')]
    bindings = {p: sha(p) for p in paths}
    for p in paths:
        if p in old_bindings:
            assert bindings[p] == old_bindings[p], p
    (OUT/'bindings-before.json').write_text(json.dumps(bindings, indent=2)+'\n')
    result = {'scope': 'offline matrix update; not full constitutive/grid simulation', 'cases': {}}
    cases = [
        ('elastic_compression', I, -.02*I, np.zeros((3,3)), np.exp(-.02)*I),
        ('elastic_unload', np.exp(-.02)*I, .02*I, np.zeros((3,3)), I),
        ('large_elastic_state_unload', 1000*I, -np.log(1000)*I, np.zeros((3,3)), I),
    ]
    W = np.array([[0.,-.2,0.],[.2,0.,0.],[0.,0.,0.]])
    F = np.diag([1.1,.95,1.0])
    cases.append(('rigid_rotation_one_step', F, np.zeros((3,3)), W, expm(W)@F))
    H0 = np.diag([.3,-.2,.1]); H1 = np.array([[.001,.002,0],[.002,-.003,.001],[0,.001,.002]])
    R0 = expm(W*2)
    cases.append(('noncoaxial_unloading', sym_exp(H0)@R0, H1-H0, W, expm(W)@sym_exp(H1)@R0))
    for name, F, E, Wc, target in cases:
        result['cases'][name] = {m: metrics(update(F,E,Wc,m),target) for m in MODES}
    result['rotation_100_steps'] = {}
    for m in MODES:
        F = I.copy()
        for _ in range(100): F = update(F, np.zeros((3,3)), W, m)
        result['rotation_100_steps'][m] = metrics(F,expm(100*W))
    # Exact homogeneous hydrostatic return: positive pressure is compression.
    Eyoung=2e6; nu=.3; alpha=(1-2*nu)/Eyoung
    Hs=[]; Ls=[]; stresses=[]; expected_E=[]; expected_P=[]; names=[]
    hydro = [('elastic_compression',0,-.005), ('compressive_yield',0,-.2),
             ('elastic_unload_from_yield',-.02,.02), ('tensile_yield',0,.002),
             ('large_state_zero_total_increment',np.log(1000),0)]
    result['hydrostatic_oracle'] = []
    for name,h,L in hydro:
        trial=h+L; p=float(np.clip(-trial/alpha,-250,100000))
        ht=-alpha*p; de=ht-h; dp=L-de
        Hs.append(h*I); Ls.append(L*I); stresses.append(p*I); expected_E.append(de*I); expected_P.append(dp*I); names.append(name)
        result['hydrostatic_oracle'].append({'name':name,'H_previous':h,'total_increment':L,'pressure':p,'H_target':ht,'elastic_increment':de,'plastic_increment':dp})
    np.savez(OUT/'homogeneous-kernel-input.npz',names=np.array(names), H=np.array(Hs), L=np.array(Ls), stress=np.array(stresses), expected_E=np.array(expected_E),expected_P=np.array(expected_P))
    # True recorded one-step input, never recursively advance a counterfactual state.
    a=np.load(AUDIT/'cohort-history.npz'); ids=a['ids']; frames=a['frames']; j=int(np.flatnonzero(ids==53753)[0])
    FF=a['particle_elastic_strain']; mu=a['material_update']
    rows=[]
    for frame in range(1039,1048):
        k=int(np.flatnonzero(frames==frame)[0]); prev=FF[k-1,j].astype(float)
        delta=mu[k,j,1:10].reshape(3,3).astype(float); G=mu[k,j,10:19].reshape(3,3).astype(float); spin=(G-G.T)/120
        H,_=polar_log(prev); eig=np.linalg.eigvalsh(H+(delta+delta.T)/2)
        row={'frame':frame,'previous_H_eigenvalues':np.linalg.eigvalsh(H).tolist(),'updated_H_eigenvalues':eig.tolist(), 'predicted_singular_range_log_update':np.exp([eig[0],eig[-1]]).tolist(), 'log_update_outside_existing_singular_guard':bool(eig[0]<np.log(1e-6) or eig[-1]>np.log(1e6)), 'updates':{}}
        for m in MODES:
            val=update(prev,delta,spin,m); s=np.linalg.svd(val,compute_uv=False)
            row['updates'][m]={'min_singular':float(s[-1]),'max_singular':float(s[0]),'determinant':float(np.linalg.det(val)),'existing_guard_rejects':bool(s[-1]<1e-6 or s[0]>1e6)}
        rows.append(row)
    result['recorded_particle_53753_one_step'] = rows
    # All original upward crossings: is exponentiation alone inside the existing admissible range?
    s=np.linalg.svd(FF,compute_uv=False)[:,:,0]; ts,js=np.nonzero((s[1:]>2)&(s[:-1]<=2));ts+=1
    prev=FF[ts-1,js].astype(float); U,v,V=np.linalg.svd(prev)
    H=(U*np.log(v)[:,None,:])@U.transpose(0,2,1)
    delta=mu[ts,js,1:10].reshape(-1,3,3).astype(float)
    target_eig=np.linalg.eigvalsh(H+(delta+delta.transpose(0,2,1))/2)
    result['recorded_crossings_log_target']={'events':len(ts),'below_existing_guard':int((target_eig[:,0]<np.log(1e-6)).sum()),'above_existing_guard':int((target_eig[:,-1]>np.log(1e6)).sum()),'minimum_eigenvalue_quantiles':np.quantile(target_eig[:,0],[0,.01,.5,.99,1]).tolist(),'maximum_eigenvalue_quantiles':np.quantile(target_eig[:,-1],[0,.01,.5,.99,1]).tolist()}
    checks={
        'log_reconstruction_exact_all_analytic_cases':all(c['log_reconstruction']['F_error']<1e-10 for c in result['cases'].values()),
        'finite_rotation_only_does_not_fix_large_unload':result['cases']['large_elastic_state_unload']['finite_rotation_only']['F_error']>100,
        'exponential_increment_noncoaxial_residual_exists':result['cases']['noncoaxial_unloading']['exp_increment']['Hencky_error']>1e-5,
        'finite_rotation_no_artificial_stretch':result['rotation_100_steps']['log_reconstruction']['F_error']<1e-10,
        'sources_unchanged':all(sha(p)==s for p,s in bindings.items()),
    }
    result['checks']=checks
    (OUT/'benchmark-report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'checks':checks,'crossings':result['recorded_crossings_log_target'],'sample':rows},indent=2))
    assert all(checks.values())

if __name__=='__main__':main()
