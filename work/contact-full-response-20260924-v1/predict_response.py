"""Closed-chain four-port finite-horizon response, no data-fitted parameters."""
import json,subprocess,os,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import expm
W=Path(__file__).resolve().parent;ROOT=W.parents[1]
OLD=ROOT/'work/contact-effective-response-20260924-v1'
sys.path.insert(0,str(OLD))
from diagnose import OUT as PRIOR,BASE,BUNDLE,DT,LENGTH
OUT=ROOT/'runs/contact-full-response-20260924-v1'
def predict(coeff,method='native_euler'):
    M=np.array(coeff['mass']).reshape(4,4);J=np.array(coeff['jacobian']).reshape(6,4).copy();J[3:]*=LENGTH
    P=np.array(coeff['passive_damping']).reshape(4,4);K=np.diag(coeff['stiffness']);D=np.diag(coeff['drive_damping'])
    if method=='native_euler':
        h=1/4800;A=np.linalg.inv(M+h*P);V=np.zeros((4,4));Q=np.zeros((4,4))
        for _ in range(80):V=A@(M@V+h*(np.eye(4)-D@V-K@Q));Q+=h*V
    else:
        A=np.zeros((12,12));A[:4,4:8]=np.eye(4);A[4:8,:4]=-np.linalg.solve(M,K);A[4:8,4:8]=-np.linalg.solve(M,P+D);A[4:8,8:]=np.linalg.inv(M)
        V=expm(A*DT)[4:8,8:]
    return J@V@J.T/DT

def errors(pred,ref):
    norms=np.linalg.norm(ref,axis=0);delta=pred-ref
    return dict(relative=float(np.linalg.norm(delta)/np.linalg.norm(ref)),column_relative=(np.linalg.norm(delta,axis=0)/np.maximum(norms,.01*max(norms))).tolist(),symmetry=float(np.linalg.norm(pred-pred.T)/np.linalg.norm(pred)),eigenvalues=np.linalg.eigvalsh((pred+pred.T)/2).tolist())
def main():
    OUT.mkdir(exist_ok=True);raw=json.loads((PRIOR/'exact-response.json').read_text());lines=[]
    for c in raw['contexts']:
        state=c['trials'][0]['states'][0];q=[*state['mechanism_qpos'],*[x for r in state['rocks'] for x in r['qpos']]];v=[*state['mechanism_qvel'],*[x for r in state['rocks'] for x in r['qvel']]]
        lines.append(' '.join(format(x,'.17g') for x in [*c['origin_world_m'],*q,*v]))
    cmd=json.loads((OLD/'compile-review.json').read_text())['command'];cmd[cmd.index(str(OLD.relative_to(ROOT)/'endpoint_velocity.cc'))]=str(W/'export_response.cc');cmd[-1]=str(OUT/'export-response')
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True);(W/'compile-review.json').write_text(json.dumps(dict(command=cmd,returncode=p.returncode,stderr=p.stderr),indent=2)+'\n');p.check_returncode()
    env=dict(os.environ,EXCAVATOR_MUJOCO_MESH_DIR=str(BUNDLE/'excavator-resources/mesh'),EXCAVATOR_MUJOCO_TEXTURE_DIR=str(BUNDLE/'excavator-resources/texture'))
    p=subprocess.run([str(OUT/'export-response'),str(BASE/'physics/runtime/audit.xml')],input='\n'.join(lines)+'\n',capture_output=True,text=True,env=env,timeout=15)
    if p.returncode:raise RuntimeError(p.stderr)
    coeffs=[json.loads(l) for l in p.stdout.splitlines()];results=[]
    for c,co in zip(raw['contexts'],coeffs,strict=True):
        estimates={}
        for method in ('native_euler','continuous'):
            pred=predict(co,method);err=errors(pred,np.array(c['matrices']['100.0']));estimates[method]=dict(matrix=pred.tolist(),**err,passed=err['relative']<=.2 and max(err['column_relative'])<=.3 and min(err['eigenvalues'])>=-1e-12)
        result=dict(context=c['context'],coefficients=co,predictions=estimates);results.append(result);print(c['context'],{k:{a:b for a,b in v.items() if a!='matrix'} for k,v in estimates.items()},flush=True)
    report=dict(contexts=results,native_dynamic_steps=0,MPM_steps=0,parameters_fitted=0,inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (PRIOR/'exact-response.json',BASE/'physics/runtime/audit.xml',W/'export_response.cc',W/'predict_response.py')])
    (W/'response-review.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
