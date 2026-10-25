"""Recompute work-conjugate endpoint velocities, without any dynamic steps."""
import copy,json,os,subprocess,hashlib
from pathlib import Path
import numpy as np
from diagnose import W,ROOT,OUT,BASE,BUNDLE,DT,LENGTH

def main():
    raw=json.loads((OUT/'report.json').read_text());result=copy.deepcopy(raw);lines=[]
    for context in raw['contexts']:
        for t in context['trials']:
            end=t['states'][-1]
            q=[*end['mechanism_qpos'],*[x for r in end['rocks'] for x in r['qpos']]]
            v=[*end['mechanism_qvel'],*[x for r in end['rocks'] for x in r['qvel']]]
            values=[*context['origin_world_m'],*q,*v]
            lines.append(' '.join(format(x,'.17g') for x in values))
    env=dict(os.environ,EXCAVATOR_MUJOCO_MESH_DIR=str(BUNDLE/'excavator-resources/mesh'),EXCAVATOR_MUJOCO_TEXTURE_DIR=str(BUNDLE/'excavator-resources/texture'))
    p=subprocess.run([str(W/'endpoint-velocity'),str(BASE/'physics/runtime/audit.xml')],input='\n'.join(lines)+'\n',text=True,capture_output=True,env=env,timeout=10)
    if p.returncode:raise RuntimeError(p.stderr)
    velocities=[json.loads(x) for x in p.stdout.splitlines()]
    if len(velocities)!=len(lines):raise ValueError('endpoint_velocity_count')
    index=0
    for context in result['contexts']:
        bylabel={}
        for t in context['trials']:
            v=np.array(velocities[index]);index+=1;v[3:]*=LENGTH
            t['exact_endpoint_twist_scaled_at_initial_wrench_origin']=v.tolist();bylabel[t['label']]=v
        for amplitude in (100.,200.):
            matrix=np.zeros((6,6))
            for axis in range(6):
                prefix=context['context']+f'-a{amplitude:g}-axis{axis}-sign'
                matrix[:,axis]=(bylabel[prefix+'1']-bylabel[prefix+'-1'])/(2*amplitude*DT)
            context['matrices'][str(amplitude)]=matrix.tolist()
        context['response_definition']='C=d[v_at_fixed_initial_wrench_origin,L*omega]/d[F,torque_at_same_origin/L]/dt; exact endpoint native qpos/qvel Jacobian, world coordinates.'
    result['kinematic_postprocess']=dict(native_steps=0,MPM_steps=0,work_conjugate_reference=True,inputs=[dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in (W/'endpoint_velocity.cc',W/'endpoint-velocity',OUT/'report.json')])
    (OUT/'exact-response.json').write_text(json.dumps(result,indent=2)+'\n')
    print('EXACT_ENDPOINTS',index,'native_dynamic_steps',0)
if __name__=='__main__':main()
