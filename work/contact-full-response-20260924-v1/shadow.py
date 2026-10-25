"""Persistent private MuJoCo predictor. Never advances the live process."""
import subprocess,json,os,select,time
import numpy as np
from predict_response import OUT,BASE,BUNDLE,predict,LENGTH
from diagnose import initial_line,step_line
from layered_47_cycle_bridge import controller_snapshot

class Process:
    def __init__(self,binary,model,stderr):
        self.err=open(stderr,'w');self.p=subprocess.Popen([str(binary),str(model)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True,bufsize=1)
    def send(self,line):self.p.stdin.write(line+'\n');self.p.stdin.flush()
    def receive(self,prefix='',timeout=10):
        # One response per request, no read-ahead consumer; pipe readiness bounded.
        if not select.select([self.p.stdout],[],[],timeout)[0]:raise TimeoutError('shadow_response')
        while True:
            line=self.p.stdout.readline()
            if not line:raise RuntimeError('shadow_closed:'+str(self.p.poll()))
            if line.startswith(prefix):return json.loads(line[len(prefix):])
    def close(self):
        if self.p.poll() is None:
            self.p.stdin.close()
            try:self.p.wait(timeout=3)
            except subprocess.TimeoutExpired:self.p.terminate();self.p.wait(timeout=3)
        self.p.stdout.close()
        self.err.close()

class Shadow:
    def __init__(self,output,model=None):
        model=model or BASE/'physics/runtime/audit.xml'
        self.native=Process(OUT/'shadow-bridge',model,output/'shadow-stderr.log');self.ready=self.native.receive('@READY ')
        self.coefficients=Process(OUT/'export-response',model,output/'coefficients-stderr.log')
    def forecast(self,state,command,pending=None):
        started=time.perf_counter();origin=np.array(state.get('bucket_origin_world_m',state.get('bucket_origin')))
        arrays=dict(mechanism_qpos=np.array(state['mechanism_qpos']),mechanism_qvel=np.array(state['mechanism_qvel']),rigid_qpos=np.array([r['qpos'] for r in state['rocks']]),rigid_qvel=np.array([r['qvel'] for r in state['rocks']]))
        meta=dict(controller_state=controller_snapshot(state));c=meta['controller_state']
        values=[*origin,*arrays['mechanism_qpos'],*arrays['rigid_qpos'].ravel(),*arrays['mechanism_qvel'],*arrays['rigid_qvel'].ravel()]
        self.coefficients.send(' '.join(format(x,'.17g') for x in values));coeff=self.coefficients.receive();C=predict(coeff)
        if coeff['integrator']!=0:raise ValueError('unvalidated_integrator')
        if np.max(np.abs(C-C.T))>1e-10 or np.linalg.eigvalsh(C).min()<-1e-12:raise ValueError('response_not_passive')
        scale=np.diag([1,1,1,1/LENGTH,1/LENGTH,1/LENGTH]);C=scale@C@scale
        self.native.send(initial_line(arrays,meta));restored=self.native.receive('@STATE ')
        if restored['mechanism_qpos']!=state['mechanism_qpos'] or restored['mechanism_qvel']!=state['mechanism_qvel'] or any(controller_snapshot(restored)[k]!=v for k,v in c.items() if k not in ('bucket_linear_velocity_world_m_s','bucket_angular_velocity_world_rad_s')):raise ValueError('shadow_restore_not_exact')
        wrench={'bucket':dict(force=[0.]*3,torque=[0.]*3,point=origin.tolist())}
        if pending is not None:
            imp=np.array(pending['reaction_impulse_ns']);angular=np.array(pending['reaction_angular_impulse_nms'])+np.cross(np.array(pending['point_world_m'])-origin,imp)
            wrench['bucket'].update(force=(imp*60).tolist(),torque=(angular*60).tolist())
        states=[]
        for half in range(2):
            self.native.send(step_line(c['native_time_s']+half/120,command,wrench));states.append(self.native.receive('@STATE '))
        twist=np.array(states[-1]['prediction_endpoint_twist'])
        return C,twist,dict(wall_seconds=time.perf_counter()-started,coefficients=coeff,free_twist=twist.tolist(),response_world=C.tolist(),native_steps=80,pending_force_predicted=wrench['bucket'],predicted_states=states)
    def close(self):self.native.close();self.coefficients.close()
