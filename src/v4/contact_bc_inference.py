"""Bounded CPU inference worker and causal online feature assembly for D-145."""
from __future__ import annotations
import argparse
from collections import deque
import hashlib,json,selectors,subprocess,sys,time
from pathlib import Path
import numpy as np
from contact_bc_data import STATE_FIELDS,terrain_arrays
from contact_bc_policy import SCHEMA,encode_episode,normalize,build_actor

LIMITS=np.array([.12,.0225,.0225,.0675])

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def structured_observation(*,frame,state,terrain,terrain_frame,goal,previous_action,previous_load,sequence_id=1):
    if not 0<=terrain_frame<frame or not 0<=frame-1-terrain_frame<=2:raise ValueError('noncausal_or_stale_live_terrain')
    values={}
    for key,size in STATE_FIELDS.items():
        v=np.asarray(state[key],dtype=np.float32)
        if v.shape!=(size,) or not np.isfinite(v).all():raise ValueError('live_state:'+key)
        values[key]=v
    values['terrain_height_m'],values['terrain_known']=terrain_arrays(terrain)
    values.update(terrain_origin_xy_m=np.asarray(terrain['origin_xy_m'],np.float32),terrain_resolution_m=np.float32(terrain['resolution_m']),
        terrain_age_frames=np.int64(frame-1-terrain_frame),terrain_anchor_frame=np.int64(terrain_frame),
        goal_world_m=np.asarray([*goal['entry_xy_m'],goal['surface_z_m']],np.float32),
        previous_applied_action_si=np.asarray(previous_action,np.float32),
        previous_drive_load=np.asarray([0.]*4 if previous_load is None else previous_load,np.float32),
        previous_drive_load_valid=np.bool_(previous_load is not None),frame=np.int64(frame),sequence_id=np.int64(sequence_id))
    if values['previous_applied_action_si'].shape!=(4,) or values['previous_drive_load'].shape!=(4,) or values['goal_world_m'].shape!=(3,):raise ValueError('live_vector_shape')
    if not all(np.isfinite(v).all() for v in values.values()):raise ValueError('nonfinite_live_observation')
    return values

class OnlineFeatures:
    def __init__(self,*,goal_valid=True):
        self.goal_valid=goal_valid;self.rows=deque(maxlen=10);self.last_frame=None;self.last_sequence=None
    def update(self,row):
        frame=int(row['frame']);sequence=int(row['sequence_id'])
        if self.last_frame is not None and frame<=self.last_frame:raise ValueError('nonmonotonic_live_frame')
        if self.last_frame is not None and (frame!=self.last_frame+1 or sequence!=self.last_sequence):self.rows.clear()
        self.rows.append(row);self.last_frame=frame;self.last_sequence=sequence
        arrays={key:np.stack([r[key] for r in self.rows]) for key in row}
        x,spec,digest=encode_episode(arrays,goal_valid=self.goal_valid)
        return x[-1],digest

class InferenceWorker:
    def __init__(self,python,checkpoint,expected_sha,*,stderr_path,timeout=15):
        self.timeout=timeout;self.proc=None;self.request_id=0;self.err=open(stderr_path,'w')
        self.selector=selectors.DefaultSelector()
        try:
            self.proc=subprocess.Popen([str(python),str(Path(__file__).resolve()),'--worker','--checkpoint',str(checkpoint),'--sha256',expected_sha],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True,bufsize=1)
            self.selector.register(self.proc.stdout,selectors.EVENT_READ)
            self.ready=self._receive()
            if self.ready.get('status')!='ready' or self.ready.get('checkpoint_sha256')!=expected_sha:raise ValueError('inference_worker_not_ready')
        except BaseException:self.close();raise
    def _receive(self):
        if not self.selector.select(self.timeout):raise TimeoutError('inference_worker_timeout_no_physics_step')
        line=self.proc.stdout.readline()
        if not line:raise RuntimeError('inference_worker_closed')
        r=json.loads(line)
        if 'error' in r:raise RuntimeError('inference_worker:'+r['error'])
        return r
    def predict(self,features,spec_sha):
        self.request_id+=1
        packet=dict(id=self.request_id,feature_spec_sha256=spec_sha,features=np.asarray(features,dtype=float).tolist())
        start=time.monotonic();self.proc.stdin.write(json.dumps(packet,allow_nan=False)+'\n');self.proc.stdin.flush()
        result=self._receive()
        if result.get('id')!=self.request_id:raise ValueError('inference_response_identity')
        action=np.asarray(result['action_normalized'],dtype=float)
        if action.shape!=(4,) or not np.isfinite(action).all() or np.any(np.abs(action)>1+1e-7):raise ValueError('invalid_policy_output')
        return np.clip(action,-1,1),dict(roundtrip_seconds=time.monotonic()-start,worker_seconds=result['inference_seconds'])
    def close(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                if self.proc.stdin:self.proc.stdin.close()
                try:self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.terminate()
                    try:self.proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
            if self.proc.stdout:self.proc.stdout.close()
        self.selector.close();self.err.close()
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()

def worker(checkpoint,expected_sha):
    import torch
    if sha(checkpoint)!=expected_sha:raise ValueError('policy_checkpoint_sha256')
    torch.set_num_threads(2)
    c=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if c['schema']!=SCHEMA or c['action_hz']!=60 or c['action_limits_si']!=LIMITS.tolist():raise ValueError('policy_contract')
    spec=c['feature_spec'];spec_sha=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    if spec_sha!=c['feature_spec_sha256']:raise ValueError('policy_feature_spec_sha')
    n=len(spec['feature_names']);actor=build_actor(torch,n);actor.load_state_dict(c['actor_state_dict']);actor.eval()
    mean=c['normalization_mean'].numpy();scale=c['normalization_scale'].numpy();frozen=c['normalization_frozen'].numpy()
    if any(x.shape!=(n,) for x in (mean,scale,frozen)) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('policy_normalization')
    print(json.dumps(dict(status='ready',checkpoint_sha256=expected_sha,feature_spec_sha256=spec_sha,feature_count=n,device='cpu',action_hz=60)),flush=True)
    for line in sys.stdin:
        try:
            packet=json.loads(line);x=np.asarray(packet['features'],dtype=np.float32)
            if x.shape!=(n,) or not np.isfinite(x).all() or packet['feature_spec_sha256']!=spec_sha:raise ValueError('policy_features')
            start=time.monotonic()
            with torch.inference_mode():a=actor(torch.from_numpy(normalize(x[None],mean,scale,frozen))).numpy()[0]
            if not np.isfinite(a).all():raise ValueError('policy_nonfinite')
            print(json.dumps(dict(id=packet['id'],action_normalized=a.tolist(),inference_seconds=time.monotonic()-start),allow_nan=False),flush=True)
        except BaseException as error:
            print(json.dumps(dict(error=repr(error))),flush=True);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);args=p.parse_args()
    if not args.worker:raise ValueError('worker_mode_required')
    worker(args.checkpoint,args.sha256)
