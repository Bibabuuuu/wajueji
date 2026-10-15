"""Read-only frozen-model sensitivity on cached states, not a rollout."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-qerror-dig-20261014-v1';sys.path.insert(0,str(W))
import run_qerror_second as m
import numpy as np,torch
from stable_baselines3 import SAC
torch.set_num_threads(1)
decisions=[json.loads(s) for s in (m.OUT/'policy-decisions.ndjson').read_text().splitlines()]
model=SAC.load(m.MODEL,device='cpu');updates=model._n_updates;out=[]
for frame in [0,78,207,1080,1560]:
 d=next(x for x in decisions if x['local_frame_before']==frame);original=np.asarray(d['observation'],np.float32)
 base=model.predict(original,deterministic=True)[0]
 for future_frame in [1560,3120,6000,7200]:
  obs=original.copy();obs[85]=future_frame/1560
  action=model.predict(obs,deterministic=True)[0]
  out.append(dict(cached_frame=frame,test_frame=future_frame,normalized_time=float(obs[85]),action=action.tolist(),max_abs_action_change=float(np.max(abs(action-base))),wait_strength=((action[:4]+1)/2).tolist()))
assert m.sha(m.MODEL)==m.MODEL_SHA and model._n_updates==updates
result=dict(status='cached_time_sensitivity_measured',model_sha256=m.MODEL_SHA,observation_field=85,denominator=1560,case_count=len(out),maximum_absolute_action_change=max(x['max_abs_action_change'] for x in out),cases=out,physical_steps=0,training_updates=0,frame_cap_changed=False,limitation='Synthetic time-field changes on five already observed states only. Does not validate future physical states, observation distribution, completion or model robustness; no physical budget is created.')
m.write(W/'cached-time-input-probe.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
