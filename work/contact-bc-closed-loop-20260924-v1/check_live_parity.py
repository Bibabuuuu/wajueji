"""Teacher-forced interface check only, no simulation advancement."""
from pathlib import Path
import json
import numpy as np
import torch
from contact_bc_inference import OnlineFeatures,InferenceWorker,sha
from contact_bc_policy import encode_episode,build_actor,normalize
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
policy=ROOT/'runs/contact-bc-native60-pilot-20260924-v1/pilot.pt';digest='4bd50c9f2bd9d5d31b0adee2335f2320b19056b518a097a23a11d2373bcf371c'
assert sha(policy)==digest
c=torch.load(policy,map_location='cpu',weights_only=True);torch.set_num_threads(2)
actor=build_actor(torch,len(c['feature_spec']['feature_names']));actor.load_state_dict(c['actor_state_dict']);actor.eval()
manifest=json.loads((ROOT/'local-data/contact-bc-native-tensors-20260924-v1/manifest.json').read_text());validation=c['assignment']['validation_runs'];results=[]
with InferenceWorker('/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python',policy,digest,stderr_path=OUT/'parity-worker.log') as worker:
 for e in manifest['episodes']:
  if e['run'] not in validation:continue
  with np.load(e['path']) as archive:a={k:archive[k] for k in archive.files}
  expected,spec,spec_sha=encode_episode(a);live=OnlineFeatures();max_error=0.;pred_error=0.;calls=0
  for i in range(len(a['frame'])):
   x,h=live.update({k:v[i] for k,v in a.items()});assert h==spec_sha==c['feature_spec_sha256']
   error=float(np.max(np.abs(x-expected[i])));max_error=max(max_error,error);assert error==0
   if i in (0,1,2,3,9,99,len(a['frame'])-1):
    pred,_=worker.predict(x,h)
    z=normalize(x[None],c['normalization_mean'].numpy(),c['normalization_scale'].numpy(),c['normalization_frozen'].numpy())
    with torch.inference_mode():direct=actor(torch.from_numpy(z)).numpy()[0]
    delta=float(np.max(np.abs(pred-direct)));pred_error=max(pred_error,delta);assert delta<=1e-6;calls+=1
  results.append(dict(run=e['run'],frames=len(a['frame']),maximum_feature_error=max_error,maximum_action_error=pred_error,worker_calls=calls))
result=dict(passed=True,episodes=results,frames=sum(r['frames'] for r in results),policy_sha256=digest,physics_steps=0,scope='Streaming versus batch feature identity and worker versus direct actor output; no autonomous rollout.')
(OUT/'parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
