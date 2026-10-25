"""Actual derived tensors: batch/stream parity and real isolated CPU-worker parity."""
import json
from pathlib import Path
import numpy as np
import torch
from contact_bc_inference import sha
from contact_phase_bc_inference import OnlinePhaseFeatures, PhaseInferenceWorker
from contact_phase_bc_policy import encode_episode, build_actor, normalize

ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
B=ROOT/'work/contact-derived-batch-20260924-v1'
read=lambda p:json.loads(Path(p).read_text())
def ident(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))

def main():
 report=read(B/'training-report.json');policy=Path(report['checkpoint']);digest=report['checkpoint_sha256']
 assert sha(policy)==digest
 c=torch.load(policy,map_location='cpu',weights_only=True);torch.set_num_threads(2)
 actor=build_actor(torch,len(c['feature_spec']['feature_names']));actor.load_state_dict(c['actor_state_dict']);actor.eval()
 results=[];last=None
 with PhaseInferenceWorker('/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python',policy,digest,stderr_path=W/'worker-parity.log') as worker:
  for e in read(B/'manifest.json')['episodes']:
   assert sha(e['path'])==e['sha256']
   with np.load(e['path'],allow_pickle=False) as z:a={k:z[k] for k in z.files}
   valid=e['run'] not in c['assignment']['mask_original_goal_runs']
   expected,_,h=encode_episode(a,goal_valid=valid);live=OnlinePhaseFeatures(goal_valid=valid)
   picks={0,1,2,3,9,len(a['frame'])-1}|set((np.flatnonzero(np.diff(a['phase_id']))+1).tolist())
   error=0.;calls=0
   for i in range(len(a['frame'])):
    x,spec=live.update({k:v[i] for k,v in a.items()});assert spec==h==c['feature_spec_sha256']
    np.testing.assert_array_equal(x,expected[i])
    if i in picks:
     predicted,_=worker.predict(x,h)
     inp=normalize(x[None],c['normalization_mean'].numpy(),c['normalization_scale'].numpy(),c['normalization_frozen'].numpy())
     with torch.inference_mode():direct=actor(torch.from_numpy(inp)).numpy()[0]
     delta=float(np.max(np.abs(predicted-direct)));assert delta<=1e-6;error=max(error,delta);calls+=1
    last=(x,h)
   results.append(dict(run=e['run'],rows=len(a['frame']),feature_max_abs_error=0.,action_max_abs_error=error,worker_calls=calls))
  bad=last[0].copy();bad[-5:]=0
  try:worker.predict(bad,last[1])
  except RuntimeError as exc:assert 'phase_policy_features' in str(exc)
  else:raise AssertionError('unknown_phase_was_accepted')
 # A flat checkpoint must not silently enter the distinct phase model route.
 old=read(ROOT/'work/contact-bc-policy-repair-20260924-v1/training-neutral-report.json')
 try:
  with PhaseInferenceWorker('/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python',old['checkpoint'],old['checkpoint_sha256'],stderr_path=W/'wrong-schema.log'):pass
 except RuntimeError as exc:assert 'closed' in str(exc) or 'contract' in str(exc)
 else:raise AssertionError('flat_schema_was_accepted')
 result=dict(passed=True,policy_sha256=digest,episodes=results,rows=sum(r['rows'] for r in results),
  invalid_phase_rejected=True,flat_checkpoint_rejected=True,physics_steps=0,
  inputs=[ident(B/'manifest.json'),ident(B/'training-report.json'),ident(policy)],
  code=[ident(p) for p in (Path(__file__),ROOT/'src/v4/contact_phase_bc_inference.py',ROOT/'src/v4/contact_phase_bc_policy.py',ROOT/'src/v4/contact_bc_inference.py')])
 (W/'feature-parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
