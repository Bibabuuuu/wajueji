"""Reconstruct actual rollout inputs and compare direct model outputs, no physics."""
import json
from pathlib import Path
import numpy as np
import torch
from contact_bc_inference import structured_observation, sha
from contact_phase_bc_inference import with_phase
from contact_phase_bc_policy import encode_episode, normalize, build_actor

ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
RUN=ROOT/'runs/contact-phase-bc-probe-20260924-v1/contact600'
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()]
def ident(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))

def main():
 r=read(RUN/'report.json');events=rows(RUN/'actions.ndjson');terrain={x['frame']:x['terrain'] for x in rows(RUN/'terrain.ndjson')}
 c=torch.load(r['policy']['path'],map_location='cpu',weights_only=True);assert sha(r['policy']['path'])==r['policy']['sha256']
 torch.set_num_threads(2);actor=build_actor(torch,len(c['feature_spec']['feature_names']));actor.load_state_dict(c['actor_state_dict']);actor.eval()
 structured=[]
 for e in events:
  row=structured_observation(frame=e['frame'],state=e['before_state'],terrain=terrain[e['terrain_anchor_frame']],terrain_frame=e['terrain_anchor_frame'],goal=r['selected_goal'],previous_action=e['previous_applied_action_si'],previous_load=e['previous_drive_load'])
  structured.append(with_phase(row,e['phase'],e['latest_allowed_phase_evidence_frame']))
 arrays={k:np.stack([s[k] for s in structured]) for k in structured[0]};x,_,h=encode_episode(arrays)
 assert h==c['feature_spec_sha256']==r['worker_ready']['feature_spec_sha256']
 def predict(value):
  z=normalize(value,c['normalization_mean'].numpy(),c['normalization_scale'].numpy(),c['normalization_frozen'].numpy())
  with torch.inference_mode():return actor(torch.from_numpy(z)).numpy()
 predicted=predict(x);recorded=np.asarray([e['action_normalized'] for e in events]);error=float(np.max(np.abs(predicted-recorded)))
 assert error<1e-6
 # Diagnostic only: changing a semantic input is NOT an authorized runtime override.
 probes=[]
 for frame in (120,280,281,320,378):
  feature=x[frame-1:frame].copy();outputs={}
  for i,name in enumerate(('approach_contact','penetrate','curl_load','lift','off_soil')):
   altered=feature.copy();altered[0,-5:]=np.eye(5,dtype=np.float32)[i];outputs[name]=predict(altered)[0].tolist()
  probes.append(dict(frame=frame,actual_phase=events[frame-1]['phase'],actions_if_only_phase_changed=outputs))
 result=dict(passed=True,frames=len(events),maximum_action_abs_error=error,physics_steps=0,
  phase_sensitivity_diagnostic_only=probes,phase_forcing_deployed=False,
  scope='Actual recorded state/history/terrain/causal-phase reproduces CPU-worker actions. Counterfactual phase outputs explain model sensitivity, not evidence that a forced phase is correct or safe.',
  inputs=[ident(RUN/x) for x in ('report.json','actions.ndjson','terrain.ndjson')]+[ident(r['policy']['path'])],reviewer=ident(Path(__file__)))
 (W/'rollout-prediction-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
