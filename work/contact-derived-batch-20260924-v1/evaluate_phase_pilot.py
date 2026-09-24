"""Offline comparison on the same derived states; not closed-loop evidence."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from contact_bc_policy import encode_episode as flat_encode,normalize,build_actor
from contact_phase_bc_policy import encode_episode as phase_encode
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(path,digest):
 assert sha(path)==digest
 c=torch.load(path,map_location='cpu',weights_only=True);actor=build_actor(torch,len(c['feature_spec']['feature_names']));actor.load_state_dict(c['actor_state_dict']);actor.eval()
 return c,actor

def main():
 torch.set_num_threads(2)
 report=read(ROOT/'runs/contact-phase-bc-pilot-v1/report.json');old=read(ROOT/'work/contact-bc-policy-repair-20260924-v1/training-neutral-report.json')
 manifest=read(W/'manifest.json');assignment=read(W/'assignment.json')
 baseline,old_actor=load(old['checkpoint'],old['checkpoint_sha256']);candidate,new_actor=load(report['checkpoint'],report['checkpoint_sha256'])
 output=dict(schema='same_derived_state_phase_bc_comparison_v1',baseline_checkpoint_sha256=old['checkpoint_sha256'],candidate_checkpoint_sha256=report['checkpoint_sha256'],episodes=[],physics_advanced=False,trained_on_different_data_domains=True,
  interpretation='Both policies evaluated on identical actual derived observations. New policy also gets causal stage. Since training states differ, this is not an isolated phase-only ablation. Offline metrics do not establish autonomous safety.')
 for e in manifest['episodes']:
  with np.load(e['path'],allow_pickle=False) as z:a={k:z[k] for k in z.files}
  assert sha(e['path'])==e['sha256'];predictions={}
  for name,checkpoint,actor,encoder in [('flat_baseline',baseline,old_actor,flat_encode),('phase_candidate',candidate,new_actor,phase_encode)]:
   x,spec,digest=encoder(a,goal_valid=e['run'] not in assignment['mask_original_goal_runs']);assert digest==checkpoint['feature_spec_sha256']
   x=normalize(x,checkpoint['normalization_mean'].numpy(),checkpoint['normalization_scale'].numpy(),checkpoint['normalization_frozen'].numpy())
   with torch.inference_mode():predictions[name]=actor(torch.from_numpy(x)).numpy()
  y=a['applied_action_normalized'];groups={}
  for name,phase in [('all',None),('approach_contact',0),('penetrate',1),('curl_load',2),('lift',3),('off_soil',4)]:
   mask=np.ones(len(y),bool) if phase is None else a['phase_id']==phase
   if not mask.any():continue
   neutral=mask&(np.abs(y[:,1])<.05)
   groups[name]=dict(rows=int(mask.sum()),neutral_boom_rows=int(neutral.sum()),models={})
   for model,p in predictions.items():
    groups[name]['models'][model]=dict(mae=float(np.abs(p[mask]-y[mask]).mean()),mae_by_axis=np.abs(p[mask]-y[mask]).mean(0).tolist(),
     neutral_boom_predicted_mean=None if not neutral.any() else float(p[neutral,1].mean()),
     neutral_boom_wrong_down_fraction=None if not neutral.any() else float((p[neutral,1]<-.2).mean()))
  output['episodes'].append(dict(run=e['run'],split=e['split'],groups=groups))
 (W/'offline-policy-comparison.json').write_text(json.dumps(output,indent=2)+'\n')
 for e in output['episodes']:
  if e['split']=='validation_runs':print(e['run'],json.dumps(e['groups'].get('curl_load')))
if __name__=='__main__':main()
