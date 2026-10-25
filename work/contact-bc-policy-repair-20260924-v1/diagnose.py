from pathlib import Path
import json,numpy as np,torch
from contact_bc_policy import encode_episode,normalize,build_actor
from contact_bc_inference import OnlineFeatures,structured_observation
ROOT=Path('/home/qunwang/wajueji');OUT=Path(__file__).resolve().parent
torch.set_num_threads(2)
c=torch.load(ROOT/'runs/contact-bc-native60-pilot-20260924-v1/pilot.pt',weights_only=True,map_location='cpu')
actor=build_actor(torch,len(c['feature_spec']['feature_names']));actor.load_state_dict(c['actor_state_dict']);actor.eval()
mean=c['normalization_mean'].numpy();scale=c['normalization_scale'].numpy();frozen=c['normalization_frozen'].numpy();names=c['feature_spec']['feature_names']
command_ids=[i for i,n in enumerate(names) if n.startswith('previous_applied_action_si[') or n.startswith('controller_target_velocity[')]
# History compact schema: qpos19 + qvel18 + origin3 + action4 + load4 + valid1.
command_ids += [i for i,n in enumerate(names) if any(n==f'history_minus_{h}[{j}]' for h in (3,6,9) for j in range(40,44))]
def pred(x):
 with torch.inference_mode():return actor(torch.from_numpy(normalize(x,mean,scale,frozen))).numpy()
manifest=json.loads((ROOT/'local-data/contact-bc-native-tensors-20260924-v1/manifest.json').read_text());results=[]
for e in manifest['episodes']:
 with np.load(e['path']) as f:a={k:f[k] for k in f.files}
 x,_,_=encode_episode(a,goal_valid=e['run'] not in c['assignment']['mask_original_goal_runs']);p=pred(x);y=a['applied_action_normalized']
 # Detect observed down-to-neutral transitions only for offline evaluation.
 stops=np.flatnonzero((y[1:,1]>-.1)&(y[:-1,1]<-.5)&(np.diff(a['frame'])==1))+1
 samples=[]
 for t in stops:
  ids=np.arange(max(0,t-5),min(len(y),t+16));samples.append(dict(frame=int(a['frame'][t]),expert_boom=y[ids,1].tolist(),pred_boom=p[ids,1].tolist()))
 row=dict(run=e['run'],split='development' if e['run'] in c['assignment']['development_runs'] else 'validation',mae=np.abs(p-y).mean(0).tolist(),boom_release_windows=samples)
 if e['run'].endswith('115812-1458627'):
  row['sample_frames']=[dict(frame=int(a['frame'][t]),expert=y[t].tolist(),prediction=p[t].tolist()) for t in range(len(y)) if int(a['frame'][t]) in (60,120,300,310,320,330,340,350,360,420,480,540,600)]
 results.append(row)
run=ROOT/'runs/contact-bc-contact-window-20260924-v1';actions=[json.loads(x) for x in (run/'actions.ndjson').read_text().splitlines()];terrain={x['frame']:x['terrain'] for x in [json.loads(line) for line in (run/'terrain.ndjson').read_text().splitlines()]};r=json.loads((run/'report.json').read_text());live=OnlineFeatures();live_rows=[]
for a in actions:
 s=structured_observation(frame=a['frame'],state=a['before_state'],terrain=terrain[a['terrain_anchor_frame']],terrain_frame=a['terrain_anchor_frame'],goal=r['selected_goal'],previous_action=a['previous_applied_action_si'],previous_load=a['previous_drive_load'])
 x,h=live.update(s)
 if a['frame'] in (60,120,300,330,340,360,420,480,540,600):
  masked=x.copy();masked[command_ids]=mean[command_ids]
  live_rows.append(dict(frame=a['frame'],original=pred(x[None])[0].tolist(),command_channels_mean_ablation=pred(masked[None])[0].tolist()))
result=dict(episodes=results,live_ablation=live_rows,ablation_scope='Diagnostic off-distribution perturbation, not a causal proof or deployable policy.',command_channel_count=len(command_ids))
(OUT/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
