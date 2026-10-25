from pathlib import Path
import json,numpy as np,torch
from contact_bc_policy import encode_episode,normalize,build_actor
from contact_bc_inference import OnlineFeatures,structured_observation
ROOT=Path('/home/qunwang/wajueji');OUT=Path(__file__).resolve().parent
torch.set_num_threads(2)
manifest=json.loads((ROOT/'local-data/contact-bc-native-tensors-20260924-v1/manifest.json').read_text())
result=[]
for name in ('native60-pilot','repair-control','repair-dropout','repair-scale','repair-state-masked','repair-neutral'):
 run=ROOT/f'runs/contact-bc-{name}-20260924-v1';c=torch.load(run/'pilot.pt',map_location='cpu',weights_only=True)
 mean=c['normalization_mean'].numpy();scale=c['normalization_scale'].numpy();frozen=c['normalization_frozen'].numpy()
 actor=build_actor(torch,len(mean));actor.load_state_dict(c['actor_state_dict']);actor.eval()
 def predict(x):
  with torch.inference_mode():return actor(torch.from_numpy(normalize(x,mean,scale,frozen))).numpy()
 episodes=[]
 for e in manifest['episodes']:
  with np.load(e['path']) as f:a={k:f[k] for k in f.files}
  x,_,_=encode_episode(a,goal_valid=e['run'] not in c['assignment']['mask_original_goal_runs']);p=predict(x);y=a['applied_action_normalized']
  neutral=np.abs(y[:,1])<.05
  row=dict(run=e['run'],mae=np.abs(p-y).mean(0).tolist(),boom_neutral_prediction_mean=float(p[neutral,1].mean()))
  if e['run'].endswith('115812-1458627'):
   row['boom_release_320_600_mean']=float(p[(a['frame']>=320)&(a['frame']<=600),1].mean())
   row['samples']=[dict(frame=int(a['frame'][i]),expert=y[i].tolist(),prediction=p[i].tolist()) for i in range(len(y)) if a['frame'][i] in (4,60,120,300,310,320,360,480,600)]
  episodes.append(row)
 # First live state is common to all policies and physically measured, not autoregressive fantasy.
 live=ROOT/'runs/contact-bc-contact-window-20260924-v1'
 event=json.loads((live/'actions.ndjson').read_text().splitlines()[0]);terrain=json.loads((live/'terrain.ndjson').read_text().splitlines()[0]);report=json.loads((live/'report.json').read_text())
 s=structured_observation(frame=1,state=event['before_state'],terrain=terrain['terrain'],terrain_frame=0,goal=report['selected_goal'],previous_action=[0.]*4,previous_load=None)
 x,_=OnlineFeatures().update(s);z=normalize(x[None],mean,scale,frozen)[0]
 extreme=np.argsort(abs(z))[-12:][::-1]
 result.append(dict(model=name,episodes=episodes,first_live_action=predict(x[None])[0].tolist(),startup_extreme_features=[dict(name=c['feature_spec']['feature_names'][i],normalized=float(z[i]),value=float(x[i]),mean=float(mean[i]),scale=float(scale[i])) for i in extreme]))
(OUT/'offline-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
for r in result:
 print(r['model'],'startup',r['first_live_action'],'release',r['episodes'][-1]['boom_release_320_600_mean'])
 print('validation MAE',np.mean([e['mae'] for e in r['episodes'][-3:]],axis=0))
 print('extremes',r['startup_extreme_features'][:3])
 print(r['episodes'][-1]['samples'])
