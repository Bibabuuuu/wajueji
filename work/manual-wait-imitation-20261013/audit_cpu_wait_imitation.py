"""Read-only recomputation and leakage audit of saved CPU imitation candidates."""
from pathlib import Path
import copy
import hashlib
import json
import pickle
import numpy as np
import torch
from stable_baselines3 import SAC

ROOT=Path('/home/qunwang/wajueji');RUN=ROOT/'runs/manual-wait-imitation-cpu-20261013-v1'
torch.set_num_threads(1)
report=json.loads((RUN/'report.json').read_text());budget=json.loads((RUN/'budget.json').read_text())
model=SAC.load(ROOT/'runs/manual-recovery-sac-20261013-v1/batch-v1/sac-last.zip',device='cpu')
with (ROOT/'runs/manual-recovery-sac-20261013-v1/initial-pool/initial-replay.pkl').open('rb') as f:pool=pickle.load(f)
anchors=pool.observations[1350:1750,0].copy();anchor_keys={x.tobytes() for x in anchors}
data={}
for label in ['early43','early52','loaded970']:
    with np.load(ROOT/'runs/manual-recovery-collection-20261013-v1'/label/'transitions.npz',allow_pickle=False) as d:
        w=(d['observations'][:,-1]>.5)&(d['policy_source']=='fixed_correction')
        z=(d['observations'][:,-1]<.5)&np.all(d['actions']==0,axis=1)
        data[label]=(d['observations'][w].copy(),d['actions'][w].copy(),d['observations'][z].copy())


def predict(actor,obs):
    with torch.no_grad():return actor(torch.from_numpy(obs),deterministic=True).numpy()


def zero_metrics(actions):
    return dict(count=len(actions),mean_absolute_error=float(abs(actions).mean()),
                max_absolute_error=float(abs(actions).max()),exact_zero_fraction=float((abs(actions)<=1e-6).all(axis=1).mean()))


checks=dict(completed=report['status']=='completed',budget_300=budget['seconds']==300,
            training_within_budget=report['elapsed_seconds']<300,
            no_physics_or_sac_updates=report['new_physics_frames']==report['sac_gradient_updates']==report['critic_updates']==0,
            source_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in report['sources'].items()),
            four_fixed_fits=len(report['folds'])==4 and sum(x['supervised_steps'] for x in report['folds'])==12000)
folds=[]
for fold in report['folds']:
    held=fold['heldout'];actor=copy.deepcopy(model.actor)
    actor.load_state_dict(torch.load(RUN/f'actor-leave-{held or "all-data"}.pt',map_location='cpu',weights_only=True))
    obs=np.concatenate([data[k][0] for k in fold['train_labels']]);targets=np.concatenate([data[k][1] for k in fold['train_labels']])
    act=predict(actor,obs)
    checks[f'{held}_train_metric_matches']=bool(np.isclose(np.mean((act-targets)**2),fold['train_wait']['mse'],rtol=1e-6,atol=1e-8))
    anch=predict(actor,anchors)
    checks[f'{held}_anchor_metric_matches']=bool(np.isclose(np.max(abs(anch)),fold['manual_anchor']['max_absolute_error'],rtol=1e-6,atol=1e-8))
    result=dict(heldout=held,train_wait=fold['train_wait'],manual_anchor=zero_metrics(anch))
    if held is not None:
        train_keys=anchor_keys|{x.tobytes() for x in obs};test,target,zeros=data[held]
        overlap=sum(x.tobytes() in train_keys for x in test)
        checks[f'{held}_wait_disjoint']=overlap==0 and held not in fold['train_labels']
        pred=predict(actor,test);needed=abs(test[:,-6:-2])>1e-4
        directed=(pred[:,:4]*test[:,-6:-2]>0)&(abs(pred[:,:4])>=.1)
        rate=float(directed[needed].mean())
        checks[f'{held}_wait_metric_matches']=np.isclose(rate,fold['heldout_wait']['needed_axis_direction_rate']).item()
        mask=np.asarray([x.tobytes() not in train_keys for x in zeros])
        assert mask.any()
        result.update(heldout_wait=fold['heldout_wait'],wait_training_duplicates=overlap,
                      nonwait_removed_training_duplicates=int((~mask).sum()),
                      independent_nonwait=zero_metrics(predict(actor,zeros[mask])))
    folds.append(result)
audit=dict(status='passed_with_raw_report_correction' if all(checks.values()) else 'failed',checks=checks,
           check_count=len(checks),folds=folds,new_physics_frames=0,new_training_updates=0,
           raw_report_correction='heldout_zero_states_never_used_for_training=true is false for loaded970: 300 of 374 states exactly duplicate manual anchors; independent nonwait metrics exclude them',
           model_integration_authorized=False,closed_loop_qualified=False,
           conclusion='Known wait actions largely fit, whole-trajectory transfer fails and nonwait outputs drift. Do not deploy or resume SAC from these prototype weights.')
(RUN/'read-only-audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
