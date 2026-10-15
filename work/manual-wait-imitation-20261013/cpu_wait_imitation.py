"""Disposable supervised actor-fit experiment; no simulator, critic or SAC updates."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from stable_baselines3 import SAC

ROOT=Path('/home/qunwang/wajueji')
LABELS=['early43','early52','loaded970']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)


def mean_action(actor,obs):
    features=actor.extract_features(obs,actor.features_extractor)
    return torch.tanh(actor.mu(actor.latent_pi(features)))


def measurements(actor,obs,targets,waiting):
    with torch.no_grad():pred=mean_action(actor,torch.from_numpy(obs)).numpy()
    result=dict(count=len(obs),mse=float(np.mean((pred-targets)**2)),
                mean_absolute_error=float(np.mean(abs(pred-targets))),
                max_absolute_error=float(np.max(abs(pred-targets))),
                exact_target_fraction=float(np.all(abs(pred-targets)<=1e-6,axis=1).mean()))
    if waiting:
        errors=obs[:,-6:-2]
        # Reporting threshold only: 1e-4 of each full axis SI speed, not a control rule.
        needed=abs(errors)>1e-4
        directed=(pred[:,:4]*errors>0)&(abs(pred[:,:4])>=.1)
        result.update(needed_axis_entries=int(needed.sum()),
                      needed_axis_direction_rate=float(directed[needed].mean()),
                      all_needed_axes_correct_state_rate=float(np.all(directed|~needed,axis=1).mean()),
                      no_effective_directed_axis_states=int((~(directed&needed).any(axis=1)).sum()),
                      mean_absolute_pace_residual=float(abs(pred[:,4]).mean()))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('immutable_cpu_experiment_exists')
    args.output.mkdir(parents=True)
    started=time.monotonic();deadline=started+300;epoch=time.time()
    save(args.output/'budget.json',dict(seconds=300,purpose='cpu_wait_imitation_only',epoch_start=epoch,epoch_deadline=epoch+300))
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    def check():
        if time.monotonic()>=deadline:raise TimeoutError('cpu_300_second_budget_expired')
    collection=ROOT/'runs/manual-recovery-collection-20261013-v1'
    checkpoint=ROOT/'runs/manual-recovery-sac-20261013-v1/batch-v1/sac-last.zip'
    replay=ROOT/'runs/manual-recovery-sac-20261013-v1/initial-pool/initial-replay.pkl'
    index=json.loads((collection/'dataset-index.json').read_text())
    sources={str(checkpoint):sha(checkpoint),str(replay):sha(replay),str(collection/'dataset-index.json'):sha(collection/'dataset-index.json')}
    data={}
    for label,item in zip(LABELS,index['episode_index']):
        assert item['label']==label
        path=collection/label/'transitions.npz';sources[str(path)]=sha(path)
        assert sources[str(path)]==item['source_npz_sha256']
        with np.load(path,allow_pickle=False) as d:
            wait=(d['observations'][:,-1]>.5)&(d['policy_source']=='fixed_correction')
            zero=(d['observations'][:,-1]<.5)&np.all(d['actions']==0,axis=1)
            data[label]=dict(obs=d['observations'][wait].copy(),target=d['actions'][wait].copy(),
                             zero_obs=d['observations'][zero].copy(),indices=np.flatnonzero(wait).tolist(),
                             full_episode_success=item['endpoint_success'])
        assert data[label]['obs'].shape[1]==221 and data[label]['target'].shape[1]==5
        assert len(data[label]['obs']) and len(data[label]['zero_obs'])
    with replay.open('rb') as f:pool=pickle.load(f)
    # The approved mix keeps new rows first, then old rows sorted; old baseline is first 400.
    anchors=pool.observations[1350:1750,0].copy()
    assert not pool.actions[1350:1750].any() and not anchors[:,-1].any()
    frozen=SAC.load(checkpoint,device='cpu')
    assert frozen.actor.mu.in_features==64 and frozen.actor.mu.out_features==5
    actor_template=copy.deepcopy(frozen.actor).cpu()
    report=dict(status='running',purpose='cpu_supervised_fit_diagnostic_only',new_physics_frames=0,
                sac_gradient_updates=0,critic_updates=0,production_model_changed=False,
                initialization='same saved actor architecture, fresh Linear parameters, zero mean head',
                config=dict(steps_per_fit=3000,learning_rate=.001,seed=42,loss='mean action MSE; waiting and manual anchors equal weight',
                            wait_batch=64,anchor_batch=64,needed_error_threshold_normalized=1e-4,
                            directional_action_magnitude_threshold=.1,exact_zero_tolerance=1e-6),
                sources=sources,folds=[],sample_counts={k:dict(wait=len(v['obs']),nonwait_zero=len(v['zero_obs']),
                full_episode_success=v['full_episode_success'],wait_indices=v['indices']) for k,v in data.items()},
                manual_anchor_count=len(anchors),baseline_anchor_is_training_data=True,
                heldout_zero_states_never_used_for_training=True,
                failed_episode_scope='early43 local recovery actions only; endpoint failure retained',
                failed_sac_wait_metrics={k:measurements(frozen.actor,v['obs'],v['target'],True) for k,v in data.items()})
    save(args.output/'report.json',report)
    try:
        for heldout in LABELS+[None]:
            check();train_labels=[k for k in LABELS if k!=heldout]
            assert heldout is None or heldout not in train_labels
            torch.manual_seed(42);rng=np.random.default_rng(42)
            actor=copy.deepcopy(actor_template)
            for module in actor.modules():
                if isinstance(module,torch.nn.Linear):module.reset_parameters()
            with torch.no_grad():
                actor.mu.weight.zero_();actor.mu.bias.zero_()
                actor.log_std.weight.zero_();actor.log_std.bias.fill_(-3.)
            for parameter in actor.log_std.parameters():parameter.requires_grad_(False)
            optimizer=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=.001)
            obs=np.concatenate([data[k]['obs'] for k in train_labels]);targets=np.concatenate([data[k]['target'] for k in train_labels])
            x=torch.from_numpy(obs);y=torch.from_numpy(targets);z=torch.from_numpy(anchors)
            assert torch.count_nonzero(mean_action(actor,z)).item()==0
            trace=[]
            for step in range(3000):
                check();wi=rng.integers(len(x),size=64);zi=rng.integers(len(z),size=64)
                wait_loss=torch.mean((mean_action(actor,x[wi])-y[wi])**2)
                anchor_loss=torch.mean(mean_action(actor,z[zi])**2)
                loss=wait_loss+anchor_loss
                assert torch.isfinite(loss)
                optimizer.zero_grad();loss.backward();optimizer.step()
                if (step+1)%500==0:
                    trace.append(dict(step=step+1,wait_mse=float(wait_loss.detach()),anchor_mse=float(anchor_loss.detach())))
                    print('CPU_FIT',heldout or 'all_data',step+1,'elapsed',round(time.monotonic()-started,2),flush=True)
            fold=dict(heldout=heldout,train_labels=train_labels,train_wait_count=len(obs),supervised_steps=3000,
                      train_wait=measurements(actor,obs,targets,True),
                      manual_anchor=measurements(actor,anchors,np.zeros((len(anchors),5),np.float32),False),trace=trace)
            if heldout is not None:
                v=data[heldout];fold['heldout_wait']=measurements(actor,v['obs'],v['target'],True)
                fold['heldout_nonwait']=measurements(actor,v['zero_obs'],np.zeros((len(v['zero_obs']),5),np.float32),False)
            name=heldout or 'all-data'
            torch.save(actor.state_dict(),args.output/f'actor-leave-{name}.pt')
            report['folds'].append(fold);save(args.output/'report.json',report)
        check()
        report.update(status='completed',elapsed_seconds=time.monotonic()-started,supervised_steps=12000,
                      source_files_unchanged=all(sha(path)==expected for path,expected in sources.items()),
                      scope='offline actor fit only; no closed-loop restoration, safety, SAC retention or improvement qualification')
        assert report['source_files_unchanged']
    except BaseException as error:
        report.update(status='budget_exhausted' if isinstance(error,TimeoutError) else 'failed',error=str(error),
                      elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        report['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save(args.output/'report.json',report)
    print('CPU_RESULT',report['status'],report['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
