"""Bounded offline wiring pilot; never starts a physics rollout or A30 job."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from contact_phase_bc_policy import SCHEMA,encode_episode,fit_normalization,normalize,build_actor

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def metrics(pred,y):
    return dict(mae_by_axis=np.abs(pred-y).mean(0).tolist(),mae=float(np.abs(pred-y).mean()),
                mse=float(np.square(pred-y).mean()),axes=['swing','boom','arm','bucket'])

def command_feature_indices(spec):
    """Command channels only; preserve measured pose/velocity and their history."""
    names=spec['feature_names']
    selected=[i for i,n in enumerate(names) if n.startswith(('previous_applied_action_si[','controller_target_velocity['))]
    # Compact v1: qpos19 + qvel18 + origin3 precede previous action4.
    selected += [i for i,n in enumerate(names) if any(n==f'history_minus_{h}[{j}]' for h in spec['history_offsets'] for j in range(40,44))]
    if len(selected)!=20:raise ValueError('command_feature_contract')
    return selected

def training_episodes(episodes,weights):
    values=np.asarray([weights[e['meta']['run']] for e in episodes],dtype=float)
    if not np.isfinite(values).all() or np.any(values<0):raise ValueError('invalid_episode_weights')
    result=[e for e,w in zip(episodes,values) if w>0]
    if not result:raise ValueError('no_training_episodes')
    return result

def floor_terrain_scale(scale,spec,floor_m):
    """Keep sparse/rare terrain cells from receiving arbitrarily large gain."""
    result=scale.copy()
    ids=[i for i,n in enumerate(spec['feature_names']) if n.startswith('terrain_height_m[')]
    result[ids]=np.maximum(result[ids],floor_m)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--assignment',required=True)
    p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=200)
    p.add_argument('--device',default='cuda:0');p.add_argument('--max-training-seconds',type=float,default=120)
    p.add_argument('--command-dropout',type=float,default=0.,help='Training-only probability of masking all command channels in a sampled row')
    p.add_argument('--terrain-scale-floor-m',type=float,default=0.,help='Minimum normalization scale for each height cell; stored in checkpoint')
    p.add_argument('--neutral-axis-weight',type=float,default=1.,help='Loss weight for expert axes with absolute normalized action below 0.05; never relabels actions')
    args=p.parse_args()
    if not 1<=args.steps<=1000 or not 0<args.max_training_seconds<=300:raise ValueError('bounded_pilot_budget')
    if not 0<=args.command_dropout<=1:raise ValueError('command_dropout_probability')
    if not 0<=args.terrain_scale_floor_m<=1:raise ValueError('terrain_scale_floor')
    if not 1<=args.neutral_axis_weight<=10:raise ValueError('neutral_axis_weight')
    manifest=read(args.manifest);assignment=read(args.assignment)
    if assignment['scope']!='bounded_offline_wiring_pilot_only':raise ValueError('assignment_scope')
    if sha(args.manifest)!=assignment['source_manifest_sha256']:raise ValueError('manifest_binding')
    if manifest['schema']!='contact_bc_derived_causal_phase_structured_v1':raise ValueError('tensor_schema')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(20260924);np.random.seed(20260924);torch.set_num_threads(2)
    splits={k:set(assignment[k]) for k in ('development_runs','validation_runs')}
    if splits['development_runs']&splits['validation_runs']:raise ValueError('split_overlap')
    if splits['development_runs']|splits['validation_runs']!={e['run'] for e in manifest['episodes']}:raise ValueError('split_coverage')
    episodes=[];spec=None;spec_sha=None
    for e in manifest['episodes']:
        if sha(e['path'])!=e['sha256']:raise ValueError('tensor_changed')
        with np.load(e['path'],allow_pickle=False) as archive:a={k:archive[k] for k in archive.files}
        goal_valid=e['run'] not in assignment['mask_original_goal_runs']
        x,new_spec,new_sha=encode_episode(a,goal_valid=goal_valid)
        if spec_sha is not None and spec_sha!=new_sha:raise ValueError('feature_contract_differs')
        spec,spec_sha=new_spec,new_sha
        split='development' if e['run'] in splits['development_runs'] else 'validation'
        episodes.append(dict(meta=e,x=x,y=a['applied_action_normalized'],previous=a['previous_applied_action_si']/np.array([.12,.0225,.0225,.0675]),split=split,goal_valid=goal_valid))
    dev=[e for e in episodes if e['split']=='development'];val=[e for e in episodes if e['split']=='validation']
    if {e['meta']['lineage'] for e in dev}&{e['meta']['lineage'] for e in val}:raise ValueError('lineage_leakage')
    dev=training_episodes(dev,assignment['episode_weights'])
    train_x=np.concatenate([e['x'] for e in dev]);train_y=np.concatenate([e['y'] for e in dev])
    mean,scale,frozen=fit_normalization(train_x,spec['continuous_indices'])
    scale=floor_terrain_scale(scale,spec,args.terrain_scale_floor_m)
    command_ids=command_feature_indices(spec)
    # Complete ablation must also mask inference through checkpoint normalization.
    if args.command_dropout==1:frozen[command_ids]=True
    weights=np.concatenate([np.full(len(e['x']),assignment['episode_weights'][e['meta']['run']]/len(e['x'])) for e in dev]);weights/=weights.sum()
    device=torch.device(args.device);actor=build_actor(torch,train_x.shape[1]).to(device)
    x=torch.from_numpy(normalize(train_x,mean,scale,frozen)).to(device);y=torch.from_numpy(train_y).to(device)
    optimizer=torch.optim.Adam(actor.parameters(),lr=3e-4);rng=np.random.default_rng(20260924)
    start=time.monotonic();losses=[]
    for step in range(args.steps):
        if time.monotonic()-start>args.max_training_seconds:break
        ids=rng.choice(len(train_x),size=128,replace=True,p=weights)
        batch=x[ids].clone()
        if args.command_dropout:
            mask=torch.rand((len(ids),1),device=device)<args.command_dropout
            batch[:,command_ids]=batch[:,command_ids].masked_fill(mask,0.)
        pred=actor(batch);target=y[ids]
        if args.neutral_axis_weight==1:
            loss=torch.nn.functional.mse_loss(pred,target)
        else:
            axis_weight=torch.where(target.abs()<.05,args.neutral_axis_weight,1.)
            loss=(torch.square(pred-target)*axis_weight).sum()/axis_weight.sum()
        if not torch.isfinite(loss):raise ValueError('nonfinite_training_loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.0);optimizer.step();losses.append(float(loss.detach().cpu()))
    if not losses:raise ValueError('no_training_steps')
    actor.eval();evaluation=[]
    with torch.inference_mode():
        for e in episodes:
            inp=normalize(e['x'],mean,scale,frozen);pred=actor(torch.from_numpy(inp).to(device)).cpu().numpy()
            evaluation.append(dict(run=e['meta']['run'],split=e['split'],rows=len(inp),goal_valid=e['goal_valid'],
                                   model=metrics(pred,e['y']),zero_baseline=metrics(np.zeros_like(e['y']),e['y']),previous_action_baseline=metrics(e['previous'],e['y'])))
    checkpoint=dict(schema=SCHEMA,feature_spec=spec,feature_spec_sha256=spec_sha,actor_state_dict={k:v.detach().cpu() for k,v in actor.state_dict().items()},
                    normalization_mean=torch.from_numpy(mean),normalization_scale=torch.from_numpy(scale),normalization_frozen=torch.from_numpy(frozen),
                    assignment=assignment,assignment_sha256=sha(args.assignment),manifest_sha256=sha(args.manifest),
                    training_options=dict(steps=len(losses),command_dropout=args.command_dropout,terrain_scale_floor_m=args.terrain_scale_floor_m,neutral_axis_weight=args.neutral_axis_weight),
                    action_hz=60,action_limits_si=[.12,.0225,.0225,.0675],physics_rollout_qualified=False,formal_training_admission=False)
    path=out/'pilot.pt';torch.save(checkpoint,path)
    restored=torch.load(path,map_location='cpu',weights_only=True);probe=build_actor(torch,train_x.shape[1]);probe.load_state_dict(restored['actor_state_dict']);probe.eval()
    with torch.inference_mode():
        sample=torch.from_numpy(normalize(train_x[:16],mean,scale,frozen));left=probe(sample);right=actor(sample.to(device)).cpu()
        error=float(torch.max(torch.abs(left-right)))
    if error>1e-5:raise ValueError('checkpoint_reload_mismatch')
    report=dict(schema=SCHEMA,status='bounded_offline_pilot_completed',steps=len(losses),requested_steps=args.steps,seed=20260924,
                training_wall_seconds=time.monotonic()-start,device=str(device),torch_version=torch.__version__,feature_count=train_x.shape[1],
                development_rows=len(train_x),validation_rows=sum(len(e['x']) for e in val),normalization_fit_runs=[e['meta']['run'] for e in dev],
                loss_initial=losses[0],loss_final=losses[-1],checkpoint=str(path.resolve()),checkpoint_sha256=sha(path),reload_max_abs_error=error,
                evaluation=evaluation,manifest_sha256=sha(args.manifest),assignment_sha256=sha(args.assignment),
                code_sha256={str(Path(__file__).resolve()):sha(__file__),str(Path(__file__).resolve().parents[2]/'src/v4/contact_phase_bc_policy.py'):sha(Path(__file__).resolve().parents[2]/'src/v4/contact_phase_bc_policy.py'),str(Path(__file__).resolve().parents[2]/'src/v4/contact_bc_policy.py'):sha(Path(__file__).resolve().parents[2]/'src/v4/contact_bc_policy.py')},
                physics_rollout_qualified=False,formal_training_admission=False,command_dropout=args.command_dropout,
                terrain_scale_floor_m=args.terrain_scale_floor_m,
                neutral_axis_weight=args.neutral_axis_weight,
                excluded_fit_runs=[e['meta']['run'] for e in episodes if e['split']=='development' and e['meta']['run'] not in {d['meta']['run'] for d in dev}],
                model_selection=f'fixed {args.steps}-step pilot budget; no validation-based checkpoint selection',
                scope='Pipeline smoke and development action errors, not autonomous excavation success or formal A30 training.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='evaluation'}))

if __name__=='__main__':main()
