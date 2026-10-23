"""Bounded first SAC experiment; one physical runtime, frozen independent evaluations."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'local-data/runtime/sb3-2.9.0-py312'),str(ROOT/'src/v4'),str(ROOT)]
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from manual_residual_sac_env import Budget, BudgetExpired, ManualResidualEnv, sha, write, OBS_SIZE, environment_contract, require_environment_contract


def episode(env, policy):
    try:
        observation,_=env.reset()
        while True:
            action=policy(observation)
            observation,_,done,truncated,info=env.step(action)
            if done or truncated:return info['episode_summary']
    finally:env.close()


class Chunk(BaseCallback):
    def __init__(self,budget,path,minimum_steps,reserve):
        super().__init__();self.budget=budget;self.path=path
        self.minimum_steps=minimum_steps;self.reserve=reserve;self.started_steps=None;self.last=0.
        self.stop_next=False

    def _on_training_start(self):self.started_steps=self.model.num_timesteps

    def _on_step(self):
        # SB3 calls callbacks before storing this step. Stop on the next step,
        # so a completed episode's terminal transition and update are retained.
        if self.stop_next:return False
        if time.monotonic()-self.last>=30:
            write(self.path,dict(status='training',timesteps=self.model.num_timesteps,
                    gradient_updates=self.model._n_updates,remaining_seconds=self.budget.remaining()))
            print('SAC_PROGRESS',self.model.num_timesteps,'updates',self.model._n_updates,
                  'remaining',round(self.budget.remaining()),flush=True);self.last=time.monotonic()
        if self.budget.remaining()<self.reserve:
            if any(self.locals['dones']):
                self.stop_next=True;return True
            return False
        if self.model.num_timesteps-self.started_steps>=self.minimum_steps and any(self.locals['dones']):
            self.stop_next=True
        return True


def synthetic_check(output):
    import gymnasium as gym
    class Toy(gym.Env):
        def __init__(self):
            self.action_space=gym.spaces.Box(-1.,1.,(5,),np.float32)
            self.observation_space=gym.spaces.Box(-10.,10.,(OBS_SIZE,),np.float32);self.frames=0
        def reset(self,seed=None,options=None):
            super().reset(seed=seed);self.frames=0;return np.zeros(OBS_SIZE,np.float32),{}
        def step(self,action):
            self.frames+=1
            return np.zeros(OBS_SIZE,np.float32),float(-np.sum(action*action)),self.frames>=8,False,{}
    torch.set_num_threads(1)
    env=Toy();model=SAC('MlpPolicy',env,policy_kwargs={'net_arch':[64,64]},
        device='cpu',learning_starts=0,buffer_size=128,batch_size=8,seed=42,ent_coef=1e-5,gamma=1.,replay_buffer_kwargs={'handle_timeout_termination':False})
    model.learn(24);model.save(output)
    loaded=SAC.load(output,device='cpu')
    a,_=model.predict(np.zeros(OBS_SIZE,np.float32),deterministic=True)
    b,_=loaded.predict(np.zeros(OBS_SIZE,np.float32),deterministic=True)
    if not np.array_equal(a,b) or model._n_updates!=24:raise ValueError('synthetic_sac_checkpoint_check')
    class NoPhysicsBudget:
        def remaining(self):return 10000.
    stopped=SAC('MlpPolicy',Toy(),policy_kwargs={'net_arch':[64,64]},device='cpu',
                learning_starts=0,buffer_size=128,batch_size=8,seed=42,ent_coef=1e-5,gamma=1.,replay_buffer_kwargs={'handle_timeout_termination':False})
    stopped.learn(100,callback=Chunk(NoPhysicsBudget(),output.parent/'terminal-test.json',8,1200))
    if stopped.replay_buffer.size()!=8 or stopped.replay_buffer.dones[:8].sum()!=1 or stopped._n_updates!=8:
        raise ValueError('completed_episode_terminal_transition_missing')
    print(json.dumps(dict(status='passed',new_physics_frames=0,updates=model._n_updates,
        save_reload_actions_exact=True,terminal_transition_retained=True,versions={x:importlib.metadata.version(x) for x in
        ['stable-baselines3','gymnasium','torch','numpy','cloudpickle']})))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--synthetic-check',action='store_true')
    parser.add_argument('--budget',type=Path)
    parser.add_argument('--budget-seconds',type=float,default=7200)
    parser.add_argument('--evaluation-reserve',type=float)
    parser.add_argument('--final-only-evaluation',action='store_true')
    parser.add_argument('--alignment-admission',type=Path,
                        default=ROOT/'runs/manual-progress-alignment-20261013-v2/training-admission.json')
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--reward-transfer',type=Path)
    parser.add_argument('--waiting-action-mode',choices=['signed_residual_v2','error_strength_v1'],default='signed_residual_v2')
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    admission=json.loads(args.alignment_admission.read_text()) if not args.synthetic_check else None
    if not args.synthetic_check:
        for name in ['manual_residual_sac_env.py','manual_trajectory_alignment.py','manual_trajectory_residual.py']:
            if admission['code'].get(name)!=sha(ROOT/'src/v4'/name):raise ValueError('alignment_admission_code_changed')
    if args.waiting_action_mode=='error_strength_v1':
        require_environment_contract(admission,args.waiting_action_mode)
        for path,h in admission['source_evidence'].items():
            if sha(path)!=h:raise ValueError('training_admission_evidence_changed')
        for name,h in admission['runtime_bindings'].items():
            if sha(ROOT/'src/v4'/name)!=h:raise ValueError('training_physics_changed')
    if args.check_only:
        print(json.dumps(dict(status='ready_for_training' if admission.get('training_admission') is True else 'awaiting_verified_new_collection',
                              environment_contract=environment_contract(args.waiting_action_mode),physical_frames=0,neural_updates=0)));return
    def make_env(path,budget,**options):
        return ManualResidualEnv(path,budget,waiting_action_mode=args.waiting_action_mode,**options)
    if args.reward_transfer and args.resume_from:raise ValueError('fresh_policy_transfer_excludes_resume')
    if args.output.exists():raise ValueError('immutable_batch_output_exists')
    args.output.mkdir(parents=True)
    if args.synthetic_check:synthetic_check(args.output/'model');return
    torch.set_num_threads(1)
    if args.budget is None:raise ValueError('physical_batch_requires_shared_budget_path')
    args.budget.parent.mkdir(parents=True,exist_ok=True)
    if args.evaluation_reserve is not None and not 0<args.evaluation_reserve<args.budget_seconds:
        raise ValueError('invalid_evaluation_reserve')
    budget=Budget(args.budget,seconds=args.budget_seconds);env=None;model=None
    if budget.data is not None and budget.data['seconds']!=args.budget_seconds:
        raise ValueError('existing_budget_duration_mismatch')
    if budget.data is not None and budget.data.get('purpose','training_and_evaluation')!='training_and_evaluation':
        raise ValueError('validation_budget_cannot_authorize_training')
    if admission.get('training_admission') is not True:
        raise ValueError('alignment_wait_recovery_not_qualified')
    manifest=dict(status='preflight',environment_contract=environment_contract(args.waiting_action_mode),physics_domain='CycleRuntime_D165_motion_consistency',
        decisions=dict(base='saved_applied_manual_actions',algorithm='SB3_SAC',
            actions='four axis residuals plus monotonic reference pace',residual_cap_fraction=.1,
            pace=[.5,1.5],score='minimum of three bucket display estimates over 0.5s hold',
            budget_seconds=args.budget_seconds,baseline_fallback=True,
            final_only_evaluation=args.final_only_evaluation),probes=[],training_chunks=[],evaluations=[],
        code={str(p):sha(p) for p in [Path(__file__),ROOT/'src/v4/manual_residual_sac_env.py',
                                      ROOT/'src/v4/manual_trajectory_residual.py',
                                      ROOT/'src/v4/manual_trajectory_alignment.py']},
        versions={x:importlib.metadata.version(x) for x in
                  ['stable-baselines3','gymnasium','torch','numpy','cloudpickle']})
    write(args.output/'manifest.json',manifest)
    try:
        prior=None
        if args.resume_from is not None:
            prior=json.loads((args.resume_from/'manifest.json').read_text())
            if args.waiting_action_mode=='error_strength_v1':require_environment_contract(prior,args.waiting_action_mode)
            for path,expected in prior['code'].items():
                if Path(path).name!='run_manual_residual_sac.py' and sha(path)!=expected:
                    raise ValueError('resume_environment_code_changed')
            audit_root=Path(prior['baseline']['path']).parents[1]
            audit=json.loads((audit_root/'preflight-independent-audit.json').read_text())
            if audit['status']!='passed' or not all(audit['checks'].values()):
                raise ValueError('resume_preflight_not_qualified')
            manifest['resume_from']=str(args.resume_from.resolve())
            manifest['resume_reason']='retain training and replay after terminal callback correction; missing old transitions not fabricated'
            manifest['probes']=prior['probes'];baseline=prior['baseline'];transitions=None
            for path,expected in baseline['runtime_bindings'].items():
                if sha(ROOT/'src/v4'/path)!=expected:raise ValueError('resume_physics_code_changed')
            budget.start()
        elif args.reward_transfer is not None:
            transfer=json.loads(args.reward_transfer.read_text())
            if args.waiting_action_mode=='error_strength_v1':require_environment_contract(transfer,args.waiting_action_mode)
            if transfer['status']!='passed' or transfer['receiver_env_sha256']!=sha(ROOT/'src/v4/manual_residual_sac_env.py'):
                raise ValueError('unqualified_reward_transfer')
            if sha(Path(transfer['replay_path']))!=transfer['replay_sha256']:
                raise ValueError('transferred_replay_changed')
            baseline=transfer['baseline'];transitions=[]
            for path,expected in baseline['runtime_bindings'].items():
                if sha(ROOT/'src/v4'/path)!=expected:raise ValueError('transfer_physics_changed')
            manifest['reward_transfer']=transfer
            env=make_env(args.output/'reward-zero-check',budget,max_frames=120,zero_check=True)
            result=episode(env,lambda obs:np.zeros(5));manifest['probes'].append(result)
            if result['status']!='timeout' or result['physical_frames']!=120 or not result['source_unchanged']:
                raise ValueError('reward_only_zero_physics_check_failed')
        else:
            for label,action in [('zero',np.zeros(5)),('positive',np.ones(5)),('negative',-np.ones(5))]:
                env=make_env(args.output/f'probe-{label}',budget,max_frames=120,zero_check=label=='zero')
                result=episode(env,lambda obs,action=action:action);manifest['probes'].append(result)
                write(args.output/'manifest.json',manifest)
                if result['status']!='timeout' or result['physical_frames']!=120 or not result['source_unchanged']:
                    raise ValueError('bounded_residual_probe_failed_'+label)
            env=make_env(args.output/'baseline',budget,collect=True)
            baseline=episode(env,lambda obs:np.zeros(5));transitions=env.transitions
        manifest['baseline']=baseline
        write(args.output/'manifest.json',manifest)
        if not baseline['success'] or not baseline['source_unchanged']:
            raise ValueError('held_manual_baseline_not_qualified')
        # Real zero-action RL transitions seed SAC's replay; no BC objective.
        reference_wall=max(180.,baseline['wall_seconds'])
        reserve=max(360.,reference_wall*1.8) if args.reward_transfer else max(1200.,reference_wall*3.5)
        if args.evaluation_reserve is not None:reserve=args.evaluation_reserve
        manifest['evaluation_reserve_seconds']=reserve
        train=make_env(args.output/'training',budget)
        if prior is not None:
            checkpoint=args.resume_from/'sac-interrupted.zip';replay=args.resume_from/'replay-buffer.pkl'
            manifest['resume_bindings']={str(checkpoint):sha(checkpoint),str(replay):sha(replay)}
            model=SAC.load(checkpoint,env=train,device='cpu')
            model.load_replay_buffer(replay)
            model.gamma=1.
            model.ent_coef=1e-5
            model.ent_coef_tensor=torch.tensor(1e-5,device=model.device)
            model.replay_buffer.handle_timeout_termination=False
            model.replay_buffer.timeouts.fill(0)
            manifest['objective_correction']='undiscounted episodic held volume; entropy 1e-5; contractual timeout is terminal failure'
            manifest['resumed_timesteps']=model.num_timesteps
            manifest['resumed_gradient_updates']=model._n_updates
            manifest['resumed_replay_transitions']=model.replay_buffer.size()
            manifest['replay_seed_transitions']=prior.get('replay_seed_transitions',json.loads((audit_root/'manifest.json').read_text())['replay_seed_transitions'])
        else:
            model=SAC('MlpPolicy',train,learning_rate=3e-4,buffer_size=30000,batch_size=64,
                      gamma=1.,learning_starts=0,train_freq=1,gradient_steps=1,ent_coef=1e-5,
                      replay_buffer_kwargs={'handle_timeout_termination':False},
                      policy_kwargs={'net_arch':[64,64]},device='cpu',seed=42)
            with torch.no_grad():
                model.actor.mu.weight.zero_();model.actor.mu.bias.zero_()
                model.actor.log_std.weight.zero_();model.actor.log_std.bias.fill_(-3.)
            for before,after,action,reward,done,info in transitions:
                model.replay_buffer.add(before[None],after[None],action[None],np.array([reward]),np.array([done]),[info])
            if args.reward_transfer:
                model.load_replay_buffer(Path(transfer['replay_path']))
                if model.replay_buffer.observations.shape[-1]!=OBS_SIZE or model._n_updates!=0 or model.num_timesteps!=0:
                    raise ValueError('fresh_policy_transfer_contract')
            manifest['replay_seed_transitions']=model.replay_buffer.size()
        manifest['sac_configuration']=dict(net_arch=[64,64],learning_rate=3e-4,buffer_size=30000,
                    batch_size=64,gamma=1.,learning_starts=0,train_freq=1,gradient_steps=1,
                    ent_coef=1e-5,timeout_bootstrap=False,seed=42,initial_mean='zero',initial_log_std=-3.,device='cpu')
        best=None;round_id=0
        while budget.remaining()>reserve+reference_wall*2:
            round_id+=1;env=train;model.set_env(train,force_reset=True)
            callback=Chunk(budget,args.output/'progress.json',
                           200000 if args.final_only_evaluation else (300 if args.reward_transfer else 1200),reserve)
            model.learn(200000,callback=callback,reset_num_timesteps=False)
            train.close()
            checkpoint=args.output/f'sac-round-{round_id:02d}'
            model.save(checkpoint)
            manifest['training_chunks'].append(dict(round=round_id,timesteps=model.num_timesteps,
                updates=model._n_updates,checkpoint=str(checkpoint.with_suffix('.zip')),
                completed_episodes=list(train.completed),remaining_seconds=budget.remaining()))
            write(args.output/'manifest.json',manifest)
            if args.final_only_evaluation or budget.remaining()<reserve:break
            frozen=SAC.load(checkpoint,device='cpu')
            env=make_env(args.output/f'evaluation-{round_id:02d}',budget)
            result=episode(env,lambda obs:frozen.predict(obs,deterministic=True)[0])
            result['policy_checkpoint']=str(checkpoint.with_suffix('.zip'));result['round']=round_id
            manifest['evaluations'].append(result)
            if result['success'] and (best is None or result['retained_fill_fraction']>best['retained_fill_fraction']):best=result
            manifest['best_candidate']=best;write(args.output/'manifest.json',manifest)
        train.close()
        model.save(args.output/'sac-last');model.save_replay_buffer(args.output/'replay-buffer.pkl')
        manifest['final_evaluations']=[]
        if args.final_only_evaluation and manifest['training_chunks']:
            best={'policy_checkpoint':manifest['training_chunks'][-1]['checkpoint']}
        if best is not None and budget.remaining()>reference_wall*2.5:
            frozen=SAC.load(best['policy_checkpoint'],device='cpu')
            for repetition in range(2):
                if budget.remaining()<reference_wall*1.15:break
                env=make_env(args.output/f'final-candidate-{repetition+1}',budget)
                result=episode(env,lambda obs:frozen.predict(obs,deterministic=True)[0])
                result['policy_checkpoint']=best['policy_checkpoint'];manifest['final_evaluations'].append(result)
                write(args.output/'manifest.json',manifest)
        candidate=manifest['final_evaluations']
        improved=len(candidate)==2 and all(x['success'] and x['source_unchanged'] for x in candidate) and min(
                    x['retained_fill_fraction'] for x in candidate)>baseline['retained_fill_fraction']+1e-6
        manifest.update(status='completed',accepted_improvement=improved,
              recommendation='candidate_for_transport_test' if improved else 'retain_manual_baseline',
              final_timesteps=model.num_timesteps,final_gradient_updates=model._n_updates,
              remaining_seconds=budget.remaining(),scope='single frozen scene held display fill only; no truck result or generalization')
    except BaseException as error:
        manifest.update(status='budget_exhausted' if isinstance(error,BudgetExpired) else ('controlled_interruption' if isinstance(error,KeyboardInterrupt) else 'failed'),
                        error=str(error),exception_type=type(error).__name__,recommendation='retain_manual_baseline',
                        remaining_seconds=budget.remaining())
        if model is not None:
            model.save(args.output/'sac-interrupted');model.save_replay_buffer(args.output/'replay-buffer.pkl')
        raise
    finally:
        if env is not None:
            try:env.close()
            except BaseException as close_error:
                manifest['cleanup_error']=type(close_error).__name__+':'+str(close_error)
                if manifest['status']=='completed':manifest['status']='failed_cleanup'
        manifest['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        write(args.output/'manifest.json',manifest)
        print('SAC_BATCH_RESULT',manifest['status'],manifest.get('recommendation'),flush=True)


if __name__=='__main__':main()
