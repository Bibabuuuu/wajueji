"""Diagnostic native-60Hz four-axis BC, separate from frozen legacy actors."""
from __future__ import annotations
import hashlib,json
import numpy as np

SCHEMA='contact_bc_native60_history_mlp_v1'
HISTORY_OFFSETS=(3,6,9)  # 50/100/150 ms at native 60 Hz.
STATE_KEYS=('mechanism_qpos','mechanism_qvel','controller_target','controller_target_velocity',
            'bucket_integral_effort','bucket_active_curl_effort','bucket_origin_world_m',
            'bucket_linear_velocity_world_m_s','bucket_angular_velocity_world_rad_s',
            'previous_applied_action_si','previous_drive_load','previous_drive_load_valid')
COMPACT_KEYS=('mechanism_qpos','mechanism_qvel','bucket_origin_world_m',
              'previous_applied_action_si','previous_drive_load','previous_drive_load_valid')

def matrix(arrays,keys):
    n=len(arrays['frame'])
    return np.concatenate([np.asarray(arrays[k],dtype=np.float32).reshape(n,-1) for k in keys],axis=1)

def encode_episode(arrays, *, goal_valid=True):
    """Consumes causal staging only. History is zero-masked across every gap."""
    n=len(arrays['frame']);shape=arrays['terrain_height_m'].shape[1:]
    if len(shape)!=2:raise ValueError('terrain_shape')
    parts=[];names=[];continuous=[]
    def add(name,value,normalize=True):
        value=np.asarray(value,dtype=np.float32).reshape(n,-1)
        start=len(names);names.extend(f'{name}[{j}]' for j in range(value.shape[1]))
        if normalize:continuous.extend(range(start,len(names)))
        parts.append(value)
    add('terrain_height_m',arrays['terrain_height_m'])
    add('terrain_known',arrays['terrain_known'],False)
    add('terrain_origin_xy_m',arrays['terrain_origin_xy_m'])
    add('terrain_resolution_m',arrays['terrain_resolution_m'])
    add('terrain_age_frames',arrays['terrain_age_frames'])
    for key in STATE_KEYS:add(key,arrays[key],key!='previous_drive_load_valid')
    goal=np.asarray(arrays['goal_world_m'])-np.asarray(arrays['bucket_origin_world_m'])
    add('goal_relative_m',goal if goal_valid else np.zeros_like(goal))
    add('goal_valid',np.full((n,1),goal_valid),False)
    compact=matrix(arrays,COMPACT_KEYS)
    for offset in HISTORY_OFFSETS:
        prior=np.zeros_like(compact);valid=np.zeros(n,dtype=bool)
        if n>offset:
            good=(arrays['frame'][offset:]-arrays['frame'][:-offset]==offset)&(arrays['sequence_id'][offset:]==arrays['sequence_id'][:-offset])
            valid[offset:]=good
            dest=np.flatnonzero(good)+offset;prior[dest]=compact[dest-offset]
        add(f'history_minus_{offset}',prior)
        add(f'history_minus_{offset}_valid',valid,False)
    x=np.concatenate(parts,axis=1)
    if not np.isfinite(x).all():raise ValueError('nonfinite_features')
    spec=dict(schema=SCHEMA,action_hz=60,action_axes=['swing','boom','arm','bucket'],terrain_shape=list(shape),
              history_offsets=list(HISTORY_OFFSETS),feature_names=names,continuous_indices=continuous,
              labels='applied normalized velocity; requests retained for diagnostics',
              phase_routing='none',goal_override='known manual-retarget episodes mask original goal, never substitute future coordinates')
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    return x,spec,digest

def fit_normalization(x,continuous_indices):
    mean=np.zeros(x.shape[1],np.float32);scale=np.ones_like(mean);frozen=np.zeros(x.shape[1],bool)
    ids=np.asarray(continuous_indices,dtype=int)
    mean[ids]=x[:,ids].mean(axis=0);std=x[:,ids].std(axis=0)
    frozen[ids]=std<1e-6;scale[ids]=np.maximum(std,1e-6)
    return mean,scale,frozen

def normalize(x,mean,scale,frozen):
    result=(x-mean)/scale
    result[:,frozen]=0
    return result.astype(np.float32)

def build_actor(torch,feature_count):
    return torch.nn.Sequential(torch.nn.Linear(feature_count,256),torch.nn.ReLU(),
                               torch.nn.Linear(256,256),torch.nn.ReLU(),torch.nn.Linear(256,4),torch.nn.Tanh())
