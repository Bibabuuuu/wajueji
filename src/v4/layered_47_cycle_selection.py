"""Pair observed cut strips with safe prepoints, preserving future lateral room.

Scores are development heuristics, not a proof of optimal three-scoop yield.
No particle, terrain, rock or mechanism state is modified by planning.
"""
import numpy as np
from planning.cycle import _cut_inputs, _cut_candidate
from planning.rear_lower_face import select_rear_lower_face
from planning.rear_lower_prepoint import plan_rear_lower_prepoint
from layered_excavation_approach_planner import build_fused_terrain
from layered_47_cycle_runtime import REPO


def rank_workspace_candidates(terrain, remaining_scoops):
    if remaining_scoops not in (1,2,3):raise ValueError('cycle_remaining_scoops')
    h,n,origin,cell,excluded,centers=_cut_inputs(terrain,.6,(),.58,1.35,.1)
    candidates=[]
    for iy in range(h.shape[0]):
        for ix in range(h.shape[1]):
            candidate,_=_cut_candidate(h,n,origin,cell,excluded,centers,ix,iy,.6,.58,1.35,.1)
            if candidate is not None:
                try:face=select_rear_lower_face(terrain,candidate)
                except ValueError:continue
                candidate=dict(candidate,contact=face)
                candidates.append(candidate)
    if not candidates:raise ValueError('cycle_no_current_observed_face')
    for first in candidates:
        xy=np.array(first['entry_xy_m']);radial=xy/np.linalg.norm(xy);lateral=np.array([-radial[1],radial[0]])
        options=[]
        for other in candidates:
            if abs(np.dot(np.array(other['entry_xy_m'])-xy,lateral))>=.60:
                options.append(other)
        options.sort(key=lambda c:c['score'],reverse=True)
        future=[]
        for other in options:
            if all(abs(np.dot(np.array(other['entry_xy_m'])-np.array(c['entry_xy_m']),lateral))>=.60 for c in future):
                future.append(other)
            if len(future)>=remaining_scoops-1:break
        if remaining_scoops==1:future=[]
        first['future_strip_count']=len(future)
        first['future_entry_xy_m']=[c['entry_xy_m'] for c in future]
        first['workspace_score']=first['score']+.8*sum(c['score'] for c in future)
        first['remaining_scoops_considered']=remaining_scoops
    candidates.sort(key=lambda c:(c['future_strip_count'],c['workspace_score'],c['score']),reverse=True)
    return candidates


def plan_next_prepoint(runtime, *, maximum_pairs=16):
    if runtime.metadata['cycle_index']>=3:raise ValueError('cycle_three_scoops_complete')
    q=runtime.bridge.state['mechanism_qpos'] if runtime.bridge is not None else runtime.arrays['mechanism_qpos']
    arrays=runtime.capture() if runtime.bridge is not None else runtime.arrays
    terrain,observation=build_fused_terrain(runtime.root,runtime.records,arrays['particle_q'],arrays['rigid_qpos'])
    candidates=rank_workspace_candidates(terrain,3-runtime.metadata['cycle_index'])
    with np.load(REPO/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-mesh.npz') as data:truck=data['world_vertices'].copy()
    rejections=[];tried=[]
    for candidate in candidates:
        xy=np.array(candidate['entry_xy_m'])
        if any(np.linalg.norm(xy-prev)<.14 for prev in tried):continue
        tried.append(xy)
        try:
            path=plan_rear_lower_prepoint(terrain=terrain,contact_selection=candidate['contact'],operator_qpos=q,
                open_bucket_proxy=runtime.proxy,truck_world_vertices=truck,assembly=runtime.assembly,chain=runtime.chain)
            return dict(selected=candidate,path=path,observation=observation,terrain=terrain,
                        rejected_pairs=rejections,candidate_count=len(candidates),
                        cycle_index=runtime.metadata['cycle_index']+1,
                        selection_scope='observed spatial candidates; future room heuristic; no fixed left/center/right sequence')
        except ValueError as error:
            rejections.append(dict(entry_xy_m=candidate['entry_xy_m'],failure=str(error)))
        if len(tried)>=maximum_pairs:break
    raise ValueError('cycle_no_safe_paired_prepoint:'+str(rejections))
