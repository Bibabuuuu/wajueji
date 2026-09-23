"""Pair observed cut strips with safe prepoints, preserving future lateral room.

Scores are development heuristics, not a proof of optimal three-scoop yield.
No particle, terrain, rock or mechanism state is modified by planning.
"""
import numpy as np
from planning.cycle import _cut_inputs, _cut_candidate
from planning.rear_lower_face import select_rear_lower_face
from planning.rear_lower_prepoint import plan_rear_lower_prepoint, mechanism_and_truck_clearance
from layered_47_cycle_runtime import REPO
from layered_47_cycle_geometry import TransportClearance, stationary_terrain
from planning.cycle_geometry import solve_pose, TIP
from planning.approach import sampled_approach


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
    terrain,observation=stationary_terrain(runtime, arrays)
    candidates=rank_workspace_candidates(terrain,3-runtime.metadata['cycle_index'])
    with np.load(REPO/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-mesh.npz') as data:truck=data['world_vertices'].copy()
    checker=TransportClearance(runtime,terrain=terrain)
    def check_clear(qpos,previous):
        result=checker.check(qpos,previous)
        if result['bucket_terrain_m']<.32:
            raise ValueError('cycle_return_open_stock_clearance:'+str(result))
        bodies=runtime.assembly.assemble(qpos,include_meshes=False)[1]
        mechanism=mechanism_and_truck_clearance(checker.geometries,bodies,terrain,truck)
        if any(value<.02 for value in mechanism.values()):
            raise ValueError('cycle_return_open_mechanism_clearance:'+str(mechanism))
        return result
    def advance(qpos,velocity,dt):
        runtime.chain.q=qpos.copy();return runtime.chain.advance(velocity,dt)[0]
    # Bring only the geometry preview onto the closed manifold. The physical
    # state is never assigned; execution later follows these goals by commands.
    closed=np.array(q,dtype=float)
    for _ in range(12):
        residual=runtime.chain.residual(closed)
        if np.max(np.abs(residual))<1e-8:break
        jac=[]
        for address in runtime.chain.passive:
            other=closed.copy();other[address]+=1e-6
            jac.append((runtime.chain.residual(other)-residual)/1e-6)
        delta=np.linalg.lstsq(np.array(jac).T,-residual,rcond=1e-10)[0]
        if np.max(np.abs(delta))>.05:raise ValueError('cycle_preview_passive_branch')
        closed[runtime.chain.passive]+=delta
    else:raise ValueError('cycle_preview_closure')
    rejections=[];tried=[]
    for candidate in candidates:
        xy=np.array(candidate['entry_xy_m'])
        if any(np.linalg.norm(xy-prev)<.14 for prev in tried):continue
        tried.append(xy)
        try:
            radial=xy/np.linalg.norm(xy);seed=closed.copy()
            swing=seed.copy();swing[7]=np.arctan2(xy[1],xy[0])
            transfer_goals=[('RETURN_ALIGN_HIGH',swing)]
            seed=swing.copy()
            for name,radius,height,pitch in [('RETURN_EXTEND_HIGH',5.2,2.7,.2),
                                             ('RETURN_OPEN_HIGH',5.4,2.5,-.32)]:
                seed=solve_pose(runtime.assembly,runtime.chain,seed,np.r_[radius*radial,height],pitch,TIP)
                check_clear(seed,None)
                transfer_goals.append((name,seed.copy()))
            # A cut-face median can be much lower than a rock or uncut ridge
            # elsewhere along the return corridor. Transfer height must clear
            # the current observed stock, not just the selected face median.
            high_height=float(np.nanmax(terrain['height_m']))+.55
            high_offset=max(.55,high_height-candidate['contact']['crest_surface_z_m'])
            # Screen the final approach before spending time sweeping transfers.
            path=plan_rear_lower_prepoint(terrain=terrain,contact_selection=candidate['contact'],operator_qpos=seed,
                open_bucket_proxy=runtime.proxy,truck_world_vertices=truck,assembly=runtime.assembly,chain=runtime.chain,
                transfer_profiles=((high_offset,-.65),(high_offset,-.70),(high_offset,-.75),
                                   (high_offset+.05,-.75)),
                prepoint_offsets=((.30,.30),(.35,.40),(.40,.50)))
            prefix_q=[];prefix_v=[];prefix_segments=[];start=closed.copy()
            for name,goal in transfer_goals:
                segment=sampled_approach(start,goal,axes=runtime.chain.axes,
                    speed_limits=np.array([.12,.0225,.0225,.0225]),advance=advance,check_pose=check_clear,
                    dt=1/60,max_steps=2400,validation_substeps=2)
                prefix_q.append(segment['qpos'] if not prefix_q else segment['qpos'][1:])
                prefix_v.append(segment['velocity'])
                prefix_segments.append(dict(name=name,frames=segment['steps'],goal_qpos=goal.tolist(),
                    minimum_bucket_terrain_clearance_m=min(d['bucket_terrain_m'] for d in segment['diagnostics'])))
                start=goal
            path['qpos']=np.concatenate([*prefix_q,path['qpos'][1:]],axis=0)
            path['velocity']=np.concatenate([*prefix_v,path['velocity']],axis=0)
            path['segments']=prefix_segments+path['segments']
            path['total_frames']=len(path['velocity']);path['total_duration_sim_s']=path['total_frames']/60
            return dict(selected=candidate,path=path,observation=observation,terrain=terrain,
                        rejected_pairs=rejections,candidate_count=len(candidates),
                        cycle_index=runtime.metadata['cycle_index']+1,
                        selection_scope='observed spatial candidates; future room heuristic; no fixed left/center/right sequence')
        except ValueError as error:
            rejections.append(dict(entry_xy_m=candidate['entry_xy_m'],failure=str(error)))
            print('PAIR_REJECT',candidate['entry_xy_m'],str(error),flush=True)
        if len(tried)>=maximum_pairs:break
    raise ValueError('cycle_no_safe_paired_prepoint:'+str(rejections))
