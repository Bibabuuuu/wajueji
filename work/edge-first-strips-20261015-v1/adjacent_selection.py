"""CPU planning helper; fresh terrain in, ranked adjacent candidates out.

Entry boundary gap is a nominal 0.60m bucket-width diagnostic. It is not
excavated-volume overlap. Actual first-cut traces and changed terrain remain
mandatory inputs to the later live handoff audit; this helper admits no motion.
"""
import numpy as np
VERSION='edge_inward_adjacent_entry_v1_cpu'

def rank_adjacent(terrain, previous_entry_xy, *, inward_sign):
    from layered_47_cycle_selection import rank_workspace_candidates
    previous=np.asarray(previous_entry_xy,dtype=float)
    if previous.shape!=(2,) or not np.isfinite(previous).all() or np.linalg.norm(previous)<1 or inward_sign not in (-1,1):
        raise ValueError('adjacent_previous_entry_contract')
    theta=float(np.arctan2(previous[1],previous[0]));radius=float(np.linalg.norm(previous))
    ranked=[];rejected=[]
    for item in rank_workspace_candidates(terrain,1):
        other=np.asarray(item['contact']['contact_xy_m']);angle=float(np.arctan2(other[1],other[0]))
        delta=float(np.arctan2(np.sin(angle-theta),np.cos(angle-theta)))
        if delta*inward_sign<=0:
            rejected.append(dict(anchor=item['entry_xy_m'],reason='wrong_inward_direction'));continue
        # Symmetric radial projection at the prior actual entry station.
        # A cell-centred candidate need not attain an exactly zero gap.
        gap=2*radius*np.sin(abs(delta)/2)-.6*np.cos(abs(delta)/2)
        ranked.append(dict(item,adjacency=dict(version=VERSION,previous_entry_xy_m=previous.tolist(),
            inward_sign=inward_sign,entry_station_radius_m=radius,heading_delta_rad=delta,
            nominal_entry_boundary_gap_m=float(gap),radial_contact_difference_m=float(np.linalg.norm(other)-radius),
            scope='Entry station only; actual sweep overlap/gaps and fresh full-width material require separate audit')))
    ranked.sort(key=lambda c:(abs(c['adjacency']['nominal_entry_boundary_gap_m']),-c['score']))
    if not ranked:raise ValueError('no_observed_inward_adjacent_candidate')
    return ranked,rejected

def self_check(work):
    import json
    from pathlib import Path
    from continuous_scoop import read,write,sha
    w=Path(work)
    with np.load(w/'cpu-edge-v2/terrain.npz') as f:terrain={k:f[k] for k in f.files}
    before={k:np.array(v,copy=True) for k,v in terrain.items()}
    contact=read(w/'reference-edge-v2/report.json')['rotated_reference_contact_m'][:2]
    ranked,rejected=rank_adjacent(terrain,contact,inward_sign=-1)
    # A preview on untouched stock is not a post-first-scoop observation.
    winner=ranked[0];bad_rejected=False
    try:rank_adjacent(terrain,[float('nan'),0],inward_sign=-1)
    except ValueError:bad_rejected=True
    checks=dict(initial_map_unchanged=all(np.array_equal(v,before[k],equal_nan=True) for k,v in terrain.items()),
        finite_prior_required=bad_rejected,only_inward_candidates=all(x['adjacency']['heading_delta_rad']<0 for x in ranked),
        closest_entry_boundary_ranked_first=all(abs(winner['adjacency']['nominal_entry_boundary_gap_m'])<=abs(x['adjacency']['nominal_entry_boundary_gap_m']) for x in ranked),
        preview_gap_below_map_cell=abs(winner['adjacency']['nominal_entry_boundary_gap_m'])<=float(terrain['resolution_m']),
        all_candidates_use_existing_full_width_score=all(x['selection_version']=='full_width_cut_opportunity_v1' for x in ranked))
    result=dict(status='cpu_adjacent_preview_passed' if all(checks.values()) else 'cpu_adjacent_preview_failed',checks=checks,
        selected_preview=winner,candidate_count=len(ranked),rejected=rejected,physical_steps=0,training_updates=0,
        limitation='Untouched initial-stock preview only. After real first dig, unload and return, rerun on fresh supported stock with recorded actual entry and sweep. No second-scoop plan, fill gain or third-strip qualification is granted.')
    write(w/'adjacent-preview.json',result);print(result['status'],winner['entry_xy_m'],winner['adjacency'],flush=True)
    assert all(checks.values()),checks

if __name__=='__main__':
    import sys
    sys.path[:0]=['/home/qunwang/wajueji','/home/qunwang/wajueji/work/continuous-single-scoop-20261014-v1','/home/qunwang/wajueji/src/v4']
    self_check('/home/qunwang/wajueji/work/edge-first-strips-20261015-v1')
