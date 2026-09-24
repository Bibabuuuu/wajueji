"""Same geometric admission limits, checked inside one continuous run."""
import numpy as np


def behavior_gate(actions, physical, feedback, require_curl):
    coordinates=np.asarray([feedback.coordinates(row['after_state']['mechanism_qpos']) for row in physical])
    depth=max([max(0.,e['feedback']['cut_target'][2]-c[2]) for e,c in zip(actions,coordinates) if e['feedback']['cut_target'] is not None],default=0.)
    vertical=[e for e in actions if e['feedback']['soil_plan'] is not None and e['feedback']['coordinates'][3]>=e['feedback']['soil_plan']['vertical_pitch_rad']]
    pitch=max([abs(e['feedback']['coordinates'][3]-e['feedback']['target'][3]) for e in actions if e['phase']=='approach_contact'],default=0.)
    checks=dict(all_hard=all(x['hard_guard'] for x in physical),
        material=all(x['hencky_material_guard']['safe'] for x in physical),
        sweep=all(x['moving_bucket_sweep']['safe'] for x in physical),
        ground=all(feedback.ground.measure(x['after_state']['mechanism_qpos'])['safe'] for x in physical),
        lateral_corridor=bool(np.max(abs(coordinates[:,1]))<=.10),
        cut_depth_respected=bool(depth<=.03),approach_pitch_tracking=bool(pitch<.03),
        action_matches_physics=all(e['applied_command_si']==r['applied_command_si'] for e,r in zip(actions,physical)),
        state_chain=all(actions[i]['before_state']==physical[i-1]['after_state'] for i in range(1,len(actions))),
        causal_terrain=all(0<=e['frame']-1-e['terrain_anchor_frame']<=2 for e in actions),
        causal_phase=all(e['latest_allowed_phase_evidence_frame']==e['frame']-1 for e in actions))
    if require_curl:
        checks.update(entered_curl=any(e['phase']=='curl_load' for e in actions),
            bc_controlled_after_curl=any(e['feedback']['mode']=='bc_with_lateral_depth_feedback' for e in actions),
            root_reached_soil_before_bc=all(e['feedback']['root_ready'] for e in actions if e['phase']=='curl_load'),
            root_at_vertical=bool(vertical and vertical[0]['feedback']['root_above_surface_m']<=.01))
    return dict(passed=all(checks.values()),checks=checks,frames=len(physical),maximum_cut_depth_undershoot_m=float(depth),maximum_approach_pitch_error_rad=float(pitch))
