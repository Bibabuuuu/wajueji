"""Replay recorded kinematics; evaluate only available full-snapshot gaps."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from contact_phase_candidate import ContactPhaseConfig
from contact_phase_completion import CompletionConfig, FivePhaseCandidate
from contact_phase_geometry import SolidBucketGap
from layered_excavation_approach_planner import _load_kinematics

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
KIN=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old=json.loads((OUT/'phase-candidate-replay-v1.json').read_text())
    audit=json.loads((OUT/'lift-endpoint-audit-v1.json').read_text())
    contact=ContactPhaseConfig(**json.loads((OUT/'phase-candidate-v1.json').read_text())['parameters'])
    cfg=CompletionConfig()
    config=dict(schema='contact_phase_completion_candidate_v1',parameters=cfg.__dict__,
        original_contact_config_sha256=sha(OUT/'phase-candidate-v1.json'),
        approved_semantics='Lift follows curl_load and requires sustained actual rise; off-soil gap >=3mm for consecutive fresh frames.',
        engineering_candidate_parameters='5 off-soil frames; 12-frame actual motion test sustained 30 frames plus 8cm total rise; not human boundary calibration.',
        phase_order=['approach_contact','penetrate','curl_load','lift','off_soil'],
        distance_scope='All ground-connected particle centers to existing open solid proxy; 4cm connectivity; no rock clearance',
        live_policy_connected=False,production_qualified=False)
    (OUT/'phase-completion-v1.json').write_text(json.dumps(config,indent=2)+'\n')
    geometry_inputs=audit['geometry_inputs']
    for source in geometry_inputs:assert sha(Path(source['path']))==source['sha256']
    proxy_path=Path(next(x['path'] for x in geometry_inputs if x['path'].endswith('proxy-manifest.json')))
    observer=SolidBucketGap(json.loads(proxy_path.read_text()))
    assembly,_=_load_kinematics(KIN,KIN/'scene/input/model/A.xml',KIN/'scene/input/mesh')
    episodes=[]
    for previous,checked in zip(old['episodes'],audit['episodes'],strict=True):
        assert previous['run']==checked['run']
        for source in previous['source_inputs']+[previous['feature_file'],checked['endpoint']['snapshot']]:
            assert sha(Path(source['path']))==source['sha256']
        trace=json.loads(Path(previous['feature_file']['path']).read_text())['trace']
        router=FivePhaseCandidate(contact,cfg,initial_mouth_angle_deg=trace[0]['mouth_chord_from_down_deg'])
        path=Path(checked['endpoint']['snapshot']['path'])
        with np.load(path,allow_pickle=False) as data:
            q=data['particle_q'];pose=assembly.assemble(data['mechanism_qpos'],include_meshes=False)[1]['bucket']
        start=time.perf_counter();endpoint=observer.observe(q,pose)
        endpoint['cpu_seconds']=time.perf_counter()-start
        assert sha(path)==checked['endpoint']['snapshot']['sha256']
        endpoint.update(frame_end=trace[-1]['frame'],snapshot=checked['endpoint']['snapshot'],
                        five_frame_confirmation_available=False)
        events=[];decisions=[]
        for f in trace[1:]:
            n=f['frame'];decisions.append(router.phase_for_decision(n))
            # No interpolation/last-value hold for missing full particle states.
            gap=endpoint['remaining_soil_gap_m'] if n==trace[-1]['frame'] else None
            out=router.observe_end(frame_end=n,soil_force_n=f['soil_force_n'],
                rock_contact_substeps=f['rock_substeps'],mouth_angle_deg=f['mouth_chord_from_down_deg'],
                actual_boom_cylinder_m=f['q_boom'],bucket_origin_z_m=f['origin_z_m'],
                remaining_soil_gap_m=gap,gap_frame_end=n if gap is not None else None)
            if out['event']:events.append(out['event'])
        assert events[:2]==previous['events']
        assert len(events)==3 and events[-1]['transition']=='curl_load_to_lift'
        assert router.phase=='lift' and router.gap_streak<=1
        for e in events:
            n=e['confirmation_frame_end']
            before,after=e['transition'].split('_to_')
            assert decisions[n-1]==before and decisions[n]==after
        episodes.append(dict(run=previous['run'],source_inputs=previous['source_inputs'],
            feature_file=previous['feature_file'],observed_frames=trace[-1]['frame'],
            events=events,endpoint=endpoint,final_candidate_phase=router.phase,
            off_soil_consecutive_fresh_frames=router.gap_streak,
            human_lift_boundary_confirmed=False,off_soil_boundary_confirmed=False))
        print(previous['run'],'lift',events[-1]['confirmation_frame_end'],'endpoint gap',endpoint['remaining_soil_gap_m'],flush=True)
    inputs=[OUT/'phase-candidate-replay-v1.json',OUT/'lift-endpoint-audit-v1.json',
        OUT/'phase-completion-v1.json',ROOT/'src/v4/contact_phase_candidate.py',
        ROOT/'src/v4/contact_phase_completion.py',ROOT/'src/v4/contact_phase_geometry.py',Path(__file__)]
    result=dict(schema='phase_completion_replay_v1',status='candidate_only_no_training_export',
        inputs=[dict(path=str(p),sha256=sha(p)) for p in inputs],geometry_inputs=geometry_inputs,
        physics_advanced=False,live_policy_connected=False,original_annotations_changed=False,
        qualification='Causal replay and endpoint-only distance, not continuous off-soil or controller qualification',episodes=episodes)
    (OUT/'phase-completion-replay-v1.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
