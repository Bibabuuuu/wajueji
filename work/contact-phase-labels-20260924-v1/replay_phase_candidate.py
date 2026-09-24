"""Feed recorded end states to the partial candidate; never advance physics."""
import hashlib
import json
from pathlib import Path

from contact_phase_candidate import ContactPhaseCandidate, ContactPhaseConfig

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    angle = json.loads((OUT/'curl-angle-candidate-v1.json').read_text())
    audit = json.loads((OUT/'contact-onset-audit-v1.json').read_text())
    cfg = ContactPhaseConfig(angle['angle_threshold_deg'])
    config = dict(schema='partial_contact_phase_candidate_v1', parameters=cfg.__dict__,
                  status='offline_replay_only_not_live_deployed',
                  angle_rule_sha256=sha(OUT/'curl-angle-candidate-v1.json'),
                  force_definition='strictly >10 N for 3 consecutive frame ends',
                  rock_definition='rock_bucket_contact_substeps >0 for 3 consecutive frame ends; pair presence, not calibrated force',
                  frequency_hz=60, maximum_rock_substeps_per_frame=80,
                  causal_rule='Phase before decision n uses end states only through n-1',
                  latching=True, initial_phase='approach_contact',
                  missing_frames='reject_without_mutation',
                  below_angle_at_contact='flag_unresolved; do not invent a curl transition',
                  implemented_transitions=['approach_contact_to_penetrate', 'penetrate_to_curl_load'],
                  unsupported_phases=['lift', 'off_soil'],
                  production_router_qualified=False)
    (OUT/'phase-candidate-v1.json').write_text(json.dumps(config, indent=2)+'\n')
    summaries=[]
    for episode in audit['episodes']:
        run=ROOT/'runs/layered-47-sparse-manual-contact-v1'/episode['run']
        for source in episode['source_inputs']:
            assert sha(Path(source['path'])) == source['sha256']
        rows=[json.loads(s) for s in (run/'physics.ndjson').read_text().splitlines()]
        suffix=episode['run'].removeprefix('manual-contact-20260924-')
        path=OUT/(suffix+'-features.json')
        feature=json.loads(path.read_text())
        assert feature['run'] == episode['run']
        for source in feature['inputs']:
            assert sha(Path(source['path'])) == source['sha256']
        trace=feature['trace'];assert len(trace)==len(rows)+1
        router=ContactPhaseCandidate(cfg, initial_mouth_angle_deg=trace[0]['mouth_chord_from_down_deg'])
        events=[];unresolved=[];curl_frame=None;zero_command_frames=0
        phase_counts={};decisions=[]
        for row, f in zip(rows, trace[1:], strict=True):
            frame=row['frame'];assert f['frame']==frame
            assert f['soil_force_n']==row['bucket_mpm_force_norm_n']
            before=router.phase_for_decision(frame)
            phase_counts[before]=phase_counts.get(before,0)+1
            decisions.append(before)
            result=router.observe_end(frame_end=frame, soil_force_n=row['bucket_mpm_force_norm_n'],
                                      rock_contact_substeps=row['rock_bucket_contact_substeps'],
                                      mouth_angle_deg=f['mouth_chord_from_down_deg'])
            assert result['phase_before_observation']==before
            if result['event']:
                events.append(result['event'])
                if result['event']['transition']=='penetrate_to_curl_load':curl_frame=frame
            if result['curl_boundary_unresolved']:unresolved.append(frame)
            if before!='approach_contact' and all(abs(v)<=1.e-12 for v in row['applied_command_si']):
                zero_command_frames+=1
            if curl_frame is not None and frame>=curl_frame+30:
                break
        assert len(events)==2 and not unresolved
        expected=next(t['event'] for t in episode['force_sensitivity']
                      if t['threshold_n']==10 and t['dwell_frames']==3)
        contact=events[0]
        assert contact['confirmation_frame_end']==expected['confirmation_frame_end']
        assert contact['evidence']==[dict(kind='soil_resultant_force',onset_frame_end=expected['onset_frame_end'])]
        # In particular, the confirming frame's ACTION still has the old phase.
        for event in events:
            i=event['confirmation_frame_end']
            expected_previous='approach_contact' if event is contact else 'penetrate'
            assert decisions[i-1]==expected_previous
            assert decisions[i] == ('penetrate' if event is contact else 'curl_load')
        assert sha(run/'physics.ndjson')==episode['source_inputs'][0]['sha256']
        summaries.append(dict(run=episode['run'], observed_frames=frame,
                              scope='Stop 30 frames after curl candidate; later phases not labeled',
                              events=events, phase_counts_for_decisions=phase_counts,
                              observed_zero_command_frames_after_contact=zero_command_frames,
                              unresolved_angle_frames=unresolved,
                              source_inputs=episode['source_inputs'],
                              feature_file=dict(path=str(path),sha256=sha(path)),
                              confirmed_contact_label=None))
        print(episode['run'],[(e['transition'],e['confirmation_frame_end'],e['first_available_decision_frame']) for e in events])
    report=dict(schema='partial_phase_replay_report_v1',status='recorded_prefix_replay_passed_not_physics_validation',
                config_sha256=sha(OUT/'phase-candidate-v1.json'),
                implementation_sha256=sha(ROOT/'src/v4/contact_phase_candidate.py'),
                replay_script_sha256=sha(Path(__file__)), physics_advanced=False,
                original_annotations_changed=False, live_policy_connected=False,
                stone_first_demonstration_tested=False, production_router_qualified=False,
                episodes=summaries)
    (OUT/'phase-candidate-replay-v1.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
