"""Real cached observations: rejected lookahead must not drop causal history."""
import json
import numpy as np
from check_source import ROOT,W,read,sha
from features import CausalFeatures
from contact_bc_inference import structured_observation
from contact_phase_bc_inference import with_phase

def main():
    run=ROOT/'runs/contact-multisource-lift-20260924-v1/first-candidate'
    report=read(run/'report.json');actions=[json.loads(s) for s in (run/'actions.ndjson').read_text().splitlines()][:11]
    terrains={e['frame']:e['terrain'] for e in [json.loads(s) for s in (run/'terrain.ndjson').read_text().splitlines()]}
    f=CausalFeatures();rows=[]
    for e in actions:
        row=structured_observation(frame=e['frame'],state=e['before_state'],terrain=terrains[e['terrain_anchor_frame']],
            terrain_frame=e['terrain_anchor_frame'],goal=report['selected_goal'],
            previous_action=e['previous_applied_action_si'],previous_load=e['previous_drive_load'])
        row=with_phase(row,e['phase'],e['frame']-1);rows.append(row);f.update(row)
    assert [int(r['frame']) for r in f.rows]==list(range(2,12))
    f.retain_executed(10)
    assert f.last_frame==10 and [int(r['frame']) for r in f.rows]==list(range(1,11))
    assert all(np.array_equal(f.rows[i][k],rows[i][k]) for i in range(10) for k in rows[i])
    # Recomputing frame 11 after restoration must reproduce the original feature.
    original=CausalFeatures()
    for row in rows:x,h=original.update(row)
    restored,h2=f.update(rows[-1]);assert h==h2 and np.array_equal(x,restored)
    z=CausalFeatures();z.update(rows[0]);z.retain_executed(0)
    assert z.last_frame is None and len(z.rows)==0
    result=dict(passed=True,real_observations=11,restored_frames=list(range(1,11)),feature_exact=True,
        zero_history_empty=True,physics_steps_added=0,
        inputs={str(p):sha(p) for p in [run/'report.json',run/'actions.ndjson',run/'terrain.ndjson',W/'features.py']})
    (W/'feature-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
