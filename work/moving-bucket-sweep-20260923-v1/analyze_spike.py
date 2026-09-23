from pathlib import Path
import json
import numpy as np
w=Path(__file__).resolve().parent;run=w/'spike-20';p=np.load(run/'frame-20-probe.npz');a=[json.loads(l) for l in (run/'trace.ndjson').read_text().splitlines()];b=[json.loads(l) for l in (w/'unload-180/trace.ndjson').read_text().splitlines()][:20]
keys=['after_state','native_projection','particle_max_speed_m_s','particle_minimum_z_m','original_cavity_particles_still_in_cavity']
exact=all(all(x[k]==y[k] for k in keys) for x,y in zip(a,b))
def flatten(x):
 if isinstance(x,dict):return sum((flatten(v) for v in x.values()),[])
 if isinstance(x,list):return sum((flatten(v) for v in x),[])
 return [float(x)] if isinstance(x,(float,int)) else []
errors=[float(np.max(np.abs(np.array(flatten(x['after_state']))-np.array(flatten(y['after_state']))))) for x,y in zip(a,b)]
assert max(errors)<1e-12
particle_tracks_exact=np.array_equal(np.load(run/'post-projection-world.npy'),np.load(w/'unload-180/post-projection-world.npy')[:21])
assert particle_tracks_exact
s=np.load(w/'full-887/tracking-selection.npz');cavity=set(s['ids'][s['initial_cavity']].tolist());speeds=np.linalg.norm(p['pre_projection_v'],axis=1);post=np.linalg.norm(p['protected_v'],axis=1);delta=np.linalg.norm(p['protected_q']-p['native_q'],axis=1)
ids=set(np.flatnonzero(speeds>10).tolist())|{int(delta.argmax()),int(post.argmax())};events=[]
for i in sorted(ids):
 events.append({'id':i,'original_cavity':i in cavity,'pre_projection_speed_m_s':float(speeds[i]),'native_projected_speed_m_s':float(np.linalg.norm(p['native_v'][i])),'protected_speed_m_s':float(post[i]),'candidate_position_delta_m':float(delta[i]),'sweep_hits':int(p['flags'][i]),'world_pre_projection':p['pre_projection_q'][i].tolist(),'world_protected':p['protected_q'][i].tolist()})
result={'replayed_20_recorded_states_exact':exact,'maximum_numeric_state_absolute_difference':max(errors),'tracked_particle_positions_exact':particle_tracks_exact,'input_bindings_unchanged':json.loads((run/'report.json').read_text())['bindings_unchanged'],'pre_projection_speed_over_10_count':int((speeds>10).sum()),'protected_speed_over_10_count':int((post>10).sum()),'candidate_position_delta_over_20mm_count':int((delta>.02).sum()),'events':events,'inference':'high velocity already exists before native project_outside and this frame sweep; does not exclude effects of earlier candidate dynamics','not_yet_known':'specific contact/strain trigger and origin of energy; no full new-physics qualification'}
(w/'spike-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
