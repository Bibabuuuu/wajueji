"""Offline particle migration audit; no simulation or parameter changes."""
import json
from pathlib import Path
import numpy as np

out = Path(__file__).resolve().parent
root = out.parent.parent
source = root / 'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089/approach-handoff.npz'
a0 = np.load(source)
q0 = a0['particle_q']
xylo, xyhi = q0[:, :2].min(0), q0[:, :2].max(0)
result = {'definitions': {'bottom_cohort': 'IDs whose initial z < 0.2 m', 'outside_footprint': 'outside initial particle XY AABB expanded by 0.1 m', 'voxel': 'fixed world grid 0.1 m; initial bottom cells with >=32 particles'}, 'initial_count': len(q0), 'initial_bounds': [q0.min(0).tolist(), q0.max(0).tolist()], 'snapshots': []}
bottom = q0[:, 2] < .2
result['initial_bottom_count'] = int(bottom.sum())
v0, n0 = np.unique(np.floor(q0[q0[:, 2] < .2]/.1).astype(int), axis=0, return_counts=True)
for path in sorted(out.glob('diagnostic-*.npz')):
    a = np.load(path)
    q = a['particle_q']
    d = np.linalg.norm(q-q0, axis=1)
    outside = ((q[:, :2] < xylo-.1) | (q[:, :2] > xyhi+.1)).any(1)
    vf, nf = np.unique(np.floor(q[q[:, 2] < .2]/.1).astype(int), axis=0, return_counts=True)
    counts = {tuple(v): int(n) for v,n in zip(vf,nf)}
    deficits = []
    for v,n in zip(v0,n0):
        now = counts.get(tuple(v),0)
        if n >=32 and now < .25*n:
            deficits.append({'cell_lower_xyz_m': (v*.1).tolist(), 'initial_count': int(n), 'current_count': now})
    bad = ~np.isfinite(a['particle_Jp'])
    result['snapshots'].append({'file': path.name, 'particle_count': len(q), 'flags_unchanged': bool(np.array_equal(a0['particle_flags'], a['particle_flags'])), 'nonfinite_fields': {k:int((~np.isfinite(a[k])).sum()) for k in a.files if a[k].dtype.kind in 'fc' and not np.isfinite(a[k]).all()}, 'Jp_positive_inf': int(np.isposinf(a['particle_Jp']).sum()), 'bottom_displacement_median_p99_max_m': np.quantile(d[bottom],[.5,.99,1]).tolist(), 'bottom_moved_over_20cm': int((bottom & (d>.2)).sum()), 'bottom_outside_footprint': int((bottom & outside).sum()), 'all_outside_footprint': int(outside.sum()), 'bottom_Jp_nonfinite': int((bottom & bad).sum()), 'bottom_depleted_cells': sorted(deficits,key=lambda x:x['initial_count']-x['current_count'],reverse=True)})
qprev = np.load(out/'diagnostic-001193.npz')['particle_q']
qfinal = np.load(out/'diagnostic-001329.npz')['particle_q']
dlast = np.linalg.norm(qfinal-qprev,axis=1)
result['last_136_zero_command_frames'] = {'all_moved_over_20cm': int((dlast>.2).sum()), 'initial_bottom_moved_over_20cm': int((bottom & (dlast>.2)).sum()), 'initial_bottom_displacement_p99_max_m': np.quantile(dlast[bottom],[.99,1]).tolist()}
records = [json.loads(l) for l in (out/'projection-top20.ndjson').read_text().splitlines()]
events = []
for row in records:
    for i,p,b,a,d in zip(row['ids'], range(len(row['ids'])),row['before'],row['after'],row['delta_m']):
        if row['frame']>=1194 and d>.02:
            events.append({'frame':row['frame'],'particle_id':i,'before':b,'after':a,'delta_m':d,'initial_z_m':float(q0[i,2])})
result['late_top20_projection_events_over_2cm'] = {'count':len(events),'before_z_below_20cm':sum(e['before'][2]<.2 for e in events),'initial_z_below_20cm':sum(e['initial_z_m']<.2 for e in events),'largest': sorted(events,key=lambda e:e['delta_m'],reverse=True)[:20]}
result['limits'] = ['Particle counts are not calibrated mass or volume.', 'Fixed-cell occupancy and particle displacement do not uniquely identify screenshot hole or cause.', 'Top20 projection log is not a census of all corrected particles.', 'No control/physics causal intervention was performed.']
(out/'particle-migration-audit.json').write_text(json.dumps(result,indent=2)+'\n')
for row in result['snapshots']:
    print({k:v if k!='bottom_depleted_cells' else v[:4] for k,v in row.items()})
print('LAST',result['last_136_zero_command_frames'])
print('PROJECTION',result['late_top20_projection_events_over_2cm'])
