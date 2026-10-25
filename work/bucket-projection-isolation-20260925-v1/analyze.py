"""Read-only strict fill, solid penetration and clustering audit of saved outputs."""
import argparse, json
from pathlib import Path
import numpy as np
from live_bucket_fill import load_config, MixedFillEstimator
from layered_excavation_approach_planner import _load_kinematics
import run_layered_47_sparse_manual_contact_episode as manual
import run_layered_47_sparse_motion as motion
from moving_bucket_sweep_candidate import convex_planes

root=Path('/home/qunwang/wajueji'); work=Path(__file__).resolve().parent
bundle=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1')
kin=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
parser=argparse.ArgumentParser();parser.add_argument('--run',default='ablation-600');cli=parser.parse_args();run=work/cli.run
report=json.loads((run/'report.json').read_text());frames=report['frames']
base=motion.probe.select(47)[0](bundle/'repo/configs/v4/layered-47-excavation-preflight-v1.json',bundle/'repo')
identity=motion.probe.select(47)[1];manifest=json.loads((bundle/'repo'/base['physical_geometry']['manifest_path']).read_text())
records=[r for r in manifest['rocks'] if r['layer_id'] in identity['layers']]
verts=manual._cache_rock_vertices(bundle/'repo',records)
est=MixedFillEstimator(load_config(root/'configs/v4/live-bucket-fill-local-v1.json'),[r['volume_m3'] for r in records],verts)
source=root/'runs/layered-47-loaded-handoff-v1/d129-loaded-handoff-20260923-v2/loaded-handoff.npz'
with np.load(source) as data:q0=data['particle_q'].copy();r0=data['rigid_qpos'].copy()
selection=np.load(run/'tracking-selection.npz');ids=selection['ids'];initial=selection['initial_cavity']
positions=np.load(run/'post-projection-world.npy',mmap_mode='r');poses=np.load(run/'actual-bucket-poses.npy',mmap_mode='r')
rows=[json.loads(x) for x in (run/'trace.ndjson').read_text().splitlines()]
measurements=[]
for f in sorted(set([0,*range(60,frames+1,60),frames])):
    q=q0.copy();q[ids]=positions[f];rocks=r0 if f==0 else np.asarray([r['qpos'] for r in rows[f-1]['after_state']['rocks']])
    fill=est.measure_snapshot({'frame':f,'soil':q,'pose':poses[f],'rocks':rocks})
    local=(positions[f][initial]-poses[f,:3,3])@poses[f,:3,:3]
    _,counts=np.unique(np.floor(local/.02).astype(int),axis=0,return_counts=True)
    measurements.append({'frame':f,'fill_percent':fill['fill_percent'],'liters':fill['liters'],'valid_estimate':fill['valid_estimate'],'occupied_2cm_cells':len(counts),'max_particles_per_cell':int(counts.max())})
terminal=report.get('terminal',report.get('recovery'))
with np.load(terminal['path']) as data:
    q=data['particle_q'];pose=poses[frames];local=(q-pose[:3,3])@pose[:3,:3]
    q_reconstructed=q0.copy();q_reconstructed[ids]=positions[frames]
    proxy=json.loads((root/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json').read_text())
    o,n,d,_=convex_planes(None,proxy);n=np.asarray(n);d=np.asarray(d);inside=np.zeros(len(q),bool)
    for i in range(len(o)-1): inside |= np.min(d[o[i]:o[i+1]]-local@n[o[i]:o[i+1]].T,axis=1)>1.e-4
    complete=est.measure_snapshot({'frame':frames,'soil':q,'pose':pose,'rocks':np.asarray([r['qpos'] for r in rows[-1]['after_state']['rocks']])})
result={'schema':'bucket_projection_isolation_audit_v1','status':report['status'],'frames':frames,'wall_seconds':report['wall_seconds'],'measurements':measurements,'terminal_full_particle_fill':complete,'terminal_solid_penetration_count_above_0_1mm':int(inside.sum()),'initial_cavity_particle_count':int(initial.sum()),'retained_initial_cavity':rows[-1]['original_cavity_particles_still_in_cavity'],'all_hard_guards':all(all(x['hard'].values()) for x in rows),'all_material_guards':all(x['material_safe'] for x in rows),'bindings_unchanged':report['bindings_unchanged']}
(run/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
