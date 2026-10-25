"""Bind independently audited derived episodes, retaining original split/lineage."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
ORIGINAL=ROOT/'work/contact-bc-data-preparation-20260924-v1'
SAMPLE=ROOT/'work/contact-derived-demonstration-20260924-v1'
BAD='manual-contact-20260924-093202-1362929';DONE='manual-contact-20260924-115812-1458627'
FAILED='manual-contact-20260924-005742-1081585'
DIAGNOSTIC='manual-contact-20260924-100857-1387193'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def identity(p):return dict(path=str(p),sha256=sha(p))
def main():
 original=read(ORIGINAL/'tensor-manifest.json');assignment=read(ORIGINAL/'pilot-assignment.json');episodes=[]
 for e in original['episodes']:
  run=e['run']
  if run in (BAD,FAILED):continue
  if run==DONE:
   vp=SAMPLE/'verification-v2.json';fp=SAMPLE/'fill-comparison-v2.json'
  else:vp=W/(run+'.verification.json');fp=W/(run+'.fill.json')
  verified=read(vp);fill=read(fp);assert verified['passed'] and verified['teacher_run']==run
  rp=Path(verified['report']['path']);assert sha(rp)==verified['report']['sha256'];r=read(rp)
  assert sha(r['teacher_tensor']['path'])==e['sha256'] and r['lineage']==e['lineage']
  assert r['source_unchanged'] and r['teacher_tensor_unchanged']
  tensor=r['tensors'];assert sha(tensor['path'])==tensor['sha256'] and tensor['rows']==e['rows']
  split=next(k for k in ('development_runs','validation_runs') if run in assignment[k]);assert r['split']==split
  assert fill['original']['valid_estimate'] and fill['derived']['valid_estimate']
  # This manifest records diagnostic admission candidates; no success label is invented.
  episodes.append(dict(run=run,path=tensor['path'],sha256=tensor['sha256'],rows=tensor['rows'],lineage=e['lineage'],
   split=split,data_domain='replayed_teacher_actual_observations',phase_counts=tensor['phase_counts'],
   source_teacher_tensor=e,independent_audit=identity(vp),fill_audit=identity(fp),report=identity(rp),
   off_soil_confirmation_frame_end=verified['off_soil_confirmation_frame_end'],
   post_confirmation_below_gap_frames=verified['post_confirmation_below_gap_frames'],
   actual_fill_percent=fill['derived']['fill_percent'],fill_difference_percentage_points=fill['fill_difference_percentage_points'],
   maximum_rigid_deviations=r['maximum_deviations'],endpoint_comparison=r['endpoint_comparison']))
 assert len(episodes)==8
 manifest=dict(schema='contact_bc_derived_causal_phase_structured_v1',episodes=episodes,
  source_original_manifest=identity(ORIGINAL/'tensor-manifest.json'),original_assignment=identity(ORIGINAL/'pilot-assignment.json'),
  excluded_runs=[dict(run=BAD,reason='recorded_CAD_ground_penetration',evidence=identity(W/'excluded-ground-demo.json')),
   dict(run=FAILED,reason='observer_grid_budget_failure_with_three_high_altitude_particles',evidence=identity(W/'excluded-observer-failure.json'))],
  original_data_changed=False,training_eligible=False,scope='Eight independently audited causal derived-data candidates; phase and final-result review required before pilot fitting.')
 mp=W/'manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
 derived=dict(assignment,source_manifest_sha256=sha(mp),model_route='causal_five_phase_conditioned_shared_four_axis_short_history',
  original_assignment=identity(ORIGINAL/'pilot-assignment.json'),excluded_ground_runs=[BAD],excluded_incomplete_runs=[FAILED],
  steps=1000,training_wall_budget_seconds=300)
 for key in ('development_runs','validation_runs'):derived[key]=[r for r in derived[key] if r not in (BAD,FAILED)]
 derived['episode_weights']={k:v for k,v in derived['episode_weights'].items() if k not in (BAD,FAILED)}
 derived['episode_weights'][DIAGNOSTIC]=0.0
 derived['diagnostic_only_runs']={DIAGNOSTIC:'Rock center differs by 0.351m during lift, despite similar fill and bucket geometry; retain audited actual tensors but exclude fitting and normalization in this first pilot.'}
 derived['validation_domain_caveats']={'manual-contact-20260924-113555-1443279':'Original validation membership retained. A falling outside rock differs by 1.450m while bucket CAD differs by 0.143mm. Treat as actual replay-state action validation, not faithful full-scene reproduction or original-human closed-loop evidence.'}
 (W/'assignment.json').write_text(json.dumps(derived,indent=2)+'\n')
 print('Audited derived episodes:',len(episodes),'rows:',sum(e['rows'] for e in episodes))
if __name__=='__main__':main()
