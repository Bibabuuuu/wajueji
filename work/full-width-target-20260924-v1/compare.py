"""Compare fixed source terrain, without replaying or relabelling expert data."""
import hashlib,json,itertools
from pathlib import Path
import numpy as np
from planning.contact_footprint import cut_opportunity
from layered_47_cycle_selection import rank_workspace_candidates
ROOT=Path('/home/qunwang/wajueji');HERE=Path(__file__).resolve().parent
b=json.loads((HERE/'baseline.json').read_text());terrain=b['terrain']
assert hashlib.sha256(Path(b['source']['path']).read_bytes()).hexdigest()==b['source']['sha256']
old=b['candidates'][0]
new=rank_workspace_candidates(terrain,1)
plan=json.loads((ROOT/'runs/full-width-target-20260924-v1/new-plan/report.json').read_text())
selected=plan['plan']['selected']
comparison=[]
for c in new:
 comparison.append(dict(anchor=c['entry_xy_m'],contact=c['contact']['contact_xy_m'],
  old_score=c['legacy_height_score'],opportunity=c['cut_opportunity'],
  feasible_chosen=c['entry_xy_m']==selected['entry_xy_m']))
sensitivity=[]
for length,below,band in itertools.product((.75,.90,1.05),(.25,.35,.45),(.60,.70,.80)):
 scores=[cut_opportunity(terrain,c['contact'],length=length,below_entry=below,band_height=band)['score_m3'] for c in new]
 winner=new[int(np.argmax(scores))]
 sensitivity.append(dict(length=length,below_entry=below,band_height=band,contact=winner['contact']['contact_xy_m']))
# Project the operator's curl-start heading onto the ORIGINAL entry radius and
# height to compare the same geometric query. This is not the actual tooth entry.
entry=json.loads((ROOT/'work/fast-third-scoop-review-20260924-v1/entry-review.json').read_text())
operator=next(x for x in entry['observations'] if x['label']=='curl_load')['center_reference_world_m']
direction=np.array(operator[:2])/np.linalg.norm(operator[:2])
projected=dict(old['contact'],contact_xy_m=(direction*np.linalg.norm(old['contact']['contact_xy_m'])).tolist())
report=dict(schema='full_width_target_comparison_v1',source=b['source'],physics_steps=0,
 old_target=old['contact']['contact_xy_m'],new_feasible_target=selected['contact']['contact_xy_m'],
 old_opportunity=cut_opportunity(terrain,old['contact']),new_opportunity=selected['cut_opportunity'],
 candidates=comparison,prepoint_rejections=plan['plan']['rejected'],
 projected_operator_heading=dict(contact=projected['contact_xy_m'],opportunity=cut_opportunity(terrain,projected),
  scope='Retrospective heading projection at old radius/height, not actual entry and never an online feature.'),
 sensitivity=sensitivity,all_27_rank_winners_toward_operator=all(x['contact'][1]>old['contact']['contact_xy_m'][1] for x in sensitivity),
 limits=['heightfield geometry is not mass or guaranteed fill','manual relocation is retained separately; no label rewriting',
         'existing face admission/search envelope retained','sensitivity compares score ranking; each winner still needs safe prepoint'])
(HERE/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('old',report['old_target'],'new',report['new_feasible_target'],'robust',report['all_27_rank_winners_toward_operator'])
